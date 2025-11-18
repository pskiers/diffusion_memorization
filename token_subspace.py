import os
import importlib
import itertools
import random

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from accelerate import dispatch_model
import torch
import torch.nn.functional as F
import torchvision
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from optim_utils import *
import numpy as np


class ProcessorGradientFlow():
    """
    This wraps the huggingface CLIP processor to allow backprop through the image processing step.
    The original processor forces conversion to numpy then PIL images, which is faster for image processing but breaks gradient flow.
    """
    def __init__(self, device="cuda") -> None:
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]
        self.normalize = torchvision.transforms.Normalize(
            self.image_mean,
            self.image_std
        )
        self.resize = torchvision.transforms.Resize(224)
        self.center_crop = torchvision.transforms.CenterCrop(224)
    def preprocess_img(self, images):
        images = self.center_crop(images)
        images = self.resize(images)
        images = self.center_crop(images)
        images = self.normalize(images)
        return images
    def __call__(self, images=[], **kwargs):
        processed_inputs = self.processor(images=images, **kwargs)
        processed_inputs["pixel_values"] = self.preprocess_img(images)
        processed_inputs = {key:value.to(self.device) for (key, value) in processed_inputs.items()}
        return processed_inputs


class SubspaceGetter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipe = self.setup_model(cfg)
        self.done_grads_dict = self.create_output_dir()
        set_random_seed(cfg.seed)

    def setup_model(self, cfg):
        # Import model class
        module_path, class_name = cfg.model.model_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
        # setup the model with the local diffusers pipeline
        pipe = ModelClass.from_pretrained(
            cfg.model.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            # variant="fp16",
            cache_dir="../model_cache",
        )
        pipe.vae = AutoencoderKL.from_pretrained(
            cfg.model.model_id,
            subfolder="vae",
            # variant="fp16",
            torch_dtype=torch.float32,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        # pipe.enable_sequential_cpu_offload()
        if cfg.lora_path is not None:
            print(f"Loading LoRA weights from {cfg.lora_path}")
            pipe.load_lora_weights(cfg.lora_path)
            pipe.fuse_lora()

        if cfg.model.device_map is not None:
            pipe.unet = dispatch_model(pipe.unet, device_map=cfg.model.device_map)
            pipe.text_encoder = dispatch_model(pipe.text_encoder, device_map=cfg.model.device_map)
            pipe.text_encoder_2 = dispatch_model(pipe.text_encoder_2, device_map=cfg.model.device_map)
            pipe.vae = dispatch_model(pipe.vae, device_map=cfg.model.device_map)

        if cfg.model.device_map is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            print(device)
            pipe = pipe.to(device)

        pipe.scheduler.set_timesteps(cfg.model.num_inference_steps)

        if cfg.method == "clip":
            model_id = "openai/clip-vit-base-patch32"
            self.clip = CLIPModel.from_pretrained(model_id, cache_dir="../model_cache").to("cuda")
            self.clip_image_processor = ProcessorGradientFlow(device="cuda")
        return pipe

    @staticmethod
    def image_iterator(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                img_path = os.path.join(folder_path, filename)
                yield Image.open(img_path), filename

    def create_output_dir(self):
        self.out_dir = self.cfg.out_dir
        self.name = self.cfg.name
        self.timesteps = self.cfg.data.timesteps

        output_dir = os.path.join(self.out_dir, self.name)
        grads_dir = os.path.join(output_dir, "grads")
        config_path = os.path.join(output_dir, "config.yaml")

        if os.path.exists(output_dir):
            # grads folder may not exist
            grad_dict = {}
            if os.path.exists(grads_dir):
                for timestep in self.timesteps:
                    grad_dict[timestep] = {}
                    timestep_dir = os.path.join(grads_dir, f"t{timestep}")
                    if not os.path.exists(timestep_dir):
                        os.makedirs(timestep_dir, exist_ok=True)
                        continue
                    for fname in os.listdir(timestep_dir):
                        if "_to_" in fname:
                            if fname.endswith(".npy"):
                                fname = fname[:-4]
                            name1, name2 = fname.split("_to_")
                            name2 = name2  # already split
                            if name1 not in grad_dict[timestep]:
                                grad_dict[timestep][name1] = []
                            grad_dict[timestep][name1].append(name2)
            return grad_dict
        else:
            os.makedirs(grads_dir, exist_ok=True)
            for timestep in self.timesteps:
                timestep_dir = os.path.join(grads_dir, f"t{timestep}")
                os.makedirs(timestep_dir, exist_ok=True)
            # Save hydra config to config.yaml
            with open(config_path, "w") as f:
                f.write(OmegaConf.to_yaml(self.cfg))
            return {t: {} for t in self.timesteps}

    def encode_image(self, img):
        with torch.no_grad():
            # img_low_res = img.resize((256, 256))
            img = self.pipe.image_processor.preprocess(img, width=1024, height=1024).to(dtype=self.pipe.vae.dtype)
            # img = self.pipe.image_processor.preprocess(img).to(dtype=pipe.vae.dtype)
            latent = self.pipe.vae.encode(img).latent_dist.sample()
            latent = latent * self.pipe.vae.config.scaling_factor
            latent = latent.to(dtype=torch.float16)
        return latent

    def noise_image(self, latent, timestep):
        t_noise = torch.tensor(self.pipe.scheduler.timesteps[-timestep].item())  # -t because timesteps are in decreasing order
        noise = torch.randn_like(latent)
        return self.pipe.scheduler.add_noise(latent, noise, t_noise).detach()

    def do_svd(self):
        subspace_sizes = {t: 0 for t in self.timesteps}
        for t in self.timesteps:
            grads_dir = os.path.join(self.out_dir, self.name, "grads", f"t{t}")
            grad_files = [f for f in os.listdir(grads_dir) if f.endswith(".npy")]

            grads = []
            for fname in grad_files:
                grad = np.load(os.path.join(grads_dir, fname))
                grads.append(grad.flatten())

            grad_matrix = np.stack(grads, axis=0).astype(np.float32)
            print(grad_matrix.shape)
            u, s, vh = np.linalg.svd(grad_matrix, full_matrices=False)

            # Filter out small singular values (eigenvalues)
            threshold = 1e-3  # You can adjust this threshold
            import matplotlib.pyplot as plt

            # Sort eigenvalues from biggest to smallest
            sorted_s = np.sort(s)[::-1]

            # Plot
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(sorted_s)), sorted_s)
            plt.xlabel("Eigenvalue Index")
            plt.ylabel("Eigenvalue Magnitude")
            plt.title(f"Eigenvalues for timestep {t}")
            plt.tight_layout()

            # Save plot
            eigvals_plot_path = os.path.join(self.out_dir, self.name, f"eigvals_t{t}_barplot.png")
            plt.savefig(eigvals_plot_path)
            plt.close()

            significant = (s > threshold)
            subspace_sizes[t] = int(np.sum(significant))

            print(f"Subspace size (number of significant directions): {subspace_sizes[t]}")
            eigvals_path = os.path.join(self.out_dir, self.name, f"eigvals_t{t}.npy")
            np.save(eigvals_path, s)

        subspace_size_path = os.path.join(self.out_dir, self.name, "subspace_size.txt")
        with open(subspace_size_path, "w") as f:
            f.write(str(subspace_sizes))

    @staticmethod
    def find_significant_directions(
        token_grads: torch.Tensor,
        n_directions: int = None,
        significance_threshold: float = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs PCA on a set of vectors and returns significant components.

        Also returns the total variance and covariance matrix needed for
        cross-dataset comparisons.

        Args:
            token_grads (torch.Tensor): A tensor of shape (num_vect, vect_dim).
            n_directions (int, optional): Returns the top 'n' principal directions.
            significance_threshold (float, optional): Returns all directions with
                an explained variance ratio greater than this threshold.
                (n_directions takes precedence if both are specified).

        Returns:
            A tuple containing:
            - directions (torch.Tensor): Tensor of shape (k, vect_dim) where 'k' is
            the number of directions found. Each row is a significant direction.
            - significances (torch.Tensor): Tensor of shape (k,) with the explained
            variance ratio for each corresponding direction.
            - total_variance (torch.Tensor): A scalar tensor representing the sum
            of all eigenvalues (total variance in the data).
            - covariance_matrix (torch.Tensor): The covariance matrix (C) of shape
            (vect_dim, vect_dim).
        """
        if n_directions is not None and significance_threshold is not None:
            print("Warning: 'n_directions' takes precedence over 'significance_threshold'.")
            significance_threshold = None

        num_vect, vect_dim = token_grads.shape

        # --- 1. Perform PCA ---

        # For PCA, data should be centered (mean of 0)
        # Note: Use keepdim=True for proper broadcasting
        centered_data = token_grads - token_grads.mean(dim=0, keepdim=True)

        # Handle the "N < D" case by computing covariance on the (N, N) matrix
        # This is much faster and mathematically equivalent for this scenario.
        if num_vect < vect_dim:
            # C_small = (1/(N-1)) * X_centered @ X_centered.T   (shape N, N)
            covariance_matrix_small = (1.0 / (num_vect - 1)) * (centered_data @ centered_data.T)

            # Get eigenvalues and eigenvectors of the (N, N) matrix
            eigenvalues, eigenvectors_small = torch.linalg.eigh(covariance_matrix_small)

            # Sort eigenvalues in descending order
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            sorted_eigenvalues = eigenvalues[sorted_indices]

            # Convert eigenvectors from (N, N) to (D, N)
            # eigenvector_D = (1 / sqrt(eigenvalue * (N-1))) * X_centered.T @ eigenvector_N
            # We only care about non-zero eigenvalues
            valid_indices = sorted_indices[sorted_eigenvalues > 1e-9]
            valid_eigenvalues = sorted_eigenvalues[valid_indices]
            valid_eigenvectors_small = eigenvectors_small[:, valid_indices]

            # Calculate the (D, k) eigenvectors
            scale_factors = (1.0 / torch.sqrt(valid_eigenvalues * (num_vect - 1)))
            # X_centered.T is (D, N), valid_eigenvectors_small is (N, k)
            # scale_factors must be broadcast from (k,) to (1, k)
            sorted_eigenvectors = centered_data.T @ valid_eigenvectors_small * scale_factors.unsqueeze(0)
            sorted_eigenvalues = valid_eigenvalues

            # Also need the full (D, D) covariance matrix for cross-analysis
            # This is the only slow part, but necessary for the user's request.
            covariance_matrix = (1.0 / (num_vect - 1)) * (centered_data.T @ centered_data)

        else:
            # Standard case (N >= D)
            # C = (1/(N-1)) * X_centered.T @ X_centered   (shape D, D)
            covariance_matrix = (1.0 / (num_vect - 1)) * (centered_data.T @ centered_data)

            # Get eigenvalues and eigenvectors of the (D, D) matrix
            # .eigh is for symmetric matrices (like cov) and is more stable
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

            # Sort eigenvalues and corresponding eigenvectors in descending order
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            sorted_eigenvalues = eigenvalues[sorted_indices]
            # Eigenvectors are columns, so we reorder the columns
            sorted_eigenvectors = eigenvectors[:, sorted_indices]


        # --- 2. Calculate Significance ---
        total_variance = torch.sum(sorted_eigenvalues)

        # Avoid division by zero for zero-variance data
        if total_variance == 0:
            return (torch.empty((0, vect_dim)), torch.empty((0,)),
                    total_variance, covariance_matrix)

        explained_variances = sorted_eigenvalues / total_variance

        # --- 3. Filter Results ---
        if n_directions is not None:
            num_to_return = min(n_directions, sorted_eigenvectors.shape[1])
            # Eigenvectors are columns (dim, k), transpose to (k, dim)
            top_k_directions = sorted_eigenvectors[:, :num_to_return].T
            top_k_variances = explained_variances[:num_to_return]
            return top_k_directions, top_k_variances, total_variance, covariance_matrix

        elif significance_threshold is not None:
            # Create a boolean mask for directions meeting the threshold
            mask = explained_variances > significance_threshold
            filtered_directions = sorted_eigenvectors[:, mask].T
            filtered_variances = explained_variances[mask]
            return filtered_directions, filtered_variances, total_variance, covariance_matrix

        else:
            # Default: return all sorted, non-zero directions
            return sorted_eigenvectors.T, explained_variances, total_variance, covariance_matrix

    @staticmethod
    def remove_direction(vectors: torch.Tensor, direction: torch.Tensor):
        """
        Removes the component of a direction from a set of vectors.

        This is done by subtracting the projection of each vector onto the direction.
        The resulting vectors will be orthogonal to the given direction.

        Args:
            vectors (torch.Tensor): A tensor of shape (vect_dim,) or (num_vect, vect_dim)
                                    containing the vector(s) to modify.
            direction (torch.Tensor): A 1D tensor of shape (vect_dim,) representing the
                                    direction to remove.

        Returns:
            torch.Tensor: The modified vectors with the direction component removed.
        """
        # 1. Ensure the direction is a unit vector (length 1) for the projection formula.
        # This makes the calculation robust even if the input direction isn't normalized.
        if torch.linalg.norm(direction) == 0:
            return vectors # Cannot project onto a zero vector, return original
        direction_unit = direction / torch.linalg.norm(direction)

        # 2. Calculate the projection of the vectors onto the unit direction.
        # The dot product gives the magnitude of the projection.
        # For a batch of vectors, matmul `@` efficiently calculates all dot products.
        dot_products = vectors @ direction_unit

        # Reshape dot products to allow broadcasting for the final multiplication
        if vectors.dim() > 1:
            dot_products = dot_products.unsqueeze(-1)

        projections = dot_products * direction_unit

        # 3. Subtract the projection from the original vectors.
        vectors_after_removal = vectors - projections

        return vectors_after_removal

    def get_token_pos(self, prompt, token):
        tokens = self.pipe.tokenizer.encode(prompt)
        tokens = tokens[:77]
        all_tokes = {self.pipe.tokenizer.decode(curr_token): i for i, curr_token in enumerate(tokens)}
        return all_tokes[token]

    def run(self):
        prompt = self.cfg.data.prompt
        token = self.cfg.data.token

        images = list(self.image_iterator(self.cfg.data.image_folder))
        all_index_pairs = list(itertools.product(range(len(images)), range(len(images))))
        if self.cfg.randomize_pairs:
            random.shuffle(all_index_pairs)

        for t in self.timesteps:
            print(f"Processing timestep {t}")
            num_processed = sum(len(v) for v in self.done_grads_dict[t].values())
            for i, j in all_index_pairs:
                img, name = images[i]
                ref_img, ref_name = images[j]
                if j <= i:
                    continue
                if ref_name in self.done_grads_dict[t].get(name, []):
                    continue
                if num_processed > self.cfg.max_pairs:
                    break

                print(f"  Image {name} to {ref_name}\r", end="")

                latent = self.encode_image(img)
                noised_latent = self.noise_image(latent, t)

                denoised_latent, prompt_tokens = self.pipe.get_pred_denoised_image(
                    prompt,
                    latents=noised_latent,
                    num_inference_steps=self.cfg.model.num_inference_steps,
                    guidance_scale=self.cfg.model.guidance_scale,
                    num_images_per_prompt=1,
                    start_timestep=self.cfg.model.num_inference_steps-t,
                    output_type="pt" if self.cfg.method == "clip" else "latent",
                )

                if self.cfg.method == "clip":
                    ref_img = self.pipe.image_processor.preprocess(
                        ref_img, width=1024, height=1024
                    ).to(dtype=self.clip.dtype, device=denoised_latent.device)
                    batch = torch.cat([denoised_latent, ref_img], dim=0)
                    batch = (batch + 1.0) / 2.0
                    inputs = self.clip_image_processor(
                        images=batch, return_tensors="pt"
                    )["pixel_values"].to(self.clip.device)
                    outputs = self.clip.get_image_features(pixel_values=inputs)
                    denoised_img_feat, ref_img_feat = outputs[:2]
                    loss = F.mse_loss(denoised_img_feat.float(), ref_img_feat.float())
                elif self.cfg.method == "latent":
                    loss = F.mse_loss(denoised_latent.float(), self.encode_image(ref_img).float())
                else:
                    raise ValueError(f"Unknown method {self.cfg.method}")

                tokens_grad = torch.autograd.grad(loss, [prompt_tokens], allow_unused=True)[0]

                real_tokens_grad = tokens_grad[
                    0, :len(self.pipe.tokenizer.encode(prompt))
                ].to(torch.float).cpu()

                if self.cfg.remove_common:
                    common = real_tokens_grad.mean(dim=0, keepdim=True)
                    real_tokens_grad = real_tokens_grad - common

                directions, _, _, _= self.find_significant_directions(
                    real_tokens_grad,
                    significance_threshold=2.5 / min(real_tokens_grad.shape)
                )

                token_grad = tokens_grad[:, self.get_token_pos(prompt, token), :]
                token_grad = token_grad.cpu()

                if self.cfg.remove_significant_directions:
                    for direction in directions:
                        token_grad = self.remove_direction(token_grad, direction.to(token_grad.dtype))

                token_grad = token_grad.numpy()

                grad_path = os.path.join(self.out_dir, self.name, "grads", f"t{t}", f"{name}_to_{ref_name}.npy")
                np.save(grad_path, token_grad)
                num_processed += 1

                if name not in self.done_grads_dict[t]:
                    self.done_grads_dict[t][name] = []
                self.done_grads_dict[t][name].append(ref_name)

                del denoised_latent, prompt_tokens
                torch.cuda.empty_cache()
        self.do_svd()


@hydra.main(version_base=None, config_path="configs", config_name="token_subspace")
def main(cfg: DictConfig):
    cfg = instantiate(cfg)
    subspace_getter = SubspaceGetter(cfg)
    subspace_getter.run()


from scipy.linalg import orth
from numpy.linalg import svd


def compare_subspaces_pca(dirA, dirB, prompt, token, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def load_grads(grads_dir):
        grad_files = [f for f in os.listdir(grads_dir) if f.endswith(".npy")]
        grads = []
        filenames = []
        for fname in grad_files:
            grad = np.load(os.path.join(grads_dir, fname))
            grads.append(grad.flatten())
            filenames.append(fname)
        grad_matrix = np.stack(grads, axis=0).astype(np.float32)
        grad_matrix = torch.from_numpy(grad_matrix)
        return grad_matrix, filenames

    # Load grads
    gradsA, filenames1 = load_grads(dirA)
    gradsB, filenames2 = load_grads(dirB)

    threshold = 2.5 / min(gradsA.shape)

    # PCA
    D_A, S_A, V_A, C_A = SubspaceGetter.find_significant_directions(gradsA, significance_threshold=threshold)
    D_B, S_B, V_B, C_B = SubspaceGetter.find_significant_directions(gradsB, significance_threshold=threshold)

    print(f"\nSet A: Found {D_A.shape[0]}, set B: Found {D_B.shape[0]} significant directions.\n")

    # --- 3. Test A in B ---
    print("--- Testing A's directions in Set B ---")
    common = []
    exclusive = []
    diffs = ""
    for i, d_a in enumerate(D_A):
        # d_a has shape (dim,). Need to make it (1, dim) and (dim, 1) for matmul
        d_a_col = d_a.unsqueeze(1) # (dim, 1)
        d_a_row = d_a.unsqueeze(0) # (1, dim)

        # Variance = d_a.T @ C_B @ d_a
        explained_var_in_B = d_a_row @ C_B @ d_a_col
        significance_in_B = explained_var_in_B / V_B

        # print(f"A's Dir {i} (Original Sig: {S_A[i]:.4f}) -> Sig in B: {significance_in_B.item():.4f}")
        if significance_in_B > threshold: # Your significance threshold
            common.append(i)
            # print("  -> This direction is ALSO significant in Set B.")
        else:
            exclusive.append(i)
            print(f"\t\tA's Dir {i} (Original Sig: {S_A[i]:.4f}) -> Sig in B: {significance_in_B.item():.4f}")
            diffs += f"SDXL's Dir {i} (Original Sig: {S_A[i]:.4f}) -> Sig in SDXL turbo: {significance_in_B.item():.4f}\n"
    print(f"\tExclusive directions in A: {len(exclusive)}, Common directions: {len(common)}")

    with open(os.path.join(output_dir, "directions_sdxl.txt"), "w") as f:
        f.write(str({"all": len(exclusive)+len(common), "exclusive": len(exclusive), "common": len(common)}))
        f.write("\n")
        f.write(diffs)

    for i, idx in enumerate(exclusive):
        if i >= 8:
            break
        intervention = D_A[idx]
        experiment_with_tokens(
            prompt=prompt,
            token=token,
            token_intervention=intervention,
            intervention_strenghts=[0, 10, 15, 20, 30, 50, 70],
            output_path=f"{output_dir}/sdxl_turbo_exclusive_sdxl_to_sdxl_turbo_{i}.png",
            model_id="stabilityai/sdxl-turbo",
            num_inference_steps=4,
            guidance_scale=0.0,
        )
        experiment_with_tokens(
            prompt=prompt,
            token=token,
            token_intervention=intervention,
            intervention_strenghts=[0, 10, 15, 20, 30, 50, 70],
            output_path=f"{output_dir}/sdxl_exclusive_sdxl_to_sdxl_turbo_{i}.png",
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            num_inference_steps=50,
            guidance_scale=7.0,
        )

    for i, idx in enumerate(common):
        if i >= 8:
            break
        intervention = D_A[idx]
        experiment_with_tokens(
            prompt=prompt,
            token=token,
            token_intervention=intervention,
            intervention_strenghts=[0, 10, 15, 20, 30, 50, 70],
            output_path=f"{output_dir}/sdxl_turbo_common_sdxl_to_sdxl_turbo_{i}.png",
            model_id="stabilityai/sdxl-turbo",
            num_inference_steps=4,
            guidance_scale=0.0,
        )
        experiment_with_tokens(
            prompt=prompt,
            token=token,
            token_intervention=intervention,
            intervention_strenghts=[0, 10, 15, 20, 30, 50, 70],
            output_path=f"{output_dir}/sdxl_common_sdxl_to_sdxl_turbo_{i}.png",
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            num_inference_steps=50,
            guidance_scale=7.0,
        )


    # --- 4. Test B in A ---
    print("\n--- Testing B's directions in Set A ---")
    common = []
    exclusive = []
    diffs = ""
    for j, d_b in enumerate(D_B):
        d_b_col = d_b.unsqueeze(1)
        d_b_row = d_b.unsqueeze(0)

        explained_var_in_A = d_b_row @ C_A @ d_b_col
        significance_in_A = explained_var_in_A / V_A

        # print(f"B's Dir {j} (Original Sig: {S_B[j]:.4f}) -> Sig in A: {significance_in_A.item():.4f}")
        if significance_in_A > threshold: # Your significance threshold
            common.append(j)
            # print("  -> This direction is ALSO significant in Set A.")
        else:
            exclusive.append(j)
            print(f"\t\tB's Dir {j} (Original Sig: {S_B[j]:.4f}) -> Sig in A: {significance_in_A.item():.4f}")
            diffs += f"SDXL turbo's Dir {j} (Original Sig: {S_B[j]:.4f}) -> Sig in SDXL: {significance_in_A.item():.4f}\n"
    print(f"\tExclusive directions in B: {len(exclusive)}, Common directions: {len(common)}")

    with open(os.path.join(output_dir, "directions_sdxl_turbo.txt"), "w") as f:
        f.write(str({"all": len(exclusive)+len(common), "exclusive": len(exclusive), "common": len(common)}))
        f.write("\n")
        f.write(diffs)

    for i, idx in enumerate(exclusive):
        if i >= 8:
            break
        intervention = D_B[idx]
        experiment_with_tokens(
            prompt=prompt,
            token=token,
            token_intervention=intervention,
            intervention_strenghts=[0, 10, 15, 20, 30, 50, 70],
            output_path=f"{output_dir}/sdxl_turbo_exclusive_sdxl_turbo_to_sdxl_{i}.png",
            model_id="stabilityai/sdxl-turbo",
            num_inference_steps=4,
            guidance_scale=0.0,
        )
        experiment_with_tokens(
            prompt=prompt,
            token=token,
            token_intervention=intervention,
            intervention_strenghts=[0, 10, 15, 20, 30, 50, 70],
            output_path=f"{output_dir}/sdxl_exclusive_sdxl_turbo_to_sdxl_{i}.png",
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            num_inference_steps=50,
            guidance_scale=7.0,
        )

    for i, idx in enumerate(common):
        if i >= 8:
            break
        intervention = D_B[idx]
        experiment_with_tokens(
            prompt=prompt,
            token=token,
            token_intervention=intervention,
            intervention_strenghts=[0, 10, 15, 20, 30, 50, 70],
            output_path=f"{output_dir}/sdxl_turbo_common_sdxl_turbo_to_sdxl_{i}.png",
            model_id="stabilityai/sdxl-turbo",
            num_inference_steps=4,
            guidance_scale=0.0,
        )
        experiment_with_tokens(
            prompt=prompt,
            token=token,
            token_intervention=intervention,
            intervention_strenghts=[0, 10, 15, 20, 30, 50, 70],
            output_path=f"{output_dir}/sdxl_common_sdxl_turbo_to_sdxl_{i}.png",
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            num_inference_steps=50,
            guidance_scale=7.0,
        )


def compare_subspaces_svd():
    dir1 = "/net/scratch/hscra/plgrid/plgpawel269/LID-project/diffusion_memorization/outputs/sdxl-frog/grads/t13"
    dir2 = "/net/scratch/hscra/plgrid/plgpawel269/LID-project/diffusion_memorization/outputs/sdxl-turbo-frog/grads/t1"
    threshold = 1e-3

    def load_grads(grads_dir):
        grad_files = [f for f in os.listdir(grads_dir) if f.endswith(".npy")]
        grads = []
        filenames = []
        for fname in grad_files:
            grad = np.load(os.path.join(grads_dir, fname))
            grads.append(grad.flatten())
            filenames.append(fname)
        grad_matrix = np.stack(grads, axis=0).astype(np.float32)
        return grad_matrix, filenames

    # Load grads
    grads1, filenames1 = load_grads(dir1)
    grads2, filenames2 = load_grads(dir2)

    # SVD
    u1, s1, vh1 = np.linalg.svd(grads1, full_matrices=False)
    u2, s2, vh2 = np.linalg.svd(grads2, full_matrices=False)

    # Significant directions
    sig_idx1 = np.where(s1 > threshold)[0]
    sig_idx2 = np.where(s2 > threshold)[0]
    subspace1 = vh1[sig_idx1]
    subspace2 = vh2[sig_idx2]

    # Orthonormalize
    subspace1_orth = orth(subspace1.T)
    subspace2_orth = orth(subspace2.T)

    # Compute intersection

    M = np.dot(subspace1_orth.T, subspace2_orth)
    _, s, _ = svd(M)
    common_size = np.sum(s > 1e-2)

    print(f"Common subspace size: {common_size}")
    print(f"Subspace1 size: {subspace1_orth.shape[1]}")
    print(f"Subspace2 size: {subspace2_orth.shape[1]}")
    print(f"Exclusive to subspace1: {subspace1_orth.shape[1] - common_size}")
    print(f"Exclusive to subspace2: {subspace2_orth.shape[1] - common_size}")

    # Find exclusive directions
    # Project subspace1_orth onto subspace2_orth and get residuals
    proj1_on_2 = subspace2_orth @ (subspace2_orth.T @ subspace1_orth)
    exclusive1 = subspace1_orth - proj1_on_2
    exclusive1 = orth(exclusive1)
    np.save("exclusive_to_subspace1.npy", exclusive1)

    proj2_on_1 = subspace1_orth @ (subspace1_orth.T @ subspace2_orth)
    exclusive2 = subspace2_orth - proj2_on_1
    exclusive2 = orth(exclusive2)
    np.save("exclusive_to_subspace2.npy", exclusive2)

    # Project each original vector in grads1 onto the exclusive subspace
    projections = grads1 @ exclusive1
    projection_magnitudes = np.linalg.norm(projections, axis=1)

    n = 5
    top_n_idx = np.argpartition(-projection_magnitudes, n-1)[:n]
    top_n_idx = top_n_idx[np.argsort(-projection_magnitudes[top_n_idx])]
    # print filenames and magnitudes for the top-n
    for idx in top_n_idx:
        print(f"{filenames1[idx]}: {projection_magnitudes[idx]}")
    most_exclusive_idx = np.argmax(projection_magnitudes)
    most_exclusive_vector = grads1[most_exclusive_idx]
    np.save("most_exclusive_vector.npy", most_exclusive_vector)

    print(f"{filenames1[most_exclusive_idx]}, {most_exclusive_vector}")

def experiment_with_tokens(
    prompt,
    token,
    token_intervention,
    intervention_strenghts=[0, 500, 1000, 5000, 10000, 15000],
    output_path="token_intervention_experiment.png",
    seed=42,
    num_per=10,
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    num_inference_steps=50,
    guidance_scale=7.0,
):
    from local_sdxl_pipeline import LocalStableDiffusionXLPipeline

    set_random_seed(seed)
    pipe = LocalStableDiffusionXLPipeline.from_pretrained(
        # "stabilityai/stable-diffusion-xl-base-1.0",
        # "stabilityai/sdxl-turbo",
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir="../model_cache",
    )
    pipe = pipe.to("cuda")
    tokens = pipe.tokenizer.encode(prompt)
    tokens = tokens[:77]
    all_tokes = {pipe.tokenizer.decode(curr_token): i for i, curr_token in enumerate(tokens)}
    pos = all_tokes[token]

    rows = []
    for intervention in intervention_strenghts:
        set_random_seed(seed)
        result = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            intervention=intervention,
            num_images_per_prompt=num_per,
            token_intervention=token_intervention,
            token_intervention_pos=pos,
            intervention_strenght=intervention,
        )
        imgs_row = [im.convert("RGB") for im in result.images]  # ensure consistent mode
        rows.append(imgs_row)

    # build grid where each row is one intervention and each column is an image from that run
    cols = num_per
    single_w, single_h = rows[0][0].size
    grid_w = cols * single_w
    grid_h = len(rows) * single_h

    grid = Image.new("RGB", (grid_w, grid_h))
    for r, imgs_row in enumerate(rows):
        for c, im in enumerate(imgs_row):
            grid.paste(im, (c * single_w, r * single_h))

    img = grid

    # img = pipe(
    #     "A cartoonish scene of the inside of a subway train. There are anthropomorphic frogs with big bear-like ears sitting on the seats. One of them is reading a newspaper. The window shows the river in the background.",
    #     num_inference_steps=4,
    #     guidance_scale=0.0,
    #     intervention=0,
    #     num_images_per_prompt=1
    # ).images[0]
    img.save(output_path)


if __name__ == "__main__":
    main()
    # compare_subspaces_svd()
    # compare_subspaces_pca()
    # experiment_with_tokens()

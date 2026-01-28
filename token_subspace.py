import os
import importlib
import itertools
import random
import json
import time

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from accelerate import dispatch_model
import torch
import torch.nn.functional as F
import torchvision
from diffusers import DDIMScheduler, AutoencoderKL, DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from huggingface_hub import hf_hub_download
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from optim_utils import *
import numpy as np
import matplotlib.pyplot as plt


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
        if cfg.model.model_id != "sdxl-dmd":
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
        else:
            base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            repo_name = "tianweiy/DMD2"
            ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"
            # Load model.
            unet = UNet2DConditionModel.from_config(
                base_model_id,
                subfolder="unet"
            ).to(torch.float16)
            unet.load_state_dict(
                torch.load(
                    hf_hub_download(
                        repo_name,
                        ckpt_name,
                        cache_dir="../model_cache"
                    )
                )
            )
            pipe = ModelClass.from_pretrained(
                base_model_id,
                unet=unet,
                torch_dtype=torch.float16,
                variant="fp16",
                cache_dir="../model_cache"
            )
            pipe.vae = AutoencoderKL.from_pretrained(
                base_model_id,
                subfolder="vae",
                # variant="fp16",
                torch_dtype=torch.float32,
            )
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

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
                    with open(os.path.join(timestep_dir, "dataset_info.json"), "r") as f:
                        dataset_info = json.load(f)
                    for shard in dataset_info["shards"]:
                        for src_dst in shard:
                            src, dst = src_dst[0], src_dst[1]
                            if src not in grad_dict[timestep]:
                                grad_dict[timestep][src] = []
                            grad_dict[timestep][src].append(dst)
            return grad_dict
        else:
            os.makedirs(grads_dir, exist_ok=True)
            for timestep in self.timesteps:
                timestep_dir = os.path.join(grads_dir, f"t{timestep}")
                os.makedirs(timestep_dir, exist_ok=True)
                os.makedirs(os.path.join(timestep_dir, "shards"), exist_ok=True)
                with open(os.path.join(timestep_dir, "dataset_info.json"), "w") as f:
                    json.dump(
                        {"shard_size": self.cfg.shard_size, "total": 0, "total_shards": 0, "shards": []},
                        f,
                        indent=4,
                    )

            # Save hydra config to config.yaml
            with open(config_path, "w") as f:
                f.write(OmegaConf.to_yaml(self.cfg))
            return {t: {} for t in self.timesteps}

    def encode_image(self, img, height=1024, width=1024):
        with torch.no_grad():
            img = self.pipe.image_processor.preprocess(
                img, width=width, height=height
            ).to(dtype=self.pipe.vae.dtype, device=self.pipe.vae.device)

            latent = self.pipe.vae.encode(img).latent_dist.sample()
            latent = latent * self.pipe.vae.config.scaling_factor
            latent = latent.to(dtype=torch.float16)
        return latent

    def noise_image(self, latent, timestep):
        t_noise = torch.tensor(self.pipe.scheduler.timesteps[-timestep].item())  # -t because timesteps are in decreasing order
        noise = torch.randn_like(latent)
        return self.pipe.scheduler.add_noise(latent, noise, t_noise).detach()

    def do_svd(self, threshold=1e-3):
        subspace_sizes = {t: 0 for t in self.timesteps}
        for t in self.timesteps:
            grads_dir = os.path.join(self.out_dir, self.name, "grads", f"t{t}")
            grad_files = [f for f in os.listdir(grads_dir) if f.endswith(".npy")]

            grads = []
            for fname in grad_files:
                grad = np.load(os.path.join(grads_dir, fname))
                grads.append(grad.flatten())

            grad_matrix = np.stack(grads, axis=0).astype(np.float32)
            u, s, vh = np.linalg.svd(grad_matrix, full_matrices=False)

            sorted_s = np.sort(s)[::-1]

            plt.figure(figsize=(10, 6))
            plt.bar(range(len(sorted_s)), sorted_s)
            plt.xlabel("Eigenvalue Index")
            plt.ylabel("Eigenvalue Magnitude")
            plt.title(f"Eigenvalues for timestep {t}")
            plt.tight_layout()

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

        centered_data = token_grads - token_grads.mean(dim=0, keepdim=True)  # center the data

        if num_vect < vect_dim:  # handle the "N < D" case
            # C_small = (1/(N-1)) * X_centered @ X_centered.T   (shape N, N)
            covariance_matrix_small = (1.0 / (num_vect - 1)) * (centered_data @ centered_data.T)

            eigenvalues, eigenvectors_small = torch.linalg.eigh(covariance_matrix_small)
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            sorted_eigenvalues = eigenvalues[sorted_indices]

            valid_indices = sorted_indices[sorted_eigenvalues > 1e-9]
            valid_eigenvalues = sorted_eigenvalues[valid_indices]
            valid_eigenvectors_small = eigenvectors_small[:, valid_indices]

            scale_factors = (1.0 / torch.sqrt(valid_eigenvalues * (num_vect - 1)))
            sorted_eigenvectors = centered_data.T @ valid_eigenvectors_small * scale_factors.unsqueeze(0)
            sorted_eigenvalues = valid_eigenvalues

            covariance_matrix = (1.0 / (num_vect - 1)) * (centered_data.T @ centered_data)

        # Standard case (N >= D)
        else:
            # C = (1/(N-1)) * X_centered.T @ X_centered   (shape D, D)
            covariance_matrix = (1.0 / (num_vect - 1)) * (centered_data.T @ centered_data)
            # get eigenvalues and eigenvectors of the (D, D) matrix
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            sorted_eigenvalues = eigenvalues[sorted_indices]
            # eigenvectors are columns
            sorted_eigenvectors = eigenvectors[:, sorted_indices]

        total_variance = torch.sum(sorted_eigenvalues)
        if total_variance == 0:
            return (torch.empty((0, vect_dim)), torch.empty((0,)),
                    total_variance, covariance_matrix)

        explained_variances = sorted_eigenvalues / total_variance

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
        if torch.linalg.norm(direction) == 0:
            return vectors
        direction_unit = direction / torch.linalg.norm(direction)  # normalize to unit length

        # calculate the projection of the vectors onto the direction.
        dot_products = vectors @ direction_unit

        if vectors.dim() > 1:
            dot_products = dot_products.unsqueeze(-1)

        projections = dot_products * direction_unit

        # subtract the projection from the original vectors.
        vectors_after_removal = vectors - projections

        return vectors_after_removal

    def get_token_pos(self, prompt, token):
        tokens = self.pipe.tokenizer.encode(prompt)
        tokens = tokens[:77]
        all_tokes = {self.pipe.tokenizer.decode(curr_token): i for i, curr_token in enumerate(tokens)}
        return all_tokes[token]

    @staticmethod
    def collated_pair_iterator(data_list, batch_size, shuffle=True):
        """
        Yields batches as 4 parallel lists:
        (imgs_A, names_A, imgs_B, names_B)
        """
        n = len(data_list)

        # 1. Generate index pairs (i, j) where i < j
        idx_pairs = list(itertools.combinations(range(n), 2))

        # 2. Randomize
        if shuffle:
            random.shuffle(idx_pairs)

        # 3. Iterate
        for k in range(0, len(idx_pairs), batch_size):
            batch_indices = idx_pairs[k : k + batch_size]

            # Initialize the 4 lists for this batch
            imgs_A, names_A = [], []
            imgs_B, names_B = [], []

            for i, j in batch_indices:
                # Unpack the pair from the source list
                # data_list[i] is (PIL.Image, imgname)
                img1, name1 = data_list[i]
                img2, name2 = data_list[j]

                # Append to the 4 separate lists
                imgs_A.append(img1)
                names_A.append(name1)
                imgs_B.append(img2)
                names_B.append(name2)

            # Yield the 4 lists
            yield imgs_A, names_A, imgs_B, names_B

    def run(self):
        prompt = self.cfg.data.prompt
        token = self.cfg.data.token

        images = list(self.image_iterator(self.cfg.data.image_folder))
        dataset = self.collated_pair_iterator(
            images, batch_size=self.cfg.batch_size, shuffle=self.cfg.randomize_pairs
        )

        for t in self.timesteps:
            print(f"Processing timestep {t}")
            num_processed = sum(len(v) for v in self.done_grads_dict[t].values())
            shard = np.empty((0, 2048))
            shard_pairs = []

            start_time = time.time()
            for imgs, names, ref_imgs, ref_names in dataset:
                for i, (name, ref_name) in enumerate(zip(names, ref_names)):
                    if ref_name in self.done_grads_dict[t].get(name, []):
                        imgs.pop(i)
                        names.pop(i)
                        ref_imgs.pop(i)
                        ref_names.pop(i)

                if num_processed > self.cfg.max_pairs:
                    break

                latent = self.encode_image(imgs, width=self.cfg.model.width, height=self.cfg.model.height)
                noised_latent = self.noise_image(latent, t)

                denoised_latent, prompt_tokens = self.pipe.get_pred_denoised_image(
                    [prompt] * len(noised_latent),
                    latents=noised_latent,
                    num_inference_steps=self.cfg.model.num_inference_steps,
                    guidance_scale=self.cfg.model.guidance_scale,
                    num_images_per_prompt=1,
                    start_timestep=self.cfg.model.num_inference_steps-t,
                    output_type="pt" if self.cfg.method == "clip" else "latent",
                )

                if self.cfg.method == "clip":
                    ref_imgs = self.pipe.image_processor.preprocess(
                        ref_imgs, width=self.cfg.model.width, height=self.cfg.model.height
                    ).to(dtype=self.clip.dtype, device=denoised_latent.device)
                    batch = torch.cat([denoised_latent, ref_imgs], dim=0)
                    batch = (batch + 1.0) / 2.0
                    inputs = self.clip_image_processor(
                        images=batch, return_tensors="pt"
                    )["pixel_values"].to(self.clip.device)
                    outputs = self.clip.get_image_features(pixel_values=inputs)
                    denoised_img_feat, ref_img_feat = outputs[:2]
                    loss = F.mse_loss(denoised_img_feat.float(), ref_img_feat.float())
                elif self.cfg.method == "latent":
                    loss = F.mse_loss(
                        denoised_latent.float(),
                        self.encode_image(
                            ref_imgs, width=self.cfg.model.width, height=self.cfg.model.height
                        ).float()
                    )
                else:
                    raise ValueError(f"Unknown method {self.cfg.method}")

                tokens_grad = torch.autograd.grad(loss, [prompt_tokens], allow_unused=True)[0]

                real_tokens_grad = tokens_grad[
                    :, :len(self.pipe.tokenizer.encode(prompt))
                ].to(torch.float).cpu()

                if self.cfg.remove_common:
                    common = real_tokens_grad.mean(dim=1, keepdim=True)
                    real_tokens_grad = real_tokens_grad - common

                # directions, _, _, _= self.find_significant_directions(
                #     real_tokens_grad,
                #     significance_threshold=2.5 / min(real_tokens_grad.shape)
                # )

                token_grad = real_tokens_grad[:, self.get_token_pos(prompt, token), :]
                token_grad = token_grad.cpu()

                # if self.cfg.remove_significant_directions:
                #     for direction in directions:
                #         token_grad = self.remove_direction(token_grad, direction.to(token_grad.dtype))

                token_grad = token_grad.numpy()

                shard = np.concatenate([shard, token_grad], axis=0)
                shard_pairs += [[n, rn] for n, rn in zip(names, ref_names)]
                num_processed += len(token_grad)
                if shard.shape[0] >= self.cfg.shard_size:
                    shard_path = os.path.join(
                        self.out_dir, self.name, "grads", f"t{t}", "shards", f"shard_{num_processed//self.cfg.shard_size}.npy"
                    )
                    np.save(shard_path, shard[:self.cfg.shard_size])
                    shard_info_path = os.path.join(
                        self.out_dir, self.name, "grads", f"t{t}", "dataset_info.json"
                    )
                    with open(shard_info_path, "r") as f:
                        dataset_info = json.load(f)
                    dataset_info["shards"].append(shard_pairs[:self.cfg.shard_size])
                    dataset_info["total"] += self.cfg.shard_size
                    dataset_info["total_shards"] += 1
                    with open(shard_info_path, "w") as f:
                        json.dump(dataset_info, f, indent=4)

                    shard = shard[self.cfg.shard_size :, :]
                    shard_pairs = shard_pairs[self.cfg.shard_size :]

                for name, ref_name in zip(names, ref_names):
                    if name not in self.done_grads_dict[t]:
                        self.done_grads_dict[t][name] = []
                    self.done_grads_dict[t][name].append(ref_name)

                del denoised_latent, prompt_tokens
                torch.cuda.empty_cache()

                elapsed_sec = time.time() - start_time
                expected_total_sec = (elapsed_sec / num_processed) * self.cfg.max_pairs
                print(
                    f"Grads gathered: {num_processed}/{self.cfg.max_pairs} \t Time {format_time(elapsed_sec)}/{format_time(expected_total_sec)}\r"
                )
        self.do_svd()


def format_time(seconds):
    """Converts seconds to hh:mm:ss string."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


@hydra.main(version_base=None, config_path="configs", config_name="token_subspace")
def main(cfg: DictConfig):
    cfg = instantiate(cfg)
    subspace_getter = SubspaceGetter(cfg)
    subspace_getter.run()


if __name__ == "__main__":
    main()

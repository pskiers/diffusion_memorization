import os
import argparse
import gc

import torch
from PIL import Image
from optim_utils import *
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file

from token_subspace import SubspaceGetter
from local_sdxl_pipeline import LocalStableDiffusionXLPipeline


class Comparator:
    def __init__(self, output_dir, ):
        self.output_dir = output_dir

    @staticmethod
    def check_directions_significant(
        directions,
        cov_matrix,
        total_variance,
        threshold
    ):
        common = []
        exclusive = []
        significance_list = []
        for i, direction in enumerate(directions):
            # Variance = direction.T @ C_B @ direction
            explained_var = direction.unsqueeze(0) @ cov_matrix @ direction.unsqueeze(1)
            significance = explained_var / total_variance
            significance_list.append(significance.item())

            if significance > threshold:
                common.append(i)
            else:
                exclusive.append(i)
        return exclusive, common, significance_list

    @staticmethod
    def output_results(
        exclusive,
        common,
        significance_list,
        original_significances,
        outpath,
    ):
        diffs = ""
        for idx in exclusive:
            string = f"A's Dir {idx} (Original Sig: {original_significances[idx]:.4f}) -> Sig in B: {significance_list[idx]:.4f}"
            print("\t\t" + string)
            diffs += string + "\n"
        print(f"\tExclusive directions in A: {len(exclusive)}, Common directions: {len(common)}")
        with open(outpath, "w") as f:
            f.write(str({"all": len(exclusive)+len(common), "exclusive": len(exclusive), "common": len(common)}))
            f.write("\n")
            f.write(diffs)

    @staticmethod
    def make_variance_barplot(variances, outpath):
        vars_np = variances.detach().cpu().numpy() if hasattr(variances, "detach") else np.asarray(variances)
        plt.figure(figsize=(max(6, len(vars_np) * 0.2), 4))
        plt.bar(np.arange(len(vars_np)), vars_np, color="tab:blue")
        plt.xlabel("Principal component")
        plt.ylabel("Variance")
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()

    def run(
        self,
        datasets,
        threshold=2.5,
    ):
        """
        Compare subspaces of the provided datasets using PCA.
        Args:
            datasets (list[np.array]): List of datasets to compare
            threshold (float): How many times larger than the average variance a direction
                must be to be considered significant.
        Returns:
            directions (list[torch.Tensor]): List of significant directions for each dataset
            exclusive_matrix (list[list[list[int]]]): Matrix indicating exclusive directions
                between dataset pairs. To get indices of directions in dataset A that are
                exclusive in dataset A when compared to dataset B, use exclusive_matrix[A][B].
        """
        if not len(datasets):
            return [], []

        os.makedirs(self.output_dir, exist_ok=True)

        threshold = 2.5 / min(datasets[0].shape)

        # PCA
        pca_dict = {
            i: SubspaceGetter.find_significant_directions(
                ds, significance_threshold=threshold
            )
            for i, ds in enumerate(datasets)
        }
        for k, v in pca_dict.items():
            print(f"\nSet {k}: Found {v[0].shape[0]}")
        for k, (_, variances, _, _) in pca_dict.items():
            self.make_variance_barplot(variances, os.path.join(self.output_dir, f"var_barplot_{k}.png"))

        exclusive_matrix = [[[] for _ in pca_dict.keys()] for _ in pca_dict.keys()]
        directions = []
        for i in pca_dict.keys():
            for j in pca_dict.keys():
                if i == j:
                    continue
                print(f"\n--- Comparing Set {i} to Set {j} ---")
                D_A, S_A, V_A, C_A = pca_dict[i]
                D_B, S_B, V_B, C_B = pca_dict[j]
                exclusive_in_A, common_from_A, significance_from_A_in_B = self.check_directions_significant(
                    directions=D_A,
                    cov_matrix=C_B,
                    total_variance=V_B,
                    threshold=threshold,
                )
                self.output_results(
                    exclusive=exclusive_in_A,
                    common=common_from_A,
                    significance_list=significance_from_A_in_B,
                    original_significances=S_A,
                    outpath=os.path.join(self.output_dir, f"directions_{i}_to_{j}.txt"),
                )
                exclusive_matrix[i][j] = exclusive_in_A
            directions.append(D_A)

        return directions, exclusive_matrix


class Experimenter:
    def __init__(
        self,
        models_dict: dict[str, torch.nn.Module],
        call_kwargs: dict[str, dict[str, Any]],
        output_dir: str,
        imgs_per_experiment: int = 10,
        batch_size: int = 4,
        seed: int = 42,
    ):
        self.models_dict = models_dict
        self.call_kwargs = call_kwargs
        self.output_dir = output_dir
        self.imgs_per_experiment = imgs_per_experiment
        self.seed = seed
        self.batch_size = batch_size

    def get_token_position(self, model, prompt: str, token: str) -> int:
        tokens = model.tokenizer.encode(prompt)
        tokens = tokens[:77]
        all_tokes = {model.tokenizer.decode(curr_token): i for i, curr_token in enumerate(tokens)}
        return all_tokes[token]

    def run_single_experiment(
        self,
        model_name: str,
        prompt: str,
        token: str,
        output_path: str,
        token_intervention: torch.Tensor,
        intervention_strenghts: list[int] = [0, 10, 15, 20, 30, 50, 70],
    ):

        kwargs = self.call_kwargs[model_name]
        model = self.models_dict[model_name]
        model.to("cuda")

        rows = []
        for intervention in intervention_strenghts:
            set_random_seed(self.seed)
            imgs_row = []
            for _ in range(self.imgs_per_experiment // self.batch_size):
                result = model(
                    [prompt] * self.batch_size,
                    num_images_per_prompt=1,
                    token_intervention=torch.stack([token_intervention] * self.batch_size, dim=0),
                    token_intervention_pos=torch.tensor([self.get_token_position(model, prompt, token)] * self.batch_size),
                    intervention_strenght=torch.tensor([intervention] * self.batch_size),
                    **kwargs,
                )
                gc.collect()
                torch.cuda.empty_cache()
                imgs_row += [im.convert("RGB") for im in result.images]
            if self.imgs_per_experiment % self.batch_size != 0:
                result = model(
                    [prompt] * (self.imgs_per_experiment % self.batch_size),
                    num_images_per_prompt=1,
                    token_intervention=torch.stack([token_intervention] * (self.imgs_per_experiment % self.batch_size), dim=0),
                    token_intervention_pos=torch.tensor([self.get_token_position(model, prompt, token)] * (self.imgs_per_experiment % self.batch_size)),
                    intervention_strenght=torch.tensor([intervention] * (self.imgs_per_experiment % self.batch_size)),
                    **kwargs,
                )
                gc.collect()
                torch.cuda.empty_cache()
                imgs_row += [im.convert("RGB") for im in result.images]
            rows.append(imgs_row)

        cols = self.imgs_per_experiment
        single_w, single_h = rows[0][0].size
        grid_w = cols * single_w
        grid_h = len(rows) * single_h

        grid = Image.new("RGB", (grid_w, grid_h))
        for r, imgs_row in enumerate(rows):
            for c, im in enumerate(imgs_row):
                grid.paste(im, (c * single_w, r * single_h))
        grid.save(output_path)

        model.to("cpu")

    def run(
        self,
        token_interventions,
        prompt: str,
        token: str,
        intervention_strenghts: list[int] = [0, 10, 15, 20, 30, 50, 70],
        outname_prefix: str = "exp",
    ):
        os.makedirs(self.output_dir, exist_ok=True)
        for model_name in self.models_dict.keys():
            for i, token_intervention in enumerate(token_interventions):
                output_path = os.path.join(
                    self.output_dir,
                    f"{outname_prefix}_{model_name}_{i}.png"
                )
                self.run_single_experiment(
                    model_name=model_name,
                    prompt=prompt,
                    token=token,
                    output_path=output_path,
                    token_intervention=token_intervention,
                    intervention_strenghts=intervention_strenghts,
                )


def load_grads(grads_dir, max_num=float("inf")):
    grad_files = [f for f in os.listdir(grads_dir) if f.endswith(".npy")]
    grads = []
    for fname in grad_files:
        grad = np.load(os.path.join(grads_dir, fname))
        grads.append(grad.flatten())
        if len(grads) > max_num:
            break
    grad_matrix = np.stack(grads, axis=0).astype(np.float32)
    grad_matrix = torch.from_numpy(grad_matrix)
    return grad_matrix


def load_sae_directions(sae_ckpt_path, low=-5, high=0, add_bias=True):
    tensors = load_file(sae_ckpt_path + "/doesnotmatter/sae.safetensors")
    W_dec = tensors["W_dec"]
    b_dec = tensors["b_dec"] if add_bias else 0.0
    with open(sae_ckpt_path + "/feature_sparsity.json", "r") as f:
        densities = json.load(f)
    densities = {float(k): float(v) for k, v in densities.items() if float(v) > low and float(v) < high}
    sorted_indices = sorted(densities.keys())
    directions = W_dec[sorted_indices] + b_dec
    return directions


def get_model(model_id):
    return LocalStableDiffusionXLPipeline.from_pretrained(
        model_id,
        variant="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir="../model_cache",
    ).to("cpu")


def main(
    dirs,
    prompt,
    token,
    models_dict,
    call_kwargs,
    output_dir,
    exclude_sae_from_comparison=True,
    sae_low=-5,
    sae_high=0,
    add_bias=False,
):
    comparator = Comparator(output_dir=output_dir)

    load_method_dict = {
        "sae": lambda x: load_sae_directions(x, low=sae_low, high=sae_high, add_bias=add_bias),
        "grad": load_grads,
    }

    datasets = [
        load_method_dict[method](path)
        for path, method in dirs.items()
        if (not exclude_sae_from_comparison) or method == "grad"
    ]

    directions, exclusive_matrix = comparator.run(datasets)
    sae_directions = [
        load_method_dict[method](path)
        for path, method in dirs.items()
        if exclude_sae_from_comparison and method == "sae"
    ]

    exp_dict = dict()
    for i, dirs in enumerate(directions):
        for j in range(len(directions)):
            if i == j:
                continue
            exp_dict[f"from_{i}_not_in_{j}"] = [dirs[k] for k in exclusive_matrix[i][j]]
            exp_dict[f"from_{i}_and_in_{j}"] = [
                dir
                for l, dir in enumerate(dirs)
                if l not in exclusive_matrix[i][j]
            ]
    for i, sae_dirs in enumerate(sae_directions):
        exp_dict[f"sae_dir_{i}"] = sae_dirs

    experimenter = Experimenter(
        models_dict=models_dict,
        call_kwargs=call_kwargs,
        output_dir=output_dir,
    )

    for exp_name, token_interventions in exp_dict.items():
        experimenter.run(
            token_interventions=token_interventions[:8],
            prompt=prompt,
            token=token,
            outname_prefix=exp_name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirs",
        type=str,
        required=True,
        nargs="+",
        help="Directory for set either grads or sae directions. Should be in a format grad:<path> or sae:<path>"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt used for generation")
    parser.add_argument("--token", type=str, required=True, help="Token to intervene on")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument(
        "--no_exclude_sae_from_comparison",
        action="store_true",
        required=False,
        default=False,
        help="Whether to exclude sae directions from comparison"
    )
    parser.add_argument("--sae_low", type=float, default=-5.0, help="Lowest sae density to include")
    parser.add_argument("--sae_high", type=float, default=0.0, help="Highest sae density to include")
    parser.add_argument(
        "--add_bias",
        action="store_true",
        required=False,
        default=False,
        help="Whether to add bias to sae directions"
    )
    args = parser.parse_args()

    models_dict = dict()
    models_dict["sdxl"] = get_model("stabilityai/stable-diffusion-xl-base-1.0")
    models_dict["sdxl_turbo"] = get_model("stabilityai/sdxl-turbo")

    call_kwargs = dict()
    call_kwargs["sdxl"] = dict(
        num_inference_steps=50,
        guidance_scale=7.5,
    )
    call_kwargs["sdxl_turbo"] = dict(
        num_inference_steps=4,
        guidance_scale=0.0,
    )

    dirs = {metdir.split(":")[1]: metdir.split(":")[0] for metdir in args.dirs}

    main(
        dirs=dirs,
        prompt=args.prompt,
        token=args.token,
        models_dict=models_dict,
        call_kwargs=call_kwargs,
        output_dir=args.output_dir,
        exclude_sae_from_comparison=not args.no_exclude_sae_from_comparison,
        sae_low=args.sae_low,
        sae_high=args.sae_high,
        add_bias=args.add_bias,
    )

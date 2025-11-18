import os
import argparse

import torch
from PIL import Image
from optim_utils import *
import numpy as np
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
        os.makedirs(self.output_dir, exist_ok=True)

        threshold = 2.5 / min(datasets[0].shape)

        # PCA
        pca_dict = {
            i: SubspaceGetter.find_significant_directions(
                ds, significance_threshold=threshold
            )
            for i, ds in enumerate(datasets)
        }
        for i in pca_dict.keys():
            print(f"\nSet {i}: Found {D_A.shape[0]}")

        exclusive_matrix = [[[[] for _ in pca_dict.keys()]] for _ in pca_dict.keys()]
        directions = []
        for i in pca_dict.keys():
            for j in pca_dict.keys():
                if i >= j:
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
                    outpath=os.path.join(self.output_dir, "directions_A.txt"),
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
        seed: int = 42,
    ):
        self.models_dict = models_dict
        self.call_kwargs = call_kwargs
        self.output_dir = output_dir
        self.imgs_per_experiment = imgs_per_experiment
        self.seed = seed

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
        intervention_strenghts: list[int] = [0, 500, 1000, 5000, 10000, 15000],
    ):

        kwargs = self.call_kwargs[model_name]
        model = self.models_dict[model_name]
        model.to("cuda")

        rows = []
        for intervention in intervention_strenghts:
            set_random_seed(self.seed)
            result = model(
                prompt,
                num_images_per_prompt=self.imgs_per_experiment,
                token_intervention=token_intervention,
                token_intervention_pos=self.get_token_position(model, prompt, token),
                intervention_strenght=intervention,
                **kwargs,
            )
            imgs_row = [im.convert("RGB") for im in result.images]
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
        intervention_strenghts: list[int] = [0, 500, 1000, 5000, 10000, 15000],
        outname_prefix: str = "exp",
    ):
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


def main(
    dirA,
    dirB,
    prompt,
    token,
    models_dict,
    call_kwargs,
    output_dir,
):
    comparator = Comparator(output_dir=output_dir)

    datasets = [
        load_grads(dirA),
        load_grads(dirB),
    ]
    directions, exclusive_matrix = comparator.run(datasets)
    exp_dict = dict()
    for i in range(len(directions)):
        for j in range(len(directions)):
            exp_dict[f"from_{i}_not_in_{j}"] = [
                directions[i] for i in exclusive_matrix[i][j]
            ]
            exp_dict[f"from_{i}_and_in_{j}"] = [
                directions[i]
                for i in range(directions[i].shape[0])
                if i not in exclusive_matrix[i][j]
            ]

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
    set_random_seed(seed)
    pipe = LocalStableDiffusionXLPipeline.from_pretrained(
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
    grid.save(output_path)


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


def compare_subspaces_pca(dirA, dirB, prompt, token, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load grads
    gradsA = load_grads(dirA)
    gradsB = load_grads(dirB)

    threshold = 2.5 / min(gradsA.shape)

    # PCA
    D_A, S_A, V_A, C_A = SubspaceGetter.find_significant_directions(gradsA, significance_threshold=threshold)
    D_B, S_B, V_B, C_B = SubspaceGetter.find_significant_directions(gradsB, significance_threshold=threshold)

    print(f"\nSet A: Found {D_A.shape[0]}, set B: Found {D_B.shape[0]} significant directions.\n")

    # --- 3. Test A in B ---
    print("--- Testing A's directions in Set B ---")
    exclusive_in_A, common_from_A, significance_from_A_in_B = check_directions_significant(
        directions=D_A,
        cov_matrix=C_B,
        total_variance=V_B,
        threshold=threshold,
    )
    output_results(
        exclusive=exclusive_in_A,
        common=common_from_A,
        significance_list=significance_from_A_in_B,
        original_significances=S_A,
        outpath=os.path.join(output_dir, "directions_sdxl.txt"),
    )


    for i, idx in enumerate(exclusive):
        if i >= 8:
            break
        intervention = D_A[idx]
        # experiment_with_tokens(
        #     prompt=prompt,
        #     token=token,
        #     token_intervention=intervention,
        #     intervention_strenghts=[0, 10, 15, 20, 30, 50, 70],
        #     output_path=f"{output_dir}/sdxl_turbo_exclusive_sdxl_to_sdxl_turbo_{i}.png",
        #     model_id="stabilityai/sdxl-turbo",
        #     num_inference_steps=4,
        #     guidance_scale=0.0,
        # )
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
        # experiment_with_tokens(
        #     prompt=prompt,
        #     token=token,
        #     token_intervention=intervention,
        #     intervention_strenghts=[0, 10, 15, 20, 30, 50, 70],
        #     output_path=f"{output_dir}/sdxl_turbo_common_sdxl_to_sdxl_turbo_{i}.png",
        #     model_id="stabilityai/sdxl-turbo",
        #     num_inference_steps=4,
        #     guidance_scale=0.0,
        # )
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


    # # --- 4. Test B in A ---
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
        # experiment_with_tokens(
        #     prompt=prompt,
        #     token=token,
        #     token_intervention=intervention,
        #     intervention_strenghts=[0, 10, 15, 20, 30, 50, 70],
        #     output_path=f"{output_dir}/sdxl_turbo_exclusive_sdxl_turbo_to_sdxl_{i}.png",
        #     model_id="stabilityai/sdxl-turbo",
        #     num_inference_steps=4,
        #     guidance_scale=0.0,
        # )
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
        # experiment_with_tokens(
        #     prompt=prompt,
        #     token=token,
        #     token_intervention=intervention,
        #     intervention_strenghts=[0, 10, 15, 20, 30, 50, 70],
        #     output_path=f"{output_dir}/sdxl_turbo_common_sdxl_turbo_to_sdxl_{i}.png",
        #     model_id="stabilityai/sdxl-turbo",
        #     num_inference_steps=4,
        #     guidance_scale=0.0,
        # )
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

def get_model(model_id):
    return LocalStableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir="../model_cache",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirA", type=str, required=True, help="Directory for set A grads")
    parser.add_argument("--dirB", type=str, required=True, help="Directory for set B grads")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt used for generation")
    parser.add_argument("--token", type=str, required=True, help="Token to intervene on")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
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

    main(
        dirA=args.dirA,
        dirB=args.dirB,
        prompt=args.prompt,
        token=args.token,
        models_dict=models_dict,
        call_kwargs=call_kwargs,
        output_dir=args.output_dir
    )

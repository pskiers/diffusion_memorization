"""
Run PCA on token-space gradients and sample images with:
  - SDXL (or SDXL-DMD): full 2048-dim directions
  - SD 1.5: first 768-dim CLIP slice of those directions, re-normalised

Usage:
  python sample_sdxl_and_sd15.py \
      --grads_dir outputs/sdxl-monster/grads/t50/shards \
      --prompt "a photo of a monster" \
      --token "monster" \
      --output_dir outputs/cross_model_exp \
      [--num_directions 10] \
      [--num_samples 8] \
      [--batch_size 4] \
      [--intervention_strengths 0 10 20 50] \
      [--random_n 1] \
      [--sdxl_model sdxl-dmd|sdxl|sdxl-turbo] \
      [--sd15_model_id runwayml/stable-diffusion-v1-5]
"""

import os
import argparse
import gc

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import UNet2DConditionModel, LCMScheduler
from huggingface_hub import hf_hub_download

from optim_utils import set_random_seed
from compare_subspaces import load_grads, sample_sum_normalize
from token_subspace import SubspaceGetter
from custom_pipelines.local_sdxl_pipeline import LocalStableDiffusionXLPipeline
from custom_pipelines.local_sd_pipeline import LocalStableDiffusionPipeline


CLIP_DIM = 768  # first 768 dims of the 2048-dim SDXL embedding are the CLIP encoder


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def get_sdxl_dmd_model():
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to(torch.float16)
    unet.load_state_dict(
        torch.load(hf_hub_download("tianweiy/DMD2", "dmd2_sdxl_4step_unet_fp16.bin", cache_dir="../model_cache"))
    )
    pipe = LocalStableDiffusionXLPipeline.from_pretrained(
        base_model_id, unet=unet, torch_dtype=torch.float16, variant="fp16", cache_dir="../model_cache"
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    return pipe.to("cpu")


def get_sdxl_model(model_id):
    return LocalStableDiffusionXLPipeline.from_pretrained(
        model_id,
        variant="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir="../model_cache",
    ).to("cpu")


def get_sd15_model(model_id):
    return LocalStableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir="../model_cache",
    ).to("cpu")


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def get_token_position(pipe, prompt: str, token: str) -> int:
    tokens = pipe.tokenizer.encode(prompt)[:77]
    return {pipe.tokenizer.decode(t): i for i, t in enumerate(tokens)}[token]


def sample_grid(
    pipe,
    prompt,
    token,
    directions,  # (k, D) tensor — already on CPU
    intervention_strengths,
    imgs_per_row,
    batch_size,
    seed,
    call_kwargs,
    output_path,
):
    pipe.to("cuda")
    token_pos = get_token_position(pipe, prompt, token)
    rows = []
    for strength in intervention_strengths:
        set_random_seed(seed)
        imgs_row = []
        remaining = imgs_per_row
        dir_idx = 0
        while remaining > 0:
            bs = min(batch_size, remaining)
            batch_dirs = directions[dir_idx : dir_idx + bs]
            # pad if fewer directions than batch size
            if batch_dirs.shape[0] < bs:
                batch_dirs = directions[:bs]
            result = pipe(
                [prompt] * bs,
                num_images_per_prompt=1,
                token_intervention=batch_dirs.to("cuda"),
                token_intervention_pos=[token_pos] * bs,
                intervention_strenght=torch.tensor([strength] * bs, dtype=torch.float32),
                **call_kwargs,
            )
            gc.collect()
            torch.cuda.empty_cache()
            imgs_row += [im.convert("RGB") for im in result.images]
            remaining -= bs
            dir_idx = (dir_idx + bs) % directions.shape[0]
        rows.append(imgs_row)

    single_w, single_h = rows[0][0].size
    grid = Image.new("RGB", (imgs_per_row * single_w, len(rows) * single_h))
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            grid.paste(im, (c * single_w, r * single_h))
    grid = grid.resize((grid.width // 2, grid.height // 2), Image.Resampling.LANCZOS)
    grid.save(output_path)
    pipe.to("cpu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load gradients and run PCA
    print("Loading gradients...")
    grads = load_grads(args.grads_dir)
    print(f"  Loaded {grads.shape[0]} vectors of dim {grads.shape[1]}")

    threshold = 2.5 / grads.shape[1]
    print(f"  PCA significance threshold: {threshold:.6f}")
    all_directions, significances, _, _ = SubspaceGetter.find_significant_directions(
        grads, significance_threshold=threshold
    )
    print(f"  Significant PCA directions: {all_directions.shape[0]}")

    if args.num_directions is not None:
        all_directions = all_directions[: args.num_directions]
    print(f"  Using {all_directions.shape[0]} directions")

    # 2. Prepare direction sets
    if args.random_n > 1:
        sdxl_directions = sample_sum_normalize(all_directions, k=all_directions.shape[0], n=args.random_n)
    else:
        sdxl_directions = F.normalize(all_directions, p=2, dim=1)

    # SD 1.5 uses only the CLIP part (first 768 dims), re-normalised
    clip_directions = F.normalize(sdxl_directions[:, :CLIP_DIM], p=2, dim=1)

    print(f"  SDXL directions shape: {sdxl_directions.shape}")
    print(f"  SD1.5 CLIP directions shape: {clip_directions.shape}")

    # 3. Load models
    print("\nLoading SDXL model...")
    if args.sdxl_model == "sdxl-dmd":
        sdxl_pipe = get_sdxl_dmd_model()
        sdxl_kwargs = dict(num_inference_steps=4, guidance_scale=0.0, timesteps=[999, 749, 499, 249])
    elif args.sdxl_model == "sdxl":
        sdxl_pipe = get_sdxl_model("stabilityai/stable-diffusion-xl-base-1.0")
        sdxl_kwargs = dict(num_inference_steps=50, guidance_scale=7.5)
    elif args.sdxl_model == "sdxl-turbo":
        sdxl_pipe = get_sdxl_model("stabilityai/sdxl-turbo")
        sdxl_kwargs = dict(num_inference_steps=4, guidance_scale=0.0)
    else:
        raise ValueError(f"Unknown sdxl_model: {args.sdxl_model}")

    print("Loading SD 1.5 model...")
    sd15_pipe = get_sd15_model(args.sd15_model_id)
    sd15_kwargs = dict(num_inference_steps=50, guidance_scale=7.5)

    # 4. Sample with SDXL
    print("\nSampling with SDXL...")
    sample_grid(
        pipe=sdxl_pipe,
        prompt=args.prompt,
        token=args.token,
        directions=sdxl_directions,
        intervention_strengths=args.intervention_strengths,
        imgs_per_row=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        call_kwargs=sdxl_kwargs,
        output_path=os.path.join(args.output_dir, f"sdxl_{args.sdxl_model}.png"),
    )
    print(f"  Saved SDXL grid")

    # 5. Sample with SD 1.5
    print("\nSampling with SD 1.5...")
    sample_grid(
        pipe=sd15_pipe,
        prompt=args.prompt,
        token=args.token,
        directions=clip_directions,
        intervention_strengths=args.intervention_strengths,
        imgs_per_row=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        call_kwargs=sd15_kwargs,
        output_path=os.path.join(args.output_dir, "sd15.png"),
    )
    print(f"  Saved SD 1.5 grid")

    print(f"\nDone. Results in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grads_dir", type=str, required=True, help="Directory with .npy shard files (the shards/ subfolder)"
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--token", type=str, required=True, help="Token within the prompt to intervene on")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--num_directions",
        type=int,
        default=None,
        help="Cap on significant PCA directions to use (default: all significant)",
    )
    parser.add_argument("--num_samples", type=int, default=8, help="Images per row in the output grid")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--intervention_strengths",
        type=float,
        nargs="+",
        default=[0, 10, 20, 50],
        help="Intervention strengths (rows in the output grid)",
    )
    parser.add_argument(
        "--random_n", type=int, default=1, help="Sum N random directions per sample (1 = single direction per sample)"
    )
    parser.add_argument("--sdxl_model", type=str, default="sdxl-dmd", choices=["sdxl-dmd", "sdxl", "sdxl-turbo"])
    parser.add_argument("--sd15_model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

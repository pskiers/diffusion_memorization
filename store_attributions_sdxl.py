"""
This script stores all the caption attributions in a json file in outputs/attributions/{method}.json
"""
import json
import os

import torch
from diffusers import DDIMScheduler
import json
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from local_sdxl_pipeline import LocalStableDiffusionXLPipeline
from optim_utils import *


with open('match_verbatim_captions.json') as f:
    all_mem_captions = json.load(f)
    


@hydra.main(version_base=None, config_path="configs", config_name="store_attributions_sdxl")
def main(cfg: DictConfig):
    
    cfg = instantiate(cfg)

    # ---------------------- #
    # (1) setup captions and attributions
    # load the json full of captions
    with open(cfg.mem_captions_path, 'r') as f:
        mem_captions = json.load(f)

    # load the json with the attributions
    attribution_path = os.path.join(cfg.out_dir, "attributions", f"{cfg.name}_{cfg.attribution_method.name}.json")
    if not os.path.exists(attribution_path):
        os.makedirs(os.path.dirname(attribution_path), exist_ok=True)
        with open(attribution_path, 'w') as f:
            json.dump({}, f)
    with open(attribution_path, 'r') as f:
        attributions = json.load(f)

    # ---------------------- #
    # (2) setup model
    # setup the model with the local diffusers pipeline
    device_map = {
        # text encoder
        "text_model": 3,

        # VAE
        "encoder": 3,
        "decoder": 3,
        "quant_conv": 3,
        "post_quant_conv": 3,

        # UNet
        "text_projection": 0,
        "add_embedding": 0,
        "time_embedding": 0,

        "conv_in": 0,

        "down_blocks": 1,
        "mid_block": 0,
        "up_blocks": 2,

        "conv_norm_out": 0,
        "conv_out": 0,
    }
    pipe = LocalStableDiffusionXLPipeline.from_pretrained(
        cfg.model.model_id,
        torch_dtype=torch.float16,
        cache_dir="../model_cache",
        device_map=device_map,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # ---------------------- #
    # (3) iterate over all the captions and find attributions
    for prompt_idx, prompt in enumerate(mem_captions):
        if not cfg.prompt_start_idx <= prompt_idx < cfg.prompt_end_idx:
            continue
        if prompt in attributions:
            print(f"Skipping prompt '{prompt}' as it already exists")
            continue

        set_random_seed(cfg.seed)
        token_grads, loss = pipe.get_text_cond_grad(
            prompt,
            num_inference_steps=cfg.attribution_method.num_inference_steps,
            guidance_scale=cfg.model.guidance_scale,
            num_images_per_prompt=cfg.attribution_method.num_images_per_prompt,
            target_steps=cfg.attribution_method.target_steps,
            method=cfg.attribution_method.name,
        )
        torch.cuda.empty_cache()

        prompt_tokens = pipe.tokenizer.encode(prompt)
        prompt_tokens = prompt_tokens[1:-1]
        prompt_tokens = prompt_tokens[:75]
        token_grads = token_grads[1:(1+len(prompt_tokens))]
        token_grads = token_grads.cpu().tolist()

        all_tokes = []

        for curr_token in prompt_tokens:
            all_tokes.append(pipe.tokenizer.decode(curr_token))

        attributions[prompt] = {cfg.attribution_method.name: float(loss.cpu()), "grads": [(tok, grad) for tok, grad in zip(all_tokes, token_grads)]}

        # save the attributions
        with open(attribution_path, 'w') as f:
            json.dump(attributions, f, indent=4)
            

if __name__ == "__main__":
    main()
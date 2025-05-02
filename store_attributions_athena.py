"""
This script stores all the caption attributions in a json file in outputs/attributions/{method}.json
"""
import json
import os
import time

import torch
from diffusers import DDIMScheduler
import json
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from local_sd_pipeline import LocalStableDiffusionPipeline
from local_sdxl_pipeline import LocalStableDiffusionXLPipeline
from optim_utils import *


with open('match_verbatim_captions.json') as f:
    all_mem_captions = json.load(f)
    


@hydra.main(version_base=None, config_path="configs", config_name="store_attributions")
def main(cfg: DictConfig):
    
    cfg = instantiate(cfg)

    # ---------------------- #
    # (1) setup captions and attributions
    # load the json full of captions
    with open(cfg.mem_captions_path, 'r') as f:
        mem_captions = json.load(f)

    # load the json with the attributions
    attribution_path = os.path.join(cfg.out_dir, "attributions", f"{cfg.attribution_method.name}.json")
    if not os.path.exists(attribution_path):
        os.makedirs(os.path.dirname(attribution_path), exist_ok=True)
        with open(attribution_path, 'w') as f:
            json.dump({}, f)
    with open(attribution_path, 'r') as f:
        attributions = json.load(f)

    # ---------------------- #
    # (2) setup model
    # setup the model with the local diffusers pipeline
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device_map = {
        "conv_in": 0, 
        "time_proj": 0, 
        "time_embedding": 0, 
        "down_blocks.0.attentions.0": 1,
        "down_blocks.0.attentions.1": 0,
        "down_blocks.0.resnets.0": 0,
        "down_blocks.0.resnets.1": 1,
        "down_blocks.0.downsamplers": 1,
        "down_blocks.1": 1, 
        "down_blocks.2": 1, 
        "down_blocks.3": 0,
        "up_blocks.0": 1,
        "up_blocks.1": 1,
        "up_blocks.2": 0,
        "up_blocks.3.attentions.0": 0,
        "up_blocks.3.attentions.1.norm": 0,
        "up_blocks.3.attentions.1.proj_in": 0,
        "up_blocks.3.attentions.1.transformer_blocks.0.norm1": 0,
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1": 1,
        "up_blocks.3.attentions.1.transformer_blocks.0.norm2": 0,
        "up_blocks.3.attentions.1.transformer_blocks.0.attn2": 0,
        "up_blocks.3.attentions.1.transformer_blocks.0.norm3": 0,
        "up_blocks.3.attentions.1.transformer_blocks.0.ff": 0,
        "up_blocks.3.attentions.1.proj_out": 0,
        "up_blocks.3.attentions.2.norm": 0,
        "up_blocks.3.attentions.2.proj_in": 0,
        "up_blocks.3.attentions.2.transformer_blocks.0.norm1": 0,
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_q": 0,
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_k": 0,
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_v": 1,
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_out": 0,
        "up_blocks.3.attentions.2.transformer_blocks.0.norm2": 0,
        "up_blocks.3.attentions.2.transformer_blocks.0.attn2": 0,
        "up_blocks.3.attentions.2.transformer_blocks.0.norm3": 0,
        "up_blocks.3.attentions.2.transformer_blocks.0.ff": 0,
        "up_blocks.3.attentions.2.proj_out": 0,
        "up_blocks.3.resnets.0.norm1": 1,
        "up_blocks.3.resnets.0.conv1": 1,
        "up_blocks.3.resnets.0.time_emb_proj": 1,
        "up_blocks.3.resnets.0.norm2": 1,
        "up_blocks.3.resnets.0.dropout": 1,
        "up_blocks.3.resnets.0.conv2": 1,
        "up_blocks.3.resnets.0.nonlinearity": 1,
        "up_blocks.3.resnets.0.conv_shortcut": 1,
        "up_blocks.3.resnets.1": 0,
        "up_blocks.3.resnets.2": 0,
        "up_blocks.3.downsamplers": 1,
        "mid_block": 0, 
        "conv_norm_out": 0, 
        "conv_act": 0, 
        "conv_out": 0, 
        "decoder": 0,
        "encoder": 0,
        "post_quant_conv": 0,
        "text_model": 0,
        "quant_conv": 0,
        # "text_projection": 2,
        # "add_embedding": 3,
    }
    # device_map = {
    #     "conv_in": 1, 
    #     "time_proj": 1, 
    #     "time_embedding": 1, 
    #     "down_blocks.0.attentions.0": 1,
    #     "down_blocks.0.attentions.1": 0,
    #     "down_blocks.0.resnets.0": 0,
    #     "down_blocks.0.resnets.1": 1,
    #     "down_blocks.0.downsamplers": 1,
    #     "down_blocks.1": 3, 
    #     "down_blocks.2": 1, 
    #     "down_blocks.3": 3,
    #     "up_blocks.0": 3,
    #     "up_blocks.1": 3,
    #     "up_blocks.2": 3,
    #     "up_blocks.3.attentions.0": 2,
    #     "up_blocks.3.attentions.1.norm": 2,
    #     "up_blocks.3.attentions.1.proj_in": 2,
    #     "up_blocks.3.attentions.1.transformer_blocks.0.norm1": 2,
    #     "up_blocks.3.attentions.1.transformer_blocks.0.attn1": 3,
    #     "up_blocks.3.attentions.1.transformer_blocks.0.norm2": 2,
    #     "up_blocks.3.attentions.1.transformer_blocks.0.attn2": 2,
    #     "up_blocks.3.attentions.1.transformer_blocks.0.norm3": 2,
    #     "up_blocks.3.attentions.1.transformer_blocks.0.ff": 2,
    #     "up_blocks.3.attentions.1.proj_out": 2,
    #     "up_blocks.3.attentions.2": 1,
    #     "up_blocks.3.resnets.0": 3,
    #     "up_blocks.3.resnets.1": 1,
    #     "up_blocks.3.resnets.2": 3,
    #     "up_blocks.3.downsamplers": 3,
    #     "mid_block": 0, 
    #     "conv_norm_out": 3, 
    #     "conv_act": 3, 
    #     "conv_out": 3, 
    #     "decoder": 0,
    #     "encoder": 0,
    #     "post_quant_conv": 0,
    #     "text_model": 0,
    #     "quant_conv": 0,
    #     # "text_projection": 2,
    #     # "add_embedding": 3,
    # }
    # pipe = LocalStableDiffusionXLPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0",
    #     torch_dtype=torch.float16,
    #     safety_checker=None,
    #     requires_safety_checker=False,
    #     cache_dir="/net/tscratch/people/plgpawel269/diffusers/_model_cache",
    #     device_map=device_map,
    #     # device_map="auto",
    # )
    pipe = LocalStableDiffusionPipeline.from_pretrained(
        cfg.model.model_id,
        # "../diffusers/examples/dreambooth/save_dir_sd14",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir="/net/tscratch/people/plgpawel269/diffusers/_model_cache",
        device_map=device_map,
        # device_map="auto"
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.unet.enable_gradient_checkpointing()
    # pipe.enable_sequential_cpu_offload()
    # pipe = pipe.to(device)

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
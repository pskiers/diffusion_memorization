"""
This script stores all the caption attributions in a json file in outputs/attributions/{method}.json
"""
import json
import os
import importlib

import torch
from diffusers import DDIMScheduler
import json
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from accelerate import dispatch_model

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
        cache_dir="../model_cache",
        # device_map=cfg.model.device_map,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # pipe.enable_sequential_cpu_offload()
    if cfg.lora_path is not None:
        print(f"Loading LoRA weights from {cfg.lora_path}")
        pipe.load_lora_weights(cfg.lora_path)
        pipe.fuse_lora()
    def print_module_devices(model, max_depth: int = None):
        for name, module in model.named_modules():
            # Compute depth based on number of dots in name
            depth = name.count(".")
            if max_depth is not None and depth > max_depth:
                continue

            # Get device(s)
            devices = {p.device for p in module.parameters(recurse=False)}
            if not devices:
                devices = {b.device for b in module.buffers(recurse=False)}
            device_str = ', '.join(str(d) for d in devices) if devices else "No tensors"

            print(f"{name or 'model'} (depth {depth}): {device_str}")
    pipe.unet = dispatch_model(pipe.unet, device_map=cfg.model.device_map)
    pipe.text_encoder = dispatch_model(pipe.text_encoder, device_map=cfg.model.device_map)
    pipe.text_encoder_2 = dispatch_model(pipe.text_encoder_2, device_map=cfg.model.device_map)
    pipe.vae = dispatch_model(pipe.vae, device_map=cfg.model.device_map)
    # print_module_devices(pipe.vae, max_depth=1)
    # print_module_devices(pipe.text_encoder, max_depth=1)
    # print_module_devices(pipe.text_encoder_2, max_depth=1)
    # print_module_devices(pipe.unet, max_depth=1)
    # raise ValueError()
    if cfg.model.device_map is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        pipe = pipe.to(device)

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

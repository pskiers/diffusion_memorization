import json
import os
import importlib

import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, AutoencoderKL
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from accelerate import dispatch_model

from torchvision import transforms
from typing import Iterator, Tuple
from tqdm.auto import tqdm

from optim_utils import *


def get_image_caption_iterator(
    image_dir: str,
    captions_file: str,
    size: Tuple[int,int]=(1024,1024)
) -> Iterator[Tuple[Image.Image,str]]:
    """Yields (PIL.Image, caption), resizing each to `size`."""
    captions_path = os.path.join(image_dir, captions_file)
    with open(captions_path, "r", encoding="utf-8") as f:
        captions = json.load(f)

    tf = transforms.Compose([
        # transforms.Resize(size, Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # map to [-1,1]
    ])

    for fname, caption in captions.items():
        img_path = os.path.join(image_dir, fname)
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
        # img = tf(img).unsqueeze(0)
        yield img, caption  # shape (1,3,H,W)



@hydra.main(version_base=None, config_path="configs", config_name="img_lids")
def main(cfg: DictConfig):

    cfg = instantiate(cfg)

    iterator = get_image_caption_iterator(
        image_dir=cfg.dataset_path,
        captions_file=cfg.caption_file,
        size=(1024, 1024)
    )
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
        # variant="fp16",
        cache_dir="../model_cache",
        # device_map=cfg.model.device_map,
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

    pipe.unet = dispatch_model(pipe.unet, device_map=cfg.model.device_map)
    pipe.text_encoder = dispatch_model(pipe.text_encoder, device_map=cfg.model.device_map)
    pipe.text_encoder_2 = dispatch_model(pipe.text_encoder_2, device_map=cfg.model.device_map)
    pipe.vae = dispatch_model(pipe.vae, device_map=cfg.model.device_map)

    if cfg.model.device_map is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        pipe = pipe.to(device)

    pipe.scheduler.set_timesteps(cfg.attribution_method.num_inference_steps)
    timesteps = pipe.scheduler.timesteps  # e.g. a tensor of length num_inference_steps
    t_index = int(0.95 * (len(timesteps) - 1))
    t_noise = torch.tensor(timesteps[t_index].item())
    timesteps = torch.tensor(range(t_noise, 0, -1))

    # ---------------------- #
    # (3) iterate over all the captions
    repeated = 0
    for prompt_idx, (img, prompt) in enumerate(iterator):
        if not cfg.prompt_start_idx <= prompt_idx < cfg.prompt_end_idx:
            continue
        with torch.no_grad():
            # img_low_res = img.resize((256, 256))
            img = pipe.image_processor.preprocess(img, width=1024, height=1024).to(dtype=pipe.vae.dtype)
            # img = pipe.image_processor.preprocess(img).to(dtype=pipe.vae.dtype)
            latents = pipe.vae.encode(img).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
            latents = latents.to(dtype=torch.float16)
        # 3) add noise at the chosen timestep
        noise = torch.randn_like(latents)
        noised_latents = pipe.scheduler.add_noise(latents, noise, t_noise).detach()


        if prompt+str(repeated) in attributions:
            # print(f"Skipping prompt '{prompt}' as it already exists")
            # continue
            repeated += 1
        else:
            repeated = 0

        set_random_seed(cfg.seed)
        token_grads, loss = pipe.get_text_cond_grad(
            prompt if cfg.conditional else "",
            latents=noised_latents,
            num_inference_steps=cfg.attribution_method.num_inference_steps,
            guidance_scale=cfg.model.guidance_scale if cfg.conditional else 0,
            num_images_per_prompt=cfg.attribution_method.num_images_per_prompt,
            target_steps=cfg.attribution_method.target_steps,
            method=cfg.attribution_method.name,
            start_timestep=t_index,
            # image=img_low_res
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

        attributions[prompt + str(repeated)] = {cfg.attribution_method.name: float(loss.cpu()), "grads": [(tok, grad) for tok, grad in zip(all_tokes, token_grads)]}

        # save the attributions
        with open(attribution_path, 'w') as f:
            json.dump(attributions, f, indent=4)


if __name__ == "__main__":

    main()

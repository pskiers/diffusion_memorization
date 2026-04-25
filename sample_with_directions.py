import os
import importlib
from tqdm import tqdm
import gc

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from accelerate import dispatch_model
import torch
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel, LCMScheduler
from huggingface_hub import hf_hub_download
from optim_utils import *
from compare_subspaces import load_grads, load_sae_directions, sample_sum_normalize
from token_subspace import SubspaceGetter


class SamplerWithDirections:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipe = self.setup_model(cfg)
        self.prompt = self.cfg.prompt
        self.token = self.token
        self.batch_size = self.cfg.batch_size
        self.num_samples = self.cfg.num_samples
        self.directions_path = self.cfg.directions_path
        self.kwargs = self.cfg.call_kwargs
        self.low = self.cfg.min_intervention_strenght
        self.high = self.cfg.max_intervention_strenght
        self.name = self.cfg.name
        self.outpath = os.path.join(self.cfg.out_dir, self.cfg.name)
        self.seed = self.cfg.seed
        self.direction_type = self.cfg.direction_type
        self.num_random_directions = self.cfg.num_random_directions

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
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        if cfg.model.lora_path is not None:
            print(f"Loading LoRA weights from {cfg.model.lora_path}")
            pipe.load_lora_weights(cfg.model.lora_path)
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
        return pipe

    def get_token_position(self, prompt: str, token: str) -> int:
        tokens = self.pipe.tokenizer.encode(prompt)
        tokens = tokens[:77]
        all_tokes = {self.pipe.tokenizer.decode(curr_token): i for i, curr_token in enumerate(tokens)}
        return all_tokes[token]

    def sample(self):
        if self.direction_type == "grad":
            grads = load_grads(self.directions_path)
            threshold = 2.5 / 2048  # NOTE hardcoded for sdxl
            directions = SubspaceGetter.find_significant_directions(
                grads, significance_threshold=threshold
            )[0]
        elif self.direction_type == "sae":
            directions = load_sae_directions(
                self.directions_path, low=-6, high=0, add_bias=False
            )
        else:
            raise ValueError(self.direction_type)

        set_random_seed(self.seed)
        os.makedirs(self.outpath, exist_ok=True)

        for i in tqdm(range(self.num_samples // self.batch_size)):
            if type(self.num_random_directions) == str:
                min_directions, max_directions = map(int, self.num_random_directions.split("-"))
                curr_num_random_directions = torch.randint(min_directions, max_directions + 1, (1,)).item()
                batch_directions = sample_sum_normalize(directions, k=self.batch_size, n=curr_num_random_directions)
            else:
                batch_directions = sample_sum_normalize(directions, k=self.batch_size, n=self.num_random_directions)
            batch_intervention_strenghts = torch.empty(self.batch_size).uniform_(self.low, self.high)
            images = self.pipe(
                [self.prompt] * self.batch_size,
                num_images_per_prompt=1,
                token_intervention=batch_directions,
                token_intervention_pos=[self.get_token_position(self.prompt, self.token)] * self.batch_size,
                intervention_strenght=batch_intervention_strenghts,
                **self.kwargs,
            ).images
            gc.collect()
            torch.cuda.empty_cache()
            for j, img in enumerate(images):
                img.save(os.path.join(self.outpath, f"{i * self.batch_size + j}.png"))


@hydra.main(version_base=None, config_path="configs", config_name="sample_with_directions")
def main(cfg: DictConfig):
    cfg = instantiate(cfg)
    sampler = SamplerWithDirections(cfg)
    sampler.sample()


if __name__ == "__main__":
    main()

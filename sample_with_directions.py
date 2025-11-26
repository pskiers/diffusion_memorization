import os
import importlib
import tqdm
import gc

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from accelerate import dispatch_model
import torch
from diffusers import DDIMScheduler, AutoencoderKL
from optim_utils import *
from compare_subspaces import load_grads, load_sae_directions
from token_subspace import SubspaceGetter


class SamplerWithDirections:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipe = self.setup_model(cfg)
        self.prompt = self.cfg.prompt
        self.token = self.cfg.token
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
            variant="fp16",
            cache_dir="../model_cache",
        )
        # pipe.vae = AutoencoderKL.from_pretrained(
        #     cfg.model.model_id,
        #     subfolder="vae",
        #     # variant="fp16",
        #     torch_dtype=torch.float32,
        # )
        # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        # pipe.enable_sequential_cpu_offload()
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
            threshold = 2.5 / min(datasets[0].shape)
            directions = SubspaceGetter.find_significant_directions(
                grads, significance_threshold=threshold
            )[0]
        elif self.direction_type == "sae":
            low, high = (-5, 0)
            directions = load_sae_directions(
                self.directions_path, low, high, add_bias=False
            )
        else:
            raise ValueError(self.direction_type)

        set_random_seed(self.seed)
        os.makedirs(self.outpath, exist_ok=True)

        for i in tqdm(range(self.num_samples // self.batch_size)):
            batch_directions = directions[torch.randperm(directions.size(0))[:self.batch_size]]
            batch_intervention_strenghts = torch.empty(self.batch_size).uniform_(low, high)
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

import os
import gc
from PIL import Image

from custom_pipelines.local_flux_pipeline import LocalFluxPipeline
from custom_pipelines.local_sdxl_pipeline import LocalStableDiffusionXLPipeline
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from optim_utils import *
from compare_subspaces import load_grads, load_sae_directions
from compare_subspaces import Comparator


class SamplerWithDirections:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipe = self.setup_model(cfg)
        self.prompt = "A photorealistic picture of a monster driving a car"
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

    # def setup_model(self, cfg):
    #     return LocalFluxPipeline.from_pretrained(
    #         "black-forest-labs/FLUX.1-schnell",
    #         torch_dtype=torch.bfloat16,
    #         safety_checker=None,
    #         requires_safety_checker=False,
    #         cache_dir="../model_cache",
    #     ).to("cuda")
    def setup_model(self, cfg):
        return LocalStableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            variant="fp16",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir="../model_cache",
        ).to("cuda")

    def get_token_position(self, prompt: str, token: str) -> int:
        tokens = self.pipe.tokenizer.encode(prompt)
        tokens = tokens[:77]
        all_tokes = {self.pipe.tokenizer.decode(curr_token): i for i, curr_token in enumerate(tokens)}
        return all_tokes[token]

    def sample(self):
        # direction_car = load_sae_directions(
        #     "../universal-diffsae/sae-ckpts/sae_text_token_grads/flux-schnell-car-k32-exp-factor-1_t4",
        #     low=-4, high=-2, add_bias=False
        # )
        # direction_monster = load_sae_directions(
        #     "../universal-diffsae/sae-ckpts/sae_text_token_grads/flux-schnell-monster-k32-exp-factor-1_t4",
        #     low=-6, high=-4, add_bias=False
        # )
        # for i, dir_car in enumerate(direction_car):
        #     for j, dir_mon in enumerate(direction_monster):
        i = 17
        j = 5

        comparator = Comparator(output_dir="outputs/intreventions/highlight")
        prompt = "A picture of a cat and a dog"

        direction_car = [load_grads("outputs/sdxl-dmd-cat/grads/t4/shards")]
        direction_car, _ = comparator.run(direction_car)
        direction_monster = [load_grads("outputs/sdxl-dmd-dog/grads/t4/shards")]
        direction_monster, _ = comparator.run(direction_monster)
        dir_car = F.normalize(direction_car[0][i], p=2, dim=0)
        dir_mon = F.normalize(direction_monster[0][j], p=2, dim=0)
        set_random_seed(self.seed)
        os.makedirs(self.outpath, exist_ok=True)

        strenghts = [0, 10, 20, 30]
        bs = len([0, 10, 20, 30])

        set_random_seed(45)
        noise = torch.randn((1, 4, 128, 128), device=self.pipe.device)
        noise = noise.repeat(bs, 1, 1, 1).to(torch.float16)
        image_rows = []
        for monster_strenght in strenghts:
            interventions = [
                {
                    "token_intervention": torch.stack([dir_mon] * bs, dim=0),
                    "token_intervention_pos": torch.tensor([self.get_token_position(prompt, "dog")] * bs),
                    "intervention_strenght": torch.tensor([monster_strenght] * bs),
                },
                # {
                #     "token_intervention": torch.stack([-dir_mon] * bs, dim=0),
                #     "token_intervention_pos": torch.tensor([self.get_token_position(prompt, "cat")] * bs),
                #     "intervention_strenght": torch.tensor([monster_strenght] * bs)
                # },
                {
                    "token_intervention": torch.stack([dir_car] * bs, dim=0),
                    "token_intervention_pos": torch.tensor([self.get_token_position(prompt, "cat")] * bs),
                    "intervention_strenght": torch.tensor([0, 10, 20, 30]),
                },
                # {
                #     "token_intervention": torch.stack([-dir_car] * bs, dim=0),
                #     "token_intervention_pos": torch.tensor([self.get_token_position(prompt, "dog")] * bs),
                #     "intervention_strenght": torch.tensor([0, 10, 20, 30])
                # },
            ]
            images = self.pipe(
                [prompt] * bs,
                latents=noise.to(),
                num_images_per_prompt=1,
                interventions=interventions,
                num_inference_steps=50,
                guidance_scale=7.5,
            ).images
            image_rows.append(images)
            gc.collect()
            torch.cuda.empty_cache()

        cols = bs
        single_w, single_h = image_rows[0][0].size
        grid_w = cols * single_w
        grid_h = len(image_rows) * single_h
        grid = Image.new("RGB", (grid_w, grid_h))
        for r, imgs_row in enumerate(image_rows):
            for c, im in enumerate(imgs_row):
                grid.paste(im, (c * single_w, r * single_h))
        grid = grid.resize((grid_w // 2, grid_h // 2), Image.Resampling.LANCZOS)
        grid.save("outputs/intreventions/highlight" + f"/highlight_d{j}_c{i}_pca_fix.png")


@hydra.main(version_base=None, config_path="../configs", config_name="sample_with_directions")
def main(cfg: DictConfig):
    cfg = instantiate(cfg)
    sampler = SamplerWithDirections(cfg)
    sampler.sample()


if __name__ == "__main__":
    main()

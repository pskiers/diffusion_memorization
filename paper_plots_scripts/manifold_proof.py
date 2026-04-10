from custom_pipelines.local_sdxl_pipeline import LocalStableDiffusionXLPipeline
import torch
import torch.nn.functional as F
from compare_subspaces import load_grads, sample_sum_normalize
from compare_subspaces import Comparator
from dreamsim import dreamsim


def get_token_position(pipe, prompt: str, token: str) -> int:
    tokens = pipe.tokenizer.encode(prompt)
    tokens = tokens[:77]
    all_tokes = {pipe.tokenizer.decode(curr_token): i for i, curr_token in enumerate(tokens)}
    return all_tokes[token]


pipe = LocalStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
    cache_dir="../model_cache",
).to("cuda")
ds_model, ds_preprocess = dreamsim(pretrained=True, device="cuda")

comparator = Comparator(output_dir="outputs/intreventions/highlight")
directions = [load_grads("outputs/sdxl-dmd-car/grads/t4/shards")]
directions, _ = comparator.run(directions)
directions = directions[0]

max_iter = 1000
bs = 2
prompt = "A picture of car"
token = "car"
strenght = 30

assert bs % 2 == 0
distances = []
for _ in range(max_iter):
    dirs = sample_sum_normalize(directions, bs // 2, n=3)
    # dirs = F.normalize(torch.randn((bs // 2, 2048)), p=2, dim=1)
    dirs = dirs.repeat((2, 1))
    noise = torch.randn((bs // 2, 4, 128, 128), device=pipe.device, dtype=pipe.dtype)
    noise = noise.repeat(2, 1, 1, 1)
    images = pipe(
        [prompt] * bs,
        latents=noise.to(),
        num_images_per_prompt=1,
        token_intervention=dirs,
        token_intervention_pos=torch.tensor([get_token_position(pipe, prompt, token)] * bs),
        intervention_strenght=torch.tensor([0] * (bs // 2) + [strenght] * (bs // 2)),
        num_inference_steps=50,
        guidance_scale=7.5,
    ).images
    processed_imgs = [ds_preprocess(img) for img in images]
    tensors = torch.cat(processed_imgs, dim=0).to("cuda")
    embeds = ds_model.embed(tensors)
    embeds = F.normalize(embeds, p=2, dim=1)
    original_embeds, changed_embeds = embeds.chunk(2)
    for o_emb, c_emb in zip(original_embeds, changed_embeds):
        distances.append(1 - (o_emb @ c_emb.t()))
    print(f"MEAN DREAMSIM DISTANCE: {float(sum(distances) / len(distances))}")

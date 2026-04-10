from custom_pipelines.local_sdxl_pipeline import LocalStableDiffusionXLPipeline
from diffusers import DDIMScheduler
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from optim_utils import set_random_seed
from argparse import ArgumentParser
from tqdm import tqdm

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 23,
        "axes.labelsize": 29,
        "axes.titlesize": 27,
        "xtick.labelsize": 27,
        "ytick.labelsize": 27,
        "legend.fontsize": 23,
        "axes.linewidth": 1.2,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.figsize": (24, 5),  # Wide format for side-by-side
    }
)


def process_and_plot(data_list, output_dir="plots"):
    """
    data_list: List[Dict[int, torch.Tensor]]
    output_dir: Folder to save images
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Data container: {key: {'raw': [], 'ratio': [], 'angle': []}}
    # We use a list to collect values from all samples for each key
    aggregated_stats = defaultdict(lambda: {"raw": [], "ratio": [], "angle": []})

    # --- 1. Calculation Loop ---
    for sample in data_list:
        # Sort keys to ensure consistent ordering if needed, though we store by key
        keys = sorted(sample.keys())

        # Stack vectors for easier batch processing
        # Shape: (num_keys, vector_dim)
        vectors = torch.cat([sample[k] for k in keys])

        # A. Calculate Raw Magnitudes (L2 Norm)
        magnitudes = torch.norm(vectors, p=2, dim=1)

        # B. Calculate Sample Mean Magnitude (for Ratio)
        sample_mean_mag = torch.mean(magnitudes)

        # C. Calculate Sample Mean Vector (for Angle)
        sample_mean_vec = torch.mean(vectors, dim=0)
        sample_mean_vec_norm = torch.norm(sample_mean_vec, p=2)

        # Iterate through keys to calculate specific metrics
        for i, k in enumerate(keys):
            vec = vectors[i]
            mag = magnitudes[i]

            # 1. Raw Magnitude
            aggregated_stats[k]["raw"].append(mag.item())

            # 2. Magnitude Ratio (current / mean_of_sample)
            ratio = (mag / sample_mean_mag).item()
            aggregated_stats[k]["ratio"].append(ratio)

            # 3. Angle to Mean Vector
            # Cosine Sim = (A . B) / (|A| * |B|)
            dot_product = torch.dot(vec, sample_mean_vec)
            cosine_sim = dot_product / (mag * sample_mean_vec_norm)

            # Clamp to handle float errors (-1.0000001)
            cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)

            angle_deg = np.degrees(np.arccos(cosine_sim.item()))
            aggregated_stats[k]["angle"].append(angle_deg)

    # --- 2. Aggregation & Plotting Helper ---

    # Extract x-axis (keys)
    sorted_keys = sorted(aggregated_stats.keys())

    def save_plot_with_variance(metric_name, ylabel, filename):
        means = []
        stds = []

        for k in sorted_keys:
            values = np.array(aggregated_stats[k][metric_name])
            means.append(np.mean(values))
            stds.append(np.std(values))

        means = np.array(means)
        stds = np.array(stds)

        plt.figure(figsize=(12, 9))

        # Plot Mean Line
        plt.plot(sorted_keys, means, label="Mean", color="blue", marker="o")

        # Fill Variance (Mean +/- Std)
        plt.fill_between(sorted_keys, means - stds, means + stds, color="blue", alpha=0.2, label="Standard Deviation")

        # plt.title(f"Average {ylabel} per Key")
        plt.xlabel("Timestep")
        plt.ylabel(ylabel)
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename}")
        plt.close()
        print(f"Saved: {output_dir}/{filename}")

    save_plot_with_variance("raw", "Gradient magnitude", "1_raw_magnitudes.pdf")
    save_plot_with_variance("ratio", "Ratio to Sample Mean (1.0 = Mean)", "2_mag_ratios.pdf")
    save_plot_with_variance("angle", "Degrees from Mean Vector", "3_angle_degrees.pdf")


def get_token_position(model, prompt: str, token: str) -> int:
    tokens = model.tokenizer.encode(prompt)
    tokens = tokens[:77]
    all_tokes = {model.tokenizer.decode(curr_token): i for i, curr_token in enumerate(tokens)}
    return all_tokes[token]


def noise_image(model, latent, timestep):
    t_noise = torch.tensor(model.scheduler.timesteps[-timestep].item())  # -t because timesteps are in decreasing order
    noise = torch.randn_like(latent)
    return model.scheduler.add_noise(latent, noise, t_noise).detach()


def encode_image(model, img, height=1024, width=1024):
    with torch.no_grad():
        img = model.image_processor.preprocess(img, width=width, height=height).to(dtype=model.vae.dtype)

        latent = model.vae.encode(img).latent_dist.sample()
        latent = latent * model.vae.config.scaling_factor
        latent = latent.to(dtype=torch.float16)
    return latent


argparser = ArgumentParser()
argparser.add_argument("--prompt", type=str, required=True)
argparser.add_argument("--prompt_random", type=str, required=True)
argparser.add_argument("--token", type=str, required=True)
argparser.add_argument("--out", type=str, required=True)
argparser.add_argument("--common", action="store_true", required=False, default=False)
argparser.add_argument("--seed", type=int, required=False, default=42)
args = argparser.parse_args()


prompt = args.prompt
prompt_random = args.prompt_random
token = args.token
del_common = args.common
output = args.out
seed = args.seed

model = LocalStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
    cache_dir="../model_cache",
)
model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
model.to("cuda")
model.scheduler.set_timesteps(50)
set_random_seed(seed)


all_grads = []
for _ in tqdm(list(range(50))):
    samples = model(
        [prompt] * 2,
        output_type="latent",
    ).images
    image_latent, image_random = samples[0], samples[1]

    token_grads = {}
    for t in list(range(50, 0, -2)):
        denoised_latent, prompt_tokens = model.get_pred_denoised_image(
            [prompt] * 1,
            latents=noise_image(model, image_latent, timestep=t).unsqueeze(0),
            num_inference_steps=50,
            guidance_scale=7.5,
            num_images_per_prompt=1,
            start_timestep=50 - t,
            output_type="latent",
        )

        loss = F.mse_loss(
            denoised_latent.float(),
            image_random.float().unsqueeze(0),
        )
        tokens_grad = torch.autograd.grad(loss, [prompt_tokens], allow_unused=True)[0]

        real_tokens_grad = tokens_grad[:, : len(model.tokenizer.encode(prompt))].to(torch.float).cpu()

        if del_common:
            common = real_tokens_grad.mean(dim=1, keepdim=True)
            real_tokens_grad = real_tokens_grad - common
        token_grad = real_tokens_grad[:, get_token_position(model, prompt, token), :]
        # token_grad = token_grad / token_grad.norm()
        token_grads[t] = token_grad
    all_grads.append(token_grads)

process_and_plot(all_grads, output_dir="outputs")

from custom_pipelines.local_sdxl_pipeline import LocalStableDiffusionXLPipeline
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import make_interp_spline
from PIL import Image
from typing import List
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel

# --- Publication Style Configuration ---
# These settings ensure high readability in papers
plt.rcParams.update(
    {
        "font.family": "serif",  # Matches LaTeX/Paper font
        "font.size": 14,  # Base font size
        "axes.labelsize": 16,  # Axis label size
        "axes.titlesize": 0,  # No title displayed
        "xtick.labelsize": 14,  # X-tick size
        "ytick.labelsize": 14,  # Y-tick size
        "axes.linewidth": 1.5,  # Thicker axis frame lines
        "lines.linewidth": 0.5,  # Thicker data lines
        "xtick.major.width": 1.5,  # Thicker tick marks
        "ytick.major.width": 1.5,
    }
)

# --- 1. Single-Column Publication Config ---
# Targeted for a width of 3.3 to 3.5 inches (standard 2-column paper width)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,  # Standard caption font size
        "axes.labelsize": 12,  # Slightly larger for axes
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,  # Compact legend
        "axes.linewidth": 1.0,  # Thinner spines for small plots
        "lines.linewidth": 1.5,  # Balanced line width
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.color": "#bdc3c7",
        "grid.linestyle": "--",
        "figure.dpi": 300,  # High resolution for screen check
    }
)


class TrajectoryAnalyzer:
    def __init__(self, model_type="clip", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_type = model_type
        print(f"Loading {model_type} model on {device}...")

        if model_type == "clip":
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        elif model_type == "dino":
            self.processor = AutoImageProcessor.from_pretrained("facebook/dino-vitb16")
            self.model = AutoModel.from_pretrained("facebook/dino-vitb16").to(device)

    def get_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if self.model_type == "clip":
                outputs = self.model.get_image_features(**inputs)
            elif self.model_type == "dino":
                outputs = self.model(**inputs).last_hidden_state
                outputs = outputs[:, 0, :]

        embeddings = outputs.cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)

    def smooth_trajectory(self, x, y, num_points=300):
        """
        Uses Cubic Spline Interpolation to smooth the jagged trajectory.
        """
        if len(x) < 4:
            return x, y

        # Parameterize by cumulative distance to handle loops
        points = np.vstack((x, y)).T
        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        distance = np.insert(distance, 0, 0) / distance[-1]

        spline_x = make_interp_spline(distance, x, k=3)
        spline_y = make_interp_spline(distance, y, k=3)

        alpha = np.linspace(0, 1, num_points)
        return spline_x(alpha), spline_y(alpha)

    def add_arrow_on_curve(self, ax, x_smooth, y_smooth, fraction=0.5, color="black", size=10):
        """
        Places a smaller, proportional arrow perfectly tangent to the curve.
        """
        idx = int(len(x_smooth) * fraction)
        idx = max(0, min(idx, len(x_smooth) - 2))

        x_start, y_start = x_smooth[idx], y_smooth[idx]
        x_end, y_end = x_smooth[idx + 1], y_smooth[idx + 1]

        arrow = FancyArrowPatch(
            posA=(x_start, y_start),
            posB=(x_end, y_end),
            arrowstyle="-|>",
            mutation_scale=size,  # Smaller mutation scale for small plots
            color=color,
            linewidth=0,
            zorder=10,
        )
        ax.add_patch(arrow)

    def analyze_and_save(self, list_of_sequences: List[List[Image.Image]], save_path="trajectory_plot.pdf"):
        all_embeddings = []

        print("Extracting embeddings...")
        for seq in list_of_sequences:
            all_embeddings.append(self.get_embeddings(seq))

        # PCA Projection
        X = np.vstack(all_embeddings)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_

        # --- Single Column Figure Size ---
        # 3.5 inches width is standard for IEEE/ACM/Nature single column
        fig, ax = plt.subplots(figsize=(3.5, 3.0))

        start_idx = 0

        # Professional Distinct Palette
        colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B3", "#CCB974", "#64B5CD"]

        for i, seq_embeddings in enumerate(all_embeddings):
            n_steps = len(seq_embeddings)
            end_idx = start_idx + n_steps
            raw_data = X_pca[start_idx:end_idx]

            x_raw, y_raw = raw_data[:, 0], raw_data[:, 1]
            c = colors[i % len(colors)]

            # 1. Smooth the trajectory
            x_smooth, y_smooth = self.smooth_trajectory(x_raw, y_raw)

            # 2. Plot Glow (Thinner for small plot)
            ax.plot(x_smooth, y_smooth, color=c, linewidth=4, alpha=0.2, zorder=2)

            # 3. Plot Main Line (Refined width)
            ax.plot(x_smooth, y_smooth, color=c, linewidth=1.5, alpha=1.0, zorder=3, label=f"Seq {i+1}")

            # 4. Start Marker (Smaller size)
            ax.scatter(x_raw[0], y_raw[0], s=30, facecolors="white", edgecolors=c, linewidth=1.5, zorder=4, marker="o")

            # 5. End Marker (Smaller size)
            ax.scatter(x_raw[-1], y_raw[-1], s=30, color=c, edgecolors="white", linewidth=1.0, zorder=4, marker="D")

            # 6. Tangent Arrows (Smaller size)
            self.add_arrow_on_curve(ax, x_smooth, y_smooth, fraction=0.4, color=c, size=10)
            self.add_arrow_on_curve(ax, x_smooth, y_smooth, fraction=0.8, color=c, size=10)

            start_idx = end_idx

        # --- Labels & Layout ---
        ax.set_xlabel(f"PC 1 ({explained_var[0]:.1%} var)", labelpad=4)
        ax.set_ylabel(f"PC 2 ({explained_var[1]:.1%} var)", labelpad=4)

        # Legend: Compact and placed optimally
        # 'loc=best' usually works, but 'frameon=False' saves space visually
        ax.legend(frameon=False, loc="best", handlelength=1.5, handletextpad=0.5)

        plt.tight_layout(pad=0.5)  # Tight layout to remove excess whitespace
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Publication-ready figure saved to: {save_path}")


def get_token_position(pipe, prompt: str, token: str) -> int:
    tokens = pipe.tokenizer.encode(prompt)
    tokens = tokens[:77]
    all_tokes = {pipe.tokenizer.decode(curr_token): i for i, curr_token in enumerate(tokens)}
    return all_tokes[token]


if __name__ == "__main__":
    pipe = LocalStableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        # variant="fp16",
        cache_dir="../model_cache",
    )
    pipe = pipe.to("cuda")

    prompt = "picture of a person"
    prompt_intervention = "picture of an old person"
    token = "person"
    batch_size = 4
    intervention_strengths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

    emb_tokens = pipe.encode_prompt(prompt)[0]
    emb_token = emb_tokens[:, get_token_position(pipe, prompt, token)]
    emb_tokens_intervention = pipe.encode_prompt(prompt_intervention)[0]
    emb_token_intervention = emb_tokens_intervention[:, get_token_position(pipe, prompt_intervention, token)]
    direction = emb_token_intervention - emb_token
    batch_directions = direction.repeat(batch_size, 1)
    token_positions = [get_token_position(pipe, prompt, token)] * batch_size

    # clip_avg_cosine = []
    # clip_straightness_index = []
    # clip_variance_ratio = []
    # clip_rmse = []
    # dino_avg_cosine = []
    # dino_straightness_index = []
    # dino_variance_ratio = []
    # dino_rmse = []
    # analyzer_clip = TrajectoryAnalyzer(model_type="clip") # or "dino"
    # analyzer_dino = TrajectoryAnalyzer(model_type="dino") # or "dino"
    # for i in tqdm(range(10000)):
    #     try:
    #         images = pipe(
    #             [prompt] * 4,
    #             guidance_scale=7.5,
    #             num_images_per_prompt=1,
    #             token_intervention=batch_directions,
    #             token_intervention_pos=token_positions,
    #             intervention_strenght=torch.tensor([0.0, 0.2, 0.4, 0.6,]),
    #         ).images
    #         images += pipe(
    #             [prompt] * 4,
    #             guidance_scale=7.5,
    #             num_images_per_prompt=1,
    #             token_intervention=batch_directions,
    #             token_intervention_pos=token_positions,
    #             intervention_strenght=torch.tensor([0.8, 1.0, 1.2, 1.4]),
    #         ).images
    #         avg_cos, straightness_index, variance_ratio, rmse = analyzer_clip.calculate_all_metrics(images)
    #         clip_avg_cosine.append(avg_cos)
    #         clip_straightness_index.append(straightness_index)
    #         clip_variance_ratio.append(variance_ratio)
    #         clip_rmse.append(rmse)
    #         avg_cos, straightness_index, variance_ratio, rmse = analyzer_dino.calculate_all_metrics(images)
    #         dino_avg_cosine.append(avg_cos)
    #         dino_straightness_index.append(straightness_index)
    #         dino_variance_ratio.append(variance_ratio)
    #         dino_rmse.append(rmse)

    #     except KeyboardInterrupt:
    #         break
    # print("CLIP Linearity Results over 10,000 samples:")
    # print(f"Average Cosine: {np.mean(clip_avg_cosine):.4f} ± {np.std(clip_avg_cosine):.4f}")
    # print(f"Straightness Index: {np.mean(clip_straightness_index):.4f} ± {np.std(clip_straightness_index):.4f}")
    # print(f"Variance Ratio: {np.mean(clip_variance_ratio):.4f} ± {np.std(clip_variance_ratio):.4f}")
    # print(f"RMSE: {np.mean(clip_rmse):.4f} ± {np.std(clip_rmse):.4f}")
    # set_random_seed(1)
    # latent1 = torch.randn((1, 4, 128, 128))
    # set_random_seed(2)
    # latent2 = torch.randn((1, 4, 128, 128))
    # set_random_seed(3)
    # latent3 = torch.randn((1, 4, 128, 128))
    # set_random_seed(4)
    # latent4 = torch.randn((1, 4, 128, 128))
    # latents = torch.cat([latent1, latent2, latent3, latent4], dim=0).to(pipe.device, pipe.dtype)

    sequences = [[] for _ in range(batch_size)]
    # for strength in intervention_strengths:
    #     images = pipe(
    #         [prompt] * batch_size,
    #         latents=latents,
    #         guidance_scale=7.5,
    #         num_images_per_prompt=1,
    #         token_intervention=batch_directions,
    #         token_intervention_pos=token_positions,
    #         intervention_strenght=torch.tensor([strength] * batch_size),
    #     ).images
    #     for i in range(batch_size):
    #         sequences[i].append(images[i])

    for i in range(batch_size):
        for j in range(len(intervention_strengths)):
            try:
                image = Image.open(f"outputs/motivation/seq{i}_step{j}.png")
            except:
                print(f"outputs/motivation/seq{i}_step{j}.png")
                raise Exception()
            sequences[i].append(image)

    # Plot all sequences as columns in a single image
    # num_sequences = len(sequences)
    # num_steps = len(sequences[0])

    # Get image dimensions from first image
    # img_width, img_height = sequences[0][0].size

    # Create a grid image: rows = steps, columns = sequences
    # grid_width = num_sequences * img_width
    # grid_height = num_steps * img_height
    # grid_image = Image.new('RGB', (grid_width, grid_height))

    # Fill grid with images
    # for seq_idx, sequence in enumerate(sequences):
    #     for step_idx, img in enumerate(sequence):
    #         x = seq_idx * img_width
    #         y = step_idx * img_height
    #         try:
    #             grid_image.paste(img, (x, y))
    #         except:
    #             print(f"outputs/motivation/seq{seq_idx}_step{step_idx}.png")
    #             raise Exception()
    #         img.save(f"outputs/motivation/seq{seq_idx}_step{step_idx}.png")

    # Save the figure
    # grid_image.save("sequences_grid.png")
    # print("Saved sequences grid to sequences_grid.png")

    analyzer = TrajectoryAnalyzer(model_type="clip")  # or "dino"
    analyzer.analyze_and_save(sequences, save_path="trajectory_clip.pdf")
    analyzer = TrajectoryAnalyzer(model_type="dino")  # or "dino"
    analyzer.analyze_and_save(sequences, save_path="trajectory_dino.pdf")


# --- Example Usage ---
# Assuming 'sequences' is your List[List[PIL.Image]]
# analyzer = TrajectoryAnalyzer(model_type="clip") # or "dino"
# analyzer.analyze_and_plot(sequences)

from custom_pipelines.local_sdxl_pipeline import LocalStableDiffusionXLPipeline
from diffusers import DDIMScheduler
import torch
import torch.nn.functional as F
from optim_utils import set_random_seed
from PIL import Image, ImageDraw, ImageFont
from argparse import ArgumentParser
import torch.nn as nn
from diffusers.models.attention_processor import AttnProcessor2_0
import gc
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class UNetProbe:
    """
    A unified tool to hook into a Diffusers UNet and capture:
    1. 'output': The tensor output of a layer.
    2. 'attention': The raw attention maps (requires layer to be an Attention module).
    3. 'gradcam': Activations prepared for gradient calculation (requires backward pass).
    """

    def __init__(self, unet, hook_config: dict[str, str]):
        """
        Args:
            unet: The UNet2DConditionModel.
            hook_config: Dict where Key = Layer Name (dot path), Value = Type.
                         Types: 'output', 'attention', 'gradcam_input', 'gradcam_output'
        """
        self.unet = unet
        self.config = hook_config
        self.hooks = []
        self.original_processors = {}

        # Storage for captured data
        # Structure: { "layer_name": { "output": tensor, "attention": tensor, "activation": tensor } }
        self.data = {}

        self._register_hooks()

    def _register_hooks(self):
        """Parses config and applies specific hooks/processors."""
        for name, hook_type in self.config.items():
            # traverse to get the actual module
            module = self._get_module_by_name(name)

            if hook_type == "attention":
                self._hook_attention(name, module)
            elif hook_type == "output":
                self.hooks.append(module.register_forward_hook(self._make_output_hook(name)))
            elif hook_type == "gradcam_input":
                self.hooks.append(module.register_forward_hook(self._make_gradcam_hook(name, use_input=True)))
            elif hook_type == "gradcam_output":
                self.hooks.append(module.register_forward_hook(self._make_gradcam_hook(name, use_input=False)))
            else:
                raise ValueError(f"Unknown hook type: {hook_type}")

    def _get_module_by_name(self, name):
        """Helper to traverse 'up_blocks.1.attentions.0' strings."""
        module = self.unet
        for part in name.split("."):
            module = getattr(module, part)
        return module

    def _make_output_hook(self, name):
        """Standard hook to save output tensors."""

        def hook(module, input, output):
            if name not in self.data:
                self.data[name] = {}
            # Detach to save memory unless we need to differentiate through it later
            # For pure monitoring, detach. If you need to optimize this value, remove detach.
            # if not self.data[name].get("output", False):
            #     self.data[name]["output"] = [output]
            # else:
            #     self.data[name]["output"].append(output)
            self.data[name]["output"] = output

        return hook

    def _make_gradcam_hook(self, name, use_input=False):
        """
        Hook that captures tensors and enables gradient tracking for Cam.
        """

        def hook(module, input, output):
            if name not in self.data:
                self.data[name] = {}

            # Select target tensor
            target = input[0] if use_input else output

            # CRITICAL: We need gradients for this tensor, even if it's intermediate
            if not target.requires_grad:
                target.requires_grad_(True)

            target.retain_grad()

            # Save reference (do NOT detach, or you lose the graph connection)
            # if not self.data[name].get("gradcam_target", False):
            #     self.data[name]["gradcam_target"] = [target]
            # else:
            #     self.data[name]["gradcam_target"].append(target)
            self.data[name]["gradcam_target"] = target

        return hook

    def _hook_attention(self, name, module):
        """Swaps the attention processor to capture maps."""
        # 1. Check if it has a processor (Standard Diffusers Attention)
        if not hasattr(module, "processor"):
            raise ValueError(f"Layer {name} is not a valid Attention module (no processor found).")

        # 2. Save original to restore later
        self.original_processors[name] = module.processor

        # 3. Create a capturing processor
        # We define it inline or use a helper class to bind it to this instance and layer name
        capturing_processor = self._create_capture_processor(name)
        module.set_processor(capturing_processor)

    def _create_capture_processor(self, layer_name):
        """Creates a custom processor instance for a specific layer."""
        probe_instance = self

        class CaptureProcessor(AttnProcessor2_0):
            def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0):
                # 1. Capture Logic (Replicating SDPA prep)
                # Note: This assumes standard Diffusers AttnProcessor2_0 logic.
                # For SDXL/SD1.5, this covers most cases.

                # Capture Q and K to compute map
                query = attn.to_q(hidden_states)

                enc_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                key = attn.to_k(enc_states)

                # Reshape heads
                batch_size, _, _ = hidden_states.shape
                query_dim = query.shape[-1]
                head_dim = query_dim // attn.heads

                q = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                k = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # Compute Matrix
                attn_scores = torch.matmul(q, k.transpose(-1, -2)) * attn.scale
                attn_map = attn_scores.softmax(dim=-1).detach()  # NOTE maybe needed

                # Store in Probe
                if layer_name not in probe_instance.data:
                    probe_instance.data[layer_name] = {}
                probe_instance.data[layer_name]["attention"] = attn_map

                # 2. Forward Pass (Delegate to super/standard)
                return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, scale)

        return CaptureProcessor()

    def compute_gradcam(self, layer_name):
        """
        Calculates Grad-CAM for a layer.
        MUST BE CALLED AFTER loss.backward()!
        """
        if layer_name not in self.data or "gradcam_target" not in self.data[layer_name]:
            return None

        # tensors = self.data[layer_name]["gradcam_target"]
        tensor = self.data[layer_name]["gradcam_target"]

        # cams = []
        # for tensor in tensors:
        # Check if gradients exist
        if tensor.grad is None:
            print(f"Warning: No gradients found for {layer_name}. Did you call loss.backward()?")
            return None

        # Grad-CAM = ReLU(GlobalAveragePooling(Gradients) * Activations)
        # OR Element-wise for pixel-level granularity: Activations * Gradients

        # 1. Global Average Pooling of Gradients (Standard Grad-CAM weights)
        # Assuming shape [Batch, Channels, H, W] or [Batch, Seq, Dim]
        if tensor.dim() == 4:
            weights = tensor.grad.mean(dim=(2, 3), keepdim=True)
        else:
            weights = tensor.grad.mean(dim=1, keepdim=True)

        # 2. Weighted Activations
        cam = weights * tensor

        # 3. ReLU (Focus on features having positive impact)
        cam = torch.relu(cam)
        # cams.append(cam)

        return cam

    def remove_hooks(self):
        """Removes all hooks and restores processors."""
        # Remove torch hooks
        for h in self.hooks:
            h.remove()
        self.hooks = []

        # Restore processors
        for name, processor in self.original_processors.items():
            module = self._get_module_by_name(name)
            module.set_processor(processor)

        self.original_processors = {}

    def cleanup(self):
        self.remove_hooks()
        self.data = {}

    # Context Manager Support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


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
argparser.add_argument("-t", type=int, required=True)
argparser.add_argument("--token", type=str, required=True)
argparser.add_argument("--out", type=str, required=True)
argparser.add_argument("--common", action="store_true", required=False, default=False)
argparser.add_argument("--seed", type=int, required=False, default=42)
args = argparser.parse_args()


t = args.t
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


def extract_spatial_mask(probe_data, layer_name, token_idx, target_shape=None):
    """
    Extracts a specific token's attention map from the probe data
    and reshapes it into a spatial mask.
    """
    # 1. Get raw attention: [Batch, Heads, H*W, Tokens]
    raw_attn = probe_data[layer_name]["attention"]

    # 2. Extract specific token and average across heads
    # Shape becomes: [Batch, H*W]
    # We use mean() across heads to get the general "focus"
    token_map = raw_attn[..., token_idx].mean(dim=1)

    # 3. Reshape to spatial dimensions
    # We infer H, W from the sequence length (assuming square aspect ratio)
    seq_len = token_map.shape[1]
    h = w = int(seq_len**0.5)
    spatial_map = token_map.view(-1, 1, h, w)

    # 4. Resize if target shape is provided (e.g. to match feature map size)
    if target_shape:
        spatial_map = F.interpolate(spatial_map, size=target_shape, mode="bilinear")

    # 5. Normalize (0 to 1) for easier thresholding
    spatial_map = spatial_map / (spatial_map.max() + 1e-6)

    return spatial_map


def loss_gram_matrix(feat_1, feat_2, mask_1, mask_2):
    """
    Matches TEXTURE/STYLE.
    Good for: Late layers (up_blocks).
    """
    feat_1 = feat_1.to(torch.float32)
    feat_2 = feat_2.to(torch.float32)

    def compute_gram(feat, mask):
        b, c, h, w = feat.shape
        # Flatten features: [B, C, N]
        feat_flat = feat.view(b, c, -1)
        # Flatten mask: [B, 1, N]
        mask_flat = mask.view(b, 1, -1)

        # Apply mask (Soft masking)
        feat_masked = feat_flat * mask_flat

        # Calculate Gram Matrix: G = F @ F.T
        gram = torch.bmm(feat_masked, feat_masked.transpose(1, 2))

        # Normalize by the number of "active" pixels (sum of mask)
        # This prevents smaller objects from having smaller Gram norms
        active_pixels = mask_flat.sum(dim=-1, keepdim=True) + 1e-6
        return gram / active_pixels

    g1 = compute_gram(feat_1, mask_1)
    g2 = compute_gram(feat_2, mask_2)

    return F.mse_loss(g1, g2)


def loss_chamfer_distance(feat_1, feat_2, mask_1, mask_2, threshold=0.1):
    """
    Matches SHAPE/GEOMETRY (Bag of Parts).
    Good for: Mid-layers (mid_block).
    WARNING: Expensive. mask > threshold is critical to reduce point count.
    """
    feat_1 = feat_1.to(torch.float32)
    feat_2 = feat_2.to(torch.float32)

    # Helper to get "Bag of Points" from a feature map
    def get_point_cloud(feat, mask):
        # feat: [B, C, H, W]
        b, c, h, w = feat.shape

        # Flatten spatial: [B, H*W, C] (Permuted for cdist)
        feat_flat = feat.view(b, c, -1).permute(0, 2, 1)
        mask_flat = mask.view(b, -1)

        clouds = []
        for i in range(b):
            # Select only features where mask > threshold
            points = feat_flat[i][mask_flat[i] > threshold]
            clouds.append(points)
        return clouds

    cloud_1 = get_point_cloud(feat_1, mask_1)
    cloud_2 = get_point_cloud(feat_2, mask_2)

    total_loss = 0
    valid_batches = 0

    for p1, p2 in zip(cloud_1, cloud_2):
        if len(p1) == 0 or len(p2) == 0:
            continue

        # Compute distance matrix [N1, N2]
        dists = torch.cdist(p1, p2, p=2)

        # Bidirectional Chamfer
        # Min dist from 1 to 2
        l1 = torch.min(dists, dim=1)[0].mean()
        # Min dist from 2 to 1
        l2 = torch.min(dists, dim=0)[0].mean()

        total_loss += l1 + l2
        valid_batches += 1

    if valid_batches == 0:
        return torch.tensor(0.0, device=feat_1.device, requires_grad=True)
    return total_loss / valid_batches


def calculate_multiscale_loss(probe_pred, probe_target, token_idx):
    """
    Calculates a balanced loss across the U-Net hierarchy.
    """
    total_loss = 0

    # 1. Extract the Master Mask (from Target)
    # We use the target's attention to define what we are looking for.
    # Note: We take it from the Target because we want the Pred to match the Target's structure
    # BUT: If Pred is in a different pose, we actually need the PRED's mask for the Pred features!
    # Correct Logic: Mask_Pred masks Feat_Pred. Mask_Target masks Feat_Target.

    attn_layer = "mid_block.attentions.0.transformer_blocks.0.attn2"

    # We get a base mask from mid-block (e.g. 32x32)
    # We will resize this dynamically for other layers.
    mask_pred_base = extract_spatial_mask(probe_pred.data, attn_layer, token_idx).detach()
    mask_target_base = extract_spatial_mask(probe_target.data, attn_layer, token_idx).detach()

    # 2. Define Layer Weights (Heuristics)
    # Chamfer needs lower weight (it's naturally large). Gram needs high weight (it's tiny).
    layer_settings = {
        "mid_block": {"type": "chamfer", "weight": 1.0},
        "up_blocks.0": {"type": "chamfer", "weight": 1.0},
        "up_blocks.1": {"type": "gram", "weight": 1000.0},  # Gram is usually ~1e-5
    }

    # 3. Iterate and Calculate
    for layer, settings in layer_settings.items():
        feat_p = probe_pred.data[layer]["output"]
        feat_t = probe_target.data[layer]["output"].detach()

        # Dynamic Resizing of the Master Mask
        # If feature map is 64x64, we upsample the 32x32 mask to match
        current_h, current_w = feat_p.shape[-2:]

        m_p = F.interpolate(mask_pred_base, size=(current_h, current_w), mode="nearest")
        m_t = F.interpolate(mask_target_base, size=(current_h, current_w), mode="nearest")

        if settings["type"] == "chamfer":
            # Shape Matching
            l = loss_chamfer_distance(feat_p, feat_t, m_p, m_t)
        else:
            # Texture Matching
            l = loss_gram_matrix(feat_p, feat_t, m_p, m_t)
        total_loss += l * settings["weight"]

    return total_loss


noise_random = torch.randn(
    (1, 4, 128, 128),
    dtype=model.dtype,
    device=model.device,
)
image_random_pil = model(
    prompt_random,
    latents=noise_random,
    output_type="pil",
).images[0]

image_random = model(
    prompt_random,
    latents=noise_random,
    output_type="latent",
).images[0]

noise = torch.randn(
    (1, 4, 128, 128),
    dtype=model.dtype,
    device=model.device,
)
image_latent_pil = model(
    prompt,
    latents=noise,
    output_type="pil",
).images[0]
image_latent = model(
    prompt,
    latents=noise,
    output_type="latent",
).images[0]

token_grads = {}

masks_target = []
masks_pred = []
for t in tqdm(list(range(50, 0, -3))):
    attn_layers = [
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn2",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn2",
        "down_blocks.2.attentions.0.transformer_blocks.2.attn2",
        "down_blocks.2.attentions.0.transformer_blocks.4.attn2",
        "down_blocks.2.attentions.0.transformer_blocks.6.attn2",
        "down_blocks.2.attentions.0.transformer_blocks.8.attn2",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2",
        "down_blocks.2.attentions.1.transformer_blocks.2.attn2",
        "down_blocks.2.attentions.1.transformer_blocks.4.attn2",
        "down_blocks.2.attentions.1.transformer_blocks.6.attn2",
        "down_blocks.2.attentions.1.transformer_blocks.8.attn2",
        "mid_block.attentions.0.transformer_blocks.0.attn2",
        "mid_block.attentions.0.transformer_blocks.2.attn2",
        "mid_block.attentions.0.transformer_blocks.4.attn2",
        "mid_block.attentions.0.transformer_blocks.6.attn2",
        "mid_block.attentions.0.transformer_blocks.8.attn2",
        "up_blocks.0.attentions.0.transformer_blocks.0.attn2",
        "up_blocks.0.attentions.0.transformer_blocks.2.attn2",
        "up_blocks.0.attentions.0.transformer_blocks.4.attn2",
        "up_blocks.0.attentions.0.transformer_blocks.6.attn2",
        "up_blocks.0.attentions.0.transformer_blocks.8.attn2",
        "up_blocks.0.attentions.1.transformer_blocks.0.attn2",
        "up_blocks.0.attentions.1.transformer_blocks.2.attn2",
        "up_blocks.0.attentions.1.transformer_blocks.4.attn2",
        "up_blocks.0.attentions.1.transformer_blocks.6.attn2",
        "up_blocks.0.attentions.1.transformer_blocks.8.attn2",
        "up_blocks.0.attentions.2.transformer_blocks.0.attn2",
        "up_blocks.0.attentions.2.transformer_blocks.2.attn2",
        "up_blocks.0.attentions.2.transformer_blocks.4.attn2",
        "up_blocks.0.attentions.2.transformer_blocks.6.attn2",
        "up_blocks.0.attentions.2.transformer_blocks.8.attn2",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2",
    ]
    unet_probe_target = UNetProbe(
        model.unet,
        hook_config={
            # "mid_block.attentions.0.transformer_blocks.0.attn2": "attention",
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2": "attention",
            "down_blocks.1.attentions.1.transformer_blocks.0.attn2": "attention",
            "down_blocks.2.attentions.0.transformer_blocks.0.attn2": "attention",
            "down_blocks.2.attentions.0.transformer_blocks.2.attn2": "attention",
            "down_blocks.2.attentions.0.transformer_blocks.4.attn2": "attention",
            "down_blocks.2.attentions.0.transformer_blocks.6.attn2": "attention",
            "down_blocks.2.attentions.0.transformer_blocks.8.attn2": "attention",
            "down_blocks.2.attentions.1.transformer_blocks.0.attn2": "attention",
            "down_blocks.2.attentions.1.transformer_blocks.2.attn2": "attention",
            "down_blocks.2.attentions.1.transformer_blocks.4.attn2": "attention",
            "down_blocks.2.attentions.1.transformer_blocks.6.attn2": "attention",
            "down_blocks.2.attentions.1.transformer_blocks.8.attn2": "attention",
            "mid_block.attentions.0.transformer_blocks.0.attn2": "attention",
            "mid_block.attentions.0.transformer_blocks.2.attn2": "attention",
            "mid_block.attentions.0.transformer_blocks.4.attn2": "attention",
            "mid_block.attentions.0.transformer_blocks.6.attn2": "attention",
            "mid_block.attentions.0.transformer_blocks.8.attn2": "attention",
            "up_blocks.0.attentions.0.transformer_blocks.0.attn2": "attention",
            "up_blocks.0.attentions.0.transformer_blocks.2.attn2": "attention",
            "up_blocks.0.attentions.0.transformer_blocks.4.attn2": "attention",
            "up_blocks.0.attentions.0.transformer_blocks.6.attn2": "attention",
            "up_blocks.0.attentions.0.transformer_blocks.8.attn2": "attention",
            "up_blocks.0.attentions.1.transformer_blocks.0.attn2": "attention",
            "up_blocks.0.attentions.1.transformer_blocks.2.attn2": "attention",
            "up_blocks.0.attentions.1.transformer_blocks.4.attn2": "attention",
            "up_blocks.0.attentions.1.transformer_blocks.6.attn2": "attention",
            "up_blocks.0.attentions.1.transformer_blocks.8.attn2": "attention",
            "up_blocks.0.attentions.2.transformer_blocks.0.attn2": "attention",
            "up_blocks.0.attentions.2.transformer_blocks.2.attn2": "attention",
            "up_blocks.0.attentions.2.transformer_blocks.4.attn2": "attention",
            "up_blocks.0.attentions.2.transformer_blocks.6.attn2": "attention",
            "up_blocks.0.attentions.2.transformer_blocks.8.attn2": "attention",
            "up_blocks.1.attentions.0.transformer_blocks.0.attn2": "attention",
            "up_blocks.1.attentions.1.transformer_blocks.0.attn2": "attention",
            "up_blocks.1.attentions.2.transformer_blocks.0.attn2": "attention",
            "mid_block": "output",
            "up_blocks.0": "output",
            "up_blocks.1": "output",
        },
    )
    with torch.no_grad():
        _, _ = model.get_pred_denoised_image(
            [prompt] * 1,
            latents=noise_image(model, image_random, timestep=t).unsqueeze(0),
            num_inference_steps=50,
            guidance_scale=7.5,
            num_images_per_prompt=1,
            start_timestep=50 - t,
            output_type="latent",
        )
    unet_probe_target.remove_hooks()

    unet_probe_pred = UNetProbe(
        model.unet,
        hook_config={
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2": "attention",
            "down_blocks.1.attentions.1.transformer_blocks.0.attn2": "attention",
            "down_blocks.2.attentions.0.transformer_blocks.0.attn2": "attention",
            "down_blocks.2.attentions.0.transformer_blocks.2.attn2": "attention",
            "down_blocks.2.attentions.0.transformer_blocks.4.attn2": "attention",
            "down_blocks.2.attentions.0.transformer_blocks.6.attn2": "attention",
            "down_blocks.2.attentions.0.transformer_blocks.8.attn2": "attention",
            "down_blocks.2.attentions.1.transformer_blocks.0.attn2": "attention",
            "down_blocks.2.attentions.1.transformer_blocks.2.attn2": "attention",
            "down_blocks.2.attentions.1.transformer_blocks.4.attn2": "attention",
            "down_blocks.2.attentions.1.transformer_blocks.6.attn2": "attention",
            "down_blocks.2.attentions.1.transformer_blocks.8.attn2": "attention",
            "mid_block.attentions.0.transformer_blocks.0.attn2": "attention",
            "mid_block.attentions.0.transformer_blocks.2.attn2": "attention",
            "mid_block.attentions.0.transformer_blocks.4.attn2": "attention",
            "mid_block.attentions.0.transformer_blocks.6.attn2": "attention",
            "mid_block.attentions.0.transformer_blocks.8.attn2": "attention",
            "up_blocks.0.attentions.0.transformer_blocks.0.attn2": "attention",
            "up_blocks.0.attentions.0.transformer_blocks.2.attn2": "attention",
            "up_blocks.0.attentions.0.transformer_blocks.4.attn2": "attention",
            "up_blocks.0.attentions.0.transformer_blocks.6.attn2": "attention",
            "up_blocks.0.attentions.0.transformer_blocks.8.attn2": "attention",
            "up_blocks.0.attentions.1.transformer_blocks.0.attn2": "attention",
            "up_blocks.0.attentions.1.transformer_blocks.2.attn2": "attention",
            "up_blocks.0.attentions.1.transformer_blocks.4.attn2": "attention",
            "up_blocks.0.attentions.1.transformer_blocks.6.attn2": "attention",
            "up_blocks.0.attentions.1.transformer_blocks.8.attn2": "attention",
            "up_blocks.0.attentions.2.transformer_blocks.0.attn2": "attention",
            "up_blocks.0.attentions.2.transformer_blocks.2.attn2": "attention",
            "up_blocks.0.attentions.2.transformer_blocks.4.attn2": "attention",
            "up_blocks.0.attentions.2.transformer_blocks.6.attn2": "attention",
            "up_blocks.0.attentions.2.transformer_blocks.8.attn2": "attention",
            "up_blocks.1.attentions.0.transformer_blocks.0.attn2": "attention",
            "up_blocks.1.attentions.1.transformer_blocks.0.attn2": "attention",
            "up_blocks.1.attentions.2.transformer_blocks.0.attn2": "attention",
            "mid_block": "output",
            "up_blocks.0": "output",
            "up_blocks.1": "output",
        },
    )
    denoised_latent, prompt_tokens = model.get_pred_denoised_image(
        [prompt] * 1,
        latents=noise_image(model, image_latent, timestep=t).unsqueeze(0),
        num_inference_steps=50,
        guidance_scale=7.5,
        num_images_per_prompt=1,
        start_timestep=50 - t,
        output_type="latent",
    )

    masks_row_pred = []
    masks_row_target = []
    for attn_layer in attn_layers:
        mask_pred_base = extract_spatial_mask(
            unet_probe_pred.data, attn_layer, get_token_position(model, prompt, token)
        ).detach()
        mask_target_base = extract_spatial_mask(
            unet_probe_target.data, attn_layer, get_token_position(model, prompt, token)
        ).detach()
        masks_row_pred.append(mask_pred_base.mean(dim=0).cpu())
        masks_row_target.append(mask_target_base.mean(dim=0).cpu())
    masks_target.append(masks_row_target)
    masks_pred.append(masks_row_pred)
    # loss = calculate_multiscale_loss(
    #     unet_probe_pred,
    #     unet_probe_target,
    #     get_token_position(model, prompt, token),
    # )
    # tokens_grad = torch.autograd.grad(loss, [prompt_tokens], allow_unused=True)[0]

    # real_tokens_grad = tokens_grad[
    #     :, :len(model.tokenizer.encode(prompt))
    # ].to(torch.float).cpu()

    # if del_common:
    #     common = real_tokens_grad.mean(dim=1, keepdim=True)
    #     real_tokens_grad = real_tokens_grad - common
    # token_grad = real_tokens_grad[:, get_token_position(model, prompt, token), :]
    # token_grad = token_grad / token_grad.norm()
    # token_grads[t] = token_grad

    unet_probe_pred.cleanup()
    unet_probe_target.cleanup()
    del unet_probe_pred
    del unet_probe_target
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


def visualize_grid_with_labels(
    image, masks_table, row_labels=None, col_labels=None, output_path="labeled_grid.png", alpha=0.6, cmap="jet"
):
    """
    Overlays a table of heat masks on an image with row/column labels and saves the result.

    Args:
        image (PIL.Image): The base image.
        masks_table (list of list of torch.Tensor): 2D table of masks.
        row_labels (list of str): Labels for the rows (left side).
        col_labels (list of str): Labels for the columns (top side).
        output_path (str): Save path.
        alpha (float): Transparency.
        cmap (str): Colormap.
    """

    rows = len(masks_table)
    cols = len(masks_table[0])

    # Validation: Ensure labels match dimensions if provided
    if row_labels and len(row_labels) != rows:
        raise ValueError(f"Number of row labels ({len(row_labels)}) does not match rows in table ({rows})")
    if col_labels and len(col_labels) != cols:
        raise ValueError(f"Number of col labels ({len(col_labels)}) does not match cols in table ({cols})")

    img_np = np.array(image)
    original_h, original_w = img_np.shape[:2]

    # Initialize subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    # Ensure axes is always a 2D array for consistent indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i in tqdm(range(rows)):
        for j in range(cols):
            ax = axes[i][j]
            mask_tensor = masks_table[i][j]

            # --- 1. Processing (Same as before) ---
            mask_tensor = mask_tensor.detach().cpu()
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            elif mask_tensor.dim() == 3:
                mask_tensor = mask_tensor.unsqueeze(0)

            mask_resized = F.interpolate(
                mask_tensor, size=(original_h, original_w), mode="bilinear", align_corners=False
            )
            mask_np = mask_resized.squeeze().numpy()

            mask_min, mask_max = mask_np.min(), mask_np.max()
            if mask_max - mask_min > 1e-5:
                mask_np = (mask_np - mask_min) / (mask_max - mask_min)

            # --- 2. Plotting ---
            ax.imshow(img_np)
            ax.imshow(mask_np, cmap=cmap, alpha=alpha)

            # Remove tick marks (numbers) but keep labels if we need them later
            ax.set_xticks([])
            ax.set_yticks([])

            # --- 3. Labeling Logic ---

            # Set Column Labels (Only on the top row)
            if i == 0 and col_labels:
                ax.set_title(col_labels[j], fontsize=5, pad=10, fontweight="bold")

            # Set Row Labels (Only on the left column)
            if j == 0 and row_labels:
                ax.set_ylabel(row_labels[i], fontsize=16, labelpad=10, fontweight="bold", rotation=90)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved labeled grid to {output_path}")


visualize_grid_with_labels(
    image_latent_pil,
    masks_pred,
    col_labels=attn_layers,
    row_labels=list(range(50, 0, -3)),
    output_path=f"pred_masks_{output}.jpg",
)
visualize_grid_with_labels(
    image_random_pil,
    masks_target,
    col_labels=attn_layers,
    row_labels=list(range(50, 0, -3)),
    output_path=f"target_masks_{output}.jpg",
)
exit(0)

intervention = token_grads[50]
for i in range(50, 0, -1):
    intervention = token_grads.pop(i, intervention)
    token_grads[i] = intervention
# for t in list(range(50,0,-1)):
#     denoised_latent, prompt_tokens = model.get_pred_denoised_image(
#         [prompt] * 1,
#         latents=noise_image(model, image_latent, timestep=t).unsqueeze(0),
#         num_inference_steps=50,
#         guidance_scale=7.5,
#         num_images_per_prompt=1,
#         start_timestep=50-t,
#         output_type="latent",
#     )

#     loss = F.mse_loss(
#         denoised_latent.float(),
#         image_random.float().unsqueeze(0),
#     )
#     tokens_grad = torch.autograd.grad(loss, [prompt_tokens], allow_unused=True)[0]

#     real_tokens_grad = tokens_grad[
#         :, :len(model.tokenizer.encode(prompt))
#     ].to(torch.float).cpu()

#     if del_common:
#         common = real_tokens_grad.mean(dim=1, keepdim=True)
#         real_tokens_grad = real_tokens_grad - common
#     token_grad = real_tokens_grad[:, get_token_position(model, prompt, token), :]
#     token_grad = token_grad / token_grad.norm()
#     token_grads[t] = token_grad
# intervention = token_grads[50]
# for i in range(50, 0, -1):
#     intervention = token_grads.pop(i, intervention)
#     token_grads[i] = intervention

strenghts = [i for i in range(0, 80, 10)]
result = []
col_labels = []
row_labels = ["original"]
row_labels += [f"s={i}" for i in strenghts]
row_labels.append("other")
res_col = []

for i in range(0, len(strenghts), 4):
    strghs = strenghts[i : i + 4]
    token_interventions = {ts: torch.cat([token_grad] * len(strghs), dim=0) for ts, token_grad in token_grads.items()}
    token_interventions_pos = torch.tensor([get_token_position(model, prompt, token)] * len(strghs))
    intervetion_strenghts = {ts: torch.tensor(strghs) for ts in token_grads.keys()}
    res_col += model(
        [prompt] * len(strghs),
        num_images_per_prompt=1,
        latents=torch.cat([noise] * len(strghs), dim=0),
        token_intervention=token_interventions,
        token_intervention_pos=token_interventions_pos,
        intervention_strenght=intervetion_strenghts,
    ).images
res_col.append(image_random_pil)
result.append(res_col)
res_col = []
col_labels.append("from t=all")
for i in range(0, len(strenghts), 4):
    strghs = strenghts[i : i + 4]
    mean_grad = torch.stack(list(token_grads.values()), dim=0).mean(dim=0)
    token_interventions = {ts: torch.cat([mean_grad] * len(strghs), dim=0) for ts, _ in token_grads.items()}
    token_interventions_pos = torch.tensor([get_token_position(model, prompt, token)] * len(strghs))
    intervetion_strenghts = {ts: torch.tensor(strghs) for ts in token_grads.keys()}
    res_col += model(
        [prompt] * len(strghs),
        num_images_per_prompt=1,
        latents=torch.cat([noise] * len(strghs), dim=0),
        token_intervention=token_interventions,
        token_intervention_pos=token_interventions_pos,
        intervention_strenght=intervetion_strenghts,
    ).images
res_col.append(image_random_pil)
result.append(res_col)
res_col = []
col_labels.append("mean grad")
for grad_t in range(49, 30, -3):
    res_col = []
    for i in range(0, len(strenghts), 4):
        strghs = strenghts[i : i + 4]
        token_interventions = {
            ts: torch.cat([token_grads[grad_t]] * len(strghs), dim=0) for ts, _ in token_grads.items()
        }
        token_interventions_pos = torch.tensor([get_token_position(model, prompt, token)] * len(strghs))
        intervetion_strenghts = {ts: torch.tensor(strghs) for ts in token_grads.keys()}
        res_col += model(
            [prompt] * len(strghs),
            num_images_per_prompt=1,
            latents=torch.cat([noise] * len(strghs), dim=0),
            token_intervention=token_interventions,
            token_intervention_pos=token_interventions_pos,
            intervention_strenght=intervetion_strenghts,
        ).images
    res_col.append(image_random_pil)
    result.append(res_col)
    col_labels.append(f"from t={grad_t}")


left_margin = 100
top_margin = 50
font = ImageFont.load_default(size=26)

cols = len(result)
rows = len(result[0])
single_w, single_h = result[0][0].size
grid_w = cols * single_w + left_margin
grid_h = rows * single_h + top_margin
grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))

draw = ImageDraw.Draw(grid)

for j, label in enumerate(col_labels):
    x = left_margin + (j * single_w) + (single_w / 2)
    y = top_margin / 2
    draw.text((x, y), label, fill="black", font=font, anchor="mm")

for i, label in enumerate(row_labels):
    x = left_margin / 2
    y = top_margin + (i * single_h) + (single_h / 2)
    draw.text((x, y), label, fill="black", font=font, anchor="mm")

for j, col in enumerate(result):
    for i, img in enumerate(col):
        grid.paste(img, (left_margin + j * single_w, top_margin + i * single_h))
grid.save(f"outputs/{output}.png")

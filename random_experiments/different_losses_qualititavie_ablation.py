import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
import matplotlib.pyplot as plt
from optim_utils import set_random_seed
from custom_pipelines.local_sdxl_pipeline import LocalStableDiffusionXLPipeline
from diffusers import DDIMScheduler
from PIL import Image, ImageDraw, ImageFont
import gc
import argparse
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
from dreamsim import dreamsim


class ImageComparator:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on device: {self.device}")

        # Placeholders for lazy loading (models are heavy, load only what you use)
        self._clip_model = None
        self._clip_processor = None
        self._dino_model = None
        self._dino_processor = None
        self._dreamsim_model = None
        self._dreamsim_preprocess = None

        # Store the calculated mean for the "Fixed CLIP" method
        self.clip_mean_vector = None

    def _load_clip(self):
        if self._clip_model is None:
            print("Loading CLIP (openai/clip-vit-base-patch32)...")
            model_id = "openai/clip-vit-base-patch32"
            self._clip_model = CLIPModel.from_pretrained(model_id).to(self.device)
            self._clip_processor = CLIPProcessor.from_pretrained(model_id)

    def _load_dino(self):
        if self._dino_model is None:
            print("Loading DINOv2 (facebook/dinov2-base)...")
            model_id = "facebook/dinov2-base"
            self._dino_processor = AutoImageProcessor.from_pretrained(model_id)
            self._dino_model = AutoModel.from_pretrained(model_id).to(self.device)

    def _load_dreamsim(self):
        if self._dreamsim_model is None:
            print("Loading DreamSim (ensemble)...")
            # pretrained=True downloads the weights automatically
            self._dreamsim_model, self._dreamsim_preprocess = dreamsim(pretrained=True, device=self.device)

    def calibrate_fixed_clip(self, image_list):
        """
        Calculates the mean embedding from a list of 'background' images
        to fix the CLIP modality gap.
        """
        self._load_clip()
        print(f"Calibrating CLIP mean with {len(image_list)} images...")

        embeddings = []
        with torch.no_grad():
            for img in image_list:
                inputs = self._clip_processor(images=img, return_tensors="pt").to(self.device)
                emb = self._clip_model.get_image_features(**inputs)
                emb = F.normalize(emb, p=2, dim=1)  # Normalize first
                embeddings.append(emb)

        # Calculate mean vector across all images
        all_embs = torch.cat(embeddings, dim=0)
        self.clip_mean_vector = torch.mean(all_embs, dim=0, keepdim=True)
        print("Calibration complete.")

    def compare_clip(self, img1: Image.Image, img2: Image.Image, use_fix=True) -> float:
        """
        Returns Cosine Similarity (1.0 = identical, -1.0 = opposite).
        If use_fix=True, subtracts the calibrated mean vector.
        """
        self._load_clip()

        images = [img1, img2]
        inputs = self._clip_processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Get embeddings
            embs = self._clip_model.get_image_features(**inputs)  # shape: [2, 512]

            # Apply the "Fix" (Modality Gap centering)
            if use_fix:
                if self.clip_mean_vector is None:
                    print("Warning: use_fix=True but no calibration done. Using standard CLIP.")
                else:
                    embs = embs - self.clip_mean_vector

            # Normalize and Dot Product (Cosine Similarity)
            embs = F.normalize(embs, p=2, dim=1)
            similarity = torch.mm(embs[0].unsqueeze(0), embs[1].unsqueeze(0).T).item()

        return similarity

    def compare_dino(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Returns Cosine Similarity using DINOv2 features.
        Good for structural/visual similarity.
        """
        self._load_dino()

        images = [img1, img2]
        # DINOv2 requires resizing to multiples of 14, processor handles this usually
        inputs = self._dino_processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._dino_model(**inputs)
            # DINOv2 stores the global representation in the last_hidden_state's [CLS] token (index 0)
            # or uses pooler_output if available. Standard is CLS token.
            embs = outputs.last_hidden_state[:, 0, :]

            # Normalize and Dot Product
            embs = F.normalize(embs, p=2, dim=1)
            similarity = torch.mm(embs[0].unsqueeze(0), embs[1].unsqueeze(0).T).item()

        return similarity

    def compare_dreamsim(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Returns Distance (0.0 = identical, Higher = more different).
        Note: DreamSim returns distance, not similarity!
        """
        self._load_dreamsim()

        # DreamSim has its own preprocessor
        img1_tensor = self._dreamsim_preprocess(img1).to(self.device)
        img2_tensor = self._dreamsim_preprocess(img2).to(self.device)

        with torch.no_grad():
            # DreamSim takes inputs directly and returns distance
            distance = self._dreamsim_model(img1_tensor, img2_tensor).item()

        return distance


def safe_pairwise_squared_dist(x, y):
    """
    Computes pairwise squared Euclidean distances: ||x - y||^2
    Uses expansion: ||x||^2 + ||y||^2 - 2<x, y>
    This avoids the 'sqrt' operation of cdist, preventing NaN gradients at dist=0.
    """
    # x: (N, C), y: (M, C)

    # 1. Compute dot product
    # dot: (N, M)
    dot = x @ y.T

    # 2. Compute norms squared
    # x_norm: (N, 1), y_norm: (1, M)
    x_norm = (x**2).sum(dim=1, keepdim=True)
    y_norm = (y**2).sum(dim=1, keepdim=True).T

    # 3. Combine
    # dist_sq = x^2 + y^2 - 2xy
    dist_sq = x_norm + y_norm - 2 * dot

    # Numerical stability clamp (variance can sometimes be -1e-8 due to precision)
    return dist_sq.clamp(min=0.0)


def get_distribution_statistics(tensor, mask, eps=1e-6):
    """
    Computes weighted mean and covariance (diagonal) of the features under the mask.

    Args:
        tensor: (B, C, H, W) - The feature map / image
        mask: (B, 1, H, W) - Attention mask (0 to 1)
    Returns:
        mu: (B, C) - Weighted mean of features
        var: (B, C) - Weighted variance of features
    """
    B, C, H, W = tensor.shape

    # Flatten spatial dims: (B, C, H*W)
    flat_tensor = tensor.view(B, C, -1)
    flat_mask = mask.view(B, 1, -1)

    # Normalize mask to sum to 1 (to treat as probabilities)
    mask_sum = flat_mask.sum(dim=2, keepdim=True) + eps
    mask_prob = flat_mask / mask_sum  # weights

    # 1. Weighted Mean (mu)
    # sum(x * w) over spatial dim
    mu = (flat_tensor * mask_prob).sum(dim=2)  # (B, C)

    # 2. Weighted Variance (var)
    # sum((x - mu)^2 * w)
    # Expand mu for broadcasting: (B, C, 1)
    mu_expanded = mu.unsqueeze(2)
    var = (mask_prob * (flat_tensor - mu_expanded) ** 2).sum(dim=2) + eps  # (B, C)

    return mu, var


def gaussian_kl_divergence(mu1, var1, mu2, var2):
    """
    Calculates KL(p || q) where p ~ N(mu1, var1) and q ~ N(mu2, var2).
    Closed form solution for diagonal multivariate Normals.
    """
    # term1: log(std2/std1) = 0.5 * (log_var2 - log_var1)
    term1 = 0.5 * (torch.log(var2) - torch.log(var1))

    # term2: (var1 + (mu1 - mu2)^2) / (2 * var2)
    term2 = (var1 + (mu1 - mu2) ** 2) / (2 * var2)

    # Result per channel, sum over channels (C)
    kl = term1 + term2 - 0.5
    return kl.sum(dim=1).mean()  # Average over batch


def compute_kl_divergence(t1, t2, m1, m2):
    mu1, var1 = get_distribution_statistics(t1, m1)
    mu2, var2 = get_distribution_statistics(t2, m2)

    # Symmetric KL (Jenson-Shannon-ish behavior): KL(P||Q) + KL(Q||P)
    kl_12 = gaussian_kl_divergence(mu1, var1, mu2, var2)
    kl_21 = gaussian_kl_divergence(mu2, var2, mu1, var1)
    return 0.5 * (kl_12 + kl_21)


def compute_multi_scale_mmd(t1, m1, t2, m2, binary=True, threshold=0.6):
    """
    Calculates Multi-Scale MMD between two feature maps.

    Args:
        t1, t2: (B, C, H, W) Feature tensors
        m1, m2: (B, 1, H, W) Attention masks
        binary: If True, hard-thresholds mask and removes background pixels.
                If False, uses mask as soft weights (slower but differentiable w.r.t mask).
        threshold: Cutoff for binary mask.
    """
    B, C, H, W = t1.shape

    # Flatten features: (B, H*W, C)
    f1 = t1.view(B, C, -1).permute(0, 2, 1)
    f2 = t2.view(B, C, -1).permute(0, 2, 1)

    # Flatten masks: (B, H*W)
    w1 = m1.view(B, -1)
    w2 = m2.view(B, -1)

    loss = 0
    valid_batch_count = 0

    for b in range(B):
        # --- PREPARATION ---
        mask_x = w1[b]
        mask_y = w2[b]

        if binary:
            # === HARD FILTERING PATH ===
            # Select only indices where mask > threshold
            idx_x = mask_x > threshold
            idx_y = mask_y > threshold

            # Skip if object is missing
            if idx_x.sum() == 0 or idx_y.sum() == 0:
                continue

            x = f1[b][idx_x]  # (N_valid_x, C)
            y = f2[b][idx_y]  # (N_valid_y, C)

            # Weights are uniform for binary MMD
            weights_x = None
            weights_y = None

        else:
            # === SOFT WEIGHTING PATH ===
            # Use all pixels, but compute normalized weights
            x = f1[b]
            y = f2[b]

            sum_x = mask_x.sum()
            sum_y = mask_y.sum()

            if sum_x < 1e-6 or sum_y < 1e-6:
                continue

            # Normalize weights to sum to 1
            weights_x = mask_x / sum_x  # (N,)
            weights_y = mask_y / sum_y  # (N,)

            # Reshape for broadcasting later: (N, 1)
            weights_x = weights_x.unsqueeze(1)
            weights_y = weights_y.unsqueeze(1)

        # --- DISTANCE CALCULATION (NaN Safe) ---
        # We perform this ONCE per pair to save compute

        # dist_sq_xx: (Nx, Nx)
        dist_sq_xx = safe_pairwise_squared_dist(x, x)
        dist_sq_yy = safe_pairwise_squared_dist(y, y)
        dist_sq_xy = safe_pairwise_squared_dist(x, y)

        # --- MEDIAN HEURISTIC (Auto-tune Sigma) ---
        # We use the combined set of X and Y to find a characteristic scale.
        # If binary=True, N is small. If binary=False, N is huge (H*W).
        # We subsample if N is too large to keep heuristic fast.

        if x.shape[0] + y.shape[0] > 2000:
            # Random subsample for heuristic estimation
            combined = torch.cat([x, y], dim=0)
            perm = torch.randperm(combined.shape[0])[:2000]
            subset = combined[perm]
            dist_sq_heuristic = safe_pairwise_squared_dist(subset, subset)
            median_dist = dist_sq_heuristic.median()
        else:
            # Use the already computed xy distances
            # Note: We use xy for heuristic as it captures the separation we care about
            median_dist = dist_sq_xy.median()

        # Prevent sigma from being 0 (if images are identical solid colors)
        base_sigma_sq = median_dist.detach().clamp(min=1e-6)

        # --- MULTI-SCALE KERNEL ---
        # Scales for sigma^2: 0.1, 0.5, 1, 2, 10
        # If sigma_sq scales by S, then gamma = 1 / (2 * S * base)
        scale_factors = [0.1, 0.5, 1.0, 2.0, 10.0]

        batch_mmd = 0

        for scale in scale_factors:
            # gamma = 1 / (2 * sigma^2)
            gamma = 1.0 / (2.0 * base_sigma_sq * scale)

            # Compute Gaussian Kernels: k(x,y) = exp(-gamma * ||x-y||^2)
            k_xx = torch.exp(-gamma * dist_sq_xx)
            k_yy = torch.exp(-gamma * dist_sq_yy)
            k_xy = torch.exp(-gamma * dist_sq_xy)

            # Apply Weights / Mean
            if binary:
                # Simple mean of the matrix
                term_xx = k_xx.mean()
                term_yy = k_yy.mean()
                term_xy = k_xy.mean()
            else:
                # Weighted Sum: w^T * K * w
                # weights are (N,1), K is (N,N)
                term_xx = (weights_x.T @ k_xx @ weights_x).squeeze()
                term_yy = (weights_y.T @ k_yy @ weights_y).squeeze()
                term_xy = (weights_x.T @ k_xy @ weights_y).squeeze()

            batch_mmd += term_xx + term_yy - 2 * term_xy

        loss += batch_mmd
        valid_batch_count += 1

    if valid_batch_count == 0:
        return torch.tensor(0.0, device=t1.device, requires_grad=True)

    return loss / valid_batch_count


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


def loss_chamfer_distance(feat_1, feat_2, mask_1, mask_2, threshold=0.6):
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


def calculate_aligned_loss(t1, t2, m1, m2):
    """
    Aligns t2 to t1 based on their masks (m2, m1) and calculates weighted MSE.

    Args:
        t1, t2: Image tensors of shape (1, C, H, W)
        m1, m2: Heatmap masks of shape (1, 1, H, W), values 0-1

    Returns:
        loss: Scalar tensor (MSE) capable of backward()
    """

    # --- Step 1: Find optimal shift (Cross-Correlation) ---
    # We detach because we don't want to differentiate through the *choice* of coordinates
    with torch.no_grad():
        # Pad m1 so m2 can slide over it completely (to find partial overlaps)
        # padding=(left, right, top, bottom)
        h, w = m1.shape[-2:]
        pad_h, pad_w = h // 2, w // 2
        m1_padded = F.pad(m1, (pad_w, pad_w, pad_h, pad_h))

        # Use conv2d as cross-correlation.
        # m2 acts as the "kernel" sliding over m1.
        # Output map shows "similarity" at every possible shift position.
        correlation_map = F.conv2d(m1_padded, m2)

        # Find the position of the max overlap
        max_val_idx = torch.argmax(correlation_map)

        # Unravel index to (y, x) coordinates in the correlation map
        # Note: the map shape is roughly (H, W) because of the padding we added
        corr_h, corr_w = correlation_map.shape[-2:]
        best_y = max_val_idx // corr_w
        best_x = max_val_idx % corr_w

        # Calculate the relative shift required
        # If the peak is exactly in the center, shift is 0.
        shift_y = best_y.item() - pad_h
        shift_x = best_x.item() - pad_w

        # print(f"Optimal Shift detected: dy={shift_y}, dx={shift_x}")

    # --- Step 2: Apply Shift (Differentiable) ---
    # Helper to shift a tensor with zero-padding
    def translate_tensor(t, dy, dx):
        # t shape: (B, C, H, W)
        B, C, H, W = t.shape
        shifted = torch.zeros_like(t)

        # Source and Target slicing ranges
        src_y_start = max(0, -dy)
        src_y_end = min(H, H - dy)
        src_x_start = max(0, -dx)
        src_x_end = min(W, W - dx)

        dst_y_start = max(0, dy)
        dst_y_end = min(H, H + dy)
        dst_x_start = max(0, dx)
        dst_x_end = min(W, W + dx)

        # Copy the slice (Differentiable operation!)
        if src_y_end > src_y_start and src_x_end > src_x_start:
            shifted[:, :, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = t[
                :, :, src_y_start:src_y_end, src_x_start:src_x_end
            ]

        return shifted

    # Align t2 and m2 to match t1's position
    t2_aligned = translate_tensor(t2, shift_y, shift_x)
    m2_aligned = translate_tensor(m2, shift_y, shift_x)

    # --- Step 3: Calculate Weighted Loss ---
    # "Multiply each image by the heatmap"
    t1_masked = t1 * m1
    t2_masked_aligned = t2_aligned * m2_aligned

    # Calculate MSE on the masked versions
    # This naturally ignores the black background because 0 - 0 = 0
    loss = F.mse_loss(t1_masked, t2_masked_aligned)

    return loss


class DAAMInterpreter:
    def __init__(self, pipeline, size: int = 1024):
        """
        Args:
            pipeline: The SDXL pipeline (must have a .unet attribute)
            size: The resolution of the final heatmap (default 1024 for SDXL)
        """
        self.pipeline = pipeline
        self.size = size
        self.original_processors = {}
        self.global_heatmap = None  # Shape: (Batch, Tokens, H, W)
        self.active = False

    def __enter__(self):
        self._register_hooks()
        self.reset()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._remove_hooks()

    def reset(self):
        self.global_heatmap = None

    def _register_hooks(self):
        self.active = True
        self.original_processors = {}

        # Define the hook function
        def daam_processor_call(
            attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs
        ):
            # Standard Attention Calculation
            batch_size, sequence_length, _ = hidden_states.shape

            if encoder_hidden_states is None:
                # Fallback for Self-Attention (not needed for DAAM)
                query = attn.to_q(hidden_states)
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)
            else:
                # Cross-Attention target
                query = attn.to_q(hidden_states)
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                query,
                key.transpose(-1, -2),
                beta=0,
                alpha=attn.scale,
            )
            attn_probs = attention_scores.softmax(dim=-1)

            # --- DAAM ACCUMULATION ---
            if self.active and encoder_hidden_states is not None:
                b_sz = batch_size
                # 1. Average heads
                if attn_probs.shape[0] > b_sz:
                    attn_probs_averaged = attn_probs.view(b_sz, -1, attn_probs.shape[1], attn_probs.shape[2]).mean(
                        dim=1
                    )
                else:
                    attn_probs_averaged = attn_probs

                # 2. Reshape Spatial assuming square aspect ratio
                spatial_dim = int(attn_probs.shape[1] ** 0.5)
                if spatial_dim * spatial_dim == attn_probs.shape[1]:
                    map_to_scale = attn_probs_averaged.permute(0, 2, 1).view(b_sz, -1, spatial_dim, spatial_dim)

                    # 3. Upscale to global size
                    upscaled = F.interpolate(
                        map_to_scale.to(dtype=torch.float32),
                        size=(self.size, self.size),
                        mode="bilinear",
                        align_corners=False,
                    )

                    # 4. Accumulate
                    if self.global_heatmap is None:
                        self.global_heatmap = upscaled
                    else:
                        self.global_heatmap += upscaled
            # -------------------------

            hidden_states = torch.bmm(attn_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states

        # Inject hooks into attn2 layers
        for name, module in self.pipeline.unet.named_modules():
            if name.endswith("attn2") and hasattr(module, "set_processor"):
                self.original_processors[name] = module.processor
                module.set_processor(daam_processor_call)

    def _remove_hooks(self):
        self.active = False
        count = 0
        for name, module in self.pipeline.unet.named_modules():
            if name in self.original_processors:
                module.set_processor(self.original_processors[name])
                count += 1
        print(f"Successfully hooked {count} attention layers.")
        self.original_processors = {}

    def get_token_heatmap(self, token_index: int) -> torch.Tensor:
        if self.global_heatmap is None:
            raise ValueError("No heatmap data collected.")

        # 1. Extract the raw map
        heatmap = self.global_heatmap[1, token_index, :, :]

        # 2. Check if the map is empty/flat
        if heatmap.max() == heatmap.min():
            print(
                "WARNING: This token has zero or constant attention. It might be a padding token or the index is wrong."
            )

        # 3. Safe Normalization
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        return heatmap.cpu()


def save_overlay(image: Image.Image, heatmap: torch.Tensor, filename: str, alpha=0.6, cmap="jet"):
    """
    Overlays the heatmap on the image and saves to disk.
    Args:
        image: The PIL image generated by the model.
        heatmap: The 2D tensor from get_token_heatmap.
        filename: Output path.
        alpha: Transparency of the heatmap layer (0.0 - 1.0).
        cmap: Matplotlib colormap scheme.
    """
    # Convert tensor to numpy
    heatmap_np = heatmap.numpy()

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # 1. Plot base image
    ax.imshow(image)

    # 2. Plot heatmap on top with transparency
    # Matplotlib will automatically stretch the heatmap to fit the image bounds
    ax.imshow(heatmap_np, cmap=cmap, alpha=alpha)

    # Remove axes and padding
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    # Save figure
    print(f"Saving overlay to {filename}...")
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)  # Close figure to free memory


def get_token_position(model, prompt: str, token: str) -> int:
    tokens = model.tokenizer.encode(prompt)
    tokens = tokens[:77]
    all_tokes = {model.tokenizer.decode(curr_token): i for i, curr_token in enumerate(tokens)}
    return all_tokes[token]


def noise_image(model, latent, timestep):
    t_noise = torch.tensor(model.scheduler.timesteps[-timestep].item())  # -t because timesteps are in decreasing order
    noise = torch.randn_like(latent)
    return model.scheduler.add_noise(latent, noise, t_noise).detach()


# 1. Setup
model = LocalStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
    cache_dir="../model_cache",
    device_map="balanced",
)
model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
# model.to("cuda")
model.scheduler.set_timesteps(50)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--token", type=str, required=True)
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--loss", nargs="+", choices=["mse", "kl", "mmd", "chamfer", "gram", "aligned_mse"], required=True)
parser.add_argument("--loss_weights", nargs="+", type=float, required=True)
parser.add_argument("--n_grad_mean", type=int, default=1)
args = parser.parse_args()

seed = args.seed
prompt = args.prompt
token = args.token
target_token_idx = get_token_position(model, prompt, token)
batch_size = args.batch_size
output = args.output
losses = args.loss
loss_weights = args.loss_weights
n_grad_mean = args.n_grad_mean
assert len(losses) == len(loss_weights), "Number of losses and weights must match."
loss_weight_dict = {loss_name: weight for loss_name, weight in zip(losses, loss_weights)}

set_random_seed(seed)

# 2. Run Inference with DAAM
daam = DAAMInterpreter(model, size=1024)

with daam:
    noise = torch.randn(
        (1, 4, 128, 128),
        dtype=model.dtype,
        device=model.device,
    )
    source_pil = model(
        prompt,
        latents=noise,
        output_type="pil",
    ).images[0]
source_latent = model(
    prompt,
    latents=noise,
    output_type="latent",
).images[0]
heatmap_source = daam.get_token_heatmap(target_token_idx).unsqueeze(0).unsqueeze(0).cpu()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()

with daam:
    noise_dst = torch.randn(
        (1, 4, 128, 128),
        dtype=model.dtype,
        device=model.device,
    )
    dst_pil = model(
        prompt,
        latents=noise_dst,
        output_type="pil",
    ).images[0]
dst_latent = (
    model(
        prompt,
        latents=noise_dst,
        output_type="latent",
    )
    .images[0]
    .unsqueeze(0)
)
heatmap_dst = daam.get_token_heatmap(target_token_idx).unsqueeze(0).unsqueeze(0).cpu()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()

token_grads = {}
for t in tqdm(list(range(50, 20, -1))):
    t_grads = []
    for i in range(n_grad_mean):
        denoised_latent, prompt_tokens = model.get_pred_denoised_image(
            [prompt] * 1,
            latents=noise_image(model, source_latent, timestep=t).unsqueeze(0),
            num_inference_steps=50,
            guidance_scale=7.5,
            num_images_per_prompt=1,
            start_timestep=50 - t,
            output_type="latent",
        )
        loss = torch.tensor(0.0, device=denoised_latent.device)
        if "chamfer" in loss_weight_dict:
            loss += loss_weight_dict["chamfer"] * loss_chamfer_distance(
                feat_1=denoised_latent,
                feat_2=dst_latent,
                mask_1=F.interpolate(heatmap_source, size=(128, 128), mode="bilinear").to(
                    device=denoised_latent.device
                ),
                mask_2=F.interpolate(heatmap_dst, size=(128, 128), mode="bilinear").to(device=denoised_latent.device),
            )
        if "aligned_mse" in loss_weight_dict:
            loss += loss_weight_dict["aligned_mse"] * calculate_aligned_loss(
                t1=denoised_latent,
                t2=dst_latent,
                m1=F.interpolate(heatmap_source, size=(128, 128), mode="bilinear").to(device=denoised_latent.device),
                m2=F.interpolate(heatmap_dst, size=(128, 128), mode="bilinear").to(device=denoised_latent.device),
            )
        if "mmd" in loss_weight_dict:
            loss += loss_weight_dict["mmd"] * compute_multi_scale_mmd(
                t1=denoised_latent,
                t2=dst_latent,
                m1=F.interpolate(heatmap_source, size=(128, 128), mode="bilinear").to(device=denoised_latent.device),
                m2=F.interpolate(heatmap_dst, size=(128, 128), mode="bilinear").to(device=denoised_latent.device),
                binary=True,
            )
        if "mse" in loss_weight_dict:
            loss += loss_weight_dict["mse"] * F.mse_loss(
                denoised_latent.float(),
                dst_latent.float(),
            )
        if "kl" in loss_weight_dict:
            loss += loss_weight_dict["kl"] * compute_kl_divergence(
                t1=denoised_latent,
                t2=dst_latent,
                m1=F.interpolate(heatmap_source, size=(128, 128), mode="bilinear").to(device=denoised_latent.device),
                m2=F.interpolate(heatmap_dst, size=(128, 128), mode="bilinear").to(device=denoised_latent.device),
            )
        if "gram" in loss_weight_dict:
            loss += loss_weight_dict["gram"] * loss_gram_matrix(
                feat_1=denoised_latent,
                feat_2=dst_latent,
                mask_1=F.interpolate(heatmap_source, size=(128, 128), mode="bilinear").to(
                    device=denoised_latent.device
                ),
                mask_2=F.interpolate(heatmap_dst, size=(128, 128), mode="bilinear").to(device=denoised_latent.device),
            )
        tokens_grad = torch.autograd.grad(loss, [prompt_tokens], allow_unused=True)[0]
        real_tokens_grad = tokens_grad[:, : len(model.tokenizer.encode(prompt))].to(torch.float).cpu()
        t_grads.append(real_tokens_grad)
    real_tokens_grad = torch.stack(t_grads, dim=0).mean(dim=0)
    token_grad = real_tokens_grad[:, get_token_position(model, prompt, token), :]
    token_grad = token_grad / token_grad.norm()
    token_grads[t] = token_grad
intervention = token_grads[50]
for i in range(50, 0, -1):
    intervention = token_grads.pop(i, intervention)
    token_grads[i] = intervention


strenghts = [i for i in range(0, 40, 5)]
result = []
col_labels = []
row_labels = ["original"]
row_labels += [f"s={i}" for i in strenghts[1:]]
row_labels.append("other")
res_col = []

for i in range(0, len(strenghts), batch_size):
    strghs = strenghts[i : i + batch_size]
    token_interventions = {ts: torch.cat([token_grad] * len(strghs), dim=0) for ts, token_grad in token_grads.items()}
    token_interventions_pos = torch.tensor([get_token_position(model, prompt, token)] * len(strghs))
    intervetion_strenghts = {ts: torch.tensor(strghs) for ts in token_grads.keys()}  # TODO
    res_col += model(
        [prompt] * len(strghs),
        num_images_per_prompt=1,
        latents=torch.cat([noise] * len(strghs), dim=0),
        token_intervention=token_interventions,
        token_intervention_pos=token_interventions_pos,
        intervention_strenght=intervetion_strenghts,
    ).images
res_col.append(dst_pil)
result.append(res_col)
res_col = []
col_labels.append("from t=all")
for i in range(0, len(strenghts), batch_size):
    strghs = strenghts[i : i + batch_size]
    mean_grad = torch.stack(list(token_grads.values()), dim=0).mean(dim=0)
    token_interventions = {ts: torch.cat([mean_grad] * len(strghs), dim=0) for ts, _ in token_grads.items()}
    token_interventions_pos = torch.tensor([get_token_position(model, prompt, token)] * len(strghs))
    intervetion_strenghts = {ts: torch.tensor(strghs) for ts in token_grads.keys()}  # TODO
    res_col += model(
        [prompt] * len(strghs),
        num_images_per_prompt=1,
        latents=torch.cat([noise] * len(strghs), dim=0),
        token_intervention=token_interventions,
        token_intervention_pos=token_interventions_pos,
        intervention_strenght=intervetion_strenghts,
    ).images
res_col.append(dst_pil)
result.append(res_col)
res_col = []
col_labels.append("mean grad")
for grad_t in range(49, 39, -3):
    res_col = []
    for i in range(0, len(strenghts), batch_size):
        strghs = strenghts[i : i + batch_size]
        token_interventions = {
            ts: torch.cat([token_grads[grad_t]] * len(strghs), dim=0) for ts, _ in token_grads.items()
        }
        token_interventions_pos = torch.tensor([get_token_position(model, prompt, token)] * len(strghs))
        intervetion_strenghts = {ts: torch.tensor(strghs) for ts in token_grads.keys()}  # TODO
        res_col += model(
            [prompt] * len(strghs),
            num_images_per_prompt=1,
            latents=torch.cat([noise] * len(strghs), dim=0),
            token_intervention=token_interventions,
            token_intervention_pos=token_interventions_pos,
            intervention_strenght=intervetion_strenghts,
        ).images
    res_col.append(dst_pil)
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
losses_str = ",".join([f"{l}={w}" for l, w in loss_weight_dict.items()])
grid.save(f"outputs/{output}_token={token}_losses={losses_str}_s{seed}.png")

comparator = ImageComparator()
results = "to src\n"
results += "clip\n"
results += ",\t".join(row_labels) + "\n"
for label, col in zip(col_labels, result):
    res_row = [label]
    for img in col:
        clip_sim = comparator.compare_clip(source_pil, img, use_fix=True)
        res_row.append(f"{clip_sim:.4f}")
    results += ",\t".join(res_row) + "\n"
results += "dino\n"
results += ",\t".join(row_labels) + "\n"
for label, col in zip(col_labels, result):
    res_row = [label]
    for img in col:
        dino_sim = comparator.compare_dino(source_pil, img)
        res_row.append(f"{dino_sim:.4f}")
    results += ",\t".join(res_row) + "\n"
results += "dreamsim\n"
results += ",\t".join(row_labels) + "\n"
for label, col in zip(col_labels, result):
    res_row = [label]
    for img in col:
        dreamsim_dist = comparator.compare_dreamsim(source_pil, img)
        res_row.append(f"{dreamsim_dist:.4f}")
    results += ",\t".join(res_row) + "\n"
results += "to dst\n"
results += "clip\n"
results += ",\t".join(row_labels) + "\n"
for label, col in zip(col_labels, result):
    res_row = [label]
    for img in col:
        clip_sim = comparator.compare_clip(dst_pil, img, use_fix=True)
        res_row.append(f"{clip_sim:.4f}")
    results += ",\t".join(res_row) + "\n"
results += "dino\n"
results += ",\t".join(row_labels) + "\n"
for label, col in zip(col_labels, result):
    res_row = [label]
    for img in col:
        dino_sim = comparator.compare_dino(dst_pil, img)
        res_row.append(f"{dino_sim:.4f}")
    results += ",\t".join(res_row) + "\n"
results += "dreamsim\n"
results += ",\t".join(row_labels) + "\n"
for label, col in zip(col_labels, result):
    res_row = [label]
    for img in col:
        dreamsim_dist = comparator.compare_dreamsim(dst_pil, img)
        res_row.append(f"{dreamsim_dist:.4f}")
    results += ",\t".join(res_row) + "\n"
losses_str = ",".join([f"{l}={w}" for l, w in loss_weight_dict.items()])
with open(f"outputs/{output}_token={token}_losses={losses_str}_s{seed}_similarity.txt", "w") as f:
    f.write(results)

import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
import os
from PIL import Image, ImageFile
import numpy as np
from scipy.stats import linregress
import random
from tqdm import tqdm  
import scipy.linalg
from datetime import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True #no truncated anyway

class SlicedGridDataset(Dataset):
    def __init__(self, folder_paths, grid=False, target_size=(299, 299)):
        if isinstance(folder_paths, str):
            folder_paths = [folder_paths]
            
        self.files = []
        for path in folder_paths:
            self.files.extend([os.path.join(path, f) for f in os.listdir(path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        self.grid = grid
        self.target_size = target_size
        self.factor = 1 if grid else 9
        self.truncated_count = 0 

    def __len__(self):
        return len(self.files) * self.factor

    def __getitem__(self, idx):
        file_idx = idx // self.factor
        path = self.files[file_idx]
        
        try:
            img = Image.open(path).convert('RGB')
        except (OSError, SyntaxError, Image.DecompressionBombError):
            self.truncated_count += 1 
            return self.__getitem__(0)
        
        if self.grid:
            img = img.resize(self.target_size, Image.BILINEAR)
            # Use uint8 for torchmetrics compatibility
            return torch.from_numpy(np.array(img)).permute(2, 0, 1).to(torch.uint8)
        
        sub_idx = idx % 9
        w, h = img.size
        pw, ph = w // 3, h // 3
        row, col = divmod(sub_idx, 3)
        box = (col * pw, row * ph, (col + 1) * pw, (row + 1) * ph)
        
        face = img.crop(box).resize(self.target_size, Image.BILINEAR)
        return torch.from_numpy(np.array(face)).permute(2, 0, 1).to(torch.uint8)

def calculate_fid_from_features(feat1, feat2):
    mu1, sigma1 = np.mean(feat1, axis=0), np.cov(feat1, rowvar=False)
    mu2, sigma2 = np.mean(feat2, axis=0), np.cov(feat2, rowvar=False)
    diff = mu1 - mu2
    eps = 1e-6
    offset = np.eye(sigma1.shape[0]) * eps
    covmean, _ = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))

def run_extrapolation_benchmark(real_dir, fake_folders, result_dir, grid=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subset_sizes = [1000, 2000, 4000, 6000, 8000, 10000, 12000]
    max_needed = subset_sizes[-1]
    
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(result_dir, f"fid_results_2_8_in{timestamp}.txt")

    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    real_ds = SlicedGridDataset(real_dir, grid=grid)
    real_loader = DataLoader(real_ds, batch_size=32, num_workers=0)
    all_real_features = []
    pbar_real = tqdm(total=max_needed, desc="📥 Extracting Real")
    with torch.no_grad():
        for batch in real_loader:
            # Pass the uint8 batch; torchmetrics will call inception internally
            feats = fid_metric.inception(batch.to(device))
            all_real_features.append(feats.squeeze().cpu().numpy())
            pbar_real.update(batch.shape[0])
            if len(np.concatenate(all_real_features)) >= max_needed: break
    pbar_real.close()
    real_feats = np.concatenate(all_real_features)[:max_needed]
    real_truncated = real_ds.truncated_count

    fake_ds = SlicedGridDataset(fake_folders, grid=True)
    random.shuffle(fake_ds.files) 
    fake_loader = DataLoader(fake_ds, batch_size=32, num_workers=0)
    all_fake_features = []
    pbar_fake = tqdm(total=max_needed, desc="🎨 Extracting Fake")
    with torch.no_grad():
        for batch in fake_loader:
            feats = fid_metric.inception(batch.to(device))
            all_fake_features.append(feats.squeeze().cpu().numpy())
            pbar_fake.update(batch.shape[0])
            if len(np.concatenate(all_fake_features)) >= max_needed: break
    pbar_fake.close()
    fake_feats = np.concatenate(all_fake_features)[:max_needed]
    fake_truncated = fake_ds.truncated_count

    # 4. Calculate Subsets
    fids = []
    for n in tqdm(subset_sizes, desc="🧮 Computing FID"):
        fids.append(calculate_fid_from_features(real_feats, fake_feats[:n]))

    # 5. Regression
    inv_n = [1/n for n in subset_sizes]
    slope, intercept, r_val, _, _ = linregress(inv_n, fids)

    # 6. Format and Save Results
    results_text = [
        f"FID Extrapolation Report - {timestamp}",
        f"Mode: {'Grid' if grid else 'Sliced Images'}",
        f"Input Format: uint8 (torchmetrics internal normalization)",
        f"Real Dir: {real_dir}",
        f"Fake Dirs: {fake_folders}",
        "="*40,
        f"Data Integrity Report:",
        f"Real images corrupted/truncated: {real_truncated}",
        f"Fake images corrupted/truncated: {fake_truncated}",
        "="*40,
        f"{'N (Samples)':<15} | {'FID Score':<15}",
        "-"*40
    ]
    for n, val in zip(subset_sizes, fids):
        results_text.append(f"{n:<15} | {val:<15.4f}")
    
    results_text.extend([
        "="*40,
        f"FID Infinity (N=∞) Estimate: {intercept:.4f}",
        f"R-squared (Linear Fit): {r_val**2:.4f}",
        "="*40
    ])

    final_output = "\n".join(results_text)
    print(final_output)
    with open(output_path, "w") as f:
        f.write(final_output)
    print(f"\n✅ Results saved to: {output_path}")

# --- Execution ---
real_dir = "/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/data/parrot_images"
fake_folders = [
    "/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/diffusion_memorization/outputs/sdxl-dmd-art-samples-with-directions-port-2-8-in",
    "/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/diffusion_memorization/outputs/sdxl-dmd-art-samples-with-directions-land-2-8-in"
]
fake_folder=["/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/diffusion_memorization/outputs/test"]
result_dir = "/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/diffusion_memorization/outputs/results"

run_extrapolation_benchmark(real_dir, fake_folders, result_dir, grid=False)
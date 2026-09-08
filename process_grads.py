import os
import numpy as np
from tqdm import tqdm

def process_and_save_grads(input_dir, output_dir, max_num=float("inf")):
    # 1. Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # 2. Get files and handle the max_num limit
    grad_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npy")])
    
    # Adjust file list based on max_num
    if max_num < len(grad_files):
        grad_files = grad_files[:max_num]

    print(f"Processing {len(grad_files)} files...")

    # 3. Iterate with tqdm progress bar
    for fname in tqdm(grad_files, desc="Slicing Gradients", unit="file"):
        # Load
        file_path = os.path.join(input_dir, fname)
        grad = np.load(file_path)
        
        # Slice [rows, 2048:]
        # Using .astype(np.float32) ensures consistency if originals were float64
        sliced_grad = grad[:, 2048:].astype(np.float32)
        
        # Save
        save_path = os.path.join(output_dir, fname)
        np.save(save_path, sliced_grad)

    print(f"\nDone! Processed files are in: {output_dir}")

input_dir="/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/diffusion_memorization/outputs/sdxl-art-pca/grads/t4/shards"
output_dir="/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/diffusion_memorization/outputs/sdxl-art-pca-cut/grads/t4/shards"
process_and_save_grads(input_dir, output_dir)
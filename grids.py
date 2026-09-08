import os
from PIL import Image
from tqdm import tqdm

def create_grids(input_dir, output_dir, grid_size=(3, 3), max_images=None):
    """
    Args:
        input_dir: Path to source images
        output_dir: Path to save grids
        grid_size: Tuple (cols, rows)
        max_images: Integer to limit how many images are processed from the start
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all PNG files and sort them alphabetically
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])
    
    # Slice the list if max_images is specified
    if max_images is not None:
        files = files[:max_images]
    
    n_total = len(files)
    n_per_grid = grid_size[0] * grid_size[1]
    n_grids = n_total // n_per_grid
    
    if n_grids == 0:
        print("Not enough images to create even one grid!")
        return

    print(f"Processing {n_total} images into {n_grids} grids...")

    # tqdm creates the visual progress bar
    for i in tqdm(range(0, n_grids * n_per_grid, n_per_grid), desc="Building Grids"):
        batch = files[i:i + n_per_grid]
        
        # Load first image to set grid scale
        with Image.open(os.path.join(input_dir, batch[0])) as temp_img:
            w, h = temp_img.size
        
        grid_img = Image.new('RGB', (w * grid_size[0], h * grid_size[1]))

        for index, file_name in enumerate(batch):
            with Image.open(os.path.join(input_dir, file_name)).convert('RGB') as img:
                # Calculate position (column, row)
                x = (index % grid_size[0]) * w
                y = (index // grid_size[0]) * h
                grid_img.paste(img, (x, y))

        output_name = f"grid_{i // n_per_grid:04d}.png"
        grid_img.save(os.path.join(output_dir, output_name))

    print(f"\n✅ Done! Saved {n_grids} grids to: {output_dir}")


# --- Configuration ---
input_folder = "/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/diffusion_memorization/outputs/sdxl-dmd-art-samples-with-directions-port-2-7-in-raw"
output_folder = "/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/diffusion_memorization/outputs/sdxl-dmd-art-samples-with-directions-port-2-7-in-raw-grids"

#input_folder = "/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/diffusion_memorization/outputs/sdxl-dmd-art-samples-with-directions-land-2-7-in-raw40"
#output_folder = "/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/diffusion_memorization/outputs/sdxl-dmd-art-samples-with-directions-land-2-7-in-raw40-grids"

create_grids(input_folder, output_folder, max_images=180)
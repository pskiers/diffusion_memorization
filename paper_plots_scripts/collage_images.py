import os
from PIL import Image

# --- CONFIGURATION ---
INPUT_FOLDER = (
    "/data/pskiers/concept_dimention/sdxl-dmd-monster-samples-with-directions-from-sdxl-sae/"  # Folder with your images
)
OUTPUT_FILENAME = "collage_elrond_sae_from_sdxl.pdf"  # Output file
PADDING = 2  # "Almost no padding"
BACKGROUND_COLOR = (255, 255, 255)

# Fixed Grid Settings for A4 Portrait
# 5 columns x 8 rows = 40 images max per page
COLS = 6
ROWS = 8

# Blacklist of filenames to skip
BLACKLIST = []

# A4 Dimensions at 300 DPI
A4_WIDTH = 2480
A4_HEIGHT = 3508


def create_fixed_grid_collage(input_dir, output_file, blacklist):
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_files = []

    # 1. Load Images
    try:
        files = sorted(os.listdir(input_dir))
        print(files[:50])
    except FileNotFoundError:
        print(f"Error: Folder '{input_dir}' not found.")
        return

    for f in files:
        if f in blacklist:
            continue
        if os.path.splitext(f)[1].lower() in valid_extensions:
            image_files.append(os.path.join(input_dir, f))

    # Limit to 40 images (5x8)
    max_images = COLS * ROWS
    if len(image_files) > max_images:
        print(f"Warning: Found {len(image_files)} images. Only the first {max_images} will be used.")
        image_files = image_files[:max_images]

    if not image_files:
        print("No images found.")
        return

    print(f"Processing {len(image_files)} images for a {COLS}x{ROWS} grid...")

    # 2. Calculate Tile Size
    # We calculate the max square that fits in the columns (width) vs rows (height)
    # and take the smaller of the two to ensure it fits both ways.

    avail_width = A4_WIDTH - ((COLS - 1) * PADDING)
    avail_height = A4_HEIGHT - ((ROWS - 1) * PADDING)

    tile_w = avail_width // COLS
    tile_h = avail_height // ROWS

    # Square size is the limiting dimension
    tile_size = min(tile_w, tile_h)

    print(f"Tile Size: {tile_size}x{tile_size} pixels")

    # 3. Create Canvas
    canvas = Image.new("RGB", (A4_WIDTH, A4_HEIGHT), BACKGROUND_COLOR)

    # Calculate centering offsets (to put the grid in the exact middle of the A4)
    total_grid_w = (tile_size * COLS) + (PADDING * (COLS - 1))
    total_grid_h = (tile_size * ROWS) + (PADDING * (ROWS - 1))

    start_x = (A4_WIDTH - total_grid_w) // 2
    start_y = (A4_HEIGHT - total_grid_h) // 2

    # 4. Populate Grid
    idx = 0
    for r in range(ROWS):
        for c in range(COLS):
            if idx >= len(image_files):
                break

            img_path = image_files[idx]
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")

                    # High quality resize
                    img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

                    x = start_x + c * (tile_size + PADDING)
                    y = start_y + r * (tile_size + PADDING)

                    canvas.paste(img, (x, y))
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

            idx += 1

    # 5. Save
    canvas.save(output_file, "PDF", resolution=300.0)
    print(f"Done! Saved to {output_file}")


if __name__ == "__main__":
    if not os.path.exists(INPUT_FOLDER):
        print(f"Please create the folder '{INPUT_FOLDER}' and put images inside.")
    else:
        create_fixed_grid_collage(INPUT_FOLDER, OUTPUT_FILENAME, BLACKLIST)

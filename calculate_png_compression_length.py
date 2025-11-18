import io
from PIL import Image
import os
import json
from tqdm import tqdm
from torchvision import transforms


def png_compression_length(img: Image.Image, optimize: bool = True, compress_level: int = 9) -> int:
    """
    Returns the size in bytes of the given PIL Image when saved as a PNG.

    Args:
      img:        A PIL.Image.Image instance.
      optimize:   Whether to let Pillow optimize the PNG palette/filtering.
      compress_level: PNG compression level (0–9, where 9 is slowest/higher compression).

    Returns:
      The length in bytes of the in‑memory PNG.
    """
    buffer = io.BytesIO()
    img.save(
        buffer,
        format='PNG',
        optimize=optimize,
        compress_level=compress_level
    )
    # buffer.tell() gives the current position == size in bytes
    return buffer.tell()


def get_image_caption_iterator(
    image_dir: str,
    captions_file: str,
):
    """Yields (PIL.Image, caption), resizing each to `size`."""
    captions_path = os.path.join(image_dir, captions_file)
    with open(captions_path, "r", encoding="utf-8") as f:
        captions = json.load(f)

    # tf = transforms.Resize((1024, 1024), Image.LANCZOS)
    for fname, caption in captions.items():
        img_path = os.path.join(image_dir, fname)
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
        # img = tf(img)
        yield img, caption  # shape (1,3,H,W)

def main(path):
    prompt_dict = {}
    for img, caption in tqdm(get_image_caption_iterator(path, "captions_ai.json")):
        if caption not in prompt_dict:
            size_bytes = png_compression_length(img)
            size_bytes = int(size_bytes / (img.width * img.height) * (1024 * 1024))
            prompt_dict[caption] = size_bytes
    json.dump(prompt_dict, open("png_compression_lengths_elsa_sdxl_ai_jpg.json", "w"), indent=2)
if __name__ == "__main__":
    main("../data/elsa/stabilityai/stable-diffusion-xl-base-1.0")
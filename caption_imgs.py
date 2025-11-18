import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

# CONFIG
MODEL = "Salesforce/blip2-flan-t5-xl"
IMAGE_DIR = "../data/elsa/stabilityai/stable-diffusion-xl-base-1.0"
OUT_JSON = "../data/elsa/stabilityai/stable-diffusion-xl-base-1.0/captions_ai.json"
BATCH_SIZE = 4


def main():

    processor = InstructBlipProcessor.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl",
        cache_dir="../model_cache",
    )
    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl",
        device_map="auto",
        cache_dir="../model_cache",
        low_cpu_mem_usage=True,
    )

    model.eval()

    p = Path(IMAGE_DIR)
    imgs = sorted([x for x in p.glob("*") if x.suffix.lower() in (".jpg",".jpeg",".png", ".webp")])

    to_save = {}
    for i in tqdm(range(0, len(imgs), BATCH_SIZE)):
        batch = imgs[i:i+BATCH_SIZE]
        images = [Image.open(x).convert("RGB") for x in batch]
        prompts = ["Describe only what is visibly present in this image in one concise factual paragraph (at most 77 tokens). Mention the main objects and counts, notable colors/textures, actions/relationships, and the setting/lighting if visible. Do NOT guess identities, dates, or unseen context; do not add extra commentary."] * len(images)

        inputs = processor(
            images=images, 
            text=prompts, 
            return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=5,
                    max_length=77,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
            )
            captions = processor.batch_decode(outputs, skip_special_tokens=True)
        for img_path, caption in zip(batch, captions):
            to_save[img_path.name] = caption

        for im in images:
            im.close()
    
    with open(OUT_JSON, "w", newline="", encoding="utf-8") as fout:
        json.dump(to_save, fout, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()

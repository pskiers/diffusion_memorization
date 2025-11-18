import json

with open("outputs/attributions/sdxl_dreambooth_dog_prompt_sign_flipd.json", "r") as f:
    data1 = json.load(f)
with open("outputs/attributions/sdxl_dreambooth_dog_prompt_lora_sign_flipd.json", "r") as f:
    data2 = json.load(f)
with open("outputs/attributions/sdxl_dreambooth_dog_prompt_lora_prior_preservation_sign_flipd.json", "r") as f:
    data3 = json.load(f)

word = "painting"

for data, name in [(data1, "original"), (data2, "DB no prior"), (data3, "DB prior")]:
    sks_lids = []
    no_sks_lids = []
    for prompt, entry in data.items():
        # print(prompt, entry )
        # if "sks" in prompt:

        #     sks_lids.append(entry["flipd"] if entry["flipd"] == entry["flipd"] else 0)
        # else:
        #     no_sks_lids.append(entry["flipd"] if entry["flipd"] == entry["flipd"] else 0)

        for token, value in entry.get("grads", []):
            if token == word:
                if "sks" in prompt:
                    sks_lids.append(value if value == value else 0)
                else:
                    no_sks_lids.append(value if value == value else 0)
    print(f"{name}")
    print(f"Mean flipd sks: {sum(sks_lids) / len(sks_lids)}")
    print(f"Mean flipd no sks: {sum(no_sks_lids) / len(no_sks_lids)}")

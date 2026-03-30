import os
import json
import requests
from tqdm import tqdm

os.makedirs("data/images", exist_ok=True)

with open("data/dataset.json") as f:
    dataset = json.load(f)

new_dataset = []

for product in tqdm(dataset):
    image_paths = []

    for i, url in enumerate(product["images"]):
        try:
            img_data = requests.get(url, timeout=10).content
            filename = f"{product['product_id']}_{i}.jpg"
            filepath = f"data/images/{filename}"

            with open(filepath, "wb") as f:
                f.write(img_data)

            image_paths.append(filepath)

        except Exception as e:
            print(f"Failed: {url}")

    product["images"] = image_paths
    new_dataset.append(product)

# Save updated dataset
with open("data/dataset_local.json", "w") as f:
    json.dump(new_dataset, f, indent=2)

print("✅ Images downloaded & dataset updated!")
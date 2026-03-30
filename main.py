import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# 1. Device
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------
# 2. Load CLIP Model
# -------------------------------
def load_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess


# -------------------------------
# 3. Feature Extraction
# -------------------------------
def get_embedding(model, preprocess, img_path):
    try:
        if os.path.getsize(img_path) == 0:
            return None

        img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    except Exception:
        print(f"Skipping bad image: {img_path}")
        return None

    with torch.no_grad():
        emb = model.encode_image(img)

    emb = emb.cpu().numpy().flatten()

    # Normalize (IMPORTANT improvement)
    emb = emb / np.linalg.norm(emb)

    if np.isnan(emb).any():
        return None

    return emb


# -------------------------------
# 4. Load Dataset
# -------------------------------
def load_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)


# -------------------------------
# 5. Show Categories
# -------------------------------
def show_categories(dataset):
    categories = sorted(set([item["category"] for item in dataset]))

    print("\nAvailable Categories:\n")
    for i, cat in enumerate(categories, 1):
        print(f"{i}. {cat}")

    return categories


# -------------------------------
# 6. Build Embedding Database
# -------------------------------
def build_database(model, preprocess, dataset):
    embeddings = []
    metadata = []

    for product in tqdm(dataset, desc="Indexing images"):
        for img_path in product["images"]:
            if os.path.exists(img_path):
                emb = get_embedding(model, preprocess, img_path)
                if emb is not None:
                    embeddings.append(emb)
                    metadata.append({
                        "product_id": product["product_id"],
                        "name": product["name"],
                        "category": product["category"],
                        "image": img_path
                    })

    return np.array(embeddings), metadata


# -------------------------------
# 7. Search Similar Products
# -------------------------------
def search(query_path, model, preprocess, embeddings, metadata, top_k=3, category_filter=None):
    query_emb = get_embedding(model, preprocess, query_path)

    if query_emb is None:
        raise ValueError("Query embedding failed")

    scores = cosine_similarity([query_emb], embeddings)[0]
    sorted_indices = np.argsort(scores)[::-1]

    results = []
    for idx in sorted_indices:
        if metadata[idx]["image"] == query_path:
            continue

        if category_filter and metadata[idx]["category"] != category_filter:
            continue

        result = metadata[idx].copy()
        result["similarity"] = float(scores[idx])
        results.append(result)

        if len(results) == top_k:
            break

    return results


# -------------------------------
# 8. Show Results (Improved)
# -------------------------------
def show_results(query_image, results, metadata):
    plt.figure(figsize=(14, 5))

    # Get query info
    query_item = next((item for item in metadata if item["image"] == query_image), None)
    query_category = query_item["category"] if query_item else "Unknown"
    query_id = query_item["product_id"] if query_item else "N/A"

    # Query image
    plt.subplot(1, 4, 1)
    img = Image.open(query_image)
    plt.imshow(img)
    plt.title(f"Query\nID: {query_id}\n{query_category}")
    plt.axis("off")

    # Results
    for i, r in enumerate(results):
        plt.subplot(1, 4, i + 2)
        img = Image.open(r["image"])
        plt.imshow(img)

        title_text = (
            f"ID: {r['product_id']}\n"
            f"{r['name'][:20]}\n"
            f"{r['category']} | {r['similarity']:.2f}"
        )

        plt.title(title_text, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------------
# 9. Main Pipeline
# -------------------------------
def main():
    dataset_path = "data/dataset_local.json"

    print("Loading dataset...")
    dataset = load_dataset(dataset_path)

    # NEW: Show categories
    categories = show_categories(dataset)

    print("\nLoading CLIP model...")
    model, preprocess = load_model()

    print("Building embedding database...")
    embeddings, metadata = build_database(model, preprocess, dataset)

    print(f"\nTotal indexed images: {len(embeddings)}")

    if len(metadata) == 0:
        print("No images found!")
        return

    # -------------------------------
    # Category filter (USER CONTROL)
    # -------------------------------
    use_filter = input("\nDo you want to filter by category? (y/n): ").strip().lower()

    selected_category = None
    if use_filter == "y":
        try:
            choice = int(input("Enter category number: "))
            selected_category = categories[choice - 1]
            print(f"Selected category: {selected_category}")
        except:
            print("Invalid choice. No filter applied.")

    # -------------------------------
    # Query image
    # -------------------------------
    query_image = input("\nEnter image path (or press Enter for random): ").strip()

    if query_image == "" or not os.path.exists(query_image):
        query_image = random.choice(metadata)["image"]
        print(f"Using random dataset image: {query_image}")

    # -------------------------------
    # Test embedding
    # -------------------------------
    test_emb = get_embedding(model, preprocess, query_image)

    if test_emb is None:
        print("Query embedding failed!")
        return
    else:
        print("Query embedding OK:", test_emb.shape)

    # -------------------------------
    # Search
    # -------------------------------
    print("\nSearching similar products...")
    results = search(
        query_image,
        model,
        preprocess,
        embeddings,
        metadata,
        top_k=3,
        category_filter=selected_category
    )

    print("\nTop 3 Similar Products:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['name']} (ID: {r['product_id']})")
        print(f"   Category: {r['category']}")
        print(f"   Similarity: {r['similarity']:.4f}")
        print(f"   Image: {r['image']}\n")

    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Results saved to results.json")

    # Show results
    show_results(query_image, results, metadata)


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    main()

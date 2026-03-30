# Image-Based Product Similarity (CLIP)

## Overview
This project finds similar products from an e-commerce dataset using image similarity.

## Approach
- Used OpenAI CLIP (ViT-B/32)
- Extracted image embeddings
- Used cosine similarity for retrieval
- Returned top-3 similar products

## Dataset
Custom dataset with:
- product_id
- name
- category
- images

## How to Run

```bash
pip install -r requirements.txt
python main.py
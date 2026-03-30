#  Product Similarity Search using CLIP

##  Overview
This project implements an **image-based product similarity system** for an e-commerce store.

Given a query product image, the system retrieves the **top-3 visually similar products** using **OpenAI CLIP (ViT-B/32)** and cosine similarity.

---

##  Features
-  Image similarity using CLIP embeddings  
-  Category-based filtering  
-  Visualization of query and results  
-  Fast similarity search using cosine similarity  
-  Supports multiple images per product  

---

##  Tech Stack
- Python  
- PyTorch  
- OpenAI CLIP  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

##  Project Structure
```
project/
│── data/
│   ├── images/
│   ├── dataset.json
│   ├── dataset_local.json
│
│── main.py
│── download_images.py
│── requirements.txt
│── README.md
```

##  Dataset Format

```json
{
  "product_id": "123",
  "name": "Product Name",
  "category": "Category",
  "images": ["path1.jpg", "path2.jpg"]
}
```

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python main.py
```
Enter an image path
OR press Enter to use a random image

## Output

Prints:
- Product ID  
- Name  
- Category  
- Similarity score 

## Approach

Load CLIP model (ViT-B/32)
Extract image embeddings
Build embedding database
Compute cosine similarity
Retrieve top-3 similar products

## Future Improvements

Use FAISS for faster search
Add text-based search
Deploy as a web app (Streamlit/Flask)

## Author

Harsha
---
# Visual Product Similarity & Image-Based Recommendation System

## Overview
This project builds an Amazon-style visual recommendation system that finds visually similar products using only product images.

## Objective
- Extract high-quality image embeddings using a pretrained CNN
- Build a fast similarity search engine with FAISS
- Retrieve Top-K visually similar products
- Provide an optional Streamlit web app interface

## Tech Stack
- Python
- PyTorch
- ResNet50
- FAISS
- Streamlit

## Project Structure
```text
visual-product-similarity/
│
├── requirements.txt
├── README.md
├── data/
│   ├── raw_images/
│   ├── processed/
│   └── metadata.csv
├── models/
│   ├── embeddings.npy
│   ├── image_paths.pkl
│   ├── metadata.pkl
│   └── faiss_index.bin
└── src/
    ├── config.py
    ├── utils.py
    ├── extract_embeddings.py
    ├── build_index.py
    ├── search.py
    ├── evaluate.py
    └── app.py
```

## Dataset
Use any one public dataset:
- Amazon Product Images Dataset
- Stanford Online Products Dataset
- DeepFashion Dataset

Minimum recommended size: 1000 images.

## Metadata format
Create `data/metadata.csv` like this:

```csv
image_name,product_id,category,title,price,availability
img1.jpg,P001,Shoes,Running Shoes,1999,In Stock
img2.jpg,P002,Shoes,White Sneakers,2499,In Stock
```

## Installation
```bash
pip install -r requirements.txt
```

## Run steps

### 1. Extract embeddings
```bash
python src/extract_embeddings.py
```

### 2. Build FAISS index
```bash
python src/build_index.py
```

### 3. Run evaluation
```bash
python src/evaluate.py
```

### 4. Launch Streamlit app
```bash
streamlit run src/app.py
```

## Features
- Image embedding extraction using pretrained ResNet50
- FAISS-based fast similarity retrieval
- Cosine similarity search
- Optional filtering by category, price, availability
- Streamlit UI for image upload and result display

## Evaluation
- Precision@K
- Recall@K
- Visual inspection

## Notes
- Place all product images inside `data/raw_images/`
- If metadata is unavailable, the project still works with image paths only
- For better results, use category labels and metadata filtering

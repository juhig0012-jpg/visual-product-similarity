"""
Extract ResNet50 embeddings → Build FAISS index
Run: python extract_embeddings.py
"""
import pandas as pd
import numpy as np
import faiss
import os
from utils.model import extract_embedding
from PIL import Image

print("Loading products...")
df = pd.read_csv('data/products.csv')

print(" Processing images...")
embeddings = []
valid_products = []

for idx, row in df.iterrows():
    img_path = row['image_path']
    if os.path.exists(img_path):
        try:
            emb = extract_embedding(img_path)
            embeddings.append(emb)
            valid_products.append(row)
            print(f" {row['name'][:30]}...")
        except Exception as e:
            print(f" {img_path}: {e}")
    else:
        print(f" Missing: {img_path}")

if embeddings:
    embeddings = np.vstack(embeddings)
    df_valid = pd.DataFrame(valid_products).reset_index(drop=True)
    
    # FAISS Index (Cosine Similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))
    
    # Save
    os.makedirs('models', exist_ok=True)
    faiss.write_index(index, 'models/faiss.index')
    df_valid.to_pickle('data/products.pkl')
    
    print(f"\n SUCCESS!")
    print(f" {len(df_valid)} products processed")
    print(f" Embedding dimension: {dim}")
    print(f" Saved: models/faiss.index, data/products.pkl")
    print("\n Run: streamlit run app.py")
else:
    print(" No valid images found!")
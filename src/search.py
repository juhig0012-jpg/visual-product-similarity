import os

# torch and faiss each ship their own OpenMP runtime DLL - loading both in the
# same process trips "OMP: Error #15" on Windows. Has to be set before either
# gets imported below.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
import torch

from config import FAISS_INDEX_FILE, IMAGE_PATHS_FILE, METADATA_PKL, TOP_K
from utils import (
    build_feature_extractor,
    get_device,
    get_preprocess_transform,
    l2_normalize,
    load_image,
    load_pickle,
)


class VisualSearchEngine:
    def __init__(self):
        self.index = faiss.read_index(str(FAISS_INDEX_FILE))
        self.image_paths = load_pickle(IMAGE_PATHS_FILE)
        self.metadata = load_pickle(METADATA_PKL)

        self.device = get_device()
        self.model = build_feature_extractor().to(self.device)
        self.transform = get_preprocess_transform()

    def extract_query_embedding(self, image_path):
        image = load_image(image_path)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)
            features = features.view(features.size(0), -1)
            features = features.cpu().numpy().astype("float32")

        features = l2_normalize(features)
        return features

    def search(self, query_image_path, top_k=TOP_K, category=None, min_price=None, max_price=None, availability=None):
        query_embedding = self.extract_query_embedding(query_image_path)
        # over-fetch since category/price/availability filters below can knock
        # out most of the top matches - grab 5x and trim after filtering
        scores, indices = self.index.search(query_embedding, top_k * 5)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            item = dict(self.metadata[idx]) if idx < len(self.metadata) else {}
            item["similarity_score"] = float(score)
            item["image_path"] = self.image_paths[idx]

            if category and item.get("category") != category:
                continue

            if min_price is not None:
                try:
                    if float(item.get("price", 0)) < min_price:
                        continue
                except Exception:
                    continue

            if max_price is not None:
                try:
                    if float(item.get("price", 0)) > max_price:
                        continue
                except Exception:
                    continue

            if availability and item.get("availability") != availability:
                continue

            results.append(item)

            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":
    engine = VisualSearchEngine()
    sample_query = input("Enter query image path: ").strip()
    results = engine.search(sample_query, top_k=5)

    for i, item in enumerate(results, start=1):
        print(f"{i}. {item.get('image_name')} | Score: {item['similarity_score']:.4f}")
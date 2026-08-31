import faiss
import numpy as np

from config import EMBEDDINGS_FILE, FAISS_INDEX_FILE, MODELS_DIR


def build_faiss_index():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(EMBEDDINGS_FILE).astype("float32")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_FILE))

    print(f"FAISS index saved to: {FAISS_INDEX_FILE}")
    print(f"Total vectors indexed: {index.ntotal}")
    print(f"Embedding dimension: {dimension}")


if __name__ == "__main__":
    build_faiss_index()
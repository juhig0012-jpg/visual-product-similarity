import faiss
import numpy as np

from config import EMBEDDINGS_FILE, FAISS_INDEX_FILE, MODELS_DIR


def build_faiss_index():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(EMBEDDINGS_FILE).astype("float32")
    dimension = embeddings.shape[1]

    # HNSW graph index instead of a flat index - actual ANN search instead of
    # brute force, so this doesn't fall over once the catalog is more than a
    # few thousand images. inner product on the L2-normalized embeddings we
    # already saved = cosine similarity.
    index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 40
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_FILE))

    print(f"FAISS index saved to: {FAISS_INDEX_FILE}")
    print(f"Total vectors indexed: {index.ntotal}")
    print(f"Embedding dimension: {dimension}")


if __name__ == "__main__":
    build_faiss_index()
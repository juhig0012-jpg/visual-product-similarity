from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_IMAGE_DIR = DATA_DIR / "raw_images"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_CSV = DATA_DIR / "metadata.csv"

MODELS_DIR = BASE_DIR / "models"
EMBEDDINGS_FILE = MODELS_DIR / "embeddings.npy"
IMAGE_PATHS_FILE = MODELS_DIR / "image_paths.pkl"
METADATA_PKL = MODELS_DIR / "metadata.pkl"
FAISS_INDEX_FILE = MODELS_DIR / "faiss_index.bin"

IMAGE_SIZE = 224
BATCH_SIZE = 32
TOP_K = 5
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
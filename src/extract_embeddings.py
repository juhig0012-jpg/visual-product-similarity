import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    EMBEDDINGS_FILE,
    IMAGE_PATHS_FILE,
    METADATA_CSV,
    METADATA_PKL,
    MODELS_DIR,
)
from utils import (
    attach_metadata_to_paths,
    build_feature_extractor,
    get_device,
    get_image_paths,
    get_preprocess_transform,
    l2_normalize,
    load_image,
    load_metadata,
    save_pickle,
)


class ProductImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = load_image(image_path)
        except ValueError as exc:
            # one bad file shouldn't kill a run over a few thousand images -
            # skip it and flag it so it's obvious afterward which ones failed
            print(f"Skipping unreadable image: {exc}")
            return None
        image = self.transform(image)
        return image, str(image_path)


def _skip_unreadable(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def extract_embeddings():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = get_image_paths()
    if not image_paths:
        raise ValueError("No images found in data/raw_images/")

    metadata_df = load_metadata(METADATA_CSV)
    metadata_records = attach_metadata_to_paths(image_paths, metadata_df)

    transform = get_preprocess_transform()
    dataset = ProductImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_skip_unreadable)

    device = get_device()
    model = build_feature_extractor().to(device)

    all_embeddings = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            if batch is None:
                continue
            batch_images, batch_paths = batch
            batch_images = batch_images.to(device)
            features = model(batch_images)
            features = features.view(features.size(0), -1)
            features = features.cpu().numpy()

            all_embeddings.append(features)
            all_paths.extend(batch_paths)

    embeddings = np.vstack(all_embeddings).astype("float32")
    embeddings = l2_normalize(embeddings)

    np.save(EMBEDDINGS_FILE, embeddings)
    save_pickle(all_paths, IMAGE_PATHS_FILE)
    save_pickle(metadata_records, METADATA_PKL)

    print(f"Saved embeddings: {EMBEDDINGS_FILE}")
    print(f"Saved image paths: {IMAGE_PATHS_FILE}")
    print(f"Saved metadata: {METADATA_PKL}")
    print(f"Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    extract_embeddings()
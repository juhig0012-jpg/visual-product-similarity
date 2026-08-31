import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms

from config import IMAGE_SIZE, RAW_IMAGE_DIR, SUPPORTED_EXTENSIONS


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_image_paths(image_dir=RAW_IMAGE_DIR):
    image_paths = []
    for path in Path(image_dir).rglob("*"):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(path)
    return sorted(image_paths)


def get_preprocess_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


def load_and_preprocess_image(image_path):
    image = load_image(image_path)
    transform = get_preprocess_transform()
    return transform(image)


def build_feature_extractor():
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    return feature_extractor


def l2_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return vectors / norms


def save_pickle(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_metadata(metadata_csv_path):
    if not metadata_csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(metadata_csv_path)
    return df


def attach_metadata_to_paths(image_paths, metadata_df):
    records = []
    metadata_map = {}

    if not metadata_df.empty and "image_name" in metadata_df.columns:
        metadata_map = {
            str(row["image_name"]): row.to_dict()
            for _, row in metadata_df.iterrows()
        }

    for path in image_paths:
        image_name = path.name
        row = metadata_map.get(image_name, {})
        row["image_path"] = str(path)
        row["image_name"] = image_name
        records.append(row)

    return records
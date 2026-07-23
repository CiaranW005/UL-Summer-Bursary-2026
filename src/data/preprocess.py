import sqlite3
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 

from collections.abc import Callable

from ..config.paths import DB_PATH

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class ModelData(Dataset[torch.Tensor]):
    """
    Dataset that loads images from disk and applies preprocessing transforms.
    """

    def __init__(self, paths : list[str], transform : Callable[[Image.Image], torch.Tensor]):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index: int) ->  torch.Tensor:
        path = self.paths[index]
        img = Image.open(path).convert("RGB")

        img = self.transform(img)
        
        return img

def preprocess() -> DataLoader[torch.Tensor]:
    """
    Load all image paths from the metadata database and return a DataLoader
    for embedding extraction.
    """

    conn = sqlite3.connect(DB_PATH)

    paths = pd.read_sql_query(  # pyright: ignore[reportUnknownMemberType]
        """
        SELECT path
        FROM meta
        """,
        conn
    )["path"].to_list()

    conn.close()

    dataset = ModelData(paths, transform=img_transform)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=False
    )

    return loader

if __name__ == "__main__":
    loader = preprocess()

    for images in loader:
        print(images.shape)
        break
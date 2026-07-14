import sqlite3
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ..config.paths import DB_PATH

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class ModelData(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        
        return img

def preprocess():
    conn = sqlite3.connect(DB_PATH)

    paths = pd.read_sql_query(
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

    for images, indices in loader:
        print(images.shape)
        print(indices[:5])
        break
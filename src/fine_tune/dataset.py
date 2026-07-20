from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

from .utils import get_category

class ModelData(Dataset):
    def __init__(self, paths, root, category_to_id, transform):
        self.paths = paths
        self.root = Path(root) 

        self.category_to_id = category_to_id
        self.transform = transform

        self.labels = [
            self.category_to_id[get_category(self.root / path)]
            for path in self.paths
        ]

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.root / self.paths[index]
        img = Image.open(path).convert("RGB")

        view1 = self.transform(img)
        view2 = self.transform(img)

        return view1, view2, self.labels[index]
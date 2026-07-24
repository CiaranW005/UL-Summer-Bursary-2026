import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

from collections.abc import Callable

from .utils import get_category, get_type
from .types import Sample

class ModelData(Dataset[Sample]):
    def __init__(self, 
            paths: list[str], 
            root: str, 
            category_to_id: dict[str, int], 
            transform: Callable[[Image.Image], torch.Tensor], 
            types_to_id: dict[str, int] | None = None):
        self.paths = paths
        self.root = Path(root) 

        self.transform = transform

        self.category_ids = [
            category_to_id[get_category(self.root / path)]
            for path in self.paths
        ]

        self.types_ids = None
        if types_to_id is not None:
            self.types_ids = [
                types_to_id[get_type(self.root / path)]
                for path in self.paths
            ]

            self.cat_types_ids = list(zip(
                self.category_ids,
                self.types_ids
            ))
        

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Sample:
        path = self.root / self.paths[index]
        img = Image.open(path).convert("RGB")

        view1 = self.transform(img)
        view2 = self.transform(img)

        if self.types_ids is not None:
            return view1, view2, self.category_ids[index], self.types_ids[index]
        
        return view1, view2, self.category_ids[index], None
    
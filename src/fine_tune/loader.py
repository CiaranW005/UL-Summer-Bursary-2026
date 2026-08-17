import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from PIL import Image

from bidict import bidict
from pathlib import Path
from collections.abc import Sequence, Callable

from .dataset import ModelData
from .sampler import BatchSampler
from .negative_sampler import NegativeSampler
from .types import ModelParameters

class DataLoading:
    def __init__(
            self,
            root: Path,
            params: ModelParameters,
            categories: Sequence[str],
            types: Sequence[str]
    )-> None:
        self.root = root
        self.params = params

        self.cat_to_id = bidict({c: idx for idx, c in enumerate(categories)})
        self.types_to_id = bidict({t: idx for idx, t in enumerate(types)})

    def create_train_val_loaders(
            self,
            paths: Sequence[str],
            train_transform: Callable[[Image.Image], torch.Tensor],
            eval_transform: Callable[[Image.Image], torch.Tensor],
            train_ratio: float = 0.95
        ) -> tuple[DataLoader, DataLoader, DataLoader]:
        n = len(paths)
        train_size = int(n * train_ratio)

        generator = torch.Generator().manual_seed(self.params["seed"])
        indices: list[int] = torch.randperm(n, generator=generator).tolist()

        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]

        train_paths = [paths[i] for i in train_indices]

        mahalanobis_dataset = ModelData(
            paths=paths,
            root=self.root,
            category_to_id=self.cat_to_id,
            types_to_id=None,
            transform=eval_transform
        )        

        train_dataset = ModelData(
            paths=train_paths,
            root=self.root,
            category_to_id=self.cat_to_id,
            types_to_id=None,
            transform=train_transform
        )

        val_set = Subset(mahalanobis_dataset, eval_indices)

        train_labels = train_dataset.category_ids
        val_labels = mahalanobis_dataset.category_ids[eval_indices]

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=BatchSampler(
                labels=train_labels,
                samples_per_cat=self.params["samples_per_category"],
                seed=None
            ),
            num_workers=self.params["num_workers"],
            pin_memory=self.params["pin_memory"]
        )

        val_loader = DataLoader(
            val_set,
            batch_sampler=BatchSampler(
                labels=val_labels,
                samples_per_cat=self.params["samples_per_category"],
                seed=self.params["seed"]
            ),
            num_workers=self.params["num_workers"],
            shuffle=False,
            pin_memory=self.params["pin_memory"]
        )

        mahalanobis_loader = DataLoader(
            mahalanobis_dataset,
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=self.params["num_workers"],
            pin_memory=self.params["pin_memory"]
        )

        return train_loader, val_loader, mahalanobis_loader

    def create_test_dataset(
            self, 
            test_paths: Sequence[str],
            transform: Callable[[Image.Image], torch.Tensor]
    ) -> ModelData:
        return ModelData(
            paths=test_paths,
            root=self.root,
            category_to_id=self.cat_to_id,
            types_to_id=self.types_to_id,
            transform=transform
        )

    def create_test_loader(
            self,
            dataset: ModelData,
            excluded_indices: np.ndarray
    ) -> DataLoader:
        if excluded_indices is None or excluded_indices.size == 0:
            test_set = dataset
        else:
            keep = np.ones(len(dataset), dtype=bool)
            keep[excluded_indices] = False

            test_set = Subset(dataset, np.flatnonzero(keep).tolist())

        return DataLoader(
            test_set,
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=self.params["num_workers"],
            pin_memory=self.params["pin_memory"]
        )
    
    def select_negative_indices(
        self,
        dataset: ModelData,
        samples_per_category: int = 5
    ) -> np.ndarray:
        sampler = NegativeSampler(
            labels=dataset.cat_types_ids,
            samples_per_cat=samples_per_category,
            types_to_id=self.types_to_id,
            seed=self.params["seed"]
        )

        return np.asarray(
            sampler.get_negatives(),
            dtype=np.int64
        )

    @staticmethod
    def get_neg_embeds(
            dataset: Dataset,
            neg_indices: np.ndarray,
            device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        neg_images: list[torch.Tensor] = []
        neg_labels: list[int] = []

        for idx in neg_indices:
            view1, _, cat_id, _ = dataset[idx]

            neg_images.append(view1)
            neg_labels.append(cat_id)

        images = torch.stack(neg_images).to(device)
        labels = torch.tensor(neg_labels, dtype=torch.long, device=device)

        return images, labels
        
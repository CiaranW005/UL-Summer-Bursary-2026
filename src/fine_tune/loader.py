import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from PIL import Image

from pathlib import Path
from collections.abc import Sequence, Callable

from .dataset import ModelData
from .sampler import BatchSampler
from .negative_sampler import NegativeSampler
from .types import ModelParameters

from ..model_selection.types import EmbeddingPipeline

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

        self.cat_to_id = {c: idx for idx, c in enumerate(categories)}
        self.types_to_id = {t: idx for idx, t in enumerate(types)}

    def create_train_val_loaders(
            self,
            train_paths: Sequence[str],
            transform: Callable[[Image.Image], torch.Tensor],
            train_ratio: float = 0.95
        ) -> tuple[DataLoader, DataLoader]:

        dataset = ModelData(
            paths=train_paths,
            root=self.root,
            category_to_id=self.cat_to_id,
            types_to_id=None,
            transform=transform
        )

        train_split = int(len(dataset) * train_ratio)
        val_split = len(dataset) - train_split

        generator = torch.Generator().manual_seed(self.params["seed"])
        train_set, val_set = random_split(
            dataset,
            [train_split, val_split],
            generator=generator
        )

        train_labels = dataset.category_ids[train_set.indices]
        val_labels = dataset.category_ids[val_set.indices]

        train_loader = DataLoader(
            train_set,
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

        return train_loader, val_loader

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
        
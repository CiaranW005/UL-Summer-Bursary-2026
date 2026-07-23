import random
import math

from collections import defaultdict
from typing import Iterator

from torch.utils.data import Sampler

class BatchSampler(Sampler[list[int]]):
    """Generate balanced training batches across categories.

    Each batch contains 'samples_per_cat' samples from every category. Samples
    are drawn without replacement until a category is exhausted, after which
    its pool is reshuffled and sampling continues.
    """

    def __init__(self, 
            labels: list[int], 
            samples_per_cat: int, 
            seed: int  | None = None
        ):
        self.labels = labels
        self.samples_per_cat = samples_per_cat
        self.seed = seed

        self.indices: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.indices[label].append(index)

        self.categories = list(self.indices)

        self.batches_per_epoch = max(
            math.ceil(len(cat_idx) / samples_per_cat)
            for cat_idx in self.indices.values()
        )

    def __iter__(self) -> Iterator[list[int]]:
        """Yield balanced batches of dataset indices.

        Each batch contains an equal number of samples from every category. The
        order of samples is shuffled, and category pools are reshuffled whenever
        all samples from a category have been used.
        """
        rng = random.Random(self.seed)

        pools: dict[int, list[int]] = {}
        positions: dict[int, int] = {}

        for category, category_idx in self.indices.items():
            pool = category_idx.copy()
            rng.shuffle(pool)

            pools[category] = pool
            positions[category] = 0

        for _ in range(self.batches_per_epoch):
            batch: list[int] = []

            for category in self.categories:
                pool = pools[category]
                position = positions[category]

                selected: list[int] = []

                # Keep drawing until this category has contributed its quota
                while len(selected) < self.samples_per_cat:
                    needed = self.samples_per_cat - len(selected)
                    avail = len(pool) - position

                    take = min(needed, avail)

                    selected.extend(pool[position : position + take])
                    position += take

                    # Restart the category pool once every sample has been seen. 
                    # TODO: Evaluate whether oversampling smaller categories increases overfitting.
                    if position == len(pool):
                        rng.shuffle(pool)
                        position = 0
                
                positions[category] = position
                batch.extend(selected)
            
            rng.shuffle(batch)
            yield batch
    
    def __len__(self):
        """Return the number of batches produced in one epoch."""
        return self.batches_per_epoch

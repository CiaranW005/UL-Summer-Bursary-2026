import random
import math

from collections import defaultdict

from torch.utils.data import Sampler

class BatchSampler(Sampler[list[int]]):
    def __init__(self, labels, samples_per_cat, seed=None):
        self.labels = labels
        self.samples_per_cat = samples_per_cat
        self.seed = seed

        self.indices = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.indices[label].append(index)

        self.categories = list(self.indices)

        self.batches_per_epoch = max(
            math.ceil(len(cat_idx) / samples_per_cat)
            for cat_idx in self.indices.values()
        )

    def __iter__(self):
        rng = random.Random(self.seed)

        pools = {}
        positions = {}

        for category, category_idx in self.indices.items():
            pool = category_idx.copy()
            rng.shuffle(pool)

            pools[category] = pool
            positions[category] = 0

        for _ in range(self.batches_per_epoch):
            batch = []

            for category in self.categories:
                pool = pools[category]
                position = positions[category]

                selected = []

                while len(selected) < self.samples_per_cat:
                    needed = self.samples_per_cat - len(selected)
                    avail = len(pool) - position

                    take = min(needed, avail)

                    selected.extend(pool[position : position + take])
                    position += take

                    if position == len(pool):
                        rng.shuffle(pool)
                        position = 0
                
                positions[category] = position
                batch.extend(selected)
            
            rng.shuffle(batch)
            yield batch
    
    def __len__(self):
        return self.batches_per_epoch

import random

from collections import defaultdict

class NegativeSampler:
    def __init__(self, labels, samples_per_cat, types_to_id, seed=None):
        self.samples_per_cat = samples_per_cat
        self.seed = seed
        
        self.indices = defaultdict(list)    
        self.types_by_cat = defaultdict(set)

        for index, (cat, defect_type) in enumerate(labels):
            if defect_type == types_to_id["good"]:
                continue

            self.indices[(cat, defect_type)].append(index)
            self.types_by_cat[cat].add(defect_type)
        
        self.categories = list(self.types_by_cat)

    def get_negatives(self):
        rng = random.Random(self.seed)

        batch = []
        for cat in self.categories:
            defect_types = sorted(self.types_by_cat[cat])

            base = self.samples_per_cat // len(defect_types)
            remainder = self.samples_per_cat % len(defect_types)

            extra_types = set(rng.sample(defect_types, k=remainder))

            for defect_type in defect_types:
                n_samples = base + int(defect_type in extra_types)

                pool = self.indices[(cat, defect_type)]

                if len(pool) >= n_samples:
                    selected = rng.sample(pool, k=n_samples)
                else:
                    selected = pool.copy()
                    selected.extend(rng.choices(pool, k=n_samples - len(pool)))

                batch.extend(selected)

        rng.shuffle(batch)
        return batch

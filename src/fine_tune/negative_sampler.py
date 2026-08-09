import random
import numpy as np

import logging
from collections import defaultdict
from collections.abc import Sequence

logger = logging.getLogger(__name__)
class NegativeSampler:
    """Sample negative examples while balancing defect types within each category.

    Each category contributes 'samples_per_cat' samples. The allocation is
    distributed as evenly as possible across the available defect types, with
    any remainder assigned randomly. If a defect type contains fewer available
    samples than requested, sampling with replacement is used.
    """

    def __init__(self, 
            labels : np.ndarray, 
            samples_per_cat: int, 
            types_to_id: dict[str, int], 
            seed: int | None = None
        ):
        self.samples_per_cat = samples_per_cat
        self.seed = seed
        
        self.indices : dict[tuple[int, int], list[int]] = defaultdict(list)    
        self.types_by_cat : dict[int, set[int]] = defaultdict(set)

        for index, (cat, defect_type) in enumerate(labels):
            if defect_type == types_to_id["good"]:
                continue

            self.indices[(cat, defect_type)].append(index)
            self.types_by_cat[cat].add(defect_type)
        
        self.categories = list(self.types_by_cat)

    def get_negatives(self) -> list[int]:
        """Generate a balanced list of negative sample indices.

        Returns:
            A shuffled list of dataset indices that can be used to construct a
            training batch.
        """
        rng = random.Random(self.seed)

        batch: list[int] = []
        for cat in self.categories:
            defect_types = sorted(self.types_by_cat[cat])

            if not defect_types:
                logger.warning(
                    "No defect types found in category %d",
                    cat
                )
                continue

            base = self.samples_per_cat // len(defect_types)
            remainder = self.samples_per_cat % len(defect_types)

            extra_types = set(rng.sample(defect_types, k=remainder))

            for defect_type in defect_types:
                n_samples = base + int(defect_type in extra_types)

                if n_samples == 0: 
                    # Defect type gets 0 samples this time around
                    continue

                pool = self.indices[(cat, defect_type)]

                if not pool:
                    logger.warning(
                        "No pool found for category %d at defect %d",
                        cat, defect_type
                    )
                    continue

                if len(pool) >= n_samples:
                    selected = rng.sample(pool, k=n_samples)
                else:
                    selected = pool.copy()
                    selected.extend(rng.choices(pool, k=n_samples - len(pool)))

                batch.extend(selected)

        rng.shuffle(batch)
        return batch

from collections import defaultdict

class GetNegatives:
    def __init__(self, labels, samples_per_cat, seed=None):
        self.samples_per_cat = samples_per_cat
        self.seed = seed
        
        self.indices = defaultdict(list)    
        for index, label in enumerate(self.labels):
            self.indices[label].append(index)
        
        self.categories = list(self.indices)

    def get_negatives(self):
        
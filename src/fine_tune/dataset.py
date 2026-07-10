from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ModelData(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert("RGB")

        view1 = self.transform(img)
        view2 = self.transform(img)
        
        return view1, view2, index
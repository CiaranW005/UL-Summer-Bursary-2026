import torch

from tqdm import tqdm

def get_embeddings(model, loader, device):
    model.eval()

    cls_tokens = []
    all_patches = []
    indices = []

    with torch.no_grad():
        for images, idx in tqdm(loader):
            images = images.to(device)
            
            features = model.forward_features(images)
            cls = features["x_norm_clstoken"]
            patches = features["x_norm_patchtokens"]

            cls_tokens.append(cls.cpu())
            all_patches.append(patches.cpu())
            indices.append(idx)
    
    cls_tokens = torch.cat(cls_tokens).numpy().astype("float32")
    all_patches = torch.cat(all_patches).numpy().astype("float32")
    indices = torch.cat(indices).numpy()
    
    return cls_tokens, all_patches, indices
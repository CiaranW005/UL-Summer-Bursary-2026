import torch

from tqdm import tqdm

def get_embeddings(dino, loader, device, model=None):
    dino.eval()

    if model is not None:
        model.eval()

    cls_tokens = []
    all_patches = []

    with torch.no_grad():
        for images in tqdm(loader):
            images = images.to(device)
            
            features = dino.forward_features(images)
            cls = features["x_norm_clstoken"]
            patches = features["x_norm_patchtokens"]

            if model is not None:
                cls = model(cls)
            
            cls_tokens.append(cls.cpu())
            all_patches.append(patches.cpu())
    
    cls_tokens = torch.cat(cls_tokens).numpy().astype("float32")
    all_patches = torch.cat(all_patches).numpy().astype("float32")
    
    return cls_tokens, all_patches
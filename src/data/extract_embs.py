import torch

from tqdm import tqdm

def get_embeddings(dino, loader, device, model=None):
    dino.eval()

    if model is not None:
        model.eval()

    base_tokens = []
    projected_tokens = []
    all_patches = []

    with torch.no_grad():
        for images in tqdm(loader):
            images = images.to(device)
            
            features = dino.forward_features(images)
            base_cls = features["x_norm_clstoken"]
            patches = features["x_norm_patchtokens"]

            if model is not None:
                projected_cls = model(base_cls)
            else:
                projected_cls = base_cls

            base_tokens.append(base_cls.cpu())
            projected_tokens.append(projected_cls.cpu())
            all_patches.append(patches.cpu())
    
    base_tokens = torch.cat(base_tokens).numpy().astype("float32")
    projected_tokens = torch.cat(projected_tokens).numpy().astype("float32")

    all_patches = torch.cat(all_patches).numpy().astype("float32")
    
    return base_tokens, projected_tokens, all_patches
import torch
from torch.utils.data import DataLoader

from pathlib import Path

def get_category(path: Path) -> str:
    parts = path.parts

    try:
        return parts[parts.index("mvtec_ad") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not extract MVTec category from: {path}") from exc
    
def get_type(path: Path) -> str:
    parts = path.parts

    try:
        return parts[parts.index("mvtec_ad") + 3]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not extract MVTec type from: {path}") from exc

def get_device() -> torch.device:
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device

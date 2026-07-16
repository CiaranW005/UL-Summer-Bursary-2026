from pathlib import Path

def get_category(path):
    parts = path.parts

    try:
        return parts[parts.index("mvtec_ad") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not extract MVTec category from: {path}") from exc
    

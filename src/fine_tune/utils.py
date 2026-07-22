def get_category(path):
    parts = path.parts

    try:
        return parts[parts.index("mvtec_ad") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not extract MVTec category from: {path}") from exc
    
def get_type(path):
    parts = path.parts

    try:
        return parts[parts.index("mvtec_ad") + 3]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not extract MVTec type from: {path}") from exc

def get_category_and_type(path):
    return get_category(path), get_type(path)
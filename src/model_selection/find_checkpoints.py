from pathlib import Path


def get_checkpoints(directory: Path) -> list[Path]:
    if not directory.exists():
        print(f"Found 0 {directory} checkpoints")
        return []

    paths = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_dir() and (path / "model.pt").exists()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    print(f"Found {len(paths)} checkpoints at {directory}")
    return paths

def get_latest_checkpoint(directory: Path) -> Path:
    checkpoints = get_checkpoints(directory)

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {directory}"
        )

    return checkpoints[0]
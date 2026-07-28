from pathlib import Path


def get_checkpoints(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    return sorted(
        (
            path
            for path in directory.glob("*.pt")
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def get_latest_checkpoint(directory: Path) -> Path:
    checkpoints = get_checkpoints(directory)

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {directory}"
        )

    return checkpoints[0]
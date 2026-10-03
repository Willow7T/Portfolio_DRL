from pathlib import Path


def ensure_parent_directory(path: str | Path) -> Path:
    """Create a file's parent directory and return the resolved path object."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

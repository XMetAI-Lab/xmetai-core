"""Reusable LazyConfig recipes."""

from pathlib import Path


def recipe_path(name: str) -> str:
    """Return the source path for a bundled LazyConfig recipe."""
    filename = name if name.endswith(".py") else f"{name}.py"
    path = Path(__file__).with_name(filename)
    if not path.is_file():
        raise ValueError(f"Unknown XMetAI recipe: {name}")
    return str(path)


__all__ = ["recipe_path"]

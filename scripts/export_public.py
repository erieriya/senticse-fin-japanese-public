#!/usr/bin/env python3
"""Export only explicitly reviewed files, never private Git history or data."""
import argparse
import hashlib
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "public-files.txt"
FORBIDDEN = {".csv", ".tsv", ".jsonl", ".npy", ".npz", ".log", ".ipynb", ".pt", ".pth", ".bin", ".safetensors", ".png", ".pdf"}
PRIVATE_DIRS = {".git", ".venv", "data", "data_EWS", "sentiment_vocab", "local_vocab_files", "private", "outputs", "result"}


def public_files(root=ROOT):
    paths = []
    for line in (root / "public-files.txt").read_text().splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or PRIVATE_DIRS.intersection(relative.parts):
            raise ValueError(f"Unsafe manifest entry: {name}")
        if relative.suffix.lower() in FORBIDDEN:
            raise ValueError(f"Data/artifact files cannot be exported: {name}")
        source = root / relative
        if any(part.is_symlink() for part in [source, *source.parents]) or not source.is_file():
            raise ValueError(f"Expected a regular file without symlinks: {name}")
        if relative in paths:
            raise ValueError(f"Duplicate manifest entry: {name}")
        paths.append(relative)
    return paths


def export(destination, root=ROOT):
    destination = Path(destination).absolute()
    paths = public_files(root)
    # Refuse any existing path: never merge private files or .git into an export.
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Destination must not exist: {destination}")
    destination.mkdir(parents=True)
    for relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    return len(paths)


def check(directory, root=ROOT):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Check target must be a real directory.")
    expected = set(public_files(root))
    actual = set()
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        # A separately initialized public repository can be checked as well.
        if relative.parts[0] == ".git":
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink found: {relative}")
        if path.is_file():
            actual.add(relative)
    if actual != expected:
        raise ValueError(f"Unexpected/missing files: {sorted(str(p) for p in actual ^ expected)}")
    for relative in expected:
        if hashlib.sha256((directory / relative).read_bytes()).digest() != hashlib.sha256((root / relative).read_bytes()).digest():
            raise ValueError(f"Content differs: {relative}")
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    count = check(args.destination) if args.check else export(args.destination)
    print(f"Verified {count} public files." if args.check else f"Exported {count} public files to {args.destination}.")


if __name__ == "__main__":
    main()

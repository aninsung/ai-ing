#!/usr/bin/env python3
"""Download NIH Chest X-ray metadata + image zips (HuggingFace mirror of Kaggle/NIH)."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HF = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main"
META = {
    "Data_Entry_2017.csv": f"{HF}/data/Data_Entry_2017_v2020.csv",
    "BBox_List_2017.csv": f"{HF}/data/BBox_List_2017.csv",
    "test_list.txt": f"{HF}/data/test_list.txt",
    "train_val_list.txt": f"{HF}/data/train_val_list.txt",
}


def curl(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    cmd = [
        "curl", "-L", "--fail", "--retry", "8", "--retry-delay", "5",
        "-C", "-", "-o", str(partial), url,
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    partial.replace(dest)


def download_meta(root: Path):
    for name, url in META.items():
        dest = root / name
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"skip meta {name}", flush=True)
            continue
        curl(url, dest)


def download_images(root: Path, start: int, end: int, keep_zips: bool):
    zips = root / "zips"
    raw = root / "raw"
    zips.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    for i in range(start, end + 1):
        name = f"images_{i:03d}.zip"
        dest = zips / name
        url = f"{HF}/data/images/{name}"
        marker = raw / f".extracted_{name}"
        if marker.exists():
            print(f"skip extracted {name}", flush=True)
            continue
        if not dest.exists() or dest.stat().st_size < 1_000_000:
            curl(url, dest)
        print(f"extracting {name} ...", flush=True)
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(raw)
        marker.write_text("ok\n", encoding="utf-8")
        if not keep_zips:
            dest.unlink(missing_ok=True)
            print(f"removed zip {name}", flush=True)
    # Convenience symlink for multiclass config image_root
    archive = root / "archive"
    if not archive.exists():
        archive.symlink_to("raw")
    pngs = sum(1 for _ in raw.rglob("*.png"))
    print({"png_count": pngs, "raw": str(raw)}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=12)
    p.add_argument("--keep-zips", action="store_true")
    p.add_argument("--meta-only", action="store_true")
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    download_meta(root)
    if args.meta_only:
        return
    download_images(root, args.start, args.end, args.keep_zips)


if __name__ == "__main__":
    main()

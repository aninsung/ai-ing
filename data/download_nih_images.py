#!/usr/bin/env python3
"""Download NIH ChestX-ray14 image archives and extract them one at a time.

Google Cloud Storage for this dataset is requester-pays. These are the same
PNG archives NIH publishes, which that page points to:
https://nihcc.app.box.com/v/ChestXray-NIHCC
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path("/workspace/ai-ing/data")
RAW = ROOT / "raw"
ARCHIVES = ROOT / "archives"
DONE = ROOT / ".extracted"

# file id, expected bytes, pngs inside the archive
PARTS = [
    ("images_001.tar.gz", "219764235225", 2008470987, 4999),
    ("images_002.tar.gz", "219767703471", 3952623504, 10000),
    ("images_003.tar.gz", "219770039352", 3929234850, 10000),
    ("images_004.tar.gz", "221185642661", 3838903983, 10000),
    ("images_005.tar.gz", "219776556743", 3935496531, 10000),
    ("images_006.tar.gz", "219777758783", 3986301172, 10000),
    ("images_007.tar.gz", "220610700915", 4016328426, 10000),
    ("images_008.tar.gz", "219776273384", 4018347353, 10000),
    ("images_009.tar.gz", "219782291318", 4111327929, 10000),
    ("images_010.tar.gz", "219781375034", 4181556296, 10000),
    ("images_011.tar.gz", "219777519815", 4187084020, 10000),
    ("images_012.tar.gz", "219778785923", 2914187733, 7121),
]
EXPECTED_TOTAL = 112120
URL = (
    "https://nihcc.app.box.com/index.php?rm=box_download_shared_file"
    "&vanity_name=ChestXray-NIHCC&file_id=f_{file_id}"
)


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def png_count() -> int:
    return sum(1 for path in RAW.rglob("*.png") if path.is_file())


def download(file_id: str, dest: Path, expected: int) -> None:
    url = URL.format(file_id=file_id)
    for attempt in range(1, 41):
        if dest.exists():
            got = dest.stat().st_size
            if got == expected:
                log(f"{dest.name} already complete ({got} bytes)")
                return
            if got > expected:
                log(f"{dest.name} oversize {got}, restarting")
                dest.unlink()
            else:
                with dest.open("rb") as handle:
                    magic = handle.read(2)
                if magic != b"\x1f\x8b":
                    log(f"{dest.name} is not gzip, restarting")
                    dest.unlink()
                else:
                    log(f"{dest.name} resume at {got}/{expected}")
        free = shutil.disk_usage(ROOT).free
        if free < expected + 2_000_000_000:
            raise SystemExit(f"not enough disk for {dest.name}: free={free}")
        log(f"download {dest.name} attempt {attempt}")
        result = subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "5",
                "--retry-all-errors",
                "--retry-delay",
                "5",
                "--speed-limit",
                "20000",
                "--speed-time",
                "120",
                "-C",
                "-",
                "--output",
                str(dest),
                url,
            ]
        )
        got = dest.stat().st_size if dest.exists() else 0
        if got == expected:
            log(f"downloaded {dest.name}")
            return
        log(f"incomplete {dest.name} {got}/{expected} curl={result.returncode}")
        time.sleep(min(30, 3 * attempt))
    raise SystemExit(f"could not download {dest.name}")


def extract(name: str, dest: Path, expected_pngs: int, before: int) -> int:
    log(f"extract {name}")
    result = subprocess.run(["tar", "-xzf", str(dest), "-C", str(RAW)])
    if result.returncode != 0:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"extract failed for {name}; archive removed for retry")
    dest.unlink()
    after = png_count()
    added = after - before
    log(f"extracted {name}: +{added} pngs, total {after}")
    if added != expected_pngs:
        raise SystemExit(f"{name} added {added} pngs, expected {expected_pngs}")
    return after


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    DONE.mkdir(parents=True, exist_ok=True)
    for name, file_id, size, expected_pngs in PARTS:
        mark = DONE / name
        if mark.exists():
            log(f"skip {name}")
            continue
        before = png_count()
        dest = ARCHIVES / name
        download(file_id, dest, size)
        after = extract(name, dest, expected_pngs, before)
        mark.write_text(f"{after}\n", encoding="utf-8")
        log(f"disk free {shutil.disk_usage(ROOT).free / 1e9:.1f} GB")
    total = png_count()
    log(f"all archives extracted, pngs={total}")
    if total != EXPECTED_TOTAL:
        raise SystemExit(f"expected {EXPECTED_TOTAL} pngs, found {total}")


if __name__ == "__main__":
    main()

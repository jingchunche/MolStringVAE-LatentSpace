#!/usr/bin/env python3
"""Convert SELFIES text files in a decoding result directory into SMILES."""
import os
import sys
from pathlib import Path

try:
    import selfies
except ImportError as exc:
    print("[ERROR] selfies package is required (pip install selfies)")
    raise

ROOT_DIR = Path("/home/jingchun/TransformerVAE")
RESULTS_ROOT = ROOT_DIR / "decoding" / "results"
TARGET_PREFIX = "selfies"
OUTPUT_PREFIX = "smiles"


def decode_file(src_path: Path) -> None:
    dst_path = src_path.with_name(src_path.name.replace(TARGET_PREFIX, OUTPUT_PREFIX, 1))
    valid = total = 0
    with src_path.open() as fin, dst_path.open("w") as fout:
        for line in fin:
            total += 1
            text = line.strip()
            if not text:
                fout.write("\n")
                continue
            try:
                smiles = selfies.decoder(text)
            except selfies.DecoderError:
                fout.write("INVALID\n")
            else:
                fout.write(smiles + "\n")
                valid += 1
    print(f"[INFO] {src_path.name}: valid {valid}/{total} -> {dst_path.name}")


def main(arg: str) -> None:
    target_dir = RESULTS_ROOT / arg
    if not target_dir.is_dir():
        print(f"[ERROR] Directory not found: {target_dir}")
        sys.exit(1)

    matched = sorted(target_dir.glob("selfies*.txt"))
    if not matched:
        print(f"[WARN] No SELFIES files found in {target_dir}")
        return

    for path in matched:
        decode_file(path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python decode_selfies_to_smiles.py <data_dir>")
        print("  Example: python decode_selfies_to_smiles.py selfies/train10/10000")
        sys.exit(1)
    main(sys.argv[1])

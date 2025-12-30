#!/usr/bin/env python3
"""Convert SELFIES text files in a decoding result directory into SMILES."""
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import selfies
except ImportError as exc:
    print("[ERROR] selfies package is required (pip install selfies)")
    raise

ROOT_DIR = Path("/home/jingchun/TransformerVAE")
RESULTS_ROOT = ROOT_DIR / "decoding" / "results"
#RESULTS_ROOT = ROOT_DIR / "generation" / "results"

FILE_PAIRS = [
    ("selfies.txt", "smiles.txt"),
]

# FILE_PAIRS = [
#     ("selfies_end1.txt", "smiles_end1.txt"),
#     ("selfies_mid.txt",  "smiles_mid.txt"),
#     ("selfies_end2.txt", "smiles_end2.txt"),
# ]
# FILE_PAIRS = [
#     ("selfies_near.txt", "smiles_near.txt"),
#     ("selfies_seed.txt", "smiles_seed.txt"),
# ]

def infer_output_path(src_path: Path) -> Path:
    name = src_path.name
    if "selfies" in name:
        return src_path.with_name(name.replace("selfies", "smiles", 1))
    return src_path.with_suffix(".smiles")


def decode_file(src_path: Path, dst_path: Optional[Path] = None) -> None:
    if dst_path is None:
        dst_path = infer_output_path(src_path)
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

    if FILE_PAIRS:
        for in_name, out_name in FILE_PAIRS:
            src = target_dir / in_name
            dst = target_dir / out_name
            if not src.is_file():
                print(f"[WARN] File not found: {src}")
                continue
            decode_file(src, dst)
        return

    matched = sorted(target_dir.glob("selfies_mid*.txt"))
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
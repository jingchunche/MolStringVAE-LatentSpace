#!/usr/bin/env python3
"""Evaluate decoded_smiles.txt against the fixed GuacaMol test set."""

import argparse
from pathlib import Path
from typing import List, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = REPO_ROOT / "data" / "guacamol_v1_test.smiles"
DEFAULT_RESULTS = REPO_ROOT / "recon" / "results"
DEFAULT_DECODED_RELATIVE = "recon_smiles.txt"


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle]


def _fingerprint(mol):
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def _canonical_smiles(mol) -> str:
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def resolve_results_dir(name: str) -> Path:
    candidate = Path(name)
    if candidate.is_dir():
        return candidate
    candidate = DEFAULT_RESULTS / name
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"Could not find results directory '{name}'")


def compute_average_tanimoto(ref_smiles: List[str], decoded_smiles: List[str]) -> Tuple[float, List[int], int]:
    if len(ref_smiles) != len(decoded_smiles):
        raise ValueError(
            f"Reference set has {len(ref_smiles)} entries but decoded set has {len(decoded_smiles)} entries"
        )

    invalid_indices: List[int] = []
    similarities: List[float] = []
    exact_matches = 0

    for idx, (ref, decoded) in enumerate(zip(ref_smiles, decoded_smiles), start=1):
        ref_mol = Chem.MolFromSmiles(ref)
        decoded_mol = Chem.MolFromSmiles(decoded)
        ref_fp = _fingerprint(ref_mol)
        decoded_fp = _fingerprint(decoded_mol)
        if ref_fp is None or decoded_fp is None:
            invalid_indices.append(idx)
            similarities.append(0.0)
            continue
        similarities.append(DataStructs.TanimotoSimilarity(ref_fp, decoded_fp))

        if _canonical_smiles(ref_mol) == _canonical_smiles(decoded_mol):
            exact_matches += 1

    if not similarities:
        raise ValueError("No pairs were processed; check the input files")

    avg = sum(similarities) / len(similarities)
    return avg, invalid_indices, exact_matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "name",
        help=(
            "Directory under decoding/results, or an absolute results directory, containing decoded_smiles.txt"
        ),
    )
    args = parser.parse_args()

    results_dir = resolve_results_dir(args.name)

    decoded_path = results_dir / DEFAULT_DECODED_RELATIVE
    if not decoded_path.is_file():
        raise FileNotFoundError(f"Decoded SMILES file not found: {decoded_path}")

    reference_path = DEFAULT_REFERENCE
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference SMILES file not found: {reference_path}")

    decoded_smiles = read_lines(decoded_path)
    reference_smiles = read_lines(reference_path)

    average, invalid_indices, exact_matches = compute_average_tanimoto(reference_smiles, decoded_smiles)

    total_pairs = len(decoded_smiles)
    valid_pairs = total_pairs - len(invalid_indices)
    pass_ratio = exact_matches / total_pairs if total_pairs else 0.0

    lines = [
        f"Decoded SMILES: {decoded_path} ({len(decoded_smiles)} entries)",
        f"Reference SMILES: {reference_path} ({len(reference_smiles)} entries)",
        f"Average Tanimoto similarity: {average:.6f}",
        f"Validity: {valid_pairs}/{total_pairs}",
        f"Canonical exact matches: {exact_matches}/{total_pairs} ({pass_ratio:.2%})",
    ]
    if invalid_indices:
        lines.append(
            f"{len(invalid_indices)} pair(s) contained invalid SMILES. Indices treated as zero similarity:"
        )
        lines.append(", ".join(map(str, invalid_indices)))
    else:
        lines.append("All pairs were valid.")

    similarity_path = results_dir / "recon_summary.log"
    similarity_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for line in lines:
        print(line)


if __name__ == "__main__":
    main()

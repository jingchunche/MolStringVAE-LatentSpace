#!/usr/bin/env python3
"""Generate SELFIES vocabulary from a SMILES dataset."""
import argparse
from pathlib import Path
from typing import Iterable, Set, Tuple

import selfies as sf
from rdkit import Chem

DEFAULT_CONSTRAINT_OVERRIDES = {
    "P": 7,
    "P-1": 8,
    "P+1": 6,
    "?": 12,
    "C+1": 5,
}


def iter_smiles(path: Path) -> Iterable[str]:
    with path.open() as infile:
        for line in infile:
            smile = line.strip()
            if smile:
                yield smile


def configure_constraints() -> None:
    constraints = sf.get_preset_constraints("hypervalent")
    constraints.update(DEFAULT_CONSTRAINT_OVERRIDES)
    sf.set_semantic_constraints(constraints)


def build_vocab(smiles_path: Path) -> Tuple[Set[str], int, int]:
    vocab = set()  # type: Set[str]
    total = 0
    failed = 0
    for smile in iter_smiles(smiles_path):
        total += 1
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                raise ValueError("Failed to parse SMILES")
            canonical_smile = Chem.MolToSmiles(mol, canonical=True)
            selfies_str = sf.encoder(canonical_smile)
        except Exception:
            failed += 1
            continue
        vocab.update(sf.split_selfies(selfies_str))
    return vocab, total, failed


def write_vocab(vocab: Set[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as outfile:
        for token in sorted(vocab):
            outfile.write(token + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="SMILES input file")
    parser.add_argument("output", type=Path, help="Destination vocab file")
    parser.add_argument(
        "--no-constraints",
        action="store_true",
        help="Disable SELFIES semantic constraints overrides",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.no_constraints:
        configure_constraints()
    vocab, total, failed = build_vocab(args.input)
    write_vocab(vocab, args.output)
    print(
        "Completed. Total SMILES: {total}, failed encodes: {failed}, vocab size: {vocab_size}".format(
            total=total, failed=failed, vocab_size=len(vocab)
        )
    )


if __name__ == "__main__":
    main()

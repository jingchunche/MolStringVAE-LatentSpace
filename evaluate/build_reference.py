#!/usr/bin/env python3
"""Build an exact reusable chemical reference without project imports."""
import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BYTE_POPCOUNT = np.array(
    [bin(value).count("1") for value in range(256)], dtype=np.uint8
)


@dataclass
class ReferenceConfig:
    train_smiles: str
    query_smiles: str
    output_dir: str
    neighbors: int = 10
    radius: int = 2
    nbits: int = 2048
    batch_size: int = 2000
    random_seed: int = 0
    random_replace: bool = False


def load_lines(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_lines(path, values, fmt=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write((fmt.format(value) if fmt else str(value)) + "\n")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False
        )


def _fingerprints(smiles, source, radius, nbits):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    bits = np.zeros((len(smiles), nbits), dtype=np.uint8)
    for idx, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            raise ValueError(
                "Invalid SMILES at {}:{}: {}".format(source, idx + 1, text)
            )
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            molecule, radius, nBits=nbits
        )
        bits[idx, list(fingerprint.GetOnBits())] = 1
        if (idx + 1) % 10000 == 0:
            print(
                "[reference] featurized {:,} rows from {}".format(
                    idx + 1, source
                )
            )
    return bits


def _chemical_neighbors(
    query_bits, train_bits, query_smiles, train_smiles, k, batch_size
):
    import faiss

    if k <= 0:
        raise ValueError("neighbors must be positive")
    if len(train_bits) < k:
        raise ValueError(
            "Train set ({}) is smaller than neighbors ({})".format(
                len(train_bits), k
            )
        )
    train_packed = np.packbits(train_bits, axis=1, bitorder="little")
    query_packed = np.packbits(query_bits, axis=1, bitorder="little")
    train_counts = BYTE_POPCOUNT[train_packed].sum(axis=1).astype(np.int32)
    query_counts = BYTE_POPCOUNT[query_packed].sum(axis=1).astype(np.int32)

    buckets = {}
    for bit_count in np.unique(train_counts):
        rows = np.flatnonzero(train_counts == bit_count).astype(np.int64)
        index = faiss.IndexBinaryFlat(train_bits.shape[1])
        index.add(np.ascontiguousarray(train_packed[rows]))
        buckets[int(bit_count)] = (rows, index)

    near = []
    dissimilarities = []
    for start in range(0, len(query_bits), batch_size):
        end = min(start + batch_size, len(query_bits))
        size = end - start
        best_sim = np.full((size, k), -np.inf, dtype=np.float64)
        best_idx = np.full((size, k), -1, dtype=np.int64)
        batch_counts = query_counts[start:end]

        for query_count in np.unique(batch_counts):
            group = np.flatnonzero(batch_counts == query_count)
            ordered_buckets = sorted(
                buckets.items(),
                key=lambda item: (
                    min(int(query_count), item[0])
                    / float(max(int(query_count), item[0]))
                    if max(int(query_count), item[0]) else 1.0
                ),
                reverse=True,
            )
            for train_count, (bucket_rows, index) in ordered_buckets:
                maximum = (
                    min(int(query_count), train_count)
                    / float(max(int(query_count), train_count))
                    if max(int(query_count), train_count) else 1.0
                )
                active = group[best_sim[group, -1] < maximum]
                if active.size == 0:
                    break
                search_k = min(k, len(bucket_rows))
                distances, local_indices = index.search(
                    np.ascontiguousarray(query_packed[start:end][active]),
                    search_k,
                )
                global_indices = bucket_rows[local_indices]
                hamming = distances.astype(np.float64)
                numerator = float(query_count + train_count) - hamming
                denominator = float(query_count + train_count) + hamming
                similarities = np.where(
                    denominator > 0.0, numerator / denominator, 0.0
                )

                merged_sim = np.concatenate(
                    [best_sim[active], similarities], axis=1
                )
                merged_idx = np.concatenate(
                    [best_idx[active], global_indices], axis=1
                )
                selected = np.argpartition(
                    -merged_sim, k - 1, axis=1
                )[:, :k]
                selected_sim = np.take_along_axis(
                    merged_sim, selected, axis=1
                )
                selected_idx = np.take_along_axis(
                    merged_idx, selected, axis=1
                )
                order = np.argsort(-selected_sim, axis=1, kind="stable")
                best_sim[active] = np.take_along_axis(
                    selected_sim, order, axis=1
                )
                best_idx[active] = np.take_along_axis(
                    selected_idx, order, axis=1
                )

        if np.any(best_idx < 0) or np.any(~np.isfinite(best_sim)):
            raise RuntimeError(
                "Unable to find exact chemical top-k for every query"
            )
        for row in range(size):
            for train_idx, similarity in zip(best_idx[row], best_sim[row]):
                near.append(train_smiles[int(train_idx)])
                dissimilarities.append(1.0 - float(similarity))
        print(
            "[reference] searched {:,}/{:,} queries".format(
                end, len(query_bits)
            )
        )
    return near, dissimilarities


def _random_mean_similarity(
    query_bits, train_bits, k, random_seed, replace
):
    if len(train_bits) < k and not replace:
        raise ValueError(
            "Train set is smaller than k; enable random replacement"
        )
    rng = np.random.RandomState(random_seed)
    train_bool = train_bits.astype(bool, copy=False)
    result = []
    for query in query_bits.astype(bool, copy=False):
        chosen = train_bool[
            rng.choice(len(train_bool), size=k, replace=replace)
        ]
        intersection = np.logical_and(
            chosen, query
        ).sum(axis=1, dtype=np.float32)
        union = np.logical_or(
            chosen, query
        ).sum(axis=1, dtype=np.float32)
        similarity = np.where(union > 0, intersection / union, 0.0)
        result.append(float(np.mean(similarity)))
    return result


def build_reference(config):
    if isinstance(config, dict):
        config = ReferenceConfig(**config)
    train_path = Path(config.train_smiles).expanduser().resolve()
    query_path = Path(config.query_smiles).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = load_lines(train_path)
    queries = load_lines(query_path)
    if not train or not queries:
        raise ValueError("Train and query SMILES must both be non-empty")
    print(
        "[reference] train={:,}, queries={:,}".format(
            len(train), len(queries)
        )
    )
    train_bits = _fingerprints(
        train, train_path, config.radius, config.nbits
    )
    query_bits = _fingerprints(
        queries, query_path, config.radius, config.nbits
    )
    near, dissimilarities = _chemical_neighbors(
        query_bits, train_bits, queries, train,
        config.neighbors, config.batch_size,
    )
    random_sim = _random_mean_similarity(
        query_bits, train_bits, config.neighbors,
        config.random_seed, config.random_replace,
    )
    write_lines(output_dir / "chemical_near.txt", near)
    write_lines(
        output_dir / "chemical_dissim.txt",
        dissimilarities, "{:.10f}",
    )
    write_lines(
        output_dir / "random_sim.txt", random_sim, "{:.10f}"
    )
    manifest = asdict(config)
    manifest.update({
        "train_smiles": str(train_path),
        "query_smiles": str(query_path),
        "train_sha256": sha256_file(train_path),
        "query_sha256": sha256_file(query_path),
        "train_count": len(train),
        "query_count": len(queries),
        "search_method":
            "exact_tanimoto_topk_bitcount_bucketed_faiss_hamming",
    })
    write_json(output_dir / "reference_manifest.json", manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-smiles",
        default=str(ROOT / "data" / "guacamol_v1_train.smiles"),
    )
    parser.add_argument(
        "--query-smiles", default=str(ROOT / "data" / "query1.smiles")
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Default: evaluate/reference/<query name>; "
            "guacamol_query1.smiles becomes query1."
        ),
    )
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--nbits", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--random-replace", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        query_name = Path(args.query_smiles).expanduser().stem
        if query_name.startswith("guacamol_"):
            query_name = query_name[len("guacamol_"):]
        args.output_dir = str(
            ROOT / "evaluate" / "reference" / query_name
        )
    manifest = build_reference(ReferenceConfig(**vars(args)))
    print("[reference] complete: {}".format(manifest["output_dir"]))


if __name__ == "__main__":
    main()

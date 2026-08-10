#!/usr/bin/env python3
"""Run latent-vs-chemical neighborhood evaluation as a standalone script."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parent
EXACT_CHEMICAL_SEARCH = "exact_tanimoto_topk_bitcount_bucketed_faiss_hamming"
ALIGNMENT_TOLERANCE = 1e-8


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_lines(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_csv_matrix(path):
    data = np.loadtxt(str(path), delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return np.asarray(data, dtype=np.float32)


def load_numeric(path):
    data = np.loadtxt(str(path), dtype=float)
    return data.reshape(1) if data.ndim == 0 else data


def reshape_qk(values, k, name):
    if k <= 0:
        raise ValueError("neighbors must be positive")
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        if values.shape[1] != k:
            raise ValueError(
                "{} second dimension must be {}, got {}".format(name, k, values.shape)
            )
        return values
    flat = values.reshape(-1)
    if flat.size % k:
        raise ValueError(
            "{} length {} is not divisible by {}".format(name, flat.size, k)
        )
    return flat.reshape(-1, k)


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
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)


def write_query_metrics(path, query_smiles, arrays):
    """Write one metrics row per query, preserving the input query order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    query_count = len(query_smiles)
    columns = list(arrays)
    normalized = {}
    for name in columns:
        values = np.asarray(arrays[name], dtype=float).reshape(-1)
        if values.size != query_count:
            raise ValueError(
                "Query metric {} has {} values; expected {}".format(
                    name, values.size, query_count
                )
            )
        normalized[name] = values

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_index", "query_smiles"] + columns)
        for idx, smiles in enumerate(query_smiles):
            row = [idx, smiles]
            for name in columns:
                value = normalized[name][idx]
                row.append(
                    "" if not np.isfinite(value) else "{:.17g}".format(value)
                )
            writer.writerow(row)


def latent_neighbors(
    query_vectors, train_vectors, train_smiles, k,
    cpu=False, skip_first=False, batch_size=200000,
):
    import faiss

    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
    train_vectors = np.ascontiguousarray(train_vectors, dtype=np.float32)
    if query_vectors.shape[1] != train_vectors.shape[1]:
        raise ValueError("Query/train latent dimensions differ")
    index = faiss.IndexFlatL2(train_vectors.shape[1])
    if not cpu and faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
    for start in range(0, len(train_vectors), batch_size):
        index.add(train_vectors[start:start + batch_size])
    search_k = min(len(train_vectors), k + int(skip_first))
    d2, indices = index.search(query_vectors, search_k)
    if skip_first:
        d2, indices = d2[:, 1:], indices[:, 1:]
    d2, indices = d2[:, :k], indices[:, :k]
    if indices.shape[1] != k or np.any(indices < 0):
        raise RuntimeError("Unable to retrieve k latent neighbors for every query")
    distances = np.sqrt(np.maximum(d2, 0.0))
    near_smiles = [train_smiles[int(idx)] for idx in indices.reshape(-1)]
    return indices, distances, near_smiles


def smiles_dissimilarities(query_smiles, near_smiles, k, radius, nbits):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    def fingerprint(text):
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(
            molecule, radius, nBits=nbits
        )

    query_fps = [fingerprint(text) for text in query_smiles]
    near_fps = [fingerprint(text) for text in near_smiles]
    result = np.empty((len(query_smiles), k), dtype=float)
    for i, query_fp in enumerate(query_fps):
        for j in range(k):
            near_fp = near_fps[i * k + j]
            result[i, j] = (
                1.0 if query_fp is None or near_fp is None
                else 1.0 - DataStructs.TanimotoSimilarity(query_fp, near_fp)
            )
    return result


def map_smiles_to_rows(values, reference_smiles):
    mapping = defaultdict(list)
    for idx, text in enumerate(reference_smiles):
        mapping[text].append(idx)
    rows = []
    for text in values:
        matches = mapping.get(text)
        if not matches:
            raise ValueError(
                "Chemical-neighbor SMILES not found in train set: {}".format(text)
            )
        rows.append(matches[0])
    return np.asarray(rows, dtype=np.int64)


def paired_latent_distances(query_vectors, train_vectors, indices, k):
    blocks = train_vectors[indices.reshape(-1)].reshape(
        len(query_vectors), k, -1
    )
    return np.linalg.norm(blocks - query_vectors[:, None, :], axis=2)


def _summary(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite)) if finite.size else float("nan"),
        "min": float(np.min(finite)) if finite.size else float("nan"),
        "max": float(np.max(finite)) if finite.size else float("nan"),
        "valid_queries": int(finite.size),
        "total_queries": int(values.size),
    }


def neighbor_alignment(latent_dissim, chemical_dissim, random_similarity):
    latent_mean = np.mean(1.0 - latent_dissim, axis=1)
    chemical_mean = np.mean(1.0 - chemical_dissim, axis=1)
    ties = np.abs(latent_mean - chemical_mean) <= ALIGNMENT_TOLERANCE
    latent_mean = np.where(ties, chemical_mean, latent_mean)
    violations = latent_mean > chemical_mean + ALIGNMENT_TOLERANCE
    if np.any(violations):
        excess = latent_mean[violations] - chemical_mean[violations]
        raise ValueError(
            "Chemical reference is not exact Tanimoto top-k: "
            "{} query/queries have latent similarity above the chemical "
            "optimum (maximum excess {:.6g}). Rebuild the reference.".format(
                int(np.sum(violations)), float(np.max(excess))
            )
        )
    random_similarity = np.asarray(random_similarity, dtype=float).reshape(-1)
    denominator = chemical_mean - random_similarity
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(
            np.abs(denominator) > 1e-12,
            (latent_mean - random_similarity) / denominator,
            np.nan,
        )
    return np.where(
        (values > 1.0) & (values <= 1.0 + ALIGNMENT_TOLERANCE), 1.0, values
    )


def querywise_wasserstein(
    latent_dissim, latent_dist, chemical_dissim, chemical_dist
):
    latent = np.stack([latent_dissim, latent_dist], axis=-1)
    chemical = np.stack([chemical_dissim, chemical_dist], axis=-1)
    pooled = np.concatenate(
        [latent.reshape(-1, 2), chemical.reshape(-1, 2)], axis=0
    )
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0) + 1e-12
    latent = (latent - mean) / std
    chemical = (chemical - mean) / std
    values = np.empty(len(latent), dtype=float)
    for idx in range(len(latent)):
        cost = cdist(latent[idx], chemical[idx], metric="euclidean")
        row, col = linear_sum_assignment(cost)
        values[idx] = cost[row, col].mean()
    return values


def compute_all_metrics(
    latent_dissim, latent_dist, chemical_dissim, chemical_dist,
    random_similarity,
):
    alignment = neighbor_alignment(
        latent_dissim, chemical_dissim, random_similarity
    )
    wasserstein = querywise_wasserstein(
        latent_dissim, latent_dist, chemical_dissim, chemical_dist
    )
    return {
        "arrays": {
            "latent_mean_similarity": np.mean(1.0 - latent_dissim, axis=1),
            "chemical_mean_similarity": np.mean(1.0 - chemical_dissim, axis=1),
            "latent_mean_near_distance": np.mean(latent_dist, axis=1),
            "chemical_mean_near_distance": np.mean(chemical_dist, axis=1),
            "neighbor_alignment": alignment,
            "querywise_wasserstein": wasserstein,
        },
        "summary": {
            "mean_similarity": {
                "latent": float(np.mean(1.0 - latent_dissim)),
                "chemical": float(np.mean(1.0 - chemical_dissim)),
            },
            "mean_near_distance": {
                "latent": float(np.mean(latent_dist)),
                "chemical": float(np.mean(chemical_dist)),
            },
            "neighbor_alignment": _summary(alignment),
            "querywise_wasserstein": _summary(wasserstein),
        },
    }


def validate_reference(
    reference_dir, train_path, query_path, neighbors, radius, nbits
):
    manifest_path = reference_dir / "reference_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Missing reference manifest: {}".format(manifest_path))
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    checks = [
        (manifest.get("train_sha256"), sha256_file(train_path), "train SMILES checksum"),
        (manifest.get("query_sha256"), sha256_file(query_path), "query SMILES checksum"),
        (int(manifest.get("neighbors", -1)), neighbors, "neighbors"),
        (int(manifest.get("radius", -1)), radius, "Morgan radius"),
        (int(manifest.get("nbits", -1)), nbits, "Morgan nbits"),
        (manifest.get("search_method"), EXACT_CHEMICAL_SEARCH, "chemical search method"),
    ]
    failures = [
        "{}: reference={!r}, current={!r}".format(name, old, new)
        for old, new, name in checks if old != new
    ]
    if failures:
        raise ValueError(
            "Chemical reference does not match this run:\n - "
            + "\n - ".join(failures)
        )
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--query-csv", required=True)
    parser.add_argument(
        "--train-smiles",
        default=str(ROOT / "data" / "guacamol_v1_train.smiles"),
    )
    parser.add_argument(
        "--query-smiles", default=str(ROOT / "data" / "query1.smiles")
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="Default: eval_latent/reference/<query name>.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "evaluate" / "results"),
        help="Default: evaluate/results.",
    )
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--nbits", type=int, default=2048)
    parser.add_argument("--faiss-cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=200000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.reference_dir is None:
        query_name = Path(args.query_smiles).expanduser().stem
        if query_name.startswith("guacamol_"):
            query_name = query_name[len("guacamol_"):]
        args.reference_dir = str(
            ROOT / "eval_latent" / "reference" / query_name
        )
    train_csv = Path(args.train_csv).expanduser().resolve()
    query_csv = Path(args.query_csv).expanduser().resolve()
    train_path = Path(args.train_smiles).expanduser().resolve()
    query_path = Path(args.query_smiles).expanduser().resolve()
    reference_dir = Path(args.reference_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(
            "Output directory is not empty; use --force: {}".format(output_dir)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = validate_reference(
        reference_dir, train_path, query_path,
        args.neighbors, args.radius, args.nbits,
    )
    train_smiles = load_lines(train_path)
    query_smiles = load_lines(query_path)
    train_vectors = load_csv_matrix(train_csv)
    query_vectors = load_csv_matrix(query_csv)
    if len(train_vectors) != len(train_smiles):
        raise ValueError(
            "train CSV rows ({}) != train SMILES ({})".format(
                len(train_vectors), len(train_smiles)
            )
        )
    if len(query_vectors) != len(query_smiles):
        raise ValueError(
            "query CSV rows ({}) != query SMILES ({})".format(
                len(query_vectors), len(query_smiles)
            )
        )

    k = args.neighbors
    _, latent_dist, latent_near = latent_neighbors(
        query_vectors, train_vectors, train_smiles, k,
        cpu=args.faiss_cpu, skip_first=False, batch_size=args.batch_size,
    )
    latent_dissim = smiles_dissimilarities(
        query_smiles, latent_near, k, args.radius, args.nbits
    )
    chemical_near = load_lines(reference_dir / "chemical_near.txt")
    chemical_dissim = reshape_qk(
        load_numeric(reference_dir / "chemical_dissim.txt"),
        k, "chemical_dissim",
    )
    random_sim = load_numeric(
        reference_dir / "random_sim.txt"
    ).reshape(-1)
    if len(chemical_near) != len(query_smiles) * k:
        raise ValueError(
            "chemical_near count does not equal query_count * neighbors"
        )
    if chemical_dissim.shape != (len(query_smiles), k):
        raise ValueError(
            "chemical_dissim shape mismatch: {}".format(chemical_dissim.shape)
        )
    if random_sim.size != len(query_smiles):
        raise ValueError("random_sim count does not equal query count")

    chemical_indices = map_smiles_to_rows(
        chemical_near, train_smiles
    ).reshape(len(query_smiles), k)
    chemical_dist = paired_latent_distances(
        query_vectors, train_vectors, chemical_indices, k
    )
    metrics = compute_all_metrics(
        latent_dissim, latent_dist, chemical_dissim, chemical_dist, random_sim
    )

    write_lines(output_dir / "latent_near.txt", latent_near)
    write_lines(
        output_dir / "latent_near_dist.txt",
        latent_dist.reshape(-1), "{:.10f}",
    )
    write_lines(
        output_dir / "latent_near_dissim.txt",
        latent_dissim.reshape(-1), "{:.10f}",
    )
    write_lines(
        output_dir / "chemical_near_dist.txt",
        chemical_dist.reshape(-1), "{:.10f}",
    )
    write_query_metrics(
        output_dir / "query_metrics.csv", query_smiles, metrics["arrays"]
    )
    write_json(output_dir / "metrics.json", metrics["summary"])
    config = vars(args).copy()
    config.update({
        "generated_at": datetime.now().isoformat(),
        "train_sha256": manifest["train_sha256"],
        "query_sha256": manifest["query_sha256"],
        "query_count": len(query_smiles),
        "train_count": len(train_smiles),
    })
    write_json(output_dir / "pipeline_config.json", config)

    summary = metrics["summary"]
    print("=== Mean similarity ===")
    print("Latent similarity mean   : {:.10f}".format(
        summary["mean_similarity"]["latent"]
    ))
    print("Chemical similarity mean : {:.10f}".format(
        summary["mean_similarity"]["chemical"]
    ))
    print()
    print("=== Mean near distance ===")
    print("Latent near distance mean   : {:.10f}".format(
        summary["mean_near_distance"]["latent"]
    ))
    print("Chemical near distance mean : {:.10f}".format(
        summary["mean_near_distance"]["chemical"]
    ))
    print()
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("[pipeline] results: {}".format(output_dir))


if __name__ == "__main__":
    main()

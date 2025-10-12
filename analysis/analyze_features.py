#!/usr/bin/env python3
"""Aggregate per-dimension feature statistics and plot distributions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # ensure headless backend
import matplotlib.pyplot as plt


CHUNK_SIZE = 5000
MAX_SAMPLES = 50000
RNG_SEED = 0
DEFAULT_FEATURE_ROOT = Path("/home/jingchun/TransformerVAE/featurization/results")
RESULTS_BASE_DIR = Path("/home/jingchun/TransformerVAE/analysis/results")
DEFAULT_OUTPUT_NAME = "default"


def _merge_chunk(
    count: int,
    mean: np.ndarray,
    m2: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    chunk: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update running statistics with a new data chunk."""
    chunk_count = chunk.shape[0]
    chunk_sum = chunk.sum(axis=0)
    chunk_sum_sq = np.square(chunk).sum(axis=0)
    chunk_mean = chunk_sum / chunk_count
    chunk_m2 = chunk_sum_sq - np.square(chunk_sum) / chunk_count
    chunk_min = chunk.min(axis=0)
    chunk_max = chunk.max(axis=0)

    if mean is None:
        mean = np.zeros(chunk.shape[1], dtype=np.float64)
        m2 = np.zeros_like(mean)
        min_vals = chunk_min.copy()
        max_vals = chunk_max.copy()

    delta = chunk_mean - mean
    total_count = count + chunk_count
    if total_count == 0:
        return 0, mean, m2, min_vals, max_vals

    mean += delta * (chunk_count / total_count)
    if count:
        m2 += chunk_m2 + (delta * delta) * (count * chunk_count / total_count)
    else:
        m2 += chunk_m2

    min_vals = np.minimum(min_vals, chunk_min)
    max_vals = np.maximum(max_vals, chunk_max)
    return total_count, mean, m2, min_vals, max_vals


def _update_reservoir(
    rng: np.random.Generator,
    sample_vals: np.ndarray | None,
    sample_priorities: np.ndarray | None,
    chunk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reservoir sample based on random priorities."""
    priorities = rng.random(chunk.shape[0])
    chunk_small = chunk.astype(np.float32, copy=False)

    if sample_vals is None:
        if chunk.shape[0] <= MAX_SAMPLES:
            return chunk_small.copy(), priorities
        idx = np.argpartition(priorities, -MAX_SAMPLES)[-MAX_SAMPLES:]
        return chunk_small[idx], priorities[idx]

    combined_vals = np.vstack((sample_vals, chunk_small))
    combined_priorities = np.concatenate((sample_priorities, priorities))
    if combined_vals.shape[0] <= MAX_SAMPLES:
        return combined_vals, combined_priorities

    idx = np.argpartition(combined_priorities, -MAX_SAMPLES)[-MAX_SAMPLES:]
    return combined_vals[idx], combined_priorities[idx]


def _resolve_feature_dir(feature_dir: Path) -> Path:
    """Resolve provided directory relative to default root when needed."""
    if feature_dir.is_absolute():
        return feature_dir

    cwd_candidate = (Path.cwd() / feature_dir).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (DEFAULT_FEATURE_ROOT / feature_dir).resolve()


def _resolve_output_dir(name: Path | str | None) -> Path:
    """Map the provided output name to a directory under the fixed results base."""
    RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True)

    if name is None or str(name).strip() in {"", "."}:
        relative = Path(DEFAULT_OUTPUT_NAME)
    else:
        relative = Path(name)

    if relative.is_absolute():
        raise SystemExit(
            "--output must be a relative path or name (it will be placed "
            "under /home/jingchun/TransformerVAE/analysis/results)."
        )

    output_dir = (RESULTS_BASE_DIR / relative).resolve()
    try:
        output_dir.relative_to(RESULTS_BASE_DIR)
    except ValueError as exc:
        raise SystemExit(
            "Resolved output directory escapes the fixed results base."
        ) from exc

    return output_dir


def analyze_features(feature_root: Path, output_dir: Path) -> None:
    csv_paths = sorted(feature_root.rglob("feature.csv"))
    if not csv_paths:
        raise SystemExit(f"No feature.csv files found under {feature_root}")

    rng = np.random.default_rng(RNG_SEED)
    count = 0
    mean = None
    m2 = None
    min_vals = None
    max_vals = None
    dims = None
    sample_vals = None
    sample_priorities = None

    for csv_path in csv_paths:
        for chunk_df in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
            chunk = chunk_df.to_numpy(dtype=np.float64, copy=True)
            dims = chunk.shape[1] if dims is None else dims
            count, mean, m2, min_vals, max_vals = _merge_chunk(
                count, mean, m2, min_vals, max_vals, chunk
            )
            sample_vals, sample_priorities = _update_reservoir(
                rng, sample_vals, sample_priorities, chunk
            )

    if mean is None or dims is None:
        raise SystemExit("No data processed from feature.csv files")

    if count <= 1:
        std = np.zeros_like(mean)
    else:
        std = np.sqrt(np.maximum(m2 / (count - 1), 0.0))

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "per_dimension_stats.csv"
    fig_path = output_dir / "feature_distribution.png"

    stats_df = pd.DataFrame(
        {
            "dimension": np.arange(dims, dtype=int),
            "count": count,
            "mean": mean,
            "std": std,
            "min": min_vals,
            "max": max_vals,
        }
    )

    stats_df.to_csv(stats_path, index=False)

    if sample_vals is None or sample_vals.size == 0:
        print("Warning: no sample collected; skipping distribution plot")
        print(f"Saved stats to {stats_path}")
        return

    # Build overlay plot using sampled data for tractability
    bin_count = 120
    sample_min = float(np.min(sample_vals))
    sample_max = float(np.max(sample_vals))
    # Expand bounds slightly to avoid boundary artifacts
    padding = 0.02 * (sample_max - sample_min or 1.0)
    bin_edges = np.linspace(sample_min - padding, sample_max + padding, bin_count + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    densities = []
    for dim in range(sample_vals.shape[1]):
        counts, _ = np.histogram(sample_vals[:, dim], bins=bin_edges, density=True)
        densities.append(counts)

    fig, ax = plt.subplots(figsize=(12, 6))
    for counts in densities:
        ax.plot(bin_centers, counts, color="tab:blue", alpha=0.15, linewidth=0.7)

    ax.set_xlabel("Feature value")
    ax.set_ylabel("Density (sampled)")
    ax.set_title(
        "Per-dimension feature distributions (256 dims overlay)\n"
        f"Sample size: {sample_vals.shape[0]:,} / Total vectors: {count:,}"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"Saved stats to {stats_path}")
    print(f"Saved overlay plot to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-dimension statistics and overlay distributions for feature.csv "
            "files under the specified directory."
        )
    )
    parser.add_argument(
        "--dir",
        dest="feature_dir",
        required=True,
        type=Path,
        help="Directory containing feature.csv files (searched recursively).",
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_NAME),
        help=(
            "Subdirectory name created under /home/jingchun/TransformerVAE/analysis/results "
            "for analysis outputs."
        ),
    )
    args = parser.parse_args()
    feature_dir = _resolve_feature_dir(args.feature_dir)
    if not feature_dir.is_dir():
        raise SystemExit(
            "Provided --dir path is not a directory: "
            f"{args.feature_dir} (resolved to {feature_dir})"
        )
    output_dir = _resolve_output_dir(args.output)
    analyze_features(feature_dir, output_dir)

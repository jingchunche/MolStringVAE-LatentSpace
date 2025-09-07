#!/usr/bin/env python3
import os, math, argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser(
        description="Estimate sigma* as the mean per-dimension std from large feature.csv by streaming"
    )
    ap.add_argument("--weight", required=True, help="weight name under featurization/results/")
    ap.add_argument("--step",   required=True, help="step subdir (e.g., 000100, 100000)")
    ap.add_argument("--root",   default="/home/jingchun/TransformerVAE/featurization/results",
                    help="root dir containing <weight>/<step>/feature.csv")
    ap.add_argument("--chunksize", type=int, default=100000, help="rows per chunk (default: 100k)")
    ap.add_argument("--drop_any_nan", action="store_true",
                    help="drop rows with ANY NaN (default: per-column NaNs are ignored)")
    ap.add_argument("--multipliers", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                    help="multipliers around sigma* to suggest")
    ap.add_argument("--save", action="store_true",
                    help="save calc_sigma.txt next to feature.csv")
    args = ap.parse_args()

    feat_path = os.path.join(args.root, args.weight, args.step, "feature.csv")
    if not os.path.isfile(feat_path):
        raise FileNotFoundError(f"file not found: {feat_path}")

    head = pd.read_csv(feat_path, nrows=100, dtype=np.float32)
    num_cols = head.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        raise ValueError("No numeric columns detected in feature.csv")
    d = len(num_cols)
    print(f"[info] detected d={d} numeric columns")

    # Online accumulators per dimension (float64 for numerical stability)
    cnt = np.zeros(d, dtype=np.int64)
    s1  = np.zeros(d, dtype=np.float64)  # sum
    s2  = np.zeros(d, dtype=np.float64)  # sum of squares

    total_rows = 0
    kept_rows  = 0   # rows kept only used when drop_any_nan=True
    chunk_idx  = 0

    for chunk in pd.read_csv(feat_path, chunksize=args.chunksize):
        chunk_idx += 1
        X = chunk[num_cols].to_numpy(dtype=np.float64, copy=False)
        total_rows += X.shape[0]

        if args.drop_any_nan:
            # Drop any-NaN rows entirely
            mask = np.isfinite(X).all(axis=1)
            X = X[mask]
            kept_rows += X.shape[0]
            if X.size == 0:
                continue
            # Update per-dim stats with full rows
            s1  += X.sum(axis=0, dtype=np.float64)
            s2  += np.square(X, dtype=np.float64).sum(axis=0, dtype=np.float64)
            cnt += np.int64(X.shape[0])
        else:
            # Per-dimension NaN-robust update: ignore NaNs in each column independently
            finite = np.isfinite(X)                  # shape: (n_rows, d)
            if not finite.any():
                continue
            # Replace NaNs with 0 just for summations; counts track how many were finite
            X0 = np.where(finite, X, 0.0)
            s1  += X0.sum(axis=0, dtype=np.float64)
            s2  += np.square(X0, dtype=np.float64).sum(axis=0, dtype=np.float64)
            cnt += finite.sum(axis=0, dtype=np.int64)

        if chunk_idx % 10 == 0:
            if args.drop_any_nan:
                print(f"[info] processed {total_rows:,} rows, kept {kept_rows:,}")
            else:
                avg_col_n = float(cnt.mean()) if cnt.size else 0.0
                print(f"[info] processed {total_rows:,} rows, avg per-dim count ≈ {avg_col_n:,.0f}")

    # Compute per-dimension std (sample std, ddof=1 where possible)
    valid = cnt > 1
    if not valid.any():
        raise ValueError("Not enough valid data to compute per-dimension std (need count > 1)")

    var = np.zeros(d, dtype=np.float64)
    # var = (sum(x^2) - sum(x)^2 / n) / (n - 1)
    var[valid] = (s2[valid] - np.square(s1[valid]) / cnt[valid]) / (cnt[valid] - 1)
    # guard tiny negative due to round-off
    var[var < 0] = 0.0
    std = np.sqrt(var, dtype=np.float64)

    # sigma* = mean per-dimension std over valid dims
    sigma_star = float(std[valid].mean())

    # Some diagnostics on the std distribution
    std_mean = float(std[valid].mean())
    std_med  = float(np.median(std[valid]))
    std_q25  = float(np.quantile(std[valid], 0.25))
    std_q75  = float(np.quantile(std[valid], 0.75))
    n_valid_dims = int(valid.sum())

    print(f"[info] read {feat_path}")
    print(f"[info] total rows seen = {total_rows:,}")
    if args.drop_any_nan:
        print(f"[info] rows kept (no-NaN rows) = {kept_rows:,}")
    print(f"[info] valid dimensions (count>1) = {n_valid_dims}/{d}")
    print(f"[info] per-dim std: mean={std_mean:.6f}, median={std_med:.6f}, q25={std_q25:.6f}, q75={std_q75:.6f}")
    print(f"[info] sigma* (mean per-dim std) = {sigma_star:.6f}")
    print("[info] suggestions (multipliers * sigma*):")
    for m in args.multipliers:
        print(f"  {m:g} * sigma* = {m * sigma_star:.6f}")

    if args.save:
        out_path = os.path.join(args.root, args.weight, args.step, "calc_sigma.txt")
        with open(out_path, "w") as f:
            f.write(f"d={d}\n")
            f.write(f"total_rows={total_rows}\n")
            if args.drop_any_nan:
                f.write(f"rows_kept={kept_rows}\n")
            f.write(f"valid_dims={n_valid_dims}\n")
            f.write(f"std_mean={std_mean}\n")
            f.write(f"std_median={std_med}\n")
            f.write(f"std_q25={std_q25}\n")
            f.write(f"std_q75={std_q75}\n")
            f.write(f"sigma_star={sigma_star}\n")
            f.write("candidates=" + ",".join(f"{m * sigma_star:.6f}" for m in args.multipliers) + "\n")
        print(f"[info] saved -> {out_path}")

if __name__ == "__main__":
    main()

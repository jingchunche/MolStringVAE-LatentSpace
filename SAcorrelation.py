#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import sascore as sascorer

# -------------------- 工具 --------------------
def read_lines(path):
    with open(path, "r") as f:
        return [line.strip() for line in f]

def safe_float_list(lines):
    out = []
    for s in lines:
        try:
            out.append(float(s))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=float)

def compute_sa_delta(smiles1, smiles2, out_path, scorer):
    """計算 |SA(smiles1) - SA(smiles2)| 差值"""
    deltas, valid = [], []
    with open(out_path, "w") as fw:
        for s1, s2 in zip(smiles1, smiles2):
            m1 = Chem.MolFromSmiles(s1) if s1 else None
            m2 = Chem.MolFromSmiles(s2) if s2 else None
            if m1 is None or m2 is None:
                fw.write("nan\n")
                deltas.append(np.nan)
                valid.append(False)
                continue
            try:
                sa1 = float(scorer.calculateScore(m1))
                sa2 = float(scorer.calculateScore(m2))
                d = abs(sa1 - sa2)   # SA 分數差
            except Exception:
                d = np.nan
            if np.isnan(d):
                fw.write("nan\n")
                valid.append(False)
            else:
                fw.write(f"{d}\n")
                valid.append(True)
            deltas.append(d)
    return np.array(deltas, dtype=float), np.array(valid, dtype=bool)

def _spearman(x, y):
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(x, y)
        return float(rho), float(p)
    except Exception:
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        if rx.std() == 0 or ry.std() == 0:
            return float("nan"), float("nan")
        r = np.corrcoef(rx, ry)[0, 1]
        return float(r), float("nan")

def _kendall_tau_b(x, y):
    try:
        from scipy.stats import kendalltau
        tau, p = kendalltau(x, y, variant='b')
        return float(tau), float(p)
    except Exception:
        n = len(x)
        if n < 2:
            return float("nan"), float("nan")
        C = D = T_x = T_y = 0
        for i in range(n-1):
            xi, yi = x[i], y[i]
            for j in range(i+1, n):
                xj, yj = x[j], y[j]
                sx = (xi > xj) - (xi < xj)
                sy = (yi > yj) - (yi < yj)
                if sx == 0 and sy == 0:
                    continue
                if sx == 0 and sy != 0:
                    T_x += 1
                elif sy == 0 and sx != 0:
                    T_y += 1
                else:
                    if sx == sy:
                        C += 1
                    else:
                        D += 1
        denom = math.sqrt((C + D + T_x) * (C + D + T_y))
        if denom == 0:
            return float("nan"), float("nan")
        tau = (C - D) / denom
        return float(tau), float("nan")

# -------------------- 主程式 --------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compute ΔSA-score vs latent distance and per-seed Spearman/Kendall."
    )
    ap.add_argument("--result_dir", default="./decoding/results/sample")
    ap.add_argument("--smiles1",   default="smiles_near.txt")
    ap.add_argument("--smiles2",   default="smiles_seed.txt")
    ap.add_argument("--dist",      default="dist.txt")
    ap.add_argument("--delta_out", default="sa_delta.txt")
    ap.add_argument("--stats_out", default="sa_stats.txt")
    ap.add_argument("--knn",  type=int, required=True, help="每個 seed 的鄰近數（用於 per-seed 指標）")
    args = ap.parse_args()

    result_dir = args.result_dir
    smiles1_path = os.path.join(result_dir, args.smiles1)
    smiles2_path = os.path.join(result_dir, args.smiles2)
    dist_path    = os.path.join(result_dir, args.dist)

    delta_out = os.path.join(result_dir, args.delta_out)
    stats_out = os.path.join(result_dir, args.stats_out)

    # 讀資料
    s1 = read_lines(smiles1_path)
    s2 = read_lines(smiles2_path)
    dz = read_lines(dist_path)

    n = min(len(s1), len(s2), len(dz))
    if (len(s1), len(s2), len(dz)) != (n, n, n):
        print(f"[warn] line counts differ; truncate to {n}")
        s1, s2, dz = s1[:n], s2[:n], dz[:n]

    # 計算 ΔSA
    delta_all, mol_valid = compute_sa_delta(s1, s2, delta_out, scorer=sascorer)
    print(f"[info] wrote SA deltas to {delta_out}")

    dist_all = safe_float_list(dz)
    valid = np.isfinite(delta_all) & np.isfinite(dist_all) & mol_valid
    x = delta_all[valid]   # ΔSA
    y = dist_all[valid]    # latent distance

    # 全域線性相關
    if x.size >= 2:
        try:
            from scipy.stats import pearsonr
            pr, pp = pearsonr(x, y)
        except Exception:
            pr, pp = np.corrcoef(x, y)[0, 1], float("nan")
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        pr = pp = r2 = float("nan")
        slope = intercept = float("nan")

    # ===== per-seed Spearman / Kendall =====
    K = int(args.knn)
    local_rhos, local_taus = [], []
    for start in range(0, n, K):
        end = min(n, start + K)
        mask = valid[start:end]
        if mask.sum() >= 2:
            xa = delta_all[start:end][mask]
            ya = dist_all[start:end][mask]
            rho, _ = _spearman(xa, ya)
            tau, _ = _kendall_tau_b(xa, ya)
            if np.isfinite(rho): local_rhos.append(rho)
            if np.isfinite(tau): local_taus.append(tau)

    def _summ(a):
        if len(a) == 0:
            return float("nan"), float("nan"), float("nan"), 0
        a = np.array(a, dtype=float)
        return float(np.mean(a)), float(np.median(a)), float(np.std(a, ddof=1)), len(a)

    sp_mean, sp_med, sp_std, sp_n = _summ(local_rhos)
    kt_mean, kt_med, kt_std, kt_n = _summ(local_taus)

    # 寫 stats.txt
    with open(stats_out, "w") as f:
        f.write(f"n_valid={x.size}\n")
        f.write(f"pearson_r={pr}\n")
        f.write(f"pearson_p={pp}\n")
        f.write(f"slope={slope}\n")
        f.write(f"intercept={intercept}\n")
        f.write(f"R2={r2}\n")
        if x.size:
            f.write(f"x_mean={np.mean(x)}\n")
            f.write(f"x_std={np.std(x)}\n")
            f.write(f"x_min={np.min(x)}\n")
            f.write(f"x_max={np.max(x)}\n")
            f.write(f"y_mean={np.mean(y)}\n")
            f.write(f"y_std={np.std(y)}\n")
            f.write(f"y_min={np.min(y)}\n")
            f.write(f"y_max={np.max(y)}\n")
        f.write(f"knn={K}\n")
        f.write(f"local_spearman_n={sp_n}\n")
        f.write(f"local_spearman_mean={sp_mean}\n")
        f.write(f"local_spearman_median={sp_med}\n")
        f.write(f"local_spearman_std={sp_std}\n")
        f.write(f"local_kendall_n={kt_n}\n")
        f.write(f"local_kendall_mean={kt_mean}\n")
        f.write(f"local_kendall_median={kt_med}\n")
        f.write(f"local_kendall_std={kt_std}\n")
    print(f"[info] wrote stats to {stats_out}")

if __name__ == "__main__":
    main()

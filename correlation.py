#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, argparse
import numpy as np

# 無視窗環境
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


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


def compute_dissim(smiles1, smiles2, out_path, radius=2, nbits=2048):
    """回傳 dissimilarity 向量與有效遮罩；同時把每行寫到 out_path（無效寫 'nan'）"""
    dissim, valid = [], []
    with open(out_path, "w") as fw:
        for s1, s2 in zip(smiles1, smiles2):
            m1 = Chem.MolFromSmiles(s1) if s1 else None
            m2 = Chem.MolFromSmiles(s2) if s2 else None
            if m1 is None or m2 is None:
                fw.write("nan\n")
                dissim.append(np.nan)
                valid.append(False)
                continue
            fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, radius, nBits=nbits)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, radius, nBits=nbits)
            dice = DataStructs.DiceSimilarity(fp1, fp2)
            d = 1.0 - float(dice)
            fw.write(f"{d}\n")
            dissim.append(d)
            valid.append(True)
    return np.array(dissim, dtype=float), np.array(valid, dtype=bool)


def main():
    ap = argparse.ArgumentParser(
        description="Compute 1-Dice dissimilarity, correlate with latent distance, and draw paper-style heatmap."
    )
    ap.add_argument("--result_dir", default="./decoding/results/sample")
    ap.add_argument("--smiles1",   default="decoded_smiles1.txt")
    ap.add_argument("--smiles2",   default="decoded_smiles2.txt")
    ap.add_argument("--dist",      default="dist.txt")
    ap.add_argument("--dissim_out", default="dissimilarity.txt")
    ap.add_argument("--plot_out",   default="dissim_vs_dist.png")
    ap.add_argument("--stats_out",  default="stats.txt")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--nbits",  type=int, default=2048)
    ap.add_argument("--bins",   type=int, default=140, help="2D histogram bins per axis")
    ap.add_argument("--qlo",    type=float, default=0.01, help="lower quantile for axis trimming")
    ap.add_argument("--qhi",    type=float, default=0.99, help="upper quantile for axis trimming")
    ap.add_argument("--smooth_sigma", type=float, default=1.2, help="gaussian smoothing sigma; 0=off")
    ap.add_argument("--scatter", action="store_true", help="overlay faint scatter points")
    ap.add_argument("--xlim", type=float, nargs=2, default=None)
    ap.add_argument("--ylim", type=float, nargs=2, default=None)
    args = ap.parse_args()

    # paths
    result_dir = args.result_dir
    smiles1_path = os.path.join(result_dir, args.smiles1)
    smiles2_path = os.path.join(result_dir, args.smiles2)
    dist_path    = os.path.join(result_dir, args.dist)

    dissim_out = os.path.join(result_dir, args.dissim_out)
    plot_out   = os.path.join(result_dir, args.plot_out)
    stats_out  = os.path.join(result_dir, args.stats_out)

    # 讀資料
    s1 = read_lines(smiles1_path)
    s2 = read_lines(smiles2_path)
    dz = read_lines(dist_path)

    n = min(len(s1), len(s2), len(dz))
    if (len(s1), len(s2), len(dz)) != (n, n, n):
        print(f"[warn] line counts differ; truncate to {n}")
        s1, s2, dz = s1[:n], s2[:n], dz[:n]

    # 計算 1−Dice 並輸出 dissimilarity.txt（無效保留 nan 以維持行對齊）
    dissim, mol_valid = compute_dissim(s1, s2, dissim_out, radius=args.radius, nbits=args.nbits)
    print(f"[info] wrote dissimilarities to {dissim_out}")

    # dist 轉 float
    dist_vals = safe_float_list(dz)

    # 有效資料（兩者皆非 nan）
    valid = np.isfinite(dissim) & np.isfinite(dist_vals) & mol_valid
    x = dissim[valid]      # Molecular dissimilarity (1−Dice)
    y = dist_vals[valid]   # Latent distance

    # 統計：Pearson / Spearman / 線性擬合
    if x.size >= 2:
        try:
            from scipy.stats import pearsonr, spearmanr
            pr, pp = pearsonr(x, y)
            sr, sp = spearmanr(x, y)
        except Exception:
            pr, pp = np.corrcoef(x, y)[0, 1], float("nan")
            sr, sp = float("nan"), float("nan")
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        pr = pp = sr = sp = r2 = float("nan")
        slope = intercept = float("nan")

    # 寫 stats.txt
    with open(stats_out, "w") as f:
        f.write(f"n_valid={x.size}\n")
        f.write(f"pearson_r={pr}\n")
        f.write(f"pearson_p={pp}\n")
        f.write(f"spearman_rho={sr}\n")
        f.write(f"spearman_p={sp}\n")
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
    print(f"[info] wrote stats to {stats_out}")

    # --- PAPER-STYLE PLOT ---
    if x.size >= 2:
        # 決定顯示範圍（避免離群把畫面撐太大）
        qlo, qhi = float(args.qlo), float(args.qhi)
        xlo, xhi = np.quantile(x, qlo), np.quantile(x, qhi)
        ylo, yhi = np.quantile(y, qlo), np.quantile(y, qhi)
        # 若上下界相等，稍微加一點 padding
        if not np.isfinite(xlo) or not np.isfinite(xhi) or xlo == xhi:
            xlo, xhi = np.min(x), np.max(x)
        if not np.isfinite(ylo) or not np.isfinite(yhi) or ylo == yhi:
            ylo, yhi = np.min(y), np.max(y)
        if xlo == xhi: xlo, xhi = xlo - 1e-6, xhi + 1e-6
        if ylo == yhi: ylo, yhi = ylo - 1e-6, yhi + 1e-6

        # 2D 直方圖（count）
        H, xedges, yedges = np.histogram2d(
            x, y, bins=(args.bins, args.bins), range=[[xlo, xhi], [ylo, yhi]]
        )
        H = H.astype(float)

        # 以高分位數做飽和，normalize 到 0..1（更像論文的色帶）
        top = np.percentile(H, 99.5) if H.size else 1.0
        if top > 0:
            H = np.clip(H / top, 0, 1)

        # 平滑（可關閉）
        if args.smooth_sigma and args.smooth_sigma > 0:
            try:
                from scipy.ndimage import gaussian_filter
                H = gaussian_filter(H, sigma=float(args.smooth_sigma))
            except Exception:
                pass

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.rcParams.update({"axes.linewidth": 1.2, "font.size": 12})
        fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=320)

        cmap = "turbo" if "turbo" in plt.colormaps() else ("jet" if "jet" in plt.colormaps() else "viridis")
        im = ax.imshow(
            H.T, origin="lower", aspect="auto", extent=extent,
            cmap=cmap, vmin=0.0, vmax=1.0, interpolation="bilinear"
        )

        # 粗灰色虛線回歸
        xx = np.linspace(xlo, xhi, 300)
        yy = slope * xx + intercept
        ax.plot(xx, yy, "--", color="#b9b9b9", lw=2.8, dashes=(6, 4))

        # 軸範圍/標籤
        if args.xlim: ax.set_xlim(args.xlim)
        else:         ax.set_xlim(xlo, xhi)
        if args.ylim: ax.set_ylim(args.ylim)
        else:         ax.set_ylim(ylo, yhi)

        ax.set_xlabel("Molecular dissimilarity")
        ax.set_ylabel("Latent space distance")

        # 邊框樣式
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("k")
            sp.set_linewidth(1.2)

        # colorbar
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("Normalized population")

        # 標題
        title_r = f"{pr:.3f}" if isinstance(pr, float) and not math.isnan(pr) else "N/A"
        ax.set_title(f"Dissimilarity vs. Latent distance (r={title_r})", pad=10)

        if args.scatter:
            ax.scatter(x, y, s=4, alpha=0.18, edgecolors="none", color="white")

        fig.tight_layout()
        fig.savefig(plot_out, dpi=320)
        plt.close(fig)
        print(f"[info] plot saved to {plot_out}")
    else:
        print("[warn] not enough valid points to plot.")


if __name__ == "__main__":
    main()

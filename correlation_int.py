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


def compute_dissim(smiles_ref, smiles_mid, out_path, radius=2, nbits=2048):
    """計算 1−Dice 不相似度；寫到 out_path（無效寫 'nan'），回傳 (vals, valid_mask)"""
    dissim, valid = [], []
    with open(out_path, "w") as fw:
        for s1, s2 in zip(smiles_ref, smiles_mid):
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


def panel_stats(x, y):
    """回傳 (pearson_r, spearman_rho, slope, intercept)；不足則回 nan"""
    if x.size >= 2:
        try:
            from scipy.stats import pearsonr, spearmanr
            pr, _ = pearsonr(x, y)
            sr, _ = spearmanr(x, y)
        except Exception:
            pr = np.corrcoef(x, y)[0, 1]
            sr = float("nan")
        slope, intercept = np.polyfit(x, y, 1)
        return pr, sr, slope, intercept
    return float("nan"), float("nan"), float("nan"), float("nan")


def plot_two_panels(x1, y1, x2, y2, plot_out,
                    bins=140, qlo=0.01, qhi=0.99, smooth_sigma=1.2,
                    scatter=False, xlim=None, ylim=None):
    """
    x*: dissimilarity (1−Dice), y*: latent distance
    產生單一圖片檔案（左右兩個子圖）
    """
    # 有效點
    v1 = np.isfinite(x1) & np.isfinite(y1)
    v2 = np.isfinite(x2) & np.isfinite(y2)
    x1, y1 = x1[v1], y1[v1]
    x2, y2 = x2[v2], y2[v2]

    if x1.size < 2 and x2.size < 2:
        print("[warn] not enough valid points to plot for both panels.")
        return

    # 軸範圍（用兩組的聯集分位數，確保左右一致好比）
    def qrange(x, y):
        xlo, xhi = np.quantile(x, qlo), np.quantile(x, qhi)
        ylo, yhi = np.quantile(y, qlo), np.quantile(y, qhi)
        return xlo, xhi, ylo, yhi

    xr = []; yr = []
    if x1.size >= 2: xr.append(qrange(x1, y1))
    if x2.size >= 2: xr.append(qrange(x2, y2))
    if xr:
        xlo = min(r[0] for r in xr); xhi = max(r[1] for r in xr)
        ylo = min(r[2] for r in xr); yhi = max(r[3] for r in xr)
    else:
        # 後備（理論上不會走到）
        xlo=xhi=ylo=yhi=0.0

    # 若極端情況相等，補 padding
    if not np.isfinite(xlo) or not np.isfinite(xhi) or xlo == xhi:
        allx = np.concatenate([x1, x2]) if x1.size and x2.size else (x1 if x1.size else x2)
        xlo, xhi = np.min(allx), np.max(allx)
    if not np.isfinite(ylo) or not np.isfinite(yhi) or ylo == yhi:
        ally = np.concatenate([y1, y2]) if y1.size and y2.size else (y1 if y1.size else y2)
        ylo, yhi = np.min(ally), np.max(ally)
    if xlo == xhi: xlo, xhi = xlo - 1e-6, xhi + 1e-6
    if ylo == yhi: ylo, yhi = ylo - 1e-6, yhi + 1e-6

    # 準備圖
    plt.rcParams.update({"axes.linewidth": 1.2, "font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), dpi=320, constrained_layout=True)
    cmap = "turbo" if "turbo" in plt.colormaps() else ("jet" if "jet" in plt.colormaps() else "viridis")

    def draw_panel(ax, x, y, title):
        if x.size < 2:
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
            ax.set_axis_off(); return
        H, xedges, yedges = np.histogram2d(x, y, bins=(bins, bins), range=[[xlo, xhi], [ylo, yhi]])
        H = H.astype(float)
        top = np.percentile(H, 99.5) if H.size else 1.0
        if top > 0: H = np.clip(H / top, 0, 1)
        if smooth_sigma and smooth_sigma > 0:
            try:
                from scipy.ndimage import gaussian_filter
                H = gaussian_filter(H, sigma=float(smooth_sigma))
            except Exception:
                pass
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(H.T, origin="lower", aspect="auto", extent=extent,
                       cmap=cmap, vmin=0.0, vmax=1.0, interpolation="bilinear")

        pr, sr, slope, intercept = panel_stats(x, y)
        xx = np.linspace(xlo, xhi, 300); yy = slope * xx + intercept
        ax.plot(xx, yy, "--", color="#b9b9b9", lw=2.8, dashes=(6, 4))

        ax.set_xlim(xlim if xlim else (xlo, xhi))
        ax.set_ylim(ylim if ylim else (ylo, yhi))
        ax.set_xlabel("Molecular dissimilarity (1−Dice)")
        ax.set_ylabel("Latent space distance")
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("k"); sp.set_linewidth(1.2)
        ax.set_title(f"{title}\nPearson r={pr:.3f} | Spearman ρ={sr:.3f}")

        if scatter:
            ax.scatter(x, y, s=4, alpha=0.18, edgecolors="none", color="white")

        return im

    im1 = draw_panel(axes[0], x1, y1, "to end1")
    im2 = draw_panel(axes[1], x2, y2, "to end2")

    # 共用 colorbar（用左圖的 im 作為參考）
    if isinstance(im1, matplotlib.image.AxesImage):
        cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.046, pad=0.02)
        cbar.set_label("Normalized population")

    fig.savefig(plot_out, dpi=320)
    plt.close(fig)
    print(f"[info] plot saved to {plot_out}")


def main():
    ap = argparse.ArgumentParser(
        description="Compute 1−Dice dissimilarity to both endpoints, write dissmil_to{1,2}.txt, and draw a single two-panel figure."
    )
    ap.add_argument("--result_dir", default="./decoding/results/sample")
    ap.add_argument("--end1",  default="smiles_end1.txt")
    ap.add_argument("--end2",  default="smiles_end2.txt")
    ap.add_argument("--mid",   default="smiles_mid.txt")
    ap.add_argument("--dist1", default="dist_to1.txt")
    ap.add_argument("--dist2", default="dist_to2.txt")

    # 輸出
    ap.add_argument("--dissim1_out", default="dissmil_to1.txt")
    ap.add_argument("--dissim2_out", default="dissmil_to2.txt")
    ap.add_argument("--plot_out",    default="diss_vs_dist.png")

    # 參數
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--nbits",  type=int, default=2048)
    ap.add_argument("--bins",   type=int, default=140)
    ap.add_argument("--qlo",    type=float, default=0.01)
    ap.add_argument("--qhi",    type=float, default=0.99)
    ap.add_argument("--smooth_sigma", type=float, default=1.2)
    ap.add_argument("--scatter", action="store_true")
    ap.add_argument("--xlim", type=float, nargs=2, default=None)
    ap.add_argument("--ylim", type=float, nargs=2, default=None)
    args = ap.parse_args()

    # 路徑
    R = args.result_dir
    p_end1 = os.path.join(R, args.end1)
    p_end2 = os.path.join(R, args.end2)
    p_mid  = os.path.join(R, args.mid)
    p_d1   = os.path.join(R, args.dist1)
    p_d2   = os.path.join(R, args.dist2)

    p_dis1 = os.path.join(R, args.dissim1_out)
    p_dis2 = os.path.join(R, args.dissim2_out)
    p_plot = os.path.join(R, args.plot_out)

    # 讀入
    end1 = read_lines(p_end1)
    end2 = read_lines(p_end2)
    mid  = read_lines(p_mid)
    d1_lines = read_lines(p_d1)
    d2_lines = read_lines(p_d2)

    # 對齊：mid 應為 pairs * mid_num；end1/end2 為 pairs
    pairs = min(len(end1), len(end2))
    if len(end1) != len(end2):
        print(f"[warn] end1/end2 length mismatch, using pairs={pairs}")

    if pairs == 0 or len(mid) == 0:
        raise SystemExit("[error] empty endpoints or mid list.")

    # 推回每對的 mid 數；若不能整除則向下取整並截斷
    mid_num = len(mid) // pairs
    if mid_num == 0:
        raise SystemExit("[error] mid count < pairs; cannot align. Check inputs.")
    expected_mid = pairs * mid_num
    if expected_mid != len(mid):
        print(f"[warn] mid lines ({len(mid)}) not divisible by pairs ({pairs}); truncating to {expected_mid}.")
        mid = mid[:expected_mid]
        d1_lines = d1_lines[:expected_mid]
        d2_lines = d2_lines[:expected_mid]

    # 展開端點，使其與 mid 行數對齊（pair-major）
    end1_rep = np.repeat(end1[:pairs], mid_num).tolist()
    end2_rep = np.repeat(end2[:pairs], mid_num).tolist()

    # 距離轉 float（並截斷到 mid 長度）
    d1 = safe_float_list(d1_lines[:len(mid)])
    d2 = safe_float_list(d2_lines[:len(mid)])

    # ---- 計算不相似度並輸出（逐行對應 mid）----
    dis1, valid1_mol = compute_dissim(end1_rep, mid, p_dis1, radius=args.radius, nbits=args.nbits)
    print(f"[info] wrote dissimilarities to {p_dis1}")
    dis2, valid2_mol = compute_dissim(end2_rep, mid, p_dis2, radius=args.radius, nbits=args.nbits)
    print(f"[info] wrote dissimilarities to {p_dis2}")

    # ---- 繪圖（單一檔，兩個面板）----
    plot_two_panels(
        dis1[valid1_mol], d1[valid1_mol],
        dis2[valid2_mol], d2[valid2_mol],
        plot_out=p_plot,
        bins=args.bins, qlo=args.qlo, qhi=args.qhi,
        smooth_sigma=args.smooth_sigma, scatter=args.scatter,
        xlim=args.xlim, ylim=args.ylim
    )


if __name__ == "__main__":
    main()

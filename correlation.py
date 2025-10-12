#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
from pathlib import Path

# 無視窗環境
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


def read_lines(path):
    with open(path, 'r') as f:
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
    """回傳 1−Tanimoto dissimilarity 與有效遮罩；同時把每行寫到 out_path（無效寫 'nan'）"""
    dissim, valid = [], []
    with open(out_path, 'w') as fw:
        for s1, s2 in zip(smiles1, smiles2):
            m1 = Chem.MolFromSmiles(s1) if s1 else None
            m2 = Chem.MolFromSmiles(s2) if s2 else None
            if m1 is None or m2 is None:
                fw.write('nan\n')
                dissim.append(np.nan)
                valid.append(False)
                continue
            fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, radius, nBits=nbits)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, radius, nBits=nbits)
            tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
            d = 1.0 - float(tanimoto)
            fw.write(f"{d}\n")
            dissim.append(d)
            valid.append(True)
    return np.array(dissim, dtype=float), np.array(valid, dtype=bool)


def filter_valid(dissim, dist_all, mol_valid):
    return np.isfinite(dissim) & np.isfinite(dist_all) & mol_valid


def global_stats(dissim_all, dist_all, valid_mask):
    x = dissim_all[valid_mask]
    y = dist_all[valid_mask]
    stats = {
        'n_valid': int(x.size),
        'pearson_r': float('nan'),
        'pearson_p': float('nan'),
        'slope': float('nan'),
        'intercept': float('nan'),
        'R2': float('nan'),
        'x': x,
        'y': y,
    }
    if x.size >= 2:
        try:
            from scipy.stats import pearsonr
            pr, pp = pearsonr(x, y)
        except Exception:
            pr, pp = np.corrcoef(x, y)[0, 1], float('nan')
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        stats.update({
            'pearson_r': float(pr),
            'pearson_p': float(pp),
            'slope': float(slope),
            'intercept': float(intercept),
            'R2': float(r2),
        })
    return stats


def write_stats(path, label, stats):
    x, y = stats['x'], stats['y']
    lines = [
        f"[{label}]",
        f"n_valid={stats['n_valid']}",
        f"pearson_r={stats['pearson_r']}",
        f"pearson_p={stats['pearson_p']}",
        f"slope={stats['slope']}",
        f"intercept={stats['intercept']}",
        f"R2={stats['R2']}",
    ]
    if x.size:
        lines.extend([
            f"x_mean={float(np.mean(x))}",
            f"x_std={float(np.std(x))}",
            f"x_min={float(np.min(x))}",
            f"x_max={float(np.max(x))}",
            f"y_mean={float(np.mean(y))}",
            f"y_std={float(np.std(y))}",
            f"y_min={float(np.min(y))}",
            f"y_max={float(np.max(y))}",
        ])
    lines.append('')
    with open(path, 'a') as f:
        f.write('\n'.join(lines))


def plot_heatmap(stats, plot_path, title_prefix, args):
    x = stats['x']
    y = stats['y']
    if x.size < 2:
        return

    xlo, xhi = 0.0, 1.0
    qlo, qhi = float(args.qlo), float(args.qhi)
    ylo, yhi = np.quantile(y, qlo), np.quantile(y, qhi)
    if not np.isfinite(ylo) or not np.isfinite(yhi) or ylo == yhi:
        ylo, yhi = np.min(y), np.max(y)
    if ylo == yhi:
        ylo, yhi = ylo - 1e-6, yhi + 1e-6

    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=(args.bins, args.bins),
        range=[[xlo, xhi], [ylo, yhi]]
    )
    H = H.astype(float)
    top = np.percentile(H, 99.5) if H.size else 1.0
    if top > 0:
        H = np.clip(H / top, 0, 1)
    if args.smooth_sigma and args.smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            H = gaussian_filter(H, sigma=float(args.smooth_sigma))
        except Exception:
            pass

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.rcParams.update({'axes.linewidth': 1.2, 'font.size': 12})
    fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=320)
    cmap = 'turbo' if 'turbo' in plt.colormaps() else ('jet' if 'jet' in plt.colormaps() else 'viridis')
    im = ax.imshow(
        H.T, origin='lower', aspect='auto', extent=extent,
        cmap=cmap, vmin=0.0, vmax=1.0, interpolation='bilinear'
    )

    if np.isfinite(stats['slope']):
        xx = np.linspace(xlo, xhi, 300)
        yy = stats['slope'] * xx + stats['intercept']
        ax.plot(xx, yy, '--', color='#b9b9b9', lw=2.8, dashes=(6, 4))

    ax.set_xlim(0.0, 1.0)
    if args.ylim:
        ax.set_ylim(args.ylim)
    else:
        ax.set_ylim(ylo, yhi)

    ax.set_xlabel('Molecular dissimilarity')
    ax.set_ylabel('Latent space distance')
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color('k')
        sp.set_linewidth(1.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('Normalized population')

    title_r = f"{stats['pearson_r']:.3f}" if np.isfinite(stats['pearson_r']) else 'N/A'
    ax.set_title(f'{title_prefix} vs. Latent distance (Pearson r={title_r})', pad=10)
    if args.scatter:
        ax.scatter(x, y, s=4, alpha=0.18, edgecolors='none', color='white')

    fig.tight_layout()
    fig.savefig(plot_path, dpi=320)
    plt.close(fig)
    print(f"[info] plot saved to {plot_path}")


def main():
    ap = argparse.ArgumentParser(
        description='Compute 1−Tanimoto dissimilarity for paired data and SMILES vs latent distance.'
    )
    ap.add_argument('--result_dir', default='./decoding/results/sample')
    ap.add_argument('--smiles_near', default='smiles_near.txt')
    ap.add_argument('--smiles_seed', default='smiles_seed.txt')
    ap.add_argument('--data_near',   default='data_near.txt')
    ap.add_argument('--data_seed',   default='data_seed.txt')
    ap.add_argument('--dist',        default='dist.txt')
    ap.add_argument('--en_dissim_out', default='en_dissimilarity.txt')
    ap.add_argument('--de_dissim_out', default='de_dissimilarity.txt')
    ap.add_argument('--en_plot_out',   default='en_dissim_vs_dist.png')
    ap.add_argument('--de_plot_out',   default='de_dissim_vs_dist.png')
    ap.add_argument('--stats_out',     default='stats.txt')
    ap.add_argument('--radius', type=int, default=2)
    ap.add_argument('--nbits',  type=int, default=2048)
    ap.add_argument('--bins',   type=int, default=140, help='2D histogram bins per axis')
    ap.add_argument('--qlo',    type=float, default=0.01, help='lower quantile for axis trimming')
    ap.add_argument('--qhi',    type=float, default=0.99, help='upper quantile for axis trimming')
    ap.add_argument('--smooth_sigma', type=float, default=1.2, help='gaussian smoothing sigma; 0=off')
    ap.add_argument('--scatter', action='store_true', help='overlay faint scatter points')
    ap.add_argument('--xlim', type=float, nargs=2, default=None)
    ap.add_argument('--ylim', type=float, nargs=2, default=None)
    args = ap.parse_args()

    result_dir = args.result_dir
    smiles_near_path = os.path.join(result_dir, args.smiles_near)
    smiles_seed_path = os.path.join(result_dir, args.smiles_seed)
    data_near_path   = os.path.join(result_dir, args.data_near)
    data_seed_path   = os.path.join(result_dir, args.data_seed)
    dist_path        = os.path.join(result_dir, args.dist)

    en_dissim_out = os.path.join(result_dir, args.en_dissim_out)
    de_dissim_out = os.path.join(result_dir, args.de_dissim_out)
    en_plot_out   = os.path.join(result_dir, args.en_plot_out)
    de_plot_out   = os.path.join(result_dir, args.de_plot_out)
    stats_out     = os.path.join(result_dir, args.stats_out)

    smiles_near = read_lines(smiles_near_path)
    smiles_seed = read_lines(smiles_seed_path)
    data_near   = read_lines(data_near_path)
    data_seed   = read_lines(data_seed_path)
    dz          = read_lines(dist_path)

    n = min(len(smiles_near), len(smiles_seed), len(data_near), len(data_seed), len(dz))
    shapes = (len(smiles_near), len(smiles_seed), len(data_near), len(data_seed), len(dz))
    if shapes != (n, n, n, n, n):
        print(f'[warn] line counts differ {shapes}; truncate to {n}')
        smiles_near = smiles_near[:n]
        smiles_seed = smiles_seed[:n]
        data_near   = data_near[:n]
        data_seed   = data_seed[:n]
        dz          = dz[:n]

    de_dissim_all, de_valid = compute_dissim(smiles_near, smiles_seed, de_dissim_out,
                                             radius=args.radius, nbits=args.nbits)
    print(f"[info] wrote SMILES dissimilarities to {de_dissim_out}")

    en_dissim_all, en_valid = compute_dissim(data_near, data_seed, en_dissim_out,
                                             radius=args.radius, nbits=args.nbits)
    print(f"[info] wrote data dissimilarities to {en_dissim_out}")

    dist_all = safe_float_list(dz)

    de_mask = filter_valid(de_dissim_all, dist_all, de_valid)
    en_mask = filter_valid(en_dissim_all, dist_all, en_valid)

    de_stats = global_stats(de_dissim_all, dist_all, de_mask)
    en_stats = global_stats(en_dissim_all, dist_all, en_mask)

    Path(stats_out).write_text('')
    write_stats(stats_out, 'de', de_stats)
    write_stats(stats_out, 'en', en_stats)
    print(f"[info] wrote stats to {stats_out}")

    plot_heatmap(en_stats, en_plot_out, 'EN dissimilarity', args)
    plot_heatmap(de_stats, de_plot_out, 'DE dissimilarity', args)


if __name__ == '__main__':
    main()

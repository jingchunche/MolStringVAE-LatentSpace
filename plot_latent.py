#!/usr/bin/env python3
"""Create plots from an existing eval_latent numeric result directory."""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

def load_numeric(path):
    data = np.loadtxt(str(path), dtype=float)
    if data.ndim == 0:
        data = data.reshape(1)
    return data


def reshape_qk(values, k, name):
    if k <= 0:
        raise ValueError("neighbors must be positive")
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        if values.shape[1] != k:
            raise ValueError(
                "{} second dimension must be {}, got {}".format(
                    name, k, values.shape
                )
            )
        return values
    flat = values.reshape(-1)
    if flat.size % k:
        raise ValueError(
            "{} length {} is not divisible by {}".format(name, flat.size, k)
        )
    return flat.reshape(-1, k)


def _histogram(xy, bins, x_min, x_max, y_min, y_max):
    hist, xedges, yedges = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=bins,
        range=[[x_min, x_max], [y_min, y_max]],
    )
    x = 0.5 * (xedges[:-1] + xedges[1:])
    y = 0.5 * (yedges[:-1] + yedges[1:])
    xgrid, ygrid = np.meshgrid(x, y, indexing="xy")
    return hist, xgrid, ygrid


def _levels(z):
    if z.max() <= 0:
        return None
    levels = np.unique(np.linspace(z.max() * 0.15, z.max() * 0.9, 6))
    return levels if len(levels) else None


def make_smoothed_contour_plot(latent, chemical, output, bins, sigma):
    latent_xy = latent.reshape(-1, 2)
    chemical_xy = chemical.reshape(-1, 2)
    all_xy = np.vstack([latent_xy, chemical_xy])
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    x_pad = 0.03 * (x_max - x_min + 1e-12)
    y_pad = 0.03 * (y_max - y_min + 1e-12)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    latent_hist, xgrid, ygrid = _histogram(
        latent_xy, bins, x_min, x_max, y_min, y_max
    )
    chemical_hist, _, _ = _histogram(
        chemical_xy, bins, x_min, x_max, y_min, y_max
    )
    latent_z = gaussian_filter(latent_hist, sigma=sigma).T
    chemical_z = gaussian_filter(chemical_hist, sigma=sigma).T
    latent_levels = _levels(latent_z)
    chemical_levels = _levels(chemical_z)

    plt.figure(figsize=(6, 5), dpi=200)
    if latent_z.max() > 0:
        plt.contourf(xgrid, ygrid, latent_z, levels=12, alpha=0.10, cmap="Greens")
    if chemical_z.max() > 0:
        plt.contourf(xgrid, ygrid, chemical_z, levels=12, alpha=0.10, cmap="Oranges")
    if latent_levels is not None:
        plt.contour(xgrid, ygrid, latent_z, levels=latent_levels, colors="green", linewidths=1.4)
    if chemical_levels is not None:
        plt.contour(xgrid, ygrid, chemical_z, levels=chemical_levels, colors="orange", linewidths=1.4)

    latent_mean_xy = latent_xy.mean(axis=0)
    chemical_mean_xy = chemical_xy.mean(axis=0)
    latent_mean = plt.scatter(
        latent_mean_xy[0], latent_mean_xy[1], c="green", s=70, marker="o",
        edgecolors="black", linewidths=0.6, label="Latent NNs mean",
    )
    chemical_mean = plt.scatter(
        chemical_mean_xy[0], chemical_mean_xy[1], c="orange", s=70, marker="o",
        edgecolors="black", linewidths=0.6, label="Chemical NNs mean",
    )
    latent_line = Line2D([0], [0], color="green", linewidth=1.4, label="Latent NNs")
    chemical_line = Line2D([0], [0], color="orange", linewidth=1.4, label="Chemical NNs")
    plt.xlabel("Molecular dissimilarity")
    plt.ylabel("Latent space distance")
    plt.legend(
        handles=[latent_line, chemical_line, latent_mean, chemical_mean],
        frameon=True,
    )
    plt.tight_layout(pad=0.4)
    plt.savefig(str(output), bbox_inches="tight")
    plt.close()


def make_latent_only_contour_plot(latent, output, bins, sigma):
    latent_xy = latent.reshape(-1, 2)
    x_min, y_min = latent_xy.min(axis=0)
    x_max, y_max = latent_xy.max(axis=0)
    x_pad = 0.03 * (x_max - x_min + 1e-12)
    y_pad = 0.03 * (y_max - y_min + 1e-12)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    latent_hist, xgrid, ygrid = _histogram(
        latent_xy, bins, x_min, x_max, y_min, y_max
    )
    latent_z = gaussian_filter(latent_hist, sigma=sigma).T
    latent_levels = _levels(latent_z)

    plt.figure(figsize=(6, 5), dpi=200)
    if latent_z.max() > 0:
        plt.contourf(xgrid, ygrid, latent_z, levels=12, alpha=0.14, cmap="Greens")
    if latent_levels is not None:
        plt.contour(xgrid, ygrid, latent_z, levels=latent_levels, colors="green", linewidths=1.5)
    latent_mean_xy = latent_xy.mean(axis=0)
    latent_mean = plt.scatter(
        latent_mean_xy[0], latent_mean_xy[1], c="green", s=75, marker="o",
        edgecolors="black", linewidths=0.6, label="Latent NNs mean",
    )
    latent_line = Line2D([0], [0], color="green", linewidth=1.5, label="Latent NNs")
    plt.xlabel("Molecular dissimilarity")
    plt.ylabel("Latent space distance")
    plt.legend(handles=[latent_line, latent_mean], frameon=True)
    plt.tight_layout(pad=0.4)
    plt.savefig(str(output), bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--bins", type=int, default=180)
    parser.add_argument("--sigma", type=float, default=2.0)
    args = parser.parse_args()
    result = Path(args.result_dir).expanduser().resolve()
    reference = Path(args.reference_dir).expanduser().resolve()
    k = args.neighbors
    latent_dissim = reshape_qk(load_numeric(result / "latent_near_dissim.txt"), k, "latent_dissim")
    latent_dist = reshape_qk(load_numeric(result / "latent_near_dist.txt"), k, "latent_dist")
    chemical_dissim = reshape_qk(load_numeric(reference / "chemical_dissim.txt"), k, "chemical_dissim")
    chemical_dist = reshape_qk(load_numeric(result / "chemical_near_dist.txt"), k, "chemical_dist")
    latent = np.stack([latent_dissim, latent_dist], axis=-1)
    chemical = np.stack([chemical_dissim, chemical_dist], axis=-1)

    overlay_output = result / "latent_chemical_contour.png"
    latent_output = result / "latent_contour.png"
    make_smoothed_contour_plot(latent, chemical, overlay_output, args.bins, args.sigma)
    make_latent_only_contour_plot(latent, latent_output, args.bins, args.sigma)
    print("[plot] wrote {}".format(overlay_output))
    print("[plot] wrote {}".format(latent_output))


if __name__ == "__main__":
    main()

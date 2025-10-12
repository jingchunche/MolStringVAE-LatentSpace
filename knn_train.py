#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import yaml
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.utils.path import make_result_dir
from src.utils.logger import default_logger
from src.utils.args import load_config
from src.models import Model
from src.process import get_process
from src.datasets.tokenizer import VocabularyTokenizer

import faiss  # noqa: F401


def resolve_seed(config) -> int:
    """Return a deterministic integer seed, defaulting to 0 when unspecified."""
    seed = config.get("ran_seed")
    if seed is None:
        seed = config.get("seed")
    if seed is None:
        seed = 0
    if isinstance(seed, np.generic):
        seed = seed.item()
    return int(seed)

# ----------------------------------------------------
# FAISS index builder (FlatL2) with safe contiguity
# ----------------------------------------------------

def faiss_index(X_all: torch.Tensor, use_gpu: bool, add_bs: int = 200_000) -> "faiss.Index":
    n, d = X_all.shape
    index_cpu = faiss.IndexFlatL2(d)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # GPU:0
    else:
        index = index_cpu

    for i in range(0, n, add_bs):
        j = min(i + add_bs, n)
        xb_t = X_all[i:j].to('cpu', non_blocking=True).contiguous()
        xb   = np.ascontiguousarray(xb_t.numpy(), dtype=np.float32)
        index.add(xb)
    return index


# ----------------------------------------------------
# Main
# ----------------------------------------------------

def main(config):
    result_dir = make_result_dir(**config.result_dir)
    logger = default_logger(result_dir + "/log.txt", **config.logger)
    with open(f"{result_dir}/config.yaml", 'w') as f:
        yaml.dump(config.to_dict(), f, sort_keys=False)

    # === Seed ===
    seed = resolve_seed(config)
    logger.info(f"Setting random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # === Device ===
    DEVICE = torch.device('cuda', index=config.gpuid or 0) if torch.cuda.is_available() else torch.device('cpu')
    logger.warning(f"DEVICE: {DEVICE}")

    # === Hyper / Data paths ===
    N_seeds      = int(config.num)
    k_neighbors  = int(config.knn)
    latent_path  = config.latent_path
    data_path    = config.data_path
    use_faiss_gpu = torch.cuda.is_available()
    add_bs       = 200_000

    # === Load ALL latents ===
    logger.info(f"Loading latent vectors from: {latent_path}")
    df = pd.read_csv(latent_path, dtype=np.float32)
    X_all = torch.from_numpy(df.to_numpy(np.float32)).contiguous()
    n, d  = X_all.shape
    logger.info(f"Latents shape: n={n}, d={d}")

    logger.info(f"Loading source data from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data_lines = [line.rstrip("\n") for line in f]
    n_data = len(data_lines)
    if n_data < n:
        logger.warning(f"Data lines ({n_data}) < latent vectors ({n}); indices beyond {n_data-1} will be empty.")

    # === Bounds check ===
    if N_seeds > n:
        logger.warning(f"Requested N={N_seeds} > n={n}, reducing N to {n}.")
        N_seeds = n
    if k_neighbors >= n:
        logger.warning(f"Requested k={k_neighbors} >= n={n}, reducing k to n-1={n-1}.")
        k_neighbors = n - 1

    # === Choose seeds ===
    rng = np.random.default_rng(seed)
    seed_idx = np.sort(rng.choice(n, size=N_seeds, replace=False))
    logger.info(f"Picked {N_seeds} seeds.")

    # === kNN with FAISS ===
    logger.info(f"Building FAISS index (gpu={use_faiss_gpu}, add_bs={add_bs}) ...")
    index = faiss_index(X_all, use_gpu=use_faiss_gpu, add_bs=add_bs)

    logger.info("Searching kNN with FAISS ...")
    xb_t = X_all.to('cpu', non_blocking=True).contiguous()
    xb   = np.ascontiguousarray(xb_t.numpy(), dtype=np.float32)
    xq   = np.ascontiguousarray(xb[seed_idx], dtype=np.float32)
    D2, I = index.search(xq, k_neighbors + 1) 

    I_f, D2_f = [], []
    for row, s in enumerate(seed_idx):
        ids = I[row]
        ds2 = D2[row]
        mask = (ids != s)
        I_f.append(ids[mask][:k_neighbors])
        D2_f.append(ds2[mask][:k_neighbors])
    I  = np.stack(I_f, axis=0)
    D2 = np.stack(D2_f, axis=0)

    D = np.sqrt(np.maximum(D2, 0.0)).astype(np.float32)

    # === Flatten pairs ===
    idx_seed_flat = np.repeat(seed_idx, k_neighbors)
    idx_near_flat = I.reshape(-1)
    dist_flat     = D.reshape(-1).astype(np.float32)
    total_pairs   = len(idx_seed_flat)
    logger.info(f"Formed {total_pairs} (seed, near) pairs (N={N_seeds}, k={k_neighbors}).")

    # === Model ===
    logger.info("Preparing model...")
    model = Model(logger, **config.model)
    model.load(path=config.weight_path, strict=False)
    model.to(DEVICE)
    model.eval()
    processes = [get_process(**p) for p in config.processes]

    # === Tokenizer ===
    with open(config.voc_file) as f:
        tokenizer = VocabularyTokenizer(f.read().splitlines())

    # === Decode seeds & neighbors ===
    Z_seed = X_all[idx_seed_flat]
    Z_near = X_all[idx_near_flat]

    bs = int(config.data.get("batch_size", 32))
    dl_seed = DataLoader(TensorDataset(Z_seed, torch.arange(len(Z_seed))), batch_size=bs, shuffle=False)
    dl_near = DataLoader(TensorDataset(Z_near, torch.arange(len(Z_near))), batch_size=bs, shuffle=False)

    decoded_seed, decoded_near = [], []

    logger.info("Decoding seeds...")
    with torch.no_grad():
        iterator = tqdm(dl_seed) if config.show_tqdm else dl_seed
        for batch_X, _ in iterator:
            batch = {"latent": batch_X.to(DEVICE, non_blocking=True),
                     "batch_size": int(batch_X.size(0))}
            model(batch, processes=processes)
            decoded_seed.extend((tokenizer.detokenize(tok) if tok is not None else "") for tok in batch.get("greedy", []))
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    logger.info("Decoding neighbors...")
    with torch.no_grad():
        iterator = tqdm(dl_near) if config.show_tqdm else dl_near
        for batch_X, _ in iterator:
            batch = {"latent": batch_X.to(DEVICE, non_blocking=True),
                     "batch_size": int(batch_X.size(0))}
            model(batch, processes=processes)
            decoded_near.extend((tokenizer.detokenize(tok) if tok is not None else "") for tok in batch.get("greedy", []))
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    m = min(total_pairs, len(decoded_seed), len(decoded_near))
    if m != total_pairs:
        logger.warning(
            f"Decoded counts mismatch: pairs={total_pairs}, seed={len(decoded_seed)}, near={len(decoded_near)}. Truncating to m={m}.")
        total_pairs = m
        decoded_seed = decoded_seed[:m]
        decoded_near = decoded_near[:m]
        dist_flat    = dist_flat[:m]
        idx_seed_flat = idx_seed_flat[:m]
        idx_near_flat = idx_near_flat[:m]

    # === Prepare data outputs ===
    data_seed_flat = [data_lines[i] if 0 <= i < n_data else "" for i in idx_seed_flat.tolist()]
    data_near_flat = [data_lines[i] if 0 <= i < n_data else "" for i in idx_near_flat.tolist()]

    # === Write outputs ===
    path_seed = os.path.join(result_dir, "selfies_seed.txt")
    path_near = os.path.join(result_dir, "selfies_near.txt")
    path_dist = os.path.join(result_dir, "dist.txt")
    path_data_seed = os.path.join(result_dir, "data_seed.txt")
    path_data_near = os.path.join(result_dir, "data_near.txt")

    with open(path_seed, "w") as fs, \
         open(path_near, "w") as fn, \
         open(path_dist, "w") as fd, \
         open(path_data_seed, "w") as fds, \
         open(path_data_near, "w") as fdn:
        for i in range(total_pairs):
            fs.write((decoded_seed[i] or "") + "\n")
            fn.write((decoded_near[i] or "") + "\n")
            fd.write(f"{float(dist_flat[i]):.6f}\n")
            fds.write((data_seed_flat[i] or "") + "\n")
            fdn.write((data_near_flat[i] or "") + "\n")

    logger.info(
        "Paired outputs saved:\n"
        f" - {path_seed}\n"
        f" - {path_near}\n"
        f" - {path_dist}\n"
        f" - {path_data_seed}\n"
        f" - {path_data_near}\n"
        f"(Line i in selfies_seed.txt == line i in selfies_near.txt == line i in dist.txt == line i in data_seed.txt == line i in data_near.txt)"
    )


if __name__ == '__main__':
    config = load_config(config_dir="./decoding", default_configs=['base_train'])
    main(config)

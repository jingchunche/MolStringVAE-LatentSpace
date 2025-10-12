#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import yaml
import random
from collections.abc import Mapping
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
    N_seeds       = int(config.num)
    k_neighbors   = int(config.knn)

    # 新增：允許分開指定 train / val latent；未提供時保持舊行為
    train_latent_path = config.train_latent
    val_latent_path   = config.val_latent

    # 新增：對應 train / val 原始資料位置
    data_path_cfg = config.get("data_path")
    if data_path_cfg is None:
        raise ValueError("config.data_path is required (expect str or mapping with train/val paths).")

    if isinstance(data_path_cfg, Mapping):
        shared_path = (
            data_path_cfg.get("all")
            or data_path_cfg.get("path")
            or data_path_cfg.get("both")
            or data_path_cfg.get("shared")
        )
        train_data_path = (
            data_path_cfg.get("train")
            or data_path_cfg.get("train_path")
            or data_path_cfg.get("train_data")
            or shared_path
        )
        val_data_path = (
            data_path_cfg.get("val")
            or data_path_cfg.get("val_path")
            or data_path_cfg.get("val_data")
            or shared_path
        )
    else:
        train_data_path = val_data_path = data_path_cfg

    if train_data_path is None and val_data_path is None:
        raise ValueError("data_path must provide at least one valid path.")
    if train_data_path is None:
        logger.warning("train data_path missing; fallback to val path.")
        train_data_path = val_data_path
    if val_data_path is None:
        logger.warning("val data_path missing; fallback to train path.")
        val_data_path = train_data_path

    use_faiss_gpu  = torch.cuda.is_available()
    add_bs         = 200_000

    # === Load latents ===
    logger.info(f"Loading TRAIN latent vectors from: {train_latent_path}")
    df_tr = pd.read_csv(train_latent_path, dtype=np.float32)
    X_train = torch.from_numpy(df_tr.to_numpy(np.float32)).contiguous()
    n_tr, d = X_train.shape
    logger.info(f"Train latents shape: n_tr={n_tr}, d={d}")

    logger.info(f"Loading VAL latent vectors from:   {val_latent_path}")
    df_val = pd.read_csv(val_latent_path, dtype=np.float32)
    X_val = torch.from_numpy(df_val.to_numpy(np.float32)).contiguous()
    n_val, d_val = X_val.shape
    if d_val != d:
        raise ValueError(f"Dim mismatch: train d={d} vs val d={d_val}")
    logger.info(f"Val latents shape:   n_val={n_val}, d={d_val}")

    def load_source_lines(path: str, tag: str):
        logger.info(f"Loading {tag} source data from: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    data_lines_train = load_source_lines(train_data_path, "TRAIN") if train_data_path else []
    data_lines_val   = load_source_lines(val_data_path, "VAL") if val_data_path else []

    n_data_train = len(data_lines_train)
    n_data_val   = len(data_lines_val)
    if train_data_path and n_data_train < n_tr:
        logger.warning(
            f"TRAIN data lines ({n_data_train}) < train latent vectors ({n_tr}); indices beyond range will be empty."
        )
    if val_data_path and n_data_val < n_val:
        logger.warning(
            f"VAL data lines ({n_data_val}) < val latent vectors ({n_val}); indices beyond range will be empty."
        )

    # === Bounds check ===
    if N_seeds > n_val:
        logger.warning(f"Requested N={N_seeds} > n_val={n_val}, reducing N to {n_val}.")
        N_seeds = n_val
    if k_neighbors >= n_tr:
        logger.warning(f"Requested k={k_neighbors} >= n_tr={n_tr}, reducing k to n_tr-1={n_tr-1}.")
        k_neighbors = n_tr - 1

    # === Choose seeds from VAL ===
    rng = np.random.default_rng(seed)
    seed_idx = np.sort(rng.choice(n_val, size=N_seeds, replace=False))
    logger.info(f"Picked {N_seeds} VAL seeds.")

    # === kNN: index on TRAIN, query with VAL ===
    logger.info(f"Building FAISS index on TRAIN (gpu={use_faiss_gpu}, add_bs={add_bs}) ...")
    index = faiss_index(X_train, use_gpu=use_faiss_gpu, add_bs=add_bs)

    logger.info("Searching kNN (VAL → TRAIN) with FAISS ...")
    xb_t = X_train.to('cpu', non_blocking=True).contiguous()
    xb   = np.ascontiguousarray(xb_t.numpy(), dtype=np.float32)   # database (TRAIN)
    xq_t = X_val[seed_idx].to('cpu', non_blocking=True).contiguous()
    xq   = np.ascontiguousarray(xq_t.numpy(), dtype=np.float32)   # queries (VAL seeds)

    D2, I = index.search(xq, k_neighbors + 1)  # 不保證能排除同一向量，保留後面手動裁掉

    # 由於查詢與資料庫不同集合，理論上不會撞同一 id；但仍保留「去自身」邏輯以安全
    I_f, D2_f = [], []
    for row in range(len(seed_idx)):
        ids = I[row]
        ds2 = D2[row]
        # 直接取前 k 個鄰居（資料庫是 train，不會有 val 的 index）
        I_f.append(ids[:k_neighbors])
        D2_f.append(ds2[:k_neighbors])
    I  = np.stack(I_f, axis=0)
    D2 = np.stack(D2_f, axis=0)
    D = np.sqrt(np.maximum(D2, 0.0)).astype(np.float32)

    # === Flatten pairs (VAL seed, TRAIN near) ===
    idx_seed_flat = np.repeat(seed_idx, k_neighbors)   # 索引在 VAL
    idx_near_flat = I.reshape(-1)                      # 索引在 TRAIN
    dist_flat     = D.reshape(-1).astype(np.float32)
    total_pairs   = len(idx_seed_flat)
    logger.info(f"Formed {total_pairs} (VAL seed, TRAIN near) pairs (N={N_seeds}, k={k_neighbors}).")

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

    # === Decode seeds (VAL) & neighbors (TRAIN) ===
    Z_seed = X_val[idx_seed_flat]
    Z_near = X_train[idx_near_flat]

    bs = int(config.data.get("batch_size", 32))
    dl_seed = DataLoader(TensorDataset(Z_seed, torch.arange(len(Z_seed))), batch_size=bs, shuffle=False)
    dl_near = DataLoader(TensorDataset(Z_near, torch.arange(len(Z_near))), batch_size=bs, shuffle=False)

    decoded_seed, decoded_near = [], []

    logger.info("Decoding VAL seeds...")
    with torch.no_grad():
        iterator = tqdm(dl_seed) if config.show_tqdm else dl_seed
        for batch_X, _ in iterator:
            batch = {"latent": batch_X.to(DEVICE, non_blocking=True),
                     "batch_size": int(batch_X.size(0))}
            model(batch, processes=processes)
            decoded_seed.extend((tokenizer.detokenize(tok) if tok is not None else "") for tok in batch.get("greedy", []))
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    logger.info("Decoding TRAIN neighbors...")
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

    data_seed_flat = [
        data_lines_val[i] if 0 <= i < n_data_val else ""
        for i in idx_seed_flat.tolist()
    ]
    data_near_flat = [
        data_lines_train[i] if 0 <= i < n_data_train else ""
        for i in idx_near_flat.tolist()
    ]

    # === Write outputs ===
    path_seed = os.path.join(result_dir, "selfies_seed.txt")  # VAL
    path_near = os.path.join(result_dir, "selfies_near.txt")  # TRAIN
    path_dist = os.path.join(result_dir, "dist.txt")
    path_data_seed = os.path.join(result_dir, "data_seed.txt")  # VAL data 原文
    path_data_near = os.path.join(result_dir, "data_near.txt")  # TRAIN data 原文

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
        f" - {path_seed}   (VAL)\n"
        f" - {path_near}   (TRAIN)\n"
        f" - {path_dist}\n"
        f" - {path_data_seed}   (VAL source)\n"
        f" - {path_data_near}   (TRAIN source)\n"
        f"(Line i in selfies_seed.txt == line i in selfies_near.txt == line i in dist.txt == line i in data_seed.txt == line i in data_near.txt)"
    )


if __name__ == '__main__':
    config = load_config(config_dir="./decoding", default_configs=['base_val'])
    main(config)

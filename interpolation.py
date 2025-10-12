#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
os.environ.setdefault('TOOLS_DIR', "/workspace")
sys.path += [os.environ["TOOLS_DIR"]]
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

def main(config):
    result_dir = make_result_dir(**config.result_dir)
    logger = default_logger(result_dir+"/log.txt", **config.logger)
    with open(f"{result_dir}/config.yaml", 'w') as f:
        yaml.dump(config.to_dict(), f, sort_keys=False)

    # === Seed ===
    seed = config.get("ran_seed")
    logger.info(f"Setting random seed: {seed}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # === Device ===
    DEVICE = torch.device('cuda', index=config.gpuid or 0) if torch.cuda.is_available() else torch.device('cpu')
    logger.warning(f"DEVICE: {DEVICE}")

    # === Data ===
    num         = int(config.num)       # 總共抽多少列（需為偶數，若為奇數會丟掉最後一個）
    mid_num     = int(config.mid_num)   # 每對端點要產生的中點數（不含端點）
    latent_path = config.latent_path

    # 僅抽 num 列，不載整檔
    with open(latent_path) as f:
        n = sum(1 for _ in f) - 1
    rng  = np.random.default_rng(seed)
    idx  = np.sort(rng.choice(n, size=min(num, n), replace=False))
    keep = set(idx)

    df = pd.read_csv(
        latent_path,
        dtype=np.float32,
        skiprows=lambda i: i > 0 and (i - 1) not in keep
    )
    X = torch.from_numpy(df.to_numpy(np.float32)).contiguous()  # (num_eff, d)

    # 以相鄰兩個為一對端點
    pairs = (len(df) // 2)
    if len(df) % 2 == 1:
        logger.warning("num is odd; the last sample is dropped from pairing.")
        X = X[:2 * pairs]

    # 取出每對端點 (pair-major)
    E1 = X[0:2*pairs:2]            # (pairs, d)
    E2 = X[1:2*pairs:2]            # (pairs, d)

    # 建立所有中點（pair-major；每對連續 mid_num 行）
    t = torch.linspace(0.0, 1.0, mid_num + 2, dtype=torch.float32)[1:-1]  # (mid_num,)
    Z_mid = torch.lerp(E1.unsqueeze(0), E2.unsqueeze(0), t.view(-1, 1, 1))  # (mid_num, pairs, d)
    Z_mid = Z_mid.permute(1, 0, 2).reshape(pairs * mid_num, -1).contiguous()  # (pairs*mid_num, d)

    # 端點按 pair-major 展平成 (e1_0,e2_0,e1_1,e2_1,...) 方便一次解碼
    Z_end = torch.stack([E1, E2], dim=1).reshape(2 * pairs, -1).contiguous()  # (2*pairs, d)

    # === Model ===
    logger.info("Preparing model...")
    model = Model(logger, **config.model)
    model.load(path=config.weight_path, strict=False)
    model.to(DEVICE); model.eval()
    processes = [get_process(**p) for p in config.processes]

    # === Tokenizer ===
    with open(config.voc_file) as f:
        tokenizer = VocabularyTokenizer(f.read().splitlines())

    # === Decode endpoints ===
    bs = int(config.data.get("batch_size", 32))
    dl_end = DataLoader(TensorDataset(Z_end, torch.arange(Z_end.size(0))), batch_size=bs, shuffle=False)
    decoded_end = []
    with torch.no_grad():
        iterator = tqdm(dl_end) if config.show_tqdm else dl_end
        for batch_X, _ in iterator:
            batch = {"latent": batch_X.to(DEVICE, non_blocking=True),
                     "batch_size": int(batch_X.size(0))}
            model(batch, processes=processes)
            decoded_end.extend(tokenizer.detokenize(tok) for tok in batch["greedy"])
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    # === Decode midpoints ===
    dl_mid = DataLoader(TensorDataset(Z_mid, torch.arange(Z_mid.size(0))), batch_size=bs, shuffle=False)
    decoded_mid = []
    with torch.no_grad():
        iterator = tqdm(dl_mid) if config.show_tqdm else dl_mid
        for batch_X, _ in iterator:
            batch = {"latent": batch_X.to(DEVICE, non_blocking=True),
                     "batch_size": int(batch_X.size(0))}
            model(batch, processes=processes)
            decoded_mid.extend(tokenizer.detokenize(tok) for tok in batch["greedy"])
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    # === Write outputs ===
    path1 = os.path.join(result_dir, "selfies_end1.txt")
    path2 = os.path.join(result_dir, "selfies_end2.txt")
    pathm = os.path.join(result_dir, "selfies_mid.txt")

    with open(path1, "w") as f1, open(path2, "w") as f2:
        for k in range(pairs):
            f1.write(decoded_end[2*k] + "\n")
            f2.write(decoded_end[2*k+1] + "\n")

    with open(pathm, "w") as fm:
        # pair-major: pair0 有 mid_num 行，pair1 接著 mid_num 行，以此類推
        for s in decoded_mid:
            fm.write(s + "\n")

    logger.info(f"Outputs saved:\n - {path1}\n - {path2}\n - {pathm}")

if __name__ == '__main__':
    config = load_config(config_dir="./decoding", default_configs=['base_int'])
    main(config)

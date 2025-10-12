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

# ---- SLERP ----
def slerp_midpoints(E1: torch.Tensor, E2: torch.Tensor, t: torch.Tensor,
                    radius_mode: str = "linear"):
    """
    E1,E2: (pairs, d)
    t: (m,) in (0,1) 的中點比例（不含端點）
    回傳：
      Z_mid: (pairs*m, d)  pair-major
      dist_to1, dist_to2: list 長度 = pairs*m，與 Z_mid/中點順序完全一致
    """
    eps = 1e-8
    pairs, d = E1.size(0), E1.size(1)

    # 範數與單位方向
    r1 = E1.norm(dim=1, keepdim=True).clamp_min(eps)   # (pairs,1)
    r2 = E2.norm(dim=1, keepdim=True).clamp_min(eps)   # (pairs,1)
    u1 = E1 / r1                                       # (pairs,d)
    u2 = E2 / r2                                       # (pairs,d)

    # 角度（避免邊界）
    dot   = (u1 * u2).sum(dim=1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)  # (pairs,1)
    theta = torch.acos(dot)                                                 # (pairs,1)
    sin_th = torch.sin(theta).clamp_min(1e-6)                               # (pairs,1)

    # 對齊形狀與裝置
    t_     = t.view(1, -1, 1).to(E1)       # (1,m,1)
    theta  = theta.unsqueeze(1)            # (pairs,1,1)
    sin_th = sin_th.unsqueeze(1)           # (pairs,1,1)
    u1e    = u1.unsqueeze(1)               # (pairs,1,d)
    u2e    = u2.unsqueeze(1)               # (pairs,1,d)
    r1e    = r1.unsqueeze(1)               # (pairs,1,1)
    r2e    = r2.unsqueeze(1)               # (pairs,1,1)

    # 球面方向插值
    A = torch.sin((1 - t_) * theta) / sin_th   # (pairs,m,1)
    B = torch.sin(t_ * theta) / sin_th         # (pairs,m,1)
    U = A * u1e + B * u2e                      # (pairs,m,d)

    # 半徑路徑
    if radius_mode == "linear":
        R = (1 - t_) * r1e + t_ * r2e          # (pairs,m,1)
    elif radius_mode == "const":
        R = 0.5 * (r1e + r2e).expand(-1, t.size(0), -1)  # (pairs,m,1)
    else:
        raise ValueError("radius_mode must be 'linear' or 'const'.")

    Zm = R * U                                  # (pairs,m,d)

    # 真實歐氏距離到端點（pair-major 展平）
    dist_to1 = (Zm - E1.unsqueeze(1)).norm(dim=2).reshape(-1).tolist()  # (pairs*m,)
    dist_to2 = (Zm - E2.unsqueeze(1)).norm(dim=2).reshape(-1).tolist()  # (pairs*m,)

    Z_mid = Zm.reshape(pairs * t.size(0), d).contiguous()
    return Z_mid, dist_to1, dist_to2


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

    # ---- 用 SLERP 產生中點（取代原本的 LERP）----
    t = torch.linspace(0.0, 1.0, mid_num + 2, dtype=torch.float32)[1:-1]  # (mid_num,)
    Z_mid, dist_to1, dist_to2 = slerp_midpoints(E1, E2, t, radius_mode="linear")  # (pairs*mid_num, d), lists

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
            for tok in batch["greedy"]:
                decoded_end.append(tokenizer.detokenize(tok))
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
            for tok in batch["greedy"]:
                decoded_mid.append(tokenizer.detokenize(tok))
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    # === Write outputs ===
    path1  = os.path.join(result_dir, "selfies_end1.txt")
    path2  = os.path.join(result_dir, "selfies_end2.txt")
    pathm  = os.path.join(result_dir, "selfies_mid.txt")
    pathd1 = os.path.join(result_dir, "dist_to1.txt")  # 與 selfies_mid.txt 行行對應
    pathd2 = os.path.join(result_dir, "dist_to2.txt")  # 與 selfies_mid.txt 行行對應

    # 端點：每對一行
    with open(path1, "w") as f1, open(path2, "w") as f2:
        for k in range(pairs):
            f1.write((decoded_end[2*k]   if 2*k   < len(decoded_end) else "") + "\n")
            f2.write((decoded_end[2*k+1] if 2*k+1 < len(decoded_end) else "") + "\n")

    # 中點與距離：逐行對齊（若解碼較少則截斷距離）
    m_mid = min(len(decoded_mid), len(dist_to1), len(dist_to2))
    with open(pathm, "w") as fm, open(pathd1, "w") as f1d, open(pathd2, "w") as f2d:
        for i in range(m_mid):
            fm.write((decoded_mid[i] or "") + "\n")
            f1d.write(f"{dist_to1[i]:.8f}\n")
            f2d.write(f"{dist_to2[i]:.8f}\n")

    logger.info(
        "Outputs saved:\n"
        f" - {path1}\n - {path2}\n - {pathm}\n - {pathd1}\n - {pathd2}"
    )

if __name__ == '__main__':
    config = load_config(config_dir="./decoding", default_configs=['base_int'])
    main(config)

#!/usr/bin/env python3
"""Featurize to temporary storage, then perform interpolation decoding."""

import os
import random
import tempfile

import numpy as np
import torch

import gc
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.dataset import get_dataloader
from src.datasets.tokenizer import VocabularyTokenizer
from src.models import Model
from src.process import get_process
from src.utils.logger import default_logger
from src.utils.path import make_result_dir
from src.utils.args import load_config


def _device(gpuid):
    if torch.cuda.is_available():
        return torch.device("cuda", index=gpuid or 0)
    return torch.device("cpu")


def _release_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _featurize_to_file(config, logger, device, latent_path):
    """Write feature_mu in input order to a temporary NPY memmap."""
    loader = get_dataloader(logger=logger, device=device, **config.data)
    model = Model(logger, **config.model)
    model.load(path=config.weight_path, strict=False)
    model.to(device)
    model.eval()
    processes = [get_process(**process) for process in config.processes]
    dataset_size = len(next(iter(loader.cur_dsets.values())))
    latent_file = None
    logger.info(f"Featurizing {dataset_size} molecules to temporary storage...")
    with torch.no_grad():
        for batch in tqdm(loader):
            model(batch, processes=processes)
            values = batch["mu"].detach().cpu().numpy().astype(np.float32, copy=False)
            if latent_file is None:
                latent_file = np.lib.format.open_memmap(
                    latent_path, mode="w+", dtype=np.float32,
                    shape=(dataset_size, values.shape[1]),
                )
            latent_file[np.asarray(batch["idx"], dtype=int)] = values
            del batch
    if latent_file is None:
        raise ValueError("Cannot featurize an empty dataset.")
    latent_file.flush()
    shape = latent_file.shape
    del latent_file, values, processes
    model.cpu()
    del model, loader
    _release_cuda()
    logger.info(f"Temporary latent shape: {shape}")


def _prepare_output(feature_config, decode_config):
    result_dir = make_result_dir(**decode_config.result_dir)
    logger = default_logger(result_dir + "/log.txt", **decode_config.logger)
    combined_config = {
        "featurization": feature_config.to_dict(),
        "decoding": decode_config.to_dict(),
        "latent_transport": "temporary_npy_memmap",
    }
    with open(os.path.join(result_dir, "config.yaml"), "w") as handle:
        yaml.dump(combined_config, handle, sort_keys=False)
    return result_dir, logger


def _decoder(config, logger, device):
    model = Model(logger, **config.model)
    model.load(path=config.weight_path, strict=False)
    model.to(device)
    model.eval()
    processes = [get_process(**process) for process in config.processes]
    with open(config.voc_file) as handle:
        tokenizer = VocabularyTokenizer(handle.read().splitlines())
    return model, processes, tokenizer


def _decode_batches(latent, config, model, processes, tokenizer, device):
    tensor = torch.from_numpy(np.asarray(latent, dtype=np.float32)).contiguous()
    loader = DataLoader(
        TensorDataset(tensor), batch_size=int(config.data.batch_size), shuffle=False
    )
    tokens = []
    strings = []
    with torch.no_grad():
        iterator = tqdm(loader) if config.show_tqdm else loader
        for (batch_latent,) in iterator:
            batch = {
                "latent": batch_latent.to(device, non_blocking=True),
                "batch_size": int(batch_latent.size(0)),
            }
            model(batch, processes=processes)
            batch_tokens = batch["greedy"]
            for token in batch_tokens:
                if torch.is_tensor(token):
                    token = token.detach().cpu().numpy()
                tokens.append(token)
                strings.append(tokenizer.detokenize(token))
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return tokens, strings


def parse_args():
    return load_config(config_dir="./interpolating", default_configs=["config"])


def main():
    config = parse_args()
    run_interpolate(config.featurization, config.decoding)


def run_interpolate(feature_config, decode_config):
    result_dir, logger = _prepare_output(feature_config, decode_config)
    device = _device(decode_config.gpuid)
    logger.warning(f"DEVICE: {device}")
    temp_root = os.path.abspath(
        feature_config.get("temp_dir", "./featurization/results")
    )
    os.makedirs(temp_root, exist_ok=True)
    temp_context = tempfile.TemporaryDirectory(
        prefix=".tmp-interpolate-", dir=temp_root
    )
    temp_dir = temp_context.name
    latent = None
    model = None
    processes = None
    try:
        latent_path = os.path.join(temp_dir, "feature_mu.npy")
        logger.info(f"Temporary latent file: {latent_path}")
        _featurize_to_file(feature_config, logger, device, latent_path)
        latent = np.load(latent_path, mmap_mode="r")

        seed = decode_config.get("ran_seed")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        sample_count = min(int(decode_config.num), len(latent))
        selected = np.sort(rng.choice(len(latent), size=sample_count, replace=False))
        endpoints = torch.from_numpy(
            np.asarray(latent[selected], dtype=np.float32)
        ).contiguous()
        pair_count = len(endpoints) // 2
        if len(endpoints) % 2:
            logger.warning("num is odd; the last sample is dropped from pairing.")
            endpoints = endpoints[: 2 * pair_count]
        end1 = endpoints[0::2]
        end2 = endpoints[1::2]
        # Linear interpolation: z(t) = (1 - t) * z1 + t * z2, 0 < t < 1.
        mid_num = int(decode_config.mid_num)
        fractions = torch.linspace(
            0.0, 1.0, mid_num + 2, dtype=torch.float32
        )[1:-1]
        middle = torch.lerp(
            end1.unsqueeze(0), end2.unsqueeze(0), fractions.view(-1, 1, 1)
        )
        middle = middle.permute(1, 0, 2).reshape(
            pair_count * mid_num, -1
        ).contiguous()
        endpoint_rows = torch.stack([end1, end2], dim=1).reshape(
            2 * pair_count, -1
        )

        model, processes, tokenizer = _decoder(decode_config, logger, device)
        _, decoded_end = _decode_batches(
            endpoint_rows.numpy(), decode_config, model, processes, tokenizer, device
        )
        _, decoded_mid = _decode_batches(
            middle.numpy(), decode_config, model, processes, tokenizer, device
        )
        with open(os.path.join(result_dir, "end1.txt"), "w") as first, open(
            os.path.join(result_dir, "end2.txt"), "w"
        ) as second:
            for index in range(pair_count):
                first.write(decoded_end[2 * index] + "\n")
                second.write(decoded_end[2 * index + 1] + "\n")
        with open(os.path.join(result_dir, "mid.txt"), "w") as handle:
            for value in decoded_mid:
                handle.write(value + "\n")
    finally:
        if model is not None:
            model.cpu()
        del model, processes
        if latent is not None:
            mmap = getattr(latent, "_mmap", None)
            if mmap is not None:
                mmap.close()
        del latent
        _release_cuda()
        temp_context.cleanup()
    logger.info(
        f"Interpolated {pair_count} endpoint pairs; temporary latent file removed"
    )


if __name__ == "__main__":
    main()

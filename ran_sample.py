import sys, os
os.environ.setdefault('TOOLS_DIR', "/workspace")
sys.path += [os.environ["TOOLS_DIR"]]
import yaml
import random
import numpy as np
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
    sigma = float(config.sigma)
    num, d_model = int(config.num), int(config.dmodel)
    batch_size = int(config.data.get("batch_size", 32))
    logger.info(f"Generating random dataset: {num} samples of dim {d_model}")
    X = torch.randn(num, d_model) * sigma    
    idxs = torch.arange(num)
    dataset = TensorDataset(X, idxs)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                    pin_memory=(DEVICE.type == 'cuda'))

    # === Model ===
    logger.info("Preparing model...")
    model = Model(logger, **config.model)
    model.load(path=config.weight_path, strict=False)
    model.to(DEVICE); model.eval()
    processes = [get_process(**p) for p in config.processes]

    # === Tokenizer ===
    with open(config.voc_file) as f:
        tokenizer = VocabularyTokenizer(f.read().splitlines())

    # === Decode all ===
    decoded = []  
    with torch.no_grad():
        iterator = tqdm(dl) if config.show_tqdm else dl
        for batch_X, batch_idx in iterator:
            batch = {
                "latent": batch_X.to(DEVICE, non_blocking=True),
                "idx": batch_idx.cpu().numpy(),
                "batch_size": int(batch_X.size(0)),
            }
            model(batch, processes=processes)

            for tok in batch["greedy"]:
                decoded.append(tokenizer.detokenize(tok))

            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    if len(decoded) != num:
        logger.warning(f"Decoded count ({len(decoded)}) != num ({num}); will pair by available min length.")

    pairs = (min(len(decoded), num)) // 2
    if num % 2 == 1:
        logger.warning("num is odd; the last sample is dropped from pairing.")

    path1 = os.path.join(result_dir, "decoded_selfies1.txt")
    path2 = os.path.join(result_dir, "decoded_selfies2.txt")
    pathd = os.path.join(result_dir, "dist.txt")

    with open(path1, "w") as f1, open(path2, "w") as f2, open(pathd, "w") as fd:
        for k in range(pairs):
            i = 2 * k
            j = i + 1
            f1.write((decoded[i] if i < len(decoded) else "") + "\n")
            f2.write((decoded[j] if j < len(decoded) else "") + "\n")
            dist = torch.norm(X[i] - X[j]).item()
            fd.write(f"{dist}\n")

    logger.info(f"Paired outputs saved:\n - {path1}\n - {path2}\n - {pathd}")

if __name__ == '__main__':
    config = load_config(config_dir="./decoding", default_configs=['base_ran'])
    main(config)

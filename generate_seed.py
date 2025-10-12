import yaml
import numpy as np
import torch
from tqdm import tqdm

from src.utils.path import make_result_dir
from src.utils.logger import default_logger
from src.utils.args import load_config
from src.models import Model
from src.process import get_process
from src.accumulator import get_accumulator
from src.datasets.tokenizer import VocabularyTokenizer


def load_seed_matrix(path, skip_rows=1):
    data = np.loadtxt(path, delimiter=",", skiprows=skip_rows, dtype=np.float32)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    return data


def main(config):
    result_dir = make_result_dir(**config.result_dir)
    logger = default_logger(result_dir + "/log.txt", **config.logger)
    with open(f"{result_dir}/config.yaml", "w") as f:
        yaml.dump(config.to_dict(), f)

    device = torch.device("cuda", index=config.gpuid or 0) \
        if torch.cuda.is_available() else torch.device("cpu")
    logger.warning(f"DEVICE: {device}")

    with open(config.voc_file) as f:
        tokenizer = VocabularyTokenizer(f.read().splitlines())

    model_config = config.model
    model_config.update(config.model)
    model = Model(logger, **model_config)
    model.load(**config.load)
    model.to(device)
    model.eval()

    processes = [get_process(**p) for p in config.processes]
    token_accumulator = get_accumulator(logger=logger, **config.token_accumulator)
    token_accumulator.init()

    latent_sources = load_seed_matrix(config.seed_file, skip_rows=config.seed_skiprows)
    num_sources = latent_sources.shape[0]
    logger.info(f"Loaded {num_sources} latent vectors from {config.seed_file}")

    if config.n_generation > num_sources:
        raise ValueError(
            f"Requested {config.n_generation} generations but only {num_sources} vectors available"
        )

    if config.noise_seed is not None:
        torch.manual_seed(config.noise_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.noise_seed)

    logger.info(
        f"Applying Gaussian noise with std={config.noise_std}" if config.noise_std else "Using seeds without noise"
    )

    base_batch_size = config.batch_size
    n_generation = config.n_generation
    n_iter = (n_generation - 1) // base_batch_size + 1

    with torch.no_grad():
        for i_iter in tqdm(range(n_iter)):
            start = i_iter * base_batch_size
            end = min(start + base_batch_size, n_generation)
            current_batch_size = end - start

            latent_np = latent_sources[start:end]
            latent = torch.from_numpy(latent_np).to(device)

            if config.noise_std:
                latent = latent + torch.randn_like(latent) * config.noise_std

            batch = {"batch_size": current_batch_size, "latent": latent}
            batch = model(batch, processes)
            token_accumulator(batch)

    token_accumulator.save(f"{result_dir}/tokens")
    tokens = token_accumulator.accums[:config.n_generation]

    logger.info("Tokenizing...")
    smiles = []
    with open(f"{result_dir}/selfies.txt", "w") as fw:
        for token in tokens:
            smile = tokenizer.detokenize(token)
            fw.write(smile + "\n")
            smiles.append(smile)


if __name__ == "__main__":
    config = load_config(config_dir="./generation", default_configs=["base_seed"])
    main(config)

"""Unified molecular preprocessing pipeline."""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml
from rdkit import RDLogger

from src.datasets.tokenizer import VocabularyTokenizer
from src.preprocessing.encoders import build_encoder
from src.preprocessing.modes import build_mode_outputs, empty_outputs, required_variants
from src.preprocessing.outputs import save_outputs
from src.preprocessing.validation import validate_config
from src.utils.args import load_config
from src.utils.logger import default_logger
from src.utils.path import make_result_dir


def as_plain_dict(value):
    if isinstance(value, dict):
        return {key: as_plain_dict(child) for key, child in value.items()}
    if isinstance(value, list):
        return [as_plain_dict(child) for child in value]
    return value


def configure_selfies_constraints(representation):
    if representation not in {"selfies", "group-selfies"}:
        return
    import selfies as sf

    constraints = sf.get_preset_constraints("hypervalent")
    constraints.update({"P": 7, "P-1": 8, "P+1": 6, "?": 12, "C+1": 5})
    sf.set_semantic_constraints(constraints)


def empty_stats():
    return {
        "valid": 0,
        "invalid": 0,
        "canonical_fallback": 0,
        "insufficient_variants": 0,
        "missing_variants": 0,
    }


def process(smiles, config, seed):
    configure_selfies_constraints(config["representation"])
    encoder = build_encoder(config)
    with open(config["voc_file"], encoding="utf-8") as file:
        tokenizer = VocabularyTokenizer(file.read().splitlines())

    outputs = empty_outputs()
    stats = empty_stats()
    random_state = np.random.RandomState(seed=seed)
    num_variants = config["enumeration"]["num_variants"]
    requested_variants = required_variants(config["mode"], num_variants)

    for line in smiles:
        smile = line.strip()
        if not smile:
            continue
        try:
            canonical, variants = encoder.encode(
                smile, random_state, requested_variants
            )
            canonical_tokens = tokenizer.tokenize(canonical)
            variant_tokens = []
            for value in variants:
                try:
                    variant_tokens.append(tokenizer.tokenize(value))
                except Exception:
                    continue
            sample_outputs, shortfall = build_mode_outputs(
                config["mode"], canonical_tokens, variant_tokens, num_variants
            )
        except Exception:
            stats["invalid"] += 1
            continue

        for key, values in sample_outputs.items():
            outputs[key].extend(values)
        stats["valid"] += 1
        if shortfall:
            stats["insufficient_variants"] += 1
            stats["missing_variants"] += shortfall
            if config["mode"] in {"enum2can", "enum2can+can2enum"}:
                stats["canonical_fallback"] += 1

    print("*", end="", flush=True)
    return outputs, stats


def main(config):
    config = as_plain_dict(config)
    validate_config(config)
    if not config["runtime"]["enable_rdkit_warning"]:
        RDLogger.DisableLog("rdApp.*")

    result_dir = os.path.join(config["output"]["root_dir"], config["processname"])
    make_result_dir(result_dir, duplicate=config["output"]["duplicate"])
    logger = default_logger(os.path.join(result_dir, "log.txt"))
    with open(os.path.join(result_dir, "config.yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    logger.info("Loading SMILES...")
    with open(config["input"], encoding="utf-8") as file:
        smiles = file.read().splitlines()

    chunk_size = config["runtime"]["chunk_size"]
    chunks = [
        smiles[start : start + chunk_size]
        for start in range(0, len(smiles), chunk_size)
    ]
    logger.info(f"Representation: {config['representation']}")
    logger.info(f"Mode: {config['mode']}")
    if config["representation"] == "group-selfies":
        logger.info(f"Group SELFIES strategy: {config['group_selfies']['strategy']}")
    logger.info(f"Chunk num: {len(chunks)}")

    seeds = np.random.RandomState(config["runtime"]["seed"]).randint(
        1_000_000, size=len(chunks)
    )
    if config["runtime"]["max_workers"] == 1:
        results = [
            process(chunk, config, int(seed)) for chunk, seed in zip(chunks, seeds)
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=config["runtime"]["max_workers"]
        ) as executor:
            futures = [
                executor.submit(process, chunk, config, int(seed))
                for chunk, seed in zip(chunks, seeds)
            ]
            results = [future.result() for future in futures]

    outputs = empty_outputs()
    stats = empty_stats()
    for chunk_outputs, chunk_stats in results:
        for key, values in chunk_outputs.items():
            outputs[key].extend(values)
        for key, value in chunk_stats.items():
            stats[key] += value

    print()
    logger.info(f"Valid SMILES: {stats['valid']}/{len(smiles)}")
    logger.info(f"Invalid SMILES: {stats['invalid']}")
    logger.info(f"Canonical enumeration fallbacks: {stats['canonical_fallback']}")
    logger.info(
        "Molecules below requested variant count: "
        f"{stats['insufficient_variants']}"
    )
    logger.info(f"Total missing variants: {stats['missing_variants']}")

    written = save_outputs(
        result_dir, outputs, save_empty=config["output"]["save_empty"]
    )
    for path in written:
        logger.info(f"Saved: {path}")
    return outputs, stats


if __name__ == "__main__":
    main(load_config("./preprocess", ["config"]))

import sys, os
import concurrent.futures as cf
import pickle

import yaml
import numpy as np

from src.datasets.tokenizer import VocabularyTokenizer
from src.utils.args import load_config
from src.utils.logger import default_logger
from src.utils.path import make_result_dir
from rdkit import Chem, RDLogger

from group_selfies.group_grammar import GroupGrammar
import selfies as sf
from multiprocessing import Pool

# === Grammar 初始化一次就好 ===
grammar_file = "/home/jingchun/TransformerVAE/data/guacamol_grammar.txt"
grammar = GroupGrammar.essential_set() | GroupGrammar.from_file(grammar_file)

sanitize_ops = 0
for k, v in Chem.rdmolops.SanitizeFlags.values.items():
    if v not in [Chem.rdmolops.SanitizeFlags.SANITIZE_CLEANUP,
                 Chem.rdmolops.SanitizeFlags.SANITIZE_ALL]:
        sanitize_ops |= v


def process(smiles, voc_file, seed):
    # 每個 process 裡只需要 tokenizer 和 random state，
    # grammar 可以直接用全域的，不需要再讀一次檔案
    with open(voc_file) as f:
        vocs = f.read().splitlines()
    tokenizer = VocabularyTokenizer(vocs)
    random_state = np.random.RandomState(seed=seed)

    cans = []
    rans = []
    n_valid = 0

    for smile in smiles:
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                continue

            # Canonical SELFIES
            extracted = grammar.extract_groups(mol)
            can_selfies = grammar.encoder(mol, extracted)

            # Tokenize both
            can = tokenizer.tokenize(can_selfies)
            ran = tokenizer.tokenize(can_selfies)

            cans.append(can)
            rans.append(ran)
            n_valid += 1

        except Exception as e:
            print(f"[Error] {smile}: {e}")
            continue

    print('*', end='', flush=True)
    return cans, rans, n_valid


def process_wrapper(args):
    return process(*args)


def main(config, processname, input, voc_file, max_workers, chunk_size, seed,
         enable_rdkit_warning):

    # Logging
    result_dir = os.path.join('preprocess/results', processname)
    make_result_dir(result_dir, duplicate='ask')
    logger = default_logger(os.path.join(result_dir, 'log.txt'))
    if not enable_rdkit_warning:
        RDLogger.DisableLog("rdApp.*")

    # Save params
    with open(os.path.join(result_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # Load SMILES
    logger.info("Loading SMILES...")
    with open(input, 'r') as f:
        smiles = f.read().splitlines()
    smiles_size = len(smiles)

    # 設定 SELFIES 限制
    constraints = sf.get_preset_constraints("hypervalent")
    constraints.update({"P": 7, "P-1": 8, "P+1": 6, "?": 12, "C+1": 5})
    sf.set_semantic_constraints(constraints)

    # Process SMILES
    random_state = np.random.RandomState(seed=seed)

    with Pool(processes=max_workers) as pool:
        tasks = [(smiles[chunk_start:chunk_start+chunk_size], voc_file,
                  random_state.randint(100))
                 for chunk_start in range(0, smiles_size, chunk_size)]
        results = pool.map(process_wrapper, tasks)

    cans = []
    rans = []
    n_valid = 0
    for tcans, trans, tn_valid in results:
        cans += tcans
        rans += trans
        n_valid += tn_valid
    print()
    logger.info(f"Valid SMILES: {n_valid}/{smiles_size}")

    # Save result
    cans = np.array(cans, dtype=object)
    rans = np.array(rans, dtype=object)
    with open(os.path.join(result_dir, 'can_tokens.pkl'), 'wb') as f:
        pickle.dump(cans, f)
    with open(os.path.join(result_dir, 'ran_tokens.pkl'), 'wb') as f:
        pickle.dump(rans, f)


if __name__ == '__main__':
    config = load_config("./preprocess", ["config"])
    main(config, **config)

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

sanitize_ops = 0
for k,v in Chem.rdmolops.SanitizeFlags.values.items():
    if v not in [Chem.rdmolops.SanitizeFlags.SANITIZE_CLEANUP,
                Chem.rdmolops.SanitizeFlags.SANITIZE_ALL]:
        sanitize_ops |= v

def process(smiles, voc_file, seed):
    with open(voc_file, "r", encoding="utf-8") as f:
        vocs = [ln.rstrip("\n") for ln in f]
    tokenizer = VocabularyTokenizer(vocs)
    random_state = np.random.RandomState(seed=seed)

    n_valid = 0
    cans = []
    rans = []

    for smile in smiles:
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                continue

            smi_can = Chem.MolToSmiles(mol)  
            tok_can = tokenizer.tokenize(smi_can)
            can_key = tuple(tok_can) if isinstance(tok_can, (list, tuple)) else (tok_can,)

            n_atoms = mol.GetNumAtoms()
            uniq_ran_keys = set()
            uniq_rans = []

            for _ in range(3):
                order = np.arange(n_atoms)
                random_state.shuffle(order)
                rdmol = Chem.RenumberAtoms(mol, order.tolist())

                smi_ran = Chem.MolToSmiles(rdmol, canonical=False)
                tok_ran = tokenizer.tokenize(smi_ran)
                key = tuple(tok_ran) if isinstance(tok_ran, (list, tuple)) else (tok_ran,)

                if key not in uniq_ran_keys and key != can_key:
                    uniq_ran_keys.add(key)
                    uniq_rans.append(tok_ran)

            cans.append(tok_can)
            rans.append(tok_can)
            for tok_ran in uniq_rans:
                cans.append(tok_can)
                rans.append(tok_ran)

            n_valid += 1

        except Exception:
            pass

    print('*', end='', flush=True)
    return cans, rans, n_valid

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

    # Process SMILES
    logger.info("Processing SMILES...")
    logger.info(f"Chunk num: {(smiles_size-1)//chunk_size+1}")
    random_state = np.random.RandomState(seed=seed)
    with cf.ProcessPoolExecutor(max_workers=max_workers) as e:
        futures = []
        for chunk_start in range(0, smiles_size, chunk_size):
            futures.append(e.submit(process, smiles[chunk_start:chunk_start+chunk_size],
                voc_file=voc_file, seed=random_state.randint(100)))
        cans = []
        rans = []
        n_valid = 0
        for future in futures:
            tcans, trans, tn_valid = future.result()
            cans += tcans
            rans += trans
            n_valid += tn_valid
    print()
    logger.info(f"Valid SMILES: {n_valid}/{smiles_size}")

    # Save result
    cans = np.array(cans, dtype=object)
    rans = np.array(rans, dtype=object, )
    with open(os.path.join(result_dir, 'can_tokens.pkl'), 'wb') as f:
        pickle.dump(cans, f)
    with open(os.path.join(result_dir, 'ran_tokens.pkl'), 'wb') as f:
        pickle.dump(rans, f)

if __name__ == '__main__':
    config = load_config("./preprocess", ["config"])
    main(config, **config)


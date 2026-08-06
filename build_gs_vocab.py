#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build Group-SELFIES vocabulary using an existing grammar file."""

import argparse
import os

import pandas as pd
from rdkit import Chem, RDLogger
from selfies import split_selfies

from group_selfies import GroupGrammar, get_semantic_robust_alphabet

RDLogger.DisableLog('rdApp.*')


def main():
    parser = argparse.ArgumentParser(
        description='Build Group-SELFIES vocabulary from SMILES + grammar.'
    )
    parser.add_argument('--smiles', type=str, required=True, help='SMILES input file path')
    parser.add_argument('--grammar', type=str, required=True, help='Grammar file path')
    parser.add_argument('--out', type=str, required=True, help='Output vocab file path')
    args = parser.parse_args()

    smiles_path = args.smiles
    grammar_path = args.grammar
    vocab_out = args.out

    if not os.path.exists(smiles_path):
        raise FileNotFoundError(f'SMILES file not found: {smiles_path}')
    if not os.path.exists(grammar_path):
        raise FileNotFoundError(f'Grammar file not found: {grammar_path}')

    unique_smiles = set()
    with open(smiles_path, 'r') as f:
        for line in f:
            s = line.strip()
            if s:
                unique_smiles.add(s)
    print(f'[INFO] Unique SMILES: {len(unique_smiles)}')

    grammar = GroupGrammar.from_file(grammar_path)
    print('[INFO] Grammar loaded.')

    all_tokens = set(grammar.all_tokens()) | set(get_semantic_robust_alphabet())

    invalid = 0
    processed = 0
    for smi in unique_smiles:
        processed += 1
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            invalid += 1
            continue
        try:
            atom_level = grammar.encoder(mol, [], join=True)
            all_tokens.update(split_selfies(atom_level))
        except Exception as e:
            if invalid < 10:
                print(f'[WARN] Failed on SMILES: {smi} | err={e}')
            invalid += 1

        if processed % 100000 == 0:
            print(f'[PROGRESS] processed={processed}, invalid={invalid}, vocab={len(all_tokens)}')

    tmp_out = vocab_out + '.tmp'
    pd.DataFrame(sorted(all_tokens)).to_csv(tmp_out, index=False, header=False)
    os.replace(tmp_out, vocab_out)

    print('\n[SUMMARY]')
    print(f'  Total unique SMILES: {len(unique_smiles)}')
    print(f'  Invalid SMILES:      {invalid}')
    print(f'  Total vocab tokens:  {len(all_tokens)}')
    print(f'[OK] Vocabulary saved to: {vocab_out}')


if __name__ == '__main__':
    main()

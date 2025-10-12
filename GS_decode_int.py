#!/usr/bin/env python3
from rdkit import Chem
from group_selfies import GroupGrammar
import sys, os

GRAMMAR_FILE = '/home/jingchun/TransformerVAE/data/grammar.txt'

def decode(grammar_type, data_dir):
    if grammar_type == 'merged':
        essential = GroupGrammar.essential_set()
        grammar_frag = GroupGrammar.from_file(GRAMMAR_FILE)
        merged_grammar = grammar_frag | essential
    elif grammar_type == 'essential':
        merged_grammar = GroupGrammar.essential_set()
    else:
        raise ValueError('Invalid grammar type: {}'.format(grammar_type))

    data_path = os.path.join('decoding', 'results', data_dir)
    if not os.path.isdir(data_path):
        print(f"[WARNING] Path does not exist: {data_path}")
        return

    file_pairs = [
        ('selfies_end1.txt', 'smiles_end1.txt'),
        ('selfies_mid.txt',  'smiles_mid.txt'),
        ('selfies_end2.txt', 'smiles_end2.txt')
    ]
    
    for in_name, out_name in file_pairs:
        in_path  = os.path.join(data_path, in_name)
        out_path = os.path.join(data_path, out_name)

        if not os.path.isfile(in_path):
            print(f"[WARN] File not found: {in_path}")
            continue

        ok = 0
        tot = 0
        with open(in_path, 'r') as fin, open(out_path, 'w') as fout:
            for line in fin:
                s = line.rstrip('\n')
                tot += 1
                try:
                    mol = merged_grammar.decoder(s)
                    if mol is None:
                        fout.write("INVALID\n")
                    else:
                        smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                        fout.write(smi + "\n")
                        ok += 1
                except Exception:
                    fout.write("INVALID\n")

        print(f"[INFO] Decoded: {out_path} (valid {ok}/{tot})")

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 2:
        print('Usage: python GS_decode.py <grammar_type:{merged|essential}> <data_dir>')
        sys.exit(1)
    decode(args[0], args[1])

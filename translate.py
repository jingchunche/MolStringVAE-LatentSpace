#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auto-detect SMILES, SELFIES, or Group SELFIES and output SMILES.

This is a standalone command-line script and does not import other project
scripts. Group SELFIES always uses the fixed grammar path below.
"""

import argparse
import math
import shutil
import sys
from pathlib import Path

from rdkit import Chem, RDLogger


GRAMMAR_FILE = Path(
    '/home/jingchun/MolStringVAE-LatentSpace/data/group_selfies_grammar.txt'
)
OUTPUT_NAMES = {
    'recon_string.txt': 'recon_smiles.txt',
    'decoded_string.txt': 'decoded_smiles.txt',
    'generated_string.txt': 'generated_smiles.txt',
    'end1.txt': 'end1_smiles.txt',
    'mid.txt': 'mid_smiles.txt',
    'end2.txt': 'end2_smiles.txt',
}

RDLogger.DisableLog('rdApp.*')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Convert model output strings to SMILES with automatic output naming. '
            'The input format must be specified explicitly.'
        )
    )
    parser.add_argument(
        'input',
        help=(
            'An input file, or a directory containing recognized input TXT files.'
        ),
    )
    parser.add_argument(
        '--output',
        default=None,
        help=(
            'Optional output file for file input, or output directory for '
            'directory input. By default, write beside each input file.'
        ),
    )
    parser.add_argument(
        '--format',
        choices=['smiles', 'selfies', 'group-selfies'],
        required=True,
        help='Required input format.',
    )
    parser.add_argument(
        '--constraint',
        choices=['default', 'no'],
        default='default',
        help=(
            'Semantic constraint mode: default keeps the original decoder '
            'behavior; no uses unlimited bonding capacities.'
        ),
    )
    return parser.parse_args()


def infer_output_path(input_path):
    output_name = OUTPUT_NAMES.get(input_path.name)
    if output_name is None:
        raise ValueError(
            'Cannot infer output name from {!r}; use --output. Expected one of: {}'.format(
                input_path.name, ', '.join(sorted(OUTPUT_NAMES))
            )
        )
    return input_path.with_name(output_name)


def set_constraint_mode(module, mode):
    if mode == 'default':
        module.set_semantic_constraints('default')
        return

    constraints = module.get_semantic_constraints()
    unrestricted = {key: math.inf for key in constraints}
    unrestricted['?'] = math.inf
    module.set_semantic_constraints(unrestricted)


def decode_selfies(lines, output_path, constraint_mode):
    try:
        import selfies
    except ImportError:
        raise RuntimeError('The selfies package is required for SELFIES decoding.')

    set_constraint_mode(selfies, constraint_mode)
    valid = 0
    total = 0
    with output_path.open('w') as fout:
        for raw_line in lines:
            text = raw_line.strip()
            total += 1
            if not text:
                fout.write('\n')
                continue
            try:
                smiles = selfies.decoder(text)
            except selfies.DecoderError:
                fout.write('INVALID\n')
            else:
                fout.write(smiles + '\n')
                valid += 1
    return valid, total


def load_group_grammar():
    if not GRAMMAR_FILE.is_file():
        raise FileNotFoundError('Grammar file not found: {}'.format(GRAMMAR_FILE))
    try:
        import group_selfies
        from group_selfies import GroupGrammar
    except ImportError:
        raise RuntimeError('The group_selfies package is required for Group SELFIES decoding.')

    return group_selfies, GroupGrammar


def decode_group_selfies(lines, output_path, constraint_mode):
    group_selfies, GroupGrammar = load_group_grammar()
    set_constraint_mode(group_selfies, constraint_mode)
    fragment_grammar = GroupGrammar.from_file(str(GRAMMAR_FILE))
    grammar = fragment_grammar | GroupGrammar.essential_set()
    valid = 0
    total = 0
    with output_path.open('w') as fout:
        for raw_line in lines:
            text = raw_line.rstrip('\n')
            total += 1
            try:
                mol = grammar.decoder(text)
                if mol is None:
                    raise ValueError('Decoder returned no molecule')
                smiles = Chem.MolToSmiles(
                    mol, isomericSmiles=True, canonical=True
                )
            except Exception:
                fout.write('INVALID\n')
            else:
                fout.write(smiles + '\n')
                valid += 1
    return valid, total


def translate_file(input_path, output_path, input_format, constraint_mode):
    if input_path == output_path:
        raise ValueError('Input and output paths must be different.')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # keepends=True allows SMILES input to be copied byte-for-byte.
    with input_path.open('r') as fin:
        lines = fin.readlines()

    if input_format == 'smiles':
        shutil.copyfile(str(input_path), str(output_path))
        valid = sum(bool(line.strip()) for line in lines)
        total = len(lines)
    elif input_format == 'selfies':
        valid, total = decode_selfies(lines, output_path, constraint_mode)
    else:
        valid, total = decode_group_selfies(
            lines, output_path, constraint_mode
        )

    print('[INFO] {} -> {} (valid {}/{})'.format(
        input_path.name, output_path.name, valid, total
    ))


def collect_jobs(input_path, output_arg):
    if input_path.is_file():
        output_path = (
            Path(output_arg).expanduser().resolve()
            if output_arg
            else infer_output_path(input_path)
        )
        return [(input_path, output_path)]

    if not input_path.is_dir():
        raise FileNotFoundError('Input file or directory not found: {}'.format(input_path))

    output_dir = (
        Path(output_arg).expanduser().resolve()
        if output_arg
        else input_path
    )
    jobs = []
    for input_name, output_name in OUTPUT_NAMES.items():
        source = input_path / input_name
        if source.is_file():
            jobs.append((source, output_dir / output_name))

    if not jobs:
        raise FileNotFoundError(
            'No recognized input TXT files found in: {}'.format(input_path)
        )
    return jobs


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    jobs = collect_jobs(input_path, args.output)

    print('[INFO] Input format: {}'.format(args.format))
    print('[INFO] Constraint mode: {}'.format(args.constraint))
    print('[INFO] Files found: {}'.format(len(jobs)))

    for source, destination in jobs:
        translate_file(
            source,
            destination,
            input_format=args.format,
            constraint_mode=args.constraint,
        )


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('[ERROR] {}'.format(exc), file=sys.stderr)
        sys.exit(1)

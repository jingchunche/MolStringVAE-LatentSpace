#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone Group SELFIES grammar pipeline.

Pipeline: SMILES -> fragments -> same-core attachment union
          -> diversity selection -> sort/renumber -> grammar
"""
import argparse
import os
import sys
from collections import Counter, OrderedDict
from itertools import islice
from typing import Dict, Iterable, List, Tuple
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
GROUP_SELFIES_ROOT = '/home/jingchun/group-selfies'
if os.path.isdir(GROUP_SELFIES_ROOT) and GROUP_SELFIES_ROOT not in sys.path:
    sys.path.insert(0, GROUP_SELFIES_ROOT)
from group_selfies import Group, GroupGrammar, fragment_mols
from group_selfies.utils.group_utils import mol_to_group_s
RDLogger.DisableLog('rdApp.*')
ORDER_TO_BOND = {1: BT.SINGLE, 2: BT.DOUBLE, 3: BT.TRIPLE}



def chunked_iterable(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch


def iter_smiles(path: str) -> Iterable[str]:
    with open(path, 'r') as f:
        for line in f:
            smi = line.strip()
            if smi:
                yield smi


def dedup_smiles(path: str) -> List[str]:
    seen = set()
    smiles = []

    for smi in iter_smiles(path):
        if smi in seen:
            continue

        seen.add(smi)
        smiles.append(smi)

    return smiles


def collect_fragments_with_fragment_mols(
    smiles: List[str],
    batch_size: int,
    target_group_size: int,
    n_limit: int,
    method: str,
    n_jobs: int,
) -> List[str]:
    batches = list(chunked_iterable(smiles, batch_size))

    def process_batch(batch: List[str]) -> List[str]:
        try:
            return fragment_mols(
                batch,
                convert=True,
                n_limit=n_limit,
                target=target_group_size,
                method=method,
            )
        except Exception:
            return []

    if n_jobs == 1:
        fragment_batches = [
            process_batch(batch)
            for batch in tqdm(batches, desc='Fragmenting batches')
        ]
    else:
        try:
            from joblib import Parallel, delayed, parallel_backend
        except ImportError as exc:
            raise RuntimeError(
                'joblib is required when --n_jobs is not 1. '
                'Install joblib or rerun with --n_jobs 1.'
            ) from exc

        with parallel_backend('loky', n_jobs=n_jobs):
            fragment_batches = Parallel()(
                delayed(process_batch)(batch)
                for batch in tqdm(batches, desc='Fragmenting batches')
            )

    seen = set()
    ordered_fragments = []

    for fragments in fragment_batches:
        for fragment in fragments:
            if not fragment or fragment in seen:
                continue

            seen.add(fragment)
            ordered_fragments.append(fragment)

    return ordered_fragments


def collect_fragments_with_fragment_mols_global(
    smiles: List[str],
    target_group_size: int,
    n_limit: int,
    method: str,
) -> List[str]:
    """Run fragment_mols() once over the full input set.

    This makes n_limit/filtering, closely_contained merging, and diversity
    selection global instead of batch-local.
    """
    try:
        fragments = fragment_mols(
            smiles,
            convert=True,
            n_limit=n_limit,
            target=target_group_size,
            method=method,
        )
    except Exception:
        return []

    seen = set()
    ordered_fragments = []

    for fragment in fragments:
        if not fragment or fragment in seen:
            continue

        seen.add(fragment)
        ordered_fragments.append(fragment)

    return ordered_fragments


def build_groups(
    fragments: List[str],
) -> List:
    groups = []

    for idx, fragment in enumerate(fragments):
        try:
            group = Group(f'frag_{idx}', fragment)
        except Exception:
            continue

        groups.append(group)

    return groups


def write_grammar(path: str, groups: List):
    GroupGrammar(groups).to_file(path)
    with open(path, 'a') as f:
        f.write('\n')


def remove_attachment_points(mol: Chem.Mol) -> Chem.Mol:
    """Return the core while preserving RDKit's sanitized aromatic perception."""
    rw = Chem.RWMol(mol)

    dummy_indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 0
    ]

    for idx in sorted(dummy_indices, reverse=True):
        rw.RemoveAtom(idx)

    core = rw.GetMol()
    Chem.SanitizeMol(core)
    return core


def safe_compare_core_key(core: Chem.Mol) -> str:
    """Canonical key used only for same-core comparison."""
    core_copy = Chem.Mol(core)
    Chem.SanitizeMol(core_copy)

    smi = Chem.MolToSmiles(
        core_copy,
        canonical=True,
        isomericSmiles=True,
        kekuleSmiles=False,
    )

    return f'{core_copy.GetNumAtoms()}|{core_copy.GetNumBonds()}|{smi}'


def group_merge_info(group: Group, original_index: int):
    """Extract safe merge metadata using the original molecule as template."""
    original_mol = Chem.Mol(group.mol)

    old_to_new: Dict[int, int] = {}
    new_to_old: Dict[int, int] = {}

    core_idx = 0
    for atom in original_mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            old_idx = atom.GetIdx()
            old_to_new[old_idx] = core_idx
            new_to_old[core_idx] = old_idx
            core_idx += 1

    attachments: List[Tuple[int, int]] = []
    for atom in original_mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 1:
            raise ValueError(f'Attachment atom has {len(neighbors)} parents in {group.name}')
        parent = neighbors[0]
        valency = atom.GetIntProp('valAvailable') if atom.HasProp('valAvailable') else 1
        attachments.append((old_to_new[parent.GetIdx()], valency))

    output_core = remove_attachment_points(original_mol)
    compare_core = Chem.Mol(output_core)
    Chem.SanitizeMol(compare_core)

    return {
        'group': group,
        'original_index': original_index,
        'original_mol': original_mol,
        'output_core': output_core,
        'compare_core': compare_core,
        'compare_key': safe_compare_core_key(compare_core),
        'attachments': attachments,
        'core_to_old': new_to_old,
    }


def attachment_counter(attachments: List[Tuple[int, int]]) -> Counter:
    counts = Counter()
    for parent_core_idx, valency in attachments:
        counts[(parent_core_idx, valency)] += 1
    return counts


def choose_best_match(
    rep_core: Chem.Mol,
    query_core: Chem.Mol,
    query_attachments: List[Tuple[int, int]],
    existing_counts: Counter,
):
    """Choose the atom mapping that best preserves existing attachments."""
    matches = rep_core.GetSubstructMatches(query_core, uniquify=True)

    if not matches:
        return None

    def score(match):
        mapped_counts = Counter()
        for parent_idx, valency in query_attachments:
            mapped_counts[(match[parent_idx], valency)] += 1

        overlap = sum(
            min(existing_counts[key], count)
            for key, count in mapped_counts.items()
        )

        missing = sum(
            max(0, count - existing_counts[key])
            for key, count in mapped_counts.items()
        )

        signature = tuple(sorted(mapped_counts.items()))
        return (-overlap, missing, signature)

    return min(matches, key=score)


def validate_group_string(name: str, group_string: str) -> Group:
    group = Group(name, group_string)
    _ = GroupGrammar([group])
    return group


def merge_infos_on_original_template(
    new_name: str,
    infos: List[dict],
    group_supports: Counter,
) -> Tuple[Group, int]:
    """Merge a same-core bucket by adding missing attachments to one original."""
    rep_info = max(
        infos,
        key=lambda info: (
            group_supports[info['group'].name],
            len(info['attachments']),
            -info['original_index'],
        ),
    )

    support_count = max(
        group_supports[info['group'].name]
        for info in infos
    )

    rep_core = rep_info['compare_core']
    rw = Chem.RWMol(rep_info['original_mol'])
    existing_counts = attachment_counter(rep_info['attachments'])

    for info in infos:
        match = choose_best_match(
            rep_core=rep_core,
            query_core=info['compare_core'],
            query_attachments=info['attachments'],
            existing_counts=existing_counts,
        )
        if not match:
            continue

        current_counts = Counter()
        for parent_idx, valency in info['attachments']:
            current_counts[(match[parent_idx], valency)] += 1

        for key, count in current_counts.items():
            parent_idx, valency = key

            if valency not in ORDER_TO_BOND:
                raise ValueError(f'Unsupported attachment valency {valency} in {new_name}')

            current = existing_counts[key]
            missing = count - current

            if missing <= 0:
                continue

            parent_old_idx = rep_info['core_to_old'][parent_idx]

            for _ in range(missing):
                atom = Chem.Atom('*')
                atom.SetIntProp('valAvailable', valency)
                atom.SetAtomMapNum(valency)
                new_idx = rw.AddAtom(atom)
                rw.AddBond(parent_old_idx, new_idx, ORDER_TO_BOND[valency])

            existing_counts[key] += missing

    merged_mol = rw.GetMol()
    Chem.SanitizeMol(merged_mol)

    try:
        merged_smiles = mol_to_group_s(merged_mol)
        merged_group = validate_group_string(new_name, merged_smiles)
        merged_group.priority = max(info['group'].priority for info in infos)
        return merged_group, support_count
    except Exception:
        fallback_group = Group(new_name, rep_info['group'].canonsmiles)
        _ = GroupGrammar([fallback_group])
        fallback_group.priority = max(info['group'].priority for info in infos)
        return fallback_group, support_count


def merge_same_core_attachment_union(
    groups: List[Group],
    group_supports: Counter,
) -> Tuple[List[Group], Counter, int]:
    buckets = OrderedDict()
    skipped = 0

    for idx, group in enumerate(groups):
        try:
            info = group_merge_info(group, original_index=idx)
        except Exception as exc:
            skipped += 1
            print(f'[skip] {group.name}: {exc}', file=sys.stderr)
            continue

        buckets.setdefault(info['compare_key'], []).append(info)

    merged_groups: List[Group] = []
    merged_supports = Counter()

    for idx, infos in enumerate(buckets.values()):
        try:
            merged_group, support_count = merge_infos_on_original_template(
                new_name=f'frag_{idx}',
                infos=infos,
                group_supports=group_supports,
            )
        except Exception as exc:
            skipped += len(infos)
            names = ','.join(info['group'].name for info in infos)
            print(f'[skip] bucket {names}: {exc}', file=sys.stderr)
            continue

        merged_groups.append(merged_group)
        merged_supports[merged_group.name] = support_count

    return merged_groups, merged_supports, skipped


def group_core_mol(group: Group) -> Chem.Mol:
    core = group.mol_without_attachment_points().GetMol()
    Chem.SanitizeMol(core)

    try:
        Chem.Kekulize(core, clearAromaticFlags=True)
    except Exception:
        pass

    return core


def group_heavy_atoms(group: Group) -> int:
    return group_core_mol(group).GetNumAtoms()


def sort_and_renumber_by_heavy_atoms(groups: List[Group]) -> List[Group]:
    ordered = sorted(
        groups,
        key=lambda group: (
            group_heavy_atoms(group),
            group.priority,
            group.canonsmiles,
        ),
        reverse=True,
    )

    return [
        Group(f'frag_{idx}', group.canonsmiles, priority=group.priority)
        for idx, group in enumerate(ordered)
    ]


def parse_args():
    parser = argparse.ArgumentParser(description='Build and merge a Group SELFIES grammar from SMILES.')
    parser.add_argument('--input', required=True, help='Input SMILES file.')
    parser.add_argument('--output', required=True, help='Final grammar file.')
    parser.add_argument('--candidates_out', default=None, help='Optional unmerged candidate grammar.')
    parser.add_argument('--fragment_scope', choices=['global', 'batch'], default='global')
    parser.add_argument('--target_group_size', type=int, default=165)
    parser.add_argument('--core_min_occurrence', type=int, default=1000)
    parser.add_argument('--fragment_method', choices=['default', 'mmpa', 'fraggle'], default='default')
    parser.add_argument('--batch_size', type=int, default=50000)
    parser.add_argument('--n_jobs', type=int, default=-1)
    return parser.parse_args()


def validate_args(args):
    if not os.path.isfile(args.input):
        raise FileNotFoundError('Input not found: {}'.format(args.input))
    if args.target_group_size <= 0 or args.core_min_occurrence <= 0 or args.batch_size <= 0:
        raise ValueError('Size and occurrence arguments must be greater than zero.')
    if args.n_jobs == 0:
        raise ValueError('--n_jobs cannot be zero.')


def main():
    args = parse_args()
    validate_args(args)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    if args.candidates_out:
        os.makedirs(os.path.dirname(args.candidates_out) or '.', exist_ok=True)
    smiles = dedup_smiles(args.input)
    print('Loaded unique SMILES: {:,}'.format(len(smiles)))
    common = dict(target_group_size=args.target_group_size, n_limit=args.core_min_occurrence, method=args.fragment_method)
    if args.fragment_scope == 'global':
        print('[1/4] Global fragmentation...')
        fragments = collect_fragments_with_fragment_mols_global(smiles, **common)
    else:
        print('[1/4] Batch fragmentation...')
        fragments = collect_fragments_with_fragment_mols(smiles, batch_size=args.batch_size, n_jobs=args.n_jobs, **common)
    candidates = build_groups(fragments)
    print('[2/4] Candidate groups: {:,}'.format(len(candidates)))
    if not candidates:
        raise RuntimeError('No valid candidate groups were generated.')
    if args.candidates_out:
        write_grammar(args.candidates_out, candidates)
    supports = Counter({group.name: group.priority for group in candidates})
    merged, _supports, skipped = merge_same_core_attachment_union(candidates, supports)
    merged_count = len(merged)
    print('[3/4] Merged groups: {:,}'.format(merged_count))
    final_groups = sort_and_renumber_by_heavy_atoms(merged)
    if not final_groups:
        raise RuntimeError('No valid groups remain after merging.')
    write_grammar(args.output, final_groups)
    print('[4/4] Grammar complete.')
    print('  Candidate groups:     {:,}'.format(len(candidates)))
    print('  Merged groups:        {:,}'.format(merged_count))
    print('  Final groups:         {:,}'.format(len(final_groups)))
    print('  Skipped during merge: {:,}'.format(skipped))
    print('  Output:               {}'.format(args.output))


if __name__ == '__main__':
    main()


import argparse
import json
import logging
import os
import random
from typing import List, Optional

from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.utils.helpers import setup_default_logger


logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_ROOT = os.path.join(REPO_ROOT, 'generation', 'results')
DEFAULT_TRAIN_FILE = os.path.join(REPO_ROOT, 'data', 'guacamol_v1_train.smiles')
DEFAULT_FALLBACK_OUTPUT = os.path.join(REPO_ROOT, 'evaluation')


class PreGeneratedGenerator(DistributionMatchingGenerator):
    """Wraps a static SMILES list so guacamol can request batches of samples."""

    def __init__(self, smiles: List[str], seed: int, sample_with_replacement: bool):
        if not smiles:
            raise ValueError('SMILES list is empty; nothing to evaluate.')
        self._smiles = list(smiles)
        self._seed = seed
        self._rng = random.Random(seed)
        self._with_replacement = sample_with_replacement
        self._cursor = 0
        if not sample_with_replacement:
            self._rng.shuffle(self._smiles)

    def generate(self, number_samples: int) -> List[str]:
        if self._with_replacement:
            return [self._rng.choice(self._smiles) for _ in range(number_samples)]

        generated = []
        while len(generated) < number_samples:
            remaining = number_samples - len(generated)
            chunk = self._smiles[self._cursor:self._cursor + remaining]
            generated.extend(chunk)
            self._cursor += len(chunk)
            if len(generated) < number_samples:
                self._rng.shuffle(self._smiles)
                self._cursor = 0
        return generated


def read_smiles_file(path: str, unique: bool, max_length: Optional[int]) -> List[str]:
    with open(path, 'r') as handle:
        smiles = [line.strip() for line in handle if line.strip()]
    if unique:
        seen = set()
        deduped = []
        for s in smiles:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        smiles = deduped
    if max_length is not None:
        filtered = [s for s in smiles if len(s) <= max_length]
        removed = len(smiles) - len(filtered)
        if removed:
            logger.info('Dropped %d SMILES longer than max_length %d.', removed, max_length)
        if not filtered:
            raise ValueError('All SMILES exceed the maximum length constraint; nothing to evaluate.')
        smiles = filtered
    return smiles


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a pre-generated SMILES set on GuacaMol distribution metrics.')
    parser.add_argument('--name', default=None,
                        help='Result subdirectory relative to results_root, e.g. GSran/ran/10000.')
    parser.add_argument('--results_root', default=DEFAULT_RESULTS_ROOT,
                        help='Root directory containing generation results (default: generation/results).')
    parser.add_argument('--samples', default=None,
                        help='Path to the generated SMILES file to evaluate. Overrides --name if provided.')
    parser.add_argument('--train_file', default=DEFAULT_TRAIN_FILE,
                        help='Training SMILES file used as reference distribution.')
    parser.add_argument('--output_dir', default=None,
                        help='Directory where evaluation artifacts will be written (default: same directory as samples).')
    parser.add_argument('--suite', default='v2', help='GuacaMol benchmark suite (v1 or v2).')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for sampling.')
    parser.add_argument('--unique', action='store_true', help='Remove duplicate SMILES before evaluation.')
    parser.add_argument('--max_length', type=int, default=350,
                        help='Maximum SMILES length allowed; longer entries are dropped before evaluation.')
    parser.add_argument('--no_replacement', action='store_true',
                        help='Sample without replacement when feeding SMILES to GuacaMol.')
    args = parser.parse_args()

    args.results_root = os.path.abspath(args.results_root)
    args.train_file = os.path.abspath(args.train_file)

    if args.samples is None:
        if args.name is None:
            parser.error('Either --samples or --name must be provided to locate generated SMILES.')
        sample_dir = os.path.join(args.results_root, args.name)
        args.samples = os.path.join(sample_dir, 'smiles.txt')
    else:
        sample_dir = os.path.dirname(os.path.abspath(args.samples))
        args.samples = os.path.abspath(args.samples)

    if args.output_dir is None:
        if args.name is not None:
            args.output_dir = os.path.join(args.results_root, args.name)
        else:
            args.output_dir = sample_dir or DEFAULT_FALLBACK_OUTPUT
    args.output_dir = os.path.abspath(args.output_dir)

    setup_default_logger()

    smiles = read_smiles_file(args.samples, unique=args.unique, max_length=args.max_length)

    os.makedirs(args.output_dir, exist_ok=True)
    params_path = os.path.join(args.output_dir, 'distribution_learning_params.json')
    with open(params_path, 'w') as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    generator = PreGeneratedGenerator(smiles=smiles,
                                      seed=args.seed,
                                      sample_with_replacement=not args.no_replacement)

    json_path = os.path.join(args.output_dir, 'distribution_learning_results.json')
    assess_distribution_learning(generator,
                                 chembl_training_file=args.train_file,
                                 json_output_file=json_path,
                                 benchmark_version=args.suite)


if __name__ == '__main__':
    main()

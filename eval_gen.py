import argparse
import json
import logging
import os
import random
from typing import List

from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.utils.helpers import setup_default_logger


logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_ROOT = os.path.join(REPO_ROOT, 'generation', 'results')
DEFAULT_SAMPLES_NAME = 'generated_smiles.txt'
DEFAULT_TRAIN_FILE = os.path.join(REPO_ROOT, 'data', 'guacamol_v1_train.smiles')


class PreGeneratedGenerator(DistributionMatchingGenerator):
    """Wraps a static SMILES list so guacamol can request batches of samples."""

    def __init__(self, smiles: List[str], seed: int):
        if not smiles:
            raise ValueError('SMILES list is empty; nothing to evaluate.')
        self._smiles = list(smiles)
        self._seed = seed
        self._rng = random.Random(seed)
        self._cursor = 0
        self._rng.shuffle(self._smiles)

    def generate(self, number_samples: int) -> List[str]:
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


def read_smiles_file(path: str) -> List[str]:
    with open(path, 'r') as handle:
        return [line.strip() for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a pre-generated SMILES set on GuacaMol distribution metrics.')
    parser.add_argument(
        'name',
        help=(
            'Directory under generation/results, or an absolute results '
            'directory, containing generated_smiles.txt.'
        ),
    )
    parser.add_argument('--train_file', default=DEFAULT_TRAIN_FILE,
                        help='Training SMILES file used as reference distribution.')
    parser.add_argument('--suite', default='v2', help='GuacaMol benchmark suite (v1 or v2).')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for sampling.')
    args = parser.parse_args()

    args.train_file = os.path.abspath(args.train_file)

    candidate = os.path.abspath(os.path.expanduser(args.name))
    if os.path.isdir(candidate):
        result_dir = candidate
    else:
        result_dir = os.path.join(DEFAULT_RESULTS_ROOT, args.name)
    if not os.path.isdir(result_dir):
        raise FileNotFoundError("Could not find results directory '{}'".format(args.name))

    samples_path = os.path.join(result_dir, DEFAULT_SAMPLES_NAME)
    if not os.path.isfile(samples_path):
        raise FileNotFoundError('Generated SMILES file not found: {}'.format(samples_path))

    args.name = result_dir
    setup_default_logger()

    smiles = read_smiles_file(samples_path)

    os.makedirs(result_dir, exist_ok=True)
    params_path = os.path.join(result_dir, 'distribution_learning_params.json')
    with open(params_path, 'w') as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    generator = PreGeneratedGenerator(smiles=smiles, seed=args.seed)

    json_path = os.path.join(result_dir, 'distribution_learning_results.json')
    assess_distribution_learning(generator,
                                 chembl_training_file=args.train_file,
                                 json_output_file=json_path,
                                 benchmark_version=args.suite)


if __name__ == '__main__':
    main()

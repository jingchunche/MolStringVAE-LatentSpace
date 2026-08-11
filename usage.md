This document provides instructions for setting up the required environments, preparing molecular datasets, training Transformer-based variational autoencoders with SMILES, SELFIES, and Group SELFIES representations, running inference tasks such as generation, reconstruction, and latent-space interpolation, and reproducing the latent-space evaluation workflows.

This project is derived from [TransformerVAE](https://github.com/mizuno-group/TransformerVAE) and integrates [Group SELFIES](https://github.com/aspuru-guzik-group/group-selfies) and [GuacaMol](https://github.com/BenevolentAI/guacamol).

## Environment

This project uses three Conda environments for different tasks:

```sh
conda env create -f environments/transformervae.yml
conda env create -f environments/groupselfies.yml
conda env create -f environments/guacamol.yml
```

- `transformervae`: model training, generation, reconstruction, featurization, interpolation, and latent-space evaluation.
- `groupselfies`: preprocessing and conversion of SMILES, SELFIES, and Group SELFIES strings.
- `guacamol`: GuacaMol distribution-learning evaluation.

Run all commands from the repository root:

```sh
cd /path/to/MolStringVAE-LatentSpace
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
```

## Data

Each input file is a plain-text SMILES file without a header and contains one molecule per line. The experiments use the following files:

```text
data/guacamol_v1_train.smiles
data/guacamol_v1_valid.smiles
data/guacamol_v1_test.smiles
data/guacamol_v1_all.smiles
data/guacamol_v1_test_query.smiles
```

- The GuacaMol training, validation, and test files are the official predefined splits.
- `guacamol_v1_all.smiles` is the complete merged GuacaMol dataset.
- `guacamol_v1_test_query.smiles` is a query set randomly sampled from `guacamol_v1_test.smiles` for latent-space retrieval.
- These files are not tracked by Git because of their sizes and are available on [Google Drive](https://drive.google.com/drive/folders/152a5dwmuLWihjh5G-As5-j9mumAQJOJr?usp=sharing). Place the downloaded files in `data/`, or replace the paths in the commands with their actual locations.
- The SMILES row order must remain identical to the row order of the corresponding latent CSV.
- Vocabulary files are provided in `data/` for SMILES (`smiles_vocs.txt`, `voc_size=42`), SELFIES (`selfies_vocs.txt`, `voc_size=110`), and Group SELFIES (`gs_vocs.txt`, `voc_size=1700`), including PAD, START, and END tokens.

## Group SELFIES grammar and vocabulary

The Group SELFIES experiments use `data/gs_grammar.txt`. To rebuild the grammar and vocabulary:

```sh
conda activate groupselfies

python build_gs_grammar.py \
  --input data/guacamol_v1_train.smiles \
  --output data/gs_grammar.txt \
  --core_min_occurrence 1000 \
  --target_group_size 165

python build_gs_vocab.py \
  --smiles data/guacamol_v1_all.smiles \
  --grammar data/gs_grammar.txt \
  --out data/gs_vocs.txt
```

`build_gs_vocab.py` canonicalizes valid SMILES with RDKit before building a deduplicated vocabulary.

## Data preprocessing for training

Before training, convert and tokenize the SMILES strings. The following example prepares SMILES for an enum-to-canonical experiment:

```sh
conda activate groupselfies

python preprocess.py \
  --processname smiles_enum2can/train \
  --data data/guacamol_v1_train.smiles \
  --representation smiles \
  --mode enum2can \
  --voc_file data/smiles_vocs.txt \
  --num_variants 2
```

- `<processname>` can be any name. Results are saved to `preprocess/results/<processname>/`.
- Training and validation data must be preprocessed separately before training.
- `--representation` accepts `smiles`, `selfies`, or `group-selfies`.
- `--mode` accepts `can2can`, `enum2can`, `multi-enum2can`, or `enum2can+can2enum`.
- Group SELFIES supports `--group_strategy none|masking|traversal|masking+traversal`.

## Training

Train the Transformer VAE with the preprocessed training and validation datasets:

```sh
conda activate transformervae

python train.py \
  --name smiles_enum2can \
  --train_data smiles_enum2can/train \
  --val_data smiles_enum2can/val \
  --voc_size 42
```

- `<name>` can be any experiment name.
- Models and training logs are saved to `training/results/<name>/`.
- The default maximum number of training steps is 250000.
- Use the matching `voc_size` when changing the molecular representation.
- Pretrained weights are available on [Google Drive](https://drive.google.com/drive/folders/152a5dwmuLWihjh5G-As5-j9mumAQJOJr?usp=sharing).

## Molecule generation

Generate molecules with a trained or downloaded model:

```sh
conda activate transformervae

python generate.py \
  --name smiles_enum2can \
  --weight training/results/smiles_enum2can \
  --n_generation 30000 \
  --voc_size 42 \
  --voc_file data/smiles_vocs.txt
```

Generated strings are saved to `generation/results/<name>/generated_string.txt`.

## Molecule reconstruction

`recon.py` estimates the latent posterior mean for each input molecule and then decodes it:

```sh
conda activate transformervae

python recon.py \
  --name smiles_enum2can \
  --data smiles_enum2can/test \
  --weight training/results/smiles_enum2can \
  --voc_size 42 \
  --voc_file data/smiles_vocs.txt
```

Reconstructed strings are saved to `recon/results/<name>/recon_string.txt`.

## Featurization

`featurize.py` exports the estimated posterior mean as the latent descriptor of each molecule:

```sh
conda activate transformervae

python featurize.py \
  --name smiles_enum2can/train \
  --data smiles_enum2can/train \
  --weight training/results/smiles_enum2can \
  --voc_size 42
```

- Each molecule corresponds to one row in the latent CSV.
- Results are saved to `featurization/results/<name>/feature_mu.csv`.
- Run featurization separately for the training and query datasets before latent-space retrieval.

## Decode molecules from latent variables

`decode.py` decodes latent variables stored in CSV format into molecular strings:

```sh
conda activate transformervae

python decode.py \
  --name smiles_enum2can \
  --latent featurization/results/smiles_enum2can/feature_mu.csv \
  --weight training/results/smiles_enum2can \
  --voc_size 42 \
  --voc_file data/smiles_vocs.txt
```

Decoded strings are saved to `decoding/results/<name>/decoded_string.txt`.

## Latent-space interpolation

```sh
conda activate transformervae

python interpolate.py \
  --name smiles_enum2can \
  --data smiles_enum2can/test \
  --weight training/results/smiles_enum2can \
  --num 20000 \
  --mid_num 8 \
  --voc_size 42 \
  --voc_file data/smiles_vocs.txt
```

Results are saved to `interpolating/results/<name>/`, including `end1.txt`, `mid.txt`, and `end2.txt`.

## Translate strings to SMILES

SMILES output can be copied directly. SELFIES and Group SELFIES output must be converted with the corresponding decoder:

```sh
conda activate groupselfies

python translate.py generation/results/<name> --format smiles
python translate.py recon/results/<name> --format selfies
python translate.py interpolating/results/<name> --format group-selfies
```

## Evaluation

### Distribution-learning evaluation

First use `translate.py` to produce `generated_smiles.txt`, and then run the GuacaMol evaluation:

```sh
conda activate guacamol

python eval_gen.py <name> \
  --train_file data/guacamol_v1_train.smiles \
  --suite v2
```

### Reconstruction and interpolation

Convert the model output to SMILES before running:

```sh
conda activate transformervae

python eval_recon.py <name>
python eval_int.py <name>
```

### Latent-space retrieval

The reference construction and evaluation steps must use identical training and query SMILES, neighbor counts, and Morgan fingerprint parameters:

```sh
conda activate transformervae

python evaluate/build_reference.py \
  --train-smiles data/guacamol_v1_train.smiles \
  --query-smiles data/guacamol_v1_test_query.smiles \
  --output-dir evaluate/reference/guacamol_v1_test_query

python evaluate.py \
  --train-csv featurization/results/<name>/train/feature_mu.csv \
  --query-csv featurization/results/<name>/query/feature_mu.csv \
  --train-smiles data/guacamol_v1_train.smiles \
  --query-smiles data/guacamol_v1_test_query.smiles \
  --reference-dir evaluate/reference/guacamol_v1_test_query \
  --output-dir evaluate/results/<name>
```

- The training and query SMILES must match the row counts and row order of their respective latent CSV files.
- GPU FAISS is used by default. Add `--faiss-cpu` to force CPU execution.
- `metrics.json` stores metrics aggregated across queries.
- `query_metrics.csv` contains one row per query with the following columns:
  - `query_index`
  - `query_smiles`
  - `latent_mean_similarity`
  - `chemical_mean_similarity`
  - `latent_mean_near_distance`
  - `chemical_mean_near_distance`
  - `neighbor_alignment`
  - `querywise_wasserstein`

## Notes

- Most YAML paths are relative. Run commands from the repository root.
- The `voc_size`, `voc_file`, model dimensions, and checkpoint must match for a given model.
- SELFIES and Group SELFIES output must be translated back to SMILES before evaluation.
- `duplicate: ask` requests interactive confirmation when a result directory already exists.

## Acknowledgements

The original implementation was developed by Yasuhiro Yoshikai, Tadahaya Mizuno, Shumpei Nemoto, and Hiroyuki Kusuhara. This repository extends the original code to compare SMILES, SELFIES, and Group SELFIES
representations and to evaluate their learned latent spaces. The original copyright and MIT License notices are retained in [LICENSE](LICENSE) and [src/LICENSE](src/LICENSE).

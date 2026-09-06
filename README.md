# MolStringVAE-LatentSpace

Transformer variational autoencoders for comparing SMILES, SELFIES, and Group SELFIES molecular representations.

---

## Introduction

This repository provides the code used to preprocess molecular strings, train Transformer VAE models, generate and reconstruct molecules, extract latent representations, perform latent-space interpolation, and evaluate latent-space retrieval.

The project is derived from [TransformerVAE](https://github.com/mizuno-group/TransformerVAE) and integrates [Group SELFIES](https://github.com/aspuru-guzik-group/group-selfies) and [GuacaMol](https://github.com/BenevolentAI/guacamol).

The GuacaMol pretrained model weights are available on [Zenodo](https://doi.org/10.5281/zenodo.22304527).

---

## How to use

For environment setup, data preparation, training, inference, and evaluation, see [usage.md](usage.md).

The main workflow is:

1. Prepare the Conda environments.
2. Download the GuacaMol data and pretrained weights.
3. Preprocess molecular strings.
4. Train a model or load pretrained weights.
5. Run generation, reconstruction, featurization, decoding, or interpolation.
6. Evaluate generated molecules and latent-space neighborhoods.

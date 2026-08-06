"""Persistence helpers for preprocessing output."""

import os
import pickle

import numpy as np


def save_outputs(result_dir, outputs, save_empty=False):
    written = []
    for name, values in outputs.items():
        if not values and not save_empty:
            continue
        path = os.path.join(result_dir, f"{name}.pkl")
        with open(path, "wb") as file:
            pickle.dump(np.array(values, dtype=object), file)
        written.append(path)
    return written


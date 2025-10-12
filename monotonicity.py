#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import spearmanr, kendalltau

# mute RDKit warnings (e.g., SMILES parse errors)
RDLogger.DisableLog('rdApp.*')

def mol_from_smiles(s):
    return Chem.MolFromSmiles(s) if s else None

def fp_morgan_bit(mol, radius=2, nBits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)

def fp_morgan_count(mol, radius=2):
    # Count-based fingerprint (SparseIntVect). Tanimoto can work on this too.
    return AllChem.GetMorganFingerprint(mol, radius=radius)

def sim(a, b, metric="tanimoto"):
    if metric == "tanimoto":
        return DataStructs.TanimotoSimilarity(a, b)
    elif metric == "dice":
        return DataStructs.DiceSimilarity(a, b)
    elif metric == "cosine":
        return DataStructs.CosineSimilarity(a, b)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def count_viol_inc(xs, tol=0.0):
    return sum(xs[i+1] < xs[i] - tol for i in range(len(xs)-1))

def count_viol_dec(xs, tol=0.0):
    return sum(xs[i+1] > xs[i] + tol for i in range(len(xs)-1))

def nanmean(x):
    arr = np.array(x, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--root", default="decoding/results")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--nBits", type=int, default=2048)
    ap.add_argument("--use-counts", action="store_true", help="use count fingerprints instead of bits")
    ap.add_argument("--metric", default="tanimoto", choices=["tanimoto","dice","cosine"])
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", default="similarity_summary.txt", help="output filename (saved under root/data_dir)")
    ap.add_argument("--detail", action="store_true", help="append per-step details into the same file")
    args = ap.parse_args()

    root = os.path.join(args.root, args.data_dir)
    f_e1  = os.path.join(root, "smiles_end1.txt")
    f_e2  = os.path.join(root, "smiles_end2.txt")
    f_mid = os.path.join(root, "smiles_mid.txt")

    # If input missing, just fail silently with an error code (no console noise)
    if not all(os.path.isfile(p) for p in [f_e1, f_e2, f_mid]):
        sys.exit(2)

    with open(f_e1) as f: end1 = [ln.strip() for ln in f]
    with open(f_e2) as f: end2 = [ln.strip() for ln in f]
    with open(f_mid) as f: mids = [ln.strip() for ln in f]

    pairs = len(end1)
    if len(end2) != pairs or len(mids) % pairs != 0:
        sys.exit(3)
    mid_num = len(mids) // pairs

    # FP function & description
    if args.use_counts:
        fp_fn = lambda m: fp_morgan_count(m, radius=args.radius)
        fp_desc = f"ECFP{args.radius} (count), metric={args.metric}"
    else:
        fp_fn = lambda m: fp_morgan_bit(m, radius=args.radius, nBits=args.nBits)
        fp_desc = f"ECFP{args.radius} ({args.nBits}-bit), metric={args.metric}"

    # Parse mols
    mol_e1 = [mol_from_smiles(s) for s in end1]
    mol_e2 = [mol_from_smiles(s) for s in end2]
    mol_mid= [mol_from_smiles(s) for s in mids]

    # Invalid counts
    inv_e1 = sum(m is None for m in mol_e1)
    inv_e2 = sum(m is None for m in mol_e2)
    inv_mid= sum(m is None for m in mol_mid)

    # Fingerprints
    fp_e1  = [fp_fn(m) if m is not None else None for m in mol_e1]
    fp_e2  = [fp_fn(m) if m is not None else None for m in mol_e2]
    fp_mid = [fp_fn(m) if m is not None else None for m in mol_mid]

    # Per-pair aggregates
    all_score_dec, all_score_inc = [], []
    rho_to_start, rho_to_end = [], []
    tau_to_start, tau_to_end = [], []  # NEW: Kendall tau
    ends_sim = []

    # Prepare output path
    out_path = os.path.join(root, args.out)
    with open(out_path, "w") as fout:
        # Header / setup
        fout.write("=== Similarity / Monotonicity Report ===\n")
        fout.write(f"root={args.root}\n")
        fout.write(f"data_dir={args.data_dir}\n")
        fout.write(f"pairs={pairs}\n")
        fout.write(f"mid_num={mid_num}\n")
        fout.write(f"fingerprint={fp_desc}\n")
        fout.write(f"tol={args.tol}\n")
        fout.write(f"invalid_counts: end1={inv_e1}/{pairs}, end2={inv_e2}/{pairs}, mid={inv_mid}/{len(mids)}\n")
        fout.write("\n")

        if args.detail:
            fout.write("# pair\tj\tt_rel\tS(mid,e1)\tS(mid,e2)\tviol_start_dec\tviol_end_inc\tS(e1,e2)\n")

        # Iterate pairs
        for p in range(pairs):
            fp1, fp2 = fp_e1[p], fp_e2[p]
            if (fp1 is None) or (fp2 is None):
                ends_sim.append(np.nan)
                if args.detail:
                    for j in range(mid_num):
                        t_rel = (j+1)/(mid_num+1)
                        fout.write(f"{p}\t{j}\t{t_rel:.6f}\tNaN\tNaN\t0\t0\tNaN\n")
                continue

            s12 = sim(fp1, fp2, args.metric)
            ends_sim.append(s12)

            block = fp_mid[p*mid_num:(p+1)*mid_num]
            sim_to_start = []
            sim_to_end   = []
            for fpm in block:
                if fpm is None:
                    sim_to_start.append(np.nan)
                    sim_to_end.append(np.nan)
                else:
                    sim_to_start.append(sim(fpm, fp1, args.metric))
                    sim_to_end.append(sim(fpm, fp2, args.metric))

            # valid mask
            mask_valid = [not (np.isnan(a) or np.isnan(b)) for a,b in zip(sim_to_start, sim_to_end)]
            sts = [v for v,m in zip(sim_to_start, mask_valid) if m]
            ste = [v for v,m in zip(sim_to_end,   mask_valid) if m]

            if len(sts) >= 2:
                v_dec = count_viol_dec(sts, tol=args.tol)
                v_inc = count_viol_inc(ste, tol=args.tol)
                score_dec = 1.0 - v_dec / (len(sts)-1)
                score_inc = 1.0 - v_inc / (len(ste)-1)
                all_score_dec.append(score_dec)
                all_score_inc.append(score_inc)

                t_valid = [ (j+1)/(mid_num+1) for j,m in enumerate(mask_valid) if m ]
                # Spearman
                r1, _ = spearmanr(t_valid, sts)
                r2, _ = spearmanr(t_valid, ste)
                rho_to_start.append(float(r1) if np.isfinite(r1) else np.nan)  # ideal ≈ -1
                rho_to_end.append(float(r2) if np.isfinite(r2) else np.nan)    # ideal ≈ +1
                # Kendall (NEW)
                k1, _ = kendalltau(t_valid, sts)
                k2, _ = kendalltau(t_valid, ste)
                tau_to_start.append(float(k1) if np.isfinite(k1) else np.nan)  # ideal ≈ -1
                tau_to_end.append(float(k2) if np.isfinite(k2) else np.nan)    # ideal ≈ +1
            else:
                all_score_dec.append(np.nan)
                all_score_inc.append(np.nan)
                rho_to_start.append(np.nan)
                rho_to_end.append(np.nan)
                tau_to_start.append(np.nan)
                tau_to_end.append(np.nan)

            if args.detail:
                for j in range(mid_num):
                    t_rel = (j+1)/(mid_num+1)
                    a = sim_to_start[j] if j < len(sim_to_start) else np.nan
                    b = sim_to_end[j]   if j < len(sim_to_end)   else np.nan
                    viol_dec = int(j < mid_num-1 and np.isfinite(a) and np.isfinite(sim_to_start[j+1]) and sim_to_start[j+1] > a + args.tol)
                    viol_inc = int(j < mid_num-1 and np.isfinite(b) and np.isfinite(sim_to_end[j+1])   and sim_to_end[j+1]   < b - args.tol)
                    fout.write(f"{p}\t{j}\t{t_rel:.6f}\t{a if np.isfinite(a) else 'NaN'}\t{b if np.isfinite(b) else 'NaN'}\t{viol_dec}\t{viol_inc}\t{s12:.6f}\n")

        # Summary
        fout.write("\n=== Summary ===\n")
        fout.write(f"ends similarity mean (S(e1,e2)): {nanmean(ends_sim):.6f}\n")
        fout.write(f"monotonicity start↓ (mean score): {nanmean(all_score_dec):.6f}\n")
        fout.write(f"monotonicity end↑   (mean score): {nanmean(all_score_inc):.6f}\n")
        fout.write(f"Spearman rho vs t  (to start):    {nanmean(rho_to_start):.6f}  # ideal ≈ -1\n")
        fout.write(f"Spearman rho vs t  (to end):      {nanmean(rho_to_end):.6f}    # ideal ≈ +1\n")
        # NEW: Kendall
        fout.write(f"Kendall tau vs t   (to start):    {nanmean(tau_to_start):.6f}  # ideal ≈ -1\n")
        fout.write(f"Kendall tau vs t   (to end):      {nanmean(tau_to_end):.6f}    # ideal ≈ +1\n")

    # no prints — single output file created
    # exit code 0 means success
    return 0

if __name__ == "__main__":
    sys.exit(main())

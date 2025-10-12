#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SA-score 連續性檢查（精簡版）

輸出統計指標：
- SA(e1) mean
- SA(e2) mean
- |SA(e2) - SA(e1)| mean
- monotonicity best-of-two (mean)
- monotonicity by endpoint direction (mean)
- |Spearman rho| vs t (mean)
- |Kendall tau| vs t (mean)
"""

import os, sys, argparse
import numpy as np
from rdkit import Chem, RDLogger
from scipy.stats import spearmanr, kendalltau

RDLogger.DisableLog('rdApp.*')

def DIE(code: int, msg: str):
    print(msg, file=sys.stderr)
    sys.exit(code)

# ---- 匯入 sascorer ----
sascorer = None
try:
    import sascorer as _tmp; sascorer = _tmp
except Exception:
    try:
        from rdkit.Chem import sascore as _tmp; sascorer = _tmp
    except Exception:
        try:
            from rdkit.Chem import SA_Score as _tmp; sascorer = _tmp
        except Exception:
            SITE = "/home/jingchun/mambaforge/envs/groupselfies/lib/python3.8/site-packages"
            cand = None
            for fname in ("sascore.py", "SA_Score.py"):
                p = os.path.join(SITE, "rdkit", "Chem", fname)
                if os.path.isfile(p):
                    cand = p; break
            if cand is None:
                DIE(5, "Cannot find sascorer/SA_Score in site-packages")
            import importlib.util
            spec = importlib.util.spec_from_file_location("_sa_score_fallback", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            sascorer = mod

# ---- 工具函式 ----
def mol_from_smiles(s: str):
    return Chem.MolFromSmiles(s) if s else None

def sa_of_mol(mol) -> float:
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return float('nan')

def count_viol_inc(xs, tol=0.0) -> int:
    return sum(xs[i+1] < xs[i] - tol for i in range(len(xs)-1))

def count_viol_dec(xs, tol=0.0) -> int:
    return sum(xs[i+1] > xs[i] + tol for i in range(len(xs)-1))

def nanmean(x) -> float:
    arr = np.array(x, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")

# ---- 主流程 ----
def main():
    ap = argparse.ArgumentParser(description="Check SA score continuity (summary only).")
    ap.add_argument("data_dir")
    ap.add_argument("--root", default="decoding/results")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--dir-tol", type=float, default=0.05)
    ap.add_argument("--out", default="sa_continuity_summary.txt")
    args = ap.parse_args()

    root = os.path.join(args.root, args.data_dir)
    f_e1  = os.path.join(root, "smiles_end1.txt")
    f_e2  = os.path.join(root, "smiles_end2.txt")
    f_mid = os.path.join(root, "smiles_mid.txt")

    if not all(os.path.isfile(p) for p in [f_e1, f_e2, f_mid]):
        missing = [p for p in [f_e1, f_e2, f_mid] if not os.path.isfile(p)]
        DIE(2, f"Missing input files: {missing}")

    with open(f_e1) as f: end1 = [ln.strip() for ln in f]
    with open(f_e2) as f: end2 = [ln.strip() for ln in f]
    with open(f_mid) as f: mids = [ln.strip() for ln in f]

    pairs = len(end1)
    if len(end2) != pairs or len(mids) % pairs != 0:
        DIE(3, f"Line count mismatch: end1={len(end1)}, end2={len(end2)}, mids={len(mids)}")

    mid_num = len(mids) // pairs

    mol_e1 = [mol_from_smiles(s) for s in end1]
    mol_e2 = [mol_from_smiles(s) for s in end2]
    mol_mid= [mol_from_smiles(s) for s in mids]

    sa_e1 = [sa_of_mol(m) for m in mol_e1]
    sa_e2 = [sa_of_mol(m) for m in mol_e2]

    mono_best_all, mono_dir_all = [], []
    abs_rho_all, abs_tau_all = [], []
    sa_gap = []

    for p in range(pairs):
        m1, m2 = sa_e1[p], sa_e2[p]
        gap = (m2 - m1) if np.isfinite(m1) and np.isfinite(m2) else float('nan')
        sa_gap.append(gap)

        if np.isfinite(gap):
            if gap > args.dir_tol: dir_label = 'inc'
            elif gap < -args.dir_tol: dir_label = 'dec'
            else: dir_label = 'unk'
        else:
            dir_label = 'unk'

        block_mols = mol_mid[p*mid_num:(p+1)*mid_num]
        sa_list = [sa_of_mol(m) for m in block_mols]
        sa_valid = [v for v in sa_list if np.isfinite(v)]
        t_valid  = [(j+1)/(mid_num+1) for j,v in enumerate(sa_list) if np.isfinite(v)]

        if len(sa_valid) >= 2:
            v_inc = count_viol_inc(sa_valid, tol=args.tol)
            v_dec = count_viol_dec(sa_valid, tol=args.tol)
            mono_inc = 1.0 - v_inc / (len(sa_valid)-1)
            mono_dec = 1.0 - v_dec / (len(sa_valid)-1)
            mono_best = max(mono_inc, mono_dec)
            mono_best_all.append(mono_best)

            if dir_label == 'inc': mono_dir = mono_inc
            elif dir_label == 'dec': mono_dir = mono_dec
            else: mono_dir = float('nan')
            mono_dir_all.append(mono_dir)

            r, _ = spearmanr(t_valid, sa_valid)
            k, _ = kendalltau(t_valid, sa_valid)
            abs_rho_all.append(abs(r) if np.isfinite(r) else float('nan'))
            abs_tau_all.append(abs(k) if np.isfinite(k) else float('nan'))
        else:
            mono_best_all.append(float('nan'))
            mono_dir_all.append(float('nan'))
            abs_rho_all.append(float('nan'))
            abs_tau_all.append(float('nan'))

    out_path = os.path.join(root, args.out)
    with open(out_path, "w") as fout:
        fout.write("=== SA-score Continuity Report (summary only) ===\n")
        fout.write(f"SA(e1) mean: {nanmean(sa_e1):.6f}\n")
        fout.write(f"SA(e2) mean: {nanmean(sa_e2):.6f}\n")
        fout.write(f"|SA(e2) - SA(e1)| mean: {nanmean([abs(g) for g in sa_gap]):.6f}\n")
        fout.write(f"monotonicity best-of-two (mean): {nanmean(mono_best_all):.6f}\n")
        fout.write(f"monotonicity by endpoint direction (mean): {nanmean(mono_dir_all):.6f}\n")
        fout.write(f"|Spearman rho| vs t (mean): {nanmean(abs_rho_all):.6f}\n")
        fout.write(f"|Kendall tau| vs t (mean): {nanmean(abs_tau_all):.6f}\n")

    print(f"[info] wrote report to: {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

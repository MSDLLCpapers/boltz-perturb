"""Check ligand validity (PoseBusters) and compute per-atom RMSF across conformers.

Usage:
    python validity_and_rmsf.py \
        --input_df /path/to/ost_results.csv \
        --out_dir /path/to/output \
        --n_workers 8 \
        --chunk_by struc_id \
        --suffix _my_run \
        --use_aligned_sdf
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from posebusters import PoseBusters
from tqdm import tqdm


def calculate_rmsf(sdf_files):
    """Calculate per-atom RMSF from a list of SDF files (pre-aligned coordinates)."""
    all_coords = []
    for sdf_file in sorted(sdf_files):
        mol = Chem.MolFromMolFile(str(sdf_file), sanitize=True, removeHs=True)
        if mol is None:
            print(f"  Error: Could not read molecule from {sdf_file}")
            continue
        conf = mol.GetConformer()
        coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        all_coords.append(coords)

    if len(all_coords) < 2:
        print(f"  Skipping: not enough valid conformers ({len(all_coords)})")
        return None

    n_atoms_list = [c.shape[0] for c in all_coords]
    if len(set(n_atoms_list)) > 1:
        print(f"  Skipping: atom count mismatch: {set(n_atoms_list)}")
        return None

    all_coords_arr = np.stack(all_coords, axis=0)
    mean_coords = all_coords_arr.mean(axis=0)
    deviations = all_coords_arr - mean_coords[None, :, :]
    per_atom_rmsf = np.sqrt((deviations ** 2).sum(axis=2).mean(axis=0))
    return per_atom_rmsf


def calculate_rmsf_rdkit(sdf_files):
    """Calculate per-atom RMSF using RDKit conformer alignment."""
    mols = []
    for f in sdf_files:
        suppl = Chem.SDMolSupplier(f, removeHs=False)
        for mol in suppl:
            if mol is not None:
                mols.append(mol)
                break

    if len(mols) < 2:
        print(f"  [WARN] Only {len(mols)} valid mol(s) loaded — skipping.")
        return None

    ref_mol = Chem.RWMol(mols[0])
    for mol in mols[1:]:
        try:
            conf = mol.GetConformer()
            ref_mol.AddConformer(conf, assignId=True)
        except Exception as e:
            print(f"  [WARN] Skipping a conformer: {e}")

    if ref_mol.GetNumConformers() < 2:
        print(f"  [WARN] Not enough conformers after bundling — skipping.")
        return None

    rdMolAlign.AlignMolConformers(ref_mol)

    all_pos = np.array([conf.GetPositions() for conf in ref_mol.GetConformers()])
    mean_pos = all_pos.mean(axis=0)
    deviations = all_pos - mean_pos[np.newaxis, :, :]
    sq_dev = np.sum(deviations ** 2, axis=2)
    msf = sq_dev.mean(axis=0)
    rmsf = np.sqrt(msf)
    return rmsf


def main():
    parser = argparse.ArgumentParser(description="Check ligand validity and compute RMSF.")
    parser.add_argument("--input_df", type=str, required=True, help="Input CSV from combine_ost_results.py")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--n_workers", type=int, default=8, help="Number of PoseBusters workers")
    parser.add_argument("--top10", action="store_true", help="Only use top 10 models per structure")
    parser.add_argument("--chunk_by", type=str, default="struc_id",
                        help="Comma-separated columns to chunk validity by (default: struc_id)")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to add to output files (e.g. '_top10')")
    parser.add_argument("--use_aligned_sdf", action="store_true",
                        help="Use aligned_sdf_path for RMSF instead of sdf_file_path")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_df, quotechar='"')
    print(f"Loaded dataframe: {df.shape}")

    if args.top10:
        df = df[df["model_num"] < 10].reset_index(drop=True)
        print(f"After top10 filter: {df.shape}")

    suffix = f"_{args.suffix}" if args.suffix else ""

    if args.use_aligned_sdf:
        sdf_col = "aligned_sdf_path"
        print("Using aligned_sdf_path for RMSF calculations.")
    else:
        sdf_col = "sdf_file_path"
        print("Using sdf_file_path for RMSF calculations.")

    # ── Step 1: Validity check with PoseBusters ──
    buster = PoseBusters(config="mol", max_workers=args.n_workers)
    print(f"Running validity checks with {args.n_workers} workers...")
    validity_df = buster.bust(list(df[sdf_col]), None, None)
    validity_df = validity_df.reset_index()

    validity_df.to_csv(out_dir / f"_00_pb_valid_result{suffix}.csv", index=True)

    bool_cols = validity_df.select_dtypes(include=bool).columns.tolist()
    validity_df["pb_valid"] = validity_df[bool_cols].all(axis=1).astype(int)
    validity_df["pb_problems"] = validity_df[bool_cols].apply(
        lambda row: "; ".join([col for col in bool_cols if not row[col]]) or None, axis=1
    )
    validity_df.to_csv(out_dir / f"_00_validity_check_df{suffix}.csv", index=False)

    df_valid = pd.merge(df, validity_df[["file", "pb_valid", "pb_problems"]],
                        left_on=sdf_col, right_on="file", how="left")
    df_valid.to_csv(out_dir / f"_01_validity_check_df{suffix}.csv", index=False)

    print(f"Validity summary: {df_valid['pb_valid'].sum()} valid out of {len(df_valid)} total")

    # ── Step 2: Combined validity filter ──
    df_valid["valid"] = (
        (df_valid["pb_valid"] == 1) &
        (df_valid["ost_valid"] == 1)
    ).astype(int)

    df_valid.to_csv(out_dir / f"_01_ost_valid_pb_valid{suffix}.csv", index=False)
    df_clean = df_valid[df_valid["valid"] == 1].reset_index(drop=True)

    # ── Step 3: RMSF on valid conformers ──
    print(f"Calculating RMSF for {len(df_clean)} valid conformers...")

    per_atom_rmsf_list = []
    df_rmsf_summary = []

    for (struc_id, noise_scale), grp_df in tqdm(
        df_clean.groupby(["struc_id", "noise_scale"]), desc="RMSF"
    ):
        sdf_files = grp_df[sdf_col].tolist()

        if args.use_aligned_sdf:
            per_atom_rmsf = calculate_rmsf(sdf_files)
        else:
            per_atom_rmsf = calculate_rmsf_rdkit(sdf_files)

        if per_atom_rmsf is None:
            continue

        print(f"  {struc_id} | noise={noise_scale}: mean={per_atom_rmsf.mean():.3f}, "
              f"median={np.median(per_atom_rmsf):.3f}, max={per_atom_rmsf.max():.3f}")

        for atom_idx, atom_rmsf in enumerate(per_atom_rmsf):
            per_atom_rmsf_list.append({
                "struc_id": struc_id,
                "noise_scale": noise_scale,
                "atom_idx": atom_idx,
                "rmsf": atom_rmsf,
            })

        df_rmsf_summary.append({
            "struc_id": struc_id,
            "noise_scale": noise_scale,
            "num_conformers": len(sdf_files),
            "num_atoms": len(per_atom_rmsf),
            "mean_rmsf": per_atom_rmsf.mean(),
            "max_rmsf": per_atom_rmsf.max(),
        })

    pd.DataFrame(per_atom_rmsf_list).to_csv(out_dir / f"_02_per_atom_rmsf{suffix}.csv", index=True)
    pd.DataFrame(df_rmsf_summary).to_csv(out_dir / f"_03_rmsf_summary{suffix}.csv", index=True)
    print(f"Saved RMSF results to {out_dir}")


if __name__ == "__main__":
    main()

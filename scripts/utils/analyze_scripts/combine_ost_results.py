"""Combine OpenStructure comparison results into a single CSV.

Usage:
    python combine_ost_results.py \
        --mobile_result /path/to/predictions \
        --method s_noise \
        --model_type boltz2 \
        --noise_type default \
        --ligand_chain B \
        --pattern "*.cif" \
        --num_workers 4
"""

import re
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse


def extract_digits(s: str) -> str:
    """Extract numeric part from strings like 'noise_scale0.7', 's42', 't300'."""
    match = re.search(r"[\d.]+", s)
    return match.group() if match else s


def process_pdb(mobile_pdb: Path, method, model_type, noise_type, ligand_chain="B") -> dict:
    mobile_pdb = Path(mobile_pdb)
    ost_result = mobile_pdb.parent / "ost_comparison"
    model_sdf = mobile_pdb.parent / "unaligned_ligand_sdf" / f"{mobile_pdb.stem}_chain_{ligand_chain}.sdf"

    with open(ost_result / "ligand" / f"{mobile_pdb.stem}_ligand.json") as f:
        data_ligand = json.load(f)

    condition = mobile_pdb.parts[-5].split("_")
    noise_scale = "0"
    for cond in condition:
        if cond.startswith("n"):
            noise_scale = extract_digits(cond)

    conf_file = mobile_pdb.parent / f"confidence_{mobile_pdb.stem}.json"
    with open(conf_file) as f:
        conf_data = json.load(f)

    row = {
        "struc_id": mobile_pdb.parts[-2],
        "noise_scale": noise_scale,
        "noise_type": noise_type,
        "method": method,
        "seed": extract_digits(condition[1]) if len(condition) > 1 else "",
        "temp": extract_digits(condition[0]),
        "model_num": re.search(r"model_(\d+)$", mobile_pdb.stem).group(1),
        "model_type": model_type,
        "confidence_score": conf_data.get("confidence_score"),
        "iptm": conf_data.get("iptm"),
        "ligand_iptm": conf_data.get("ligand_iptm"),
        "file_path": str(mobile_pdb),
        "sdf_file_path": str(model_sdf),
    }

    if method == "vanilla":
        row["noise_scale"] = "0"
        row["noise_type"] = "vanilla"

    if not data_ligand.get("rmsd"):
        row["ligand_rmsd"] = -999
        row["ost_valid"] = 0
    else:
        rmsd_dict = data_ligand["rmsd"]
        sdf_key = str(model_sdf)
        if sdf_key not in rmsd_dict:
            row["ligand_rmsd"] = -999
            row["ost_valid"] = 0
        else:
            row["ligand_rmsd"] = rmsd_dict[sdf_key]["rmsd"]
            row["ost_valid"] = 1

    return row


def main():
    parser = argparse.ArgumentParser(description="Combine OST comparison results into a CSV.")
    parser.add_argument("--mobile_result", type=Path, required=True, help="Directory containing predicted structures")
    parser.add_argument("--method", type=str, required=True, help="Method label (e.g. 'vanilla', 's_noise', 'token_bias')")
    parser.add_argument("--model_type", type=str, default="boltz2", help="Model type label")
    parser.add_argument("--noise_type", type=str, default="default", help="Noise type label")
    parser.add_argument("--ligand_chain", type=str, default="B", help="Ligand chain ID")
    parser.add_argument("--pattern", type=str, default="*.cif", help="Glob pattern for prediction files")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path (default: mobile_result/ost_results.csv)")
    args = parser.parse_args()

    mobile_result = Path(args.mobile_result)
    mobile_pdbs = list(mobile_result.rglob(args.pattern))
    print(f"Found {len(mobile_pdbs)} files under {mobile_result}")

    rows = [None] * len(mobile_pdbs)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_pdb, p, args.method, args.model_type, args.noise_type, args.ligand_chain): i
            for i, p in enumerate(mobile_pdbs)
        }
        for future in tqdm(as_completed(futures), total=len(mobile_pdbs)):
            idx = futures[future]
            try:
                rows[idx] = future.result()
            except Exception as e:
                print(f"[ERROR] {mobile_pdbs[idx]}: {e}")

    rmsd_df = pd.DataFrame([r for r in rows if r is not None])
    out_path = args.output or (mobile_result / f"ost_results_{args.method}.csv")
    rmsd_df.to_csv(out_path, index=False)
    print(f"Saved {len(rmsd_df)} rows → {out_path}")


if __name__ == "__main__":
    main()

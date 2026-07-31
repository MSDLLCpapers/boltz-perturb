"""Align predicted structures to a reference and extract ligand SDFs.

Usage:
    python align_and_save_predictions.py \
        --mobile_result /path/to/predictions \
        --ref_pdb /path/to/reference.cif \
        --ref_prot_chain A \
        --mobile_prot_chain A \
        --ligand_chains B \
        --pattern "*/predictions/*/*.cif" \
        --num_workers 8
"""

from pathlib import Path
import pymol
import subprocess
import os
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


def align_and_save_struct(
    ref_pdb: Path,
    mobile_pdb: Path,
    aligned_pdb_path: Path,
    ref_prot_chain: str,
    mobile_prot_chain: str,
    save_pymol=True,
):
    pymol.cmd.reinitialize()
    pymol.cmd.set("cif_use_auth", "off")
    pymol.cmd.load(str(ref_pdb), "ref")
    pymol.cmd.load(str(mobile_pdb), "mobile")
    mobile_sel = f"mobile and chain {mobile_prot_chain}"
    target_sel = f"ref and chain {ref_prot_chain}"
    rmsd_val = -999

    try:
        pymol.cmd.align(mobile_sel, target_sel, cycles=10)
        rmsd_val = pymol.cmd.align(mobile_sel, target_sel, cycles=0, transform=0)[0]
        if save_pymol:
            pymol.cmd.save(str(aligned_pdb_path), "mobile")
    except Exception as e:
        print(f"Pymol align error: {e}")

    pymol.cmd.delete("all")
    return rmsd_val


def call_ligand_convert(input_pdb, chain_id, output_file):
    """Convert ligand chain from structure file to SDF.

    Adapt the command below to your local ligand extraction tool.
    """
    try:
        command = [
            "python", "ligand_pdb_to_sdf.py",
            str(input_pdb), chain_id, str(output_file)
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            print(f"Error converting {input_pdb}: {result.stderr}")
            return False
    except Exception as e:
        print(f"Subprocess error: {e}")
        return False


def convert_ligand_pdb_sdf(input_pdb: Path, output_dir: Path, ligand_chains):
    for ligand_chain in ligand_chains:
        sdf_file = output_dir / f"{input_pdb.stem}_chain_{ligand_chain}.sdf"
        call_ligand_convert(input_pdb, ligand_chain, sdf_file)


def process_single_pdb(mobile_pdb, ref_pdb, ref_prot_chain="A", mobile_prot_chain="A", ligand_chains=["B"]):
    os.makedirs(mobile_pdb.parent / "pymol_aligned_predictions", exist_ok=True)
    aligned_pdb_path = mobile_pdb.parent / "pymol_aligned_predictions" / f"{mobile_pdb.stem}_aligned.pdb"

    align_and_save_struct(
        ref_pdb=ref_pdb,
        mobile_pdb=mobile_pdb,
        aligned_pdb_path=aligned_pdb_path,
        ref_prot_chain=ref_prot_chain,
        mobile_prot_chain=mobile_prot_chain,
        save_pymol=True,
    )

    output_dir = aligned_pdb_path.parent / "aligned_ligand_sdf"
    output_dir.mkdir(exist_ok=True)
    convert_ligand_pdb_sdf(aligned_pdb_path, output_dir, ligand_chains)

    output_dir_unaligned = mobile_pdb.parent / "unaligned_ligand_sdf"
    output_dir_unaligned.mkdir(exist_ok=True)
    convert_ligand_pdb_sdf(mobile_pdb, output_dir_unaligned, ligand_chains)


def main():
    parser = argparse.ArgumentParser(description="Align predictions to reference and extract ligand SDFs.")
    parser.add_argument("--mobile_result", type=Path, required=True, help="Directory containing predicted structures")
    parser.add_argument("--ref_pdb", type=Path, required=True, help="Reference structure file")
    parser.add_argument("--ref_prot_chain", type=str, default="A", help="Reference protein chain ID")
    parser.add_argument("--mobile_prot_chain", type=str, default="A", help="Mobile protein chain ID")
    parser.add_argument("--ligand_chains", type=str, default="B", help="Comma-separated ligand chain IDs")
    parser.add_argument("--pattern", type=str, default="*/predictions/*/*.cif", help="Glob pattern for prediction files")
    parser.add_argument("--num_workers", type=int, default=cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    mobile_result = Path(args.mobile_result)
    ref_pdb = Path(args.ref_pdb)
    ligand_chains = args.ligand_chains.split(",")

    mobile_pdbs = list(mobile_result.rglob(args.pattern))
    print(f"Found {len(mobile_pdbs)} files. Running with {args.num_workers} workers...")

    worker_fn = partial(
        process_single_pdb,
        ref_pdb=ref_pdb,
        ref_prot_chain=args.ref_prot_chain,
        mobile_prot_chain=args.mobile_prot_chain,
        ligand_chains=ligand_chains,
    )

    with Pool(processes=args.num_workers) as pool:
        list(tqdm(
            pool.imap_unordered(worker_fn, mobile_pdbs),
            total=len(mobile_pdbs),
            desc="Aligning structures",
        ))


if __name__ == "__main__":
    main()

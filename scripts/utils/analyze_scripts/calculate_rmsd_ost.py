"""Calculate protein and ligand RMSD using OpenStructure (ost).

Usage:
    python calculate_rmsd_ost.py \
        --mobile_result /path/to/predictions \
        --ref_pdb /path/to/reference.cif \
        --ref_sdf /path/to/reference_ligand.sdf \
        --ref_chainID A \
        --model_chainID A \
        --ligand_chainID B \
        --pattern "predictions/*/*.cif" \
        --num_workers 8
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import os
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenStructureRunner:
    """Handle OpenStructure (ost) command execution."""

    def __init__(self, ost_setup_commands=None):
        """Initialize with optional environment setup commands.
            user's choice either module load..etc..
        """
        self.ost_setup_commands = ost_setup_commands or []

    def run_protein_comparison(
        self,
        model_file: Path,
        reference_file: Path,
        outdir: Path,
        name: str,
        ref_chainID: Optional[str] = None,
        model_chainID: Optional[str] = None,
        rewrite: bool = False,
    ) -> Optional[subprocess.CompletedProcess]:
        outdir.mkdir(parents=True, exist_ok=True)
        protein_out_path = outdir / f"{name}.json"

        if protein_out_path.exists() and not rewrite:
            return None

        command = (
            f"ost compare-structures "
            f"-m {model_file} "
            f"-r {reference_file} "
            f"--fault-tolerant "
            f"--min-pep-length 4 "
            f"--min-nuc-length 4 "
        )

        if ref_chainID is not None and model_chainID is not None:
            command += f"-c {ref_chainID}:{model_chainID} "

        command += (
            f"-o {protein_out_path} "
            f"--lddt --bb-lddt --qs-score --dockq --ics --ips "
            f"--rigid-scores --patch-scores --tm-score "
        )

        try:
            full_command = "; ".join(self.ost_setup_commands + [command])
            return subprocess.run(
                full_command,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                executable="/bin/bash",
                timeout=300,
            )
        except subprocess.TimeoutExpired as e:
            logger.error(f"Protein comparison timed out for {name}: {e}")
            return None

    def run_ligand_comparison(
        self,
        model_file: Path,
        reference_file: Path,
        outdir: Path,
        name: str,
        refsdf: Optional[Path] = None,
        model_sdf: Optional[Path] = None,
        rewrite: bool = False,
    ) -> Optional[subprocess.CompletedProcess]:
        outdir.mkdir(parents=True, exist_ok=True)
        ligand_out_path = outdir / f"{name}_ligand.json"

        if ligand_out_path.exists() and not rewrite:
            return None

        command = (
            f"ost compare-ligand-structures "
            f"-m {model_file} "
            f"-r {reference_file} "
            f"--fault-tolerant "
            f"--lddt-pli --rmsd --substructure-match "
            f"-o {ligand_out_path}"
        )

        if refsdf is not None:
            command += f" -rl {refsdf}"
        if model_sdf is not None:
            command += f" -ml {model_sdf}"

        try:
            full_command = "; ".join(self.ost_setup_commands + [command])
            return subprocess.run(
                full_command,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                executable="/bin/bash",
                timeout=300,
            )
        except subprocess.TimeoutExpired as e:
            logger.error(f"Ligand comparison timed out for {name}: {e}")
            return None


def process_single_pdb(mobile_pdb, ref_pdb, ref_sdf, ref_chainID="A", model_chainID="A", ligand_chainID="B"):
    mobile_pdb = Path(mobile_pdb)
    ref_pdb = Path(ref_pdb)
    ref_sdf = Path(ref_sdf)

    runner = OpenStructureRunner()
    os.makedirs(mobile_pdb.parent / "ost_comparison", exist_ok=True)

    runner.run_protein_comparison(
        model_file=mobile_pdb,
        reference_file=ref_pdb,
        ref_chainID=ref_chainID,
        model_chainID=model_chainID,
        outdir=mobile_pdb.parent / "ost_comparison" / "protein",
        name=mobile_pdb.stem,
        rewrite=True,
    )

    model_sdf = mobile_pdb.parent / "unaligned_ligand_sdf" / f"{mobile_pdb.stem}_chain_{ligand_chainID}.sdf"
    if not model_sdf.exists():
        print(f"  [WARNING] model SDF not found, skipping ligand comparison: {model_sdf}")
        return

    runner.run_ligand_comparison(
        model_file=mobile_pdb,
        reference_file=ref_pdb,
        outdir=mobile_pdb.parent / "ost_comparison" / "ligand",
        name=mobile_pdb.stem,
        refsdf=ref_sdf,
        model_sdf=model_sdf,
        rewrite=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Calculate protein/ligand RMSD using OpenStructure.")
    parser.add_argument("--mobile_result", type=Path, required=True, help="Directory containing predicted structures")
    parser.add_argument("--ref_pdb", type=Path, required=True, help="Reference structure file")
    parser.add_argument("--ref_sdf", type=Path, required=True, help="Reference ligand SDF file")
    parser.add_argument("--ref_chainID", type=str, default="A", help="Reference protein chain ID")
    parser.add_argument("--model_chainID", type=str, default="A", help="Model protein chain ID")
    parser.add_argument("--ligand_chainID", type=str, default="B", help="Ligand chain ID")
    parser.add_argument("--pattern", type=str, default="predictions/*/*.cif", help="Glob pattern for prediction files")
    parser.add_argument("--num_workers", type=int, default=cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    mobile_result = Path(args.mobile_result)
    mobile_pdbs = list(mobile_result.rglob(args.pattern))

    print(f"Found {len(mobile_pdbs)} files. Running with {args.num_workers} workers...")

    worker_fn = partial(
        process_single_pdb,
        ref_pdb=args.ref_pdb,
        ref_sdf=args.ref_sdf,
        ref_chainID=args.ref_chainID,
        model_chainID=args.model_chainID,
        ligand_chainID=args.ligand_chainID,
    )

    with Pool(processes=args.num_workers) as pool:
        list(tqdm(
            pool.imap_unordered(worker_fn, mobile_pdbs),
            total=len(mobile_pdbs),
            desc="Computing RMSD",
        ))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [CPU] {msg}", flush=True)


def find_fasta_for_protein(fasta_root: Path, protein: str) -> Path:
    """
    Recreates behavior of FASTA_DIR=\"$FASTA_ROOT/$PROT\":
    - If that path is a file, use it.
    - If it's a directory, pick the first *.fasta / *.fa file inside.
    """
    candidate = fasta_root / protein
    if candidate.is_file():
        return candidate

    if candidate.is_dir():
        for ext in ("*.fasta", "*.fa", "*.faa"):
            files = list(candidate.glob(ext))
            if files:
                return files[0]

    raise FileNotFoundError(
        f"Could not find FASTA for protein '{protein}' under '{fasta_root}'. "
        f"Checked '{candidate}'."
    )


def run_jackhmmer(jackhmmer_bin: str, fasta_path: Path, uniref90_db: Path,
                  out_dir: Path, tmpdir: Path):
    """
    Run JackHMMER against UniRef90, write alignment to uniref90.sto.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sto_path = out_dir / "uniref90.sto"

    cmd = [
        jackhmmer_bin,
        "-N", "3",                 # number of iterations (adjust to taste)
        "-E", "0.0001",            # e-value (adjust)
        "-A", str(sto_path),       # alignment output
        str(fasta_path),
        str(uniref90_db),
    ]
    env = os.environ.copy()
    env["TMPDIR"] = str(tmpdir)
    log(f"Running JackHMMER: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)
    return sto_path


def run_kalign(kalign_bin: str, sto_path: Path, out_dir: Path):
    """
    Convert Stockholm to A3M-style alignment.
    This is a simplified placeholder; in a production pipeline you may
    want to mimic Alphafold/OpenFold's exact reformatting steps.
    """
    a3m_path = out_dir / "uniref90.a3m"
    cmd = [
        kalign_bin,
        "-i", str(sto_path),
        "-o", str(a3m_path),
    ]
    log(f"Running Kalign: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return a3m_path


def run_hhsearch(hhsearch_bin: str, a3m_path: Path, pdb70_db: Path, out_dir: Path):
    """
    Run HHsearch against PDB70 database to produce templates.
    """
    hhr_path = out_dir / "pdb70.hhr"
    cmd = [
        hhsearch_bin,
        "-i", str(a3m_path),
        "-o", str(hhr_path),
        "-d", str(pdb70_db),
    ]
    log(f"Running HHsearch: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return hhr_path


def main():
    parser = argparse.ArgumentParser(
        description="CPU-only pipeline: JackHMMER + HHsearch + Kalign "
                    "to generate precomputed alignments for OpenFold."
    )
    parser.add_argument(
        "--protein-id",
        required=True,
        help="Protein identifier (e.g. protein1); used to locate FASTA and write alignments.",
    )
    parser.add_argument(
        "--fasta-root",
        default=os.environ.get("FASTA_ROOT", "/data/input_fasta_single"),
        help="Root directory containing protein FASTA files or subdirectories "
             "(default: env FASTA_ROOT or /data/input_fasta_single).",
    )
    parser.add_argument(
        "--precomputed-alignments-dir",
        default=os.environ.get("PRECOMPUTED_ALIGNMENT_DIR", "/data/databases/embeddings_output_dir"),
        help="Root directory to write precomputed alignments for each protein.",
    )
    parser.add_argument(
        "--uniref90-db",
        default=os.environ.get("UNIREF90_DB", "/data/databases/uniref90/uniref90.fasta"),
        help="Path to UniRef90 FASTA.",
    )
    parser.add_argument(
        "--pdb70-db",
        default=os.environ.get("PDB70_DB", "/data/databases/pdb70/pdb70"),
        help="Path to PDB70 HHsearch database.",
    )
    parser.add_argument(
        "--jackhmmer-bin",
        default=os.environ.get("JACKHMMER_BIN", "/opt/conda/bin/jackhmmer"),
        help="Path to jackhmmer binary.",
    )
    parser.add_argument(
        "--hhsearch-bin",
        default=os.environ.get("HHSEARCH_BIN", "/opt/conda/bin/hhsearch"),
        help="Path to hhsearch binary.",
    )
    parser.add_argument(
        "--kalign-bin",
        default=os.environ.get("KALIGN_BIN", "/opt/conda/bin/kalign"),
        help="Path to kalign binary.",
    )
    parser.add_argument(
        "--tmpdir",
        default=os.environ.get("TMPDIR", "/tmp"),
        help="Temporary directory (default: env TMPDIR or /tmp).",
    )

    args = parser.parse_args()

    protein_id = args.protein_id
    fasta_root = Path(args.fasta_root)
    precomputed_root = Path(args.precomputed_alignments_dir)
    uniref90_db = Path(args.uniref90_db)
    pdb70_db = Path(args.pdb70_db)
    tmpdir = Path(args.tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Where this protein's alignments live (what GPU script will use)
    protein_align_dir = precomputed_root / protein_id

    log(f"Starting CPU pipeline for protein '{protein_id}'")
    start_time = time.time()

    try:
        fasta_path = find_fasta_for_protein(fasta_root, protein_id)
        log(f"Using FASTA: {fasta_path}")

        sto_path = run_jackhmmer(args.jackhmmer_bin, fasta_path, uniref90_db,
                                 protein_align_dir, tmpdir)
        a3m_path = run_kalign(args.kalign_bin, sto_path, protein_align_dir)
        hhr_path = run_hhsearch(args.hhsearch_bin, a3m_path, pdb70_db, protein_align_dir)

        log(f"Generated alignments for '{protein_id}' in {protein_align_dir}")
        log(f"  JackHMMER STO: {sto_path}")
        log(f"  A3M:           {a3m_path}")
        log(f"  HHsearch HHR:  {hhr_path}")

    except Exception as e:
        log(f"ERROR in CPU pipeline for '{protein_id}': {e}")
        raise

    end_time = time.time()
    log(f"CPU pipeline for '{protein_id}' completed in {end_time - start_time:.1f} seconds")


if __name__ == "__main__":
    # Mirror your original LD_LIBRARY_PATH / LIBRARY_PATH defaults if needed
    os.environ["LD_LIBRARY_PATH"] = "/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LIBRARY_PATH"] = "/lib/x86_64-linux-gnu:" + os.environ.get("LIBRARY_PATH", "")
    main()
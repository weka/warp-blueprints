#!/usr/bin/env python3
import argparse
import os
import subprocess
import time
from pathlib import Path


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [GPU] {msg}", flush=True)


def find_fasta_for_protein(fasta_root: Path, protein: str) -> Path:
    """
    Same as in CPU script: try FASTA_ROOT/protein (file or dir).
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


def main():
    parser = argparse.ArgumentParser(
        description="GPU-only OpenFold inference using precomputed alignments."
    )
    parser.add_argument(
        "--protein-id",
        required=True,
        help="Protein identifier (e.g. protein1); used to locate FASTA, "
             "precomputed alignments and output directory.",
    )
    parser.add_argument(
        "--fasta-root",
        default=os.environ.get("FASTA_ROOT", "/data/input_fasta_single"),
        help="Root directory containing protein FASTA files or subdirectories "
             "(default: env FASTA_ROOT or /data/input_fasta_single).",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("OUTPUT_ROOT", "/data/predictions"),
        help="Root directory for model prediction outputs "
             "(default: env OUTPUT_ROOT or /data/predictions).",
    )
    parser.add_argument(
        "--precomputed-alignments-dir",
        default=os.environ.get("PRECOMPUTED_ALIGNMENT_DIR", "/data/databases/embeddings_output_dir"),
        help="Root directory where CPU step wrote precomputed alignments.",
    )
    parser.add_argument(
        "--mmcif-dir",
        default=os.environ.get("MMCIF_DIR", "/data/databases/pdb_mmcif/mmcif_files"),
        help="Directory containing PDB mmCIF files.",
    )
    parser.add_argument(
        "--openfold-checkpoint-path",
        default=os.environ.get(
            "OPENFOLD_CHECKPOINT_PATH",
            "openfold/resources/openfold_msa_params/openfold_params/finetuning_ptm_1.pt",
        ),
        help="Path to OpenFold checkpoint (same as in your bash script).",
    )
    parser.add_argument(
        "--config-preset",
        default=os.environ.get("CONFIG_PRESET", "model_1_ptm"),
        help="OpenFold config preset (default: model_1_ptm).",
    )
    parser.add_argument(
        "--model-device",
        default=os.environ.get("MODEL_DEVICE", "cuda:0"),
        help="Model device, e.g. 'cuda:0' (default).",
    )
    parser.add_argument(
        "--gpu-id",
        default=os.environ.get("GPU_ID"),
        help="Optional GPU ID to set CUDA_VISIBLE_DEVICES (e.g. 0, 1, ...). "
             "If not set, uses whatever is visible in the container.",
    )
    parser.add_argument(
        "--run-pretrained-script",
        default=os.environ.get("RUN_PRETRAINED_SCRIPT", "run_pretrained_openfold.py"),
        help="Path to run_pretrained_openfold.py inside the container.",
    )

    args = parser.parse_args()

    protein_id = args.protein_id
    fasta_root = Path(args.fasta_root)
    output_root = Path(args.output_root)
    precomputed_root = Path(args.precomputed_alignments_dir)
    mmcif_dir = Path(args.mmcif_dir)
    checkpoint_path = Path(args.openfold_checkpoint_path)

    # Where this protein's outputs go
    outdir = output_root / protein_id
    outdir.mkdir(parents=True, exist_ok=True)

    # Where CPU step wrote alignments for this protein
    align_dir = precomputed_root / protein_id
    if not align_dir.exists():
        raise FileNotFoundError(
            f"Precomputed alignment directory not found for '{protein_id}': {align_dir}"
        )

    try:
        fasta_path = find_fasta_for_protein(fasta_root, protein_id)
    except FileNotFoundError as e:
        log(str(e))
        raise

    if args.gpu_id is not None:
        # Mirror your original CUDA_VISIBLE_DEVICES behavior, if you want
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        log(f"Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    log(f"Starting GPU inference for protein '{protein_id}'")
    log(f"  FASTA:          {fasta_path}")
    log(f"  MMCIF_DIR:      {mmcif_dir}")
    log(f"  OUTPUT_DIR:     {outdir}")
    log(f"  ALIGNMENTS_DIR: {align_dir}")
    log(f"  CHECKPOINT:     {checkpoint_path}")
    log(f"  CONFIG_PRESET:  {args.config_preset}")
    log(f"  MODEL_DEVICE:   {args.model_device}")

    start_time = time.time()

    # This closely mirrors your original call, but uses precomputed alignments
    cmd = [
        "python3",
        args.run_pretrained_script,
        str(fasta_path),
        str(mmcif_dir),
        "--output_dir", str(outdir),
        "--model_device", args.model_device,
        "--config_preset", args.config_preset,
        "--openfold_checkpoint_path", str(checkpoint_path),
        "--use_precomputed_alignments", str(align_dir),
        # Note: we deliberately do NOT pass uniref90/pdb70/jackhmmer/hhsearch/kalign
        # so that the script relies entirely on precomputed alignments.
    ]

    log(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log(f"ERROR during GPU inference for '{protein_id}': {e}")
        raise

    end_time = time.time()
    log(f"GPU inference for '{protein_id}' completed in {end_time - start_time:.1f} seconds")


if __name__ == "__main__":
    # Same library path tweaks you used in bash
    os.environ["LD_LIBRARY_PATH"] = "/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LIBRARY_PATH"] = "/lib/x86_64-linux-gnu:" + os.environ.get("LIBRARY_PATH", "")
    main()
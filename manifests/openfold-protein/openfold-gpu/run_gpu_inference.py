#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import time
from pathlib import Path

from prometheus_client import start_http_server, Histogram, Counter, Gauge


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [GPU] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

# Histogram buckets for GPU inference + relaxation latency (seconds)
INF_BUCKETS = (1, 2, 5, 10, 20, 40, 80, 160, 320, 600, 1200)

# End-to-end GPU stage latency (OpenFold inference + relaxation, as run by this script)
inference_latency = Histogram(
    "openfold_inference_total_latency_seconds",
    "Total latency of OpenFold GPU stage (inference + relaxation) per request",
    buckets=INF_BUCKETS,
)

# Counters and gauge
inference_requests_total = Counter(
    "openfold_inference_requests_total",
    "Total number of OpenFold GPU inference requests",
)

inference_failures_total = Counter(
    "openfold_inference_failures_total",
    "Total number of failed OpenFold GPU inference requests",
)

inference_inflight = Gauge(
    "openfold_inference_inflight_requests",
    "Number of in-flight OpenFold GPU inference requests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_fasta_for_protein(fasta_root: Path, protein: str) -> Path:
    """
    Find the FASTA file for a given protein ID.

    Tries:
      1) <fasta_root>/<protein> as a file
      2) <fasta_root>/<protein> as a directory containing *.fasta / *.fa / *.faa
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


def get_first_fasta_tag(fasta_path: Path) -> str:
    """
    Return the first FASTA header (without '>') as the tag used by OpenFold.
    This must match what the CPU step uses to name the alignment directory.
    Falls back to 'query' if no header is found.
    """
    with fasta_path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:].strip()
                return name if name else "query"
    return "query"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU-only OpenFold inference using precomputed alignments."
    )

    parser.add_argument(
        "--protein-id",
        required=True,
        help=(
            "Protein identifier (e.g. protein1, protein1.fasta); used to locate "
            "FASTA and choose the output directory. The actual alignment directory "
            "is derived from the FASTA header (e.g. '>protein1')."
        ),
    )
    parser.add_argument(
        "--fasta-root",
        default=os.environ.get("FASTA_ROOT", "/data/input_fasta_single"),
        help=(
            "Root directory containing protein FASTA files or subdirectories "
            "(default: env FASTA_ROOT or /data/input_fasta_single)."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("OUTPUT_ROOT", "/data/predictions"),
        help=(
            "Root directory for model prediction outputs "
            "(default: env OUTPUT_ROOT or /data/predictions)."
        ),
    )
    parser.add_argument(
        "--precomputed-alignments-dir",
        default=os.environ.get(
            "PRECOMPUTED_ALIGNMENT_DIR",
            "/data/databases/embeddings_output_dir",
        ),
        help=(
            "ROOT directory where the CPU step wrote precomputed alignments. "
            "The CPU step creates per-target subdirectories:\n"
            "  <precomputed-alignments-dir>/<target_id>/\n"
            "where <target_id> is derived from the first FASTA header "
            "(e.g. '>protein1' -> 'protein1'). This script passes the ROOT "
            "to OpenFold via --use_precomputed_alignments and OpenFold appends "
            "<target_id> internally."
        ),
    )
    parser.add_argument(
        "--mmcif-dir",
        default=os.environ.get(
            "MMCIF_DIR",
            "/data/databases/pdb_mmcif/mmcif_files",
        ),
        help="Directory containing PDB mmCIF files.",
    )
    parser.add_argument(
        "--openfold-checkpoint-path",
        default=os.environ.get(
            "OPENFOLD_CHECKPOINT_PATH",
            "/data/openfold_weights/finetuning_ptm_1.pt",
        ),
        help=(
            "Path to OpenFold finetuned checkpoint (.pt file). "
            "Default: /data/openfold_weights/finetuning_ptm_1.pt"
        ),
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
        help=(
            "Optional GPU ID to set CUDA_VISIBLE_DEVICES (e.g. 0, 1, ...). "
            "If not set, uses whatever is visible in the container."
        ),
    )
    parser.add_argument(
        "--run-pretrained-script",
        default=os.environ.get("RUN_PRETRAINED_SCRIPT", "run_pretrained_openfold.py"),
        help="Path to run_pretrained_openfold.py inside the container.",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=int(os.environ.get("OPENFOLD_METRICS_PORT", "9102")),
        help=(
            "Port to expose Prometheus metrics on (0 to disable). "
            "Default: 9102 or $OPENFOLD_METRICS_PORT."
        ),
    )

    args = parser.parse_args()

    # Start Prometheus metrics server (if enabled)
    if args.metrics_port and args.metrics_port > 0:
        start_http_server(args.metrics_port)
        log(f"Prometheus metrics server listening on :{args.metrics_port}")

    # Normalise paths
    protein_id = args.protein_id
    fasta_root = Path(args.fasta_root)
    output_root = Path(args.output_root)
    alignments_root = Path(args.precomputed_alignments_dir)
    mmcif_dir = Path(args.mmcif_dir)
    checkpoint_path = Path(args.openfold_checkpoint_path)

    # Where this protein's outputs go (per-protein output dir is still fine)
    outdir = output_root / protein_id
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure the alignments ROOT exists
    if not alignments_root.is_dir():
        log(f"ERROR: Precomputed alignments ROOT directory not found: {alignments_root}")
        # Treat as failed inference request
        inference_requests_total.inc()
        inference_failures_total.inc()
        raise FileNotFoundError(
            f"Precomputed alignments ROOT directory not found: {alignments_root}"
        )

    # Ensure we can find at least one FASTA for this protein (for sanity + logging)
    try:
        fasta_path = find_fasta_for_protein(fasta_root, protein_id)
    except FileNotFoundError as e:
        log(str(e))
        inference_requests_total.inc()
        inference_failures_total.inc()
        raise

    # Derive the OpenFold target ID from the FASTA header
    target_id = get_first_fasta_tag(fasta_path)
    target_align_dir = alignments_root / target_id

    if not target_align_dir.is_dir():
        msg = (
            f"Expected alignment directory for target '{target_id}' not found.\n"
            f"  FASTA:              {fasta_path}\n"
            f"  ALIGNMENTS_ROOT:    {alignments_root}\n"
            f"  EXPECTED SUBDIR:    {target_align_dir}\n"
            "Ensure the CPU step wrote alignments under "
            "<precomputed-alignments-dir>/<target_id>/ and that the "
            "FASTA header used here matches the CPU's."
        )
        log("ERROR: " + msg.replace("\n", " "))
        inference_requests_total.inc()
        inference_failures_total.inc()
        raise FileNotFoundError(msg)

    # Create a per-protein FASTA directory so OpenFold only sees this one sequence
    # e.g. /data/tmp/fasta_dirs/protein1/
    per_protein_fasta_dir = Path("/data/tmp/fasta_dirs") / target_id
    per_protein_fasta_dir.mkdir(parents=True, exist_ok=True)
    per_protein_fasta_path = per_protein_fasta_dir / fasta_path.name

    # Symlink is cheap; copy if your FS doesn't like symlinks
    if not per_protein_fasta_path.exists():
        try:
            per_protein_fasta_path.symlink_to(fasta_path)
        except OSError:
            # Fallback to a copy if symlinks aren't supported
            log(f"Symlink failed for {per_protein_fasta_path}, copying instead.")
            per_protein_fasta_path.write_bytes(fasta_path.read_bytes())

    # Check that checkpoint exists early, instead of failing deep in torch.load
    if not checkpoint_path.is_file():
        log(
            f"ERROR: OpenFold checkpoint not found at '{checkpoint_path}'. "
            "Ensure it is present in /data/openfold_weights or update "
            "--openfold-checkpoint-path / OPENFOLD_CHECKPOINT_PATH."
        )
        inference_requests_total.inc()
        inference_failures_total.inc()
        raise FileNotFoundError(
            f"Missing OpenFold checkpoint: {checkpoint_path}"
        )

    if args.gpu_id is not None:
        # Mirror original CUDA_VISIBLE_DEVICES behavior, if desired
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        log(f"Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    log(f"Starting GPU inference for protein '{protein_id}'")
    log(f"  FASTA:             {fasta_path}")
    log(f"  PER_PROTEIN_DIR:   {per_protein_fasta_dir}")
    log(f"  PER_PROTEIN_FASTA: {per_protein_fasta_path}")
    log(f"  FASTA_ROOT (all):  {fasta_root}")
    log(f"  TARGET_ID:         {target_id}")
    log(f"  MMCIF_DIR:         {mmcif_dir}")
    log(f"  OUTPUT_DIR:        {outdir}")
    log(f"  ALIGNMENTS_ROOT:   {alignments_root}")
    log(f"  TARGET_ALIGN_DIR:  {target_align_dir}")
    log(f"  CHECKPOINT:        {checkpoint_path}")
    log(f"  CONFIG_PRESET:     {args.config_preset}")
    log(f"  MODEL_DEVICE:      {args.model_device}")

    # Build the command for run_pretrained_openfold.py
    python_exe = sys.executable
    print(f"[GPU] Using Python interpreter: {python_exe}", flush=True)

    cmd = [
        python_exe,
        args.run_pretrained_script,
        str(per_protein_fasta_dir),   # fasta_dir (per-protein directory)
        str(mmcif_dir),
        "--output_dir", str(outdir),
        "--model_device", args.model_device,
        "--config_preset", args.config_preset,
        "--openfold_checkpoint_path", str(checkpoint_path),
        "--use_precomputed_alignments", str(alignments_root),
        # We deliberately do NOT pass jackhmmer/hhblits/mgnify/bfd/uniref90/etc
        # so OpenFold uses the precomputed alignments from the CPU MMseqs2 step.
    ]

    log(f"Running: {' '.join(cmd)}")

    # Metrics for this GPU inference "job"
    inference_requests_total.inc()
    inference_inflight.inc()
    start_time = time.time()

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        inference_latency.observe(duration)
        inference_failures_total.inc()
        inference_inflight.dec()
        log(f"ERROR during GPU inference for '{protein_id}': {e}")
        log(f"GPU inference for '{protein_id}' failed after {duration:.1f} seconds")
        raise
    else:
        duration = time.time() - start_time
        inference_latency.observe(duration)
        inference_inflight.dec()
        log(f"GPU inference for '{protein_id}' completed in {duration:.1f} seconds")


if __name__ == "__main__":
    # Keep your library path tweaks
    os.environ["LD_LIBRARY_PATH"] = "/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LIBRARY_PATH"] = "/lib/x86_64-linux-gnu:" + os.environ.get("LIBRARY_PATH", "")
    main()
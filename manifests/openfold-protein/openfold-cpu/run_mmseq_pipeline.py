#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [CPU] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Subprocess wrapper
# ---------------------------------------------------------------------------

def run_cmd(cmd, env=None, check=True) -> subprocess.CompletedProcess:
    """
    Run a command with logging. Raises CalledProcessError if check=True and it fails.
    """
    log(f"Running: {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, env=env, check=check)


# ---------------------------------------------------------------------------
# FASTA helpers
# ---------------------------------------------------------------------------

def get_first_fasta_tag(fasta_path: Path) -> str:
    """
    Return the first FASTA header (without '>') as the tag used by OpenFold.
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
# Trivial single-sequence A3M writer
# ---------------------------------------------------------------------------

def write_trivial_a3m(fasta_path: Path, a3m_out: Path) -> None:
    """
    When MMseqs pipeline fails or yields no hits, write a trivial
    single-sequence A3M file from the query FASTA so that downstream
    OpenFold / HHsearch still have something to consume.
    """
    seq_id = get_first_fasta_tag(fasta_path)
    log(f"Writing trivial single-sequence A3M to '{a3m_out}' (tag='{seq_id}')")
    a3m_out.parent.mkdir(parents=True, exist_ok=True)

    seq_lines = []
    with fasta_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_lines.append(line)

    seq = "".join(seq_lines) if seq_lines else ""

    with a3m_out.open("w") as out:
        out.write(f">{seq_id}\n")
        out.write(seq + "\n")


# ---------------------------------------------------------------------------
# MMseqs2 pipeline
# ---------------------------------------------------------------------------

def check_mmseqs_db(mmseqs_bin: str, db_path: Path) -> None:
    """
    Log MMseqs db type to confirm DB is valid/indexed.
    """
    cmd = [mmseqs_bin, "dbtype", str(db_path)]
    log(f"Checking MMseqs DB type: {' '.join(cmd)}")
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        log(f"MMseqs DB type for '{db_path}': {out.decode('utf-8', errors='ignore').strip()}")
    except subprocess.CalledProcessError as e:
        log(f"WARNING: mmseqs dbtype failed for '{db_path}': {e.output.decode('utf-8', errors='ignore')}")


def safe_rmdb(mmseqs_bin: str, db_prefix: Path) -> None:
    """
    If an MMseqs DB exists (by prefix), remove it with rmdb.
    """
    existing = []
    for suffix in ["", ".dbtype", ".index", ".lookup", "_h", "_h.dbtype", "_h.index"]:
        p = db_prefix.with_name(db_prefix.name + suffix)
        if p.exists():
            existing.append(p.name)

    if existing:
        log(f"Cleaning up existing MMseqs tmp DB '{db_prefix}' (files: {existing})")
        try:
            run_cmd([mmseqs_bin, "rmdb", str(db_prefix)], check=True)
        except subprocess.CalledProcessError as e:
            log(f"WARNING: mmseqs rmdb failed for '{db_prefix}': {e}")


def run_mmseqs(
    fasta_path: Path,
    uniref_db: Path,
    tmp_dir: Path,
    output_dir: Path,
    mmseqs_bin: str,
    threads: int = 32,
    db_load_mode: int = 2,
    search_evalue: float = 1.0,
    min_hit_cov: float = 0.3,
    result2msa_max_seq_id: float = 0.95,
) -> Path:
    """
    CPU-only MMseqs2 pipeline:

      1) createdb (FASTA → queryDB)
      2) search (queryDB vs UniRef/MMseqs DB)
      3) result2msa (→ A3M DB, msa-format-mode=6)
      4) unpackdb (A3M DB → per-query .a3m file)
    """
    fasta_path = fasta_path.resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    protein_name = fasta_path.name
    log(f"Starting CPU-only MMseqs2 pipeline for {protein_name}")
    log(f"Using FASTA: {fasta_path}")
    log(f"MMseqs output_dir: {output_dir}")

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    query_db = tmp_dir / f"{protein_name}_queryDB"
    result_db = tmp_dir / f"{protein_name}_resultDB"
    msa_db = tmp_dir / f"{protein_name}_msaA3M_DB"
    unpack_dir = tmp_dir / f"{protein_name}_a3m_tmp"

    final_a3m = output_dir / f"{protein_name}.a3m"

    # ------------------------------------------------------------------
    # Step 0: Check UniRef DB
    # ------------------------------------------------------------------
    check_mmseqs_db(mmseqs_bin, uniref_db)

    # ------------------------------------------------------------------
    # Step 1: Clean any old tmp DBs
    # ------------------------------------------------------------------
    safe_rmdb(mmseqs_bin, result_db)
    safe_rmdb(mmseqs_bin, msa_db)
    if unpack_dir.exists():
        log(f"Removing old unpack dir '{unpack_dir}'")
        for child in unpack_dir.iterdir():
            try:
                child.unlink()
            except Exception:
                pass
        try:
            unpack_dir.rmdir()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Step 2: createdb
    # ------------------------------------------------------------------
    cmd_createdb = [
        mmseqs_bin,
        "createdb",
        str(fasta_path),
        str(query_db),
    ]
    run_cmd(cmd_createdb)

    # ------------------------------------------------------------------
    # Step 3: search (with tunable evalue / coverage / db-load-mode)
    # ------------------------------------------------------------------
    log(f"Running mmseqs search (CPU, db-load-mode={db_load_mode})")
    cmd_search = [
        mmseqs_bin,
        "search",
        str(query_db),
        str(uniref_db),
        str(result_db),
        str(tmp_dir),
        "--threads", str(threads),
        "--db-load-mode", str(db_load_mode),
        # Tunable thresholds for depth vs quality:
        "-e", str(search_evalue),
        "-c", str(min_hit_cov),
        "--cov-mode", "0",
    ]
    run_cmd(cmd_search)

    # ------------------------------------------------------------------
    # Optional: dbsize on result DB (purely informative)
    # ------------------------------------------------------------------
    log(f"Inspecting Result DB size with: {mmseqs_bin} dbsize {result_db}")
    try:
        dbsize_out = subprocess.check_output(
            [mmseqs_bin, "dbsize", str(result_db)],
            stderr=subprocess.STDOUT
        )
        dbsize_text = dbsize_out.decode("utf-8", errors="ignore").strip()
        log(f"Result DB size info:\n{dbsize_text}")
    except subprocess.CalledProcessError as e:
        log(f"WARNING: mmseqs dbsize failed for '{result_db}': {e}")

    # ------------------------------------------------------------------
    # Step 4: result2msa (→ A3M DB, msa-format-mode=6)
    # ------------------------------------------------------------------
    log("Running mmseqs result2msa (→ A3M DB)")
    cmd_result2msa = [
        mmseqs_bin,
        "result2msa",
        str(query_db),
        str(uniref_db),
        str(result_db),
        str(msa_db),
        "--threads", str(threads),
        "--db-load-mode", str(db_load_mode),
        "--msa-format-mode", "6",                    # A3M-like DB
        "--max-seq-id", str(result2msa_max_seq_id),  # keep more similar seqs
        "--filter-msa", "0",                         # don't aggressively thin MSAs here
    ]

    try:
        run_cmd(cmd_result2msa)
    except subprocess.CalledProcessError as e:
        log(f"ERROR in result2msa: {e}")
        # Fallback: trivial A3M
        write_trivial_a3m(fasta_path, final_a3m)
        return final_a3m

    # ------------------------------------------------------------------
    # Step 5: unpackdb (A3M DB → .a3m files)
    # ------------------------------------------------------------------
    log("Running mmseqs unpackdb (A3M DB → .a3m files)")
    cmd_unpack = [
        mmseqs_bin,
        "unpackdb",
        str(msa_db),
        str(unpack_dir),
        "--unpack-name-mode", "0",
        "--unpack-suffix", ".a3m",
    ]

    try:
        run_cmd(cmd_unpack)
    except subprocess.CalledProcessError as e:
        log(f"ERROR in unpackdb: {e}")
        # Fallback: trivial A3M
        write_trivial_a3m(fasta_path, final_a3m)
        return final_a3m

    # Pick the first .a3m from unpack_dir
    if not unpack_dir.exists():
        log(f"ERROR: unpack dir '{unpack_dir}' does not exist after unpackdb. Falling back to trivial A3M.")
        write_trivial_a3m(fasta_path, final_a3m)
        return final_a3m

    a3m_files = sorted(unpack_dir.glob("*.a3m"))
    if not a3m_files:
        log(f"ERROR: no .a3m files found in '{unpack_dir}'. Falling back to trivial A3M.")
        write_trivial_a3m(fasta_path, final_a3m)
        return final_a3m

    chosen_a3m = a3m_files[0]
    log(f"Using A3M file '{chosen_a3m}' → '{final_a3m}'")
    final_a3m.parent.mkdir(parents=True, exist_ok=True)
    final_a3m.write_bytes(chosen_a3m.read_bytes())

    return final_a3m


# ---------------------------------------------------------------------------
# HHsearch helper
# ---------------------------------------------------------------------------

def run_hhsearch(
    hhsearch_bin: str,
    a3m_path: Path,
    pdb70_db: Path,
    hhr_out: Path,
) -> None:
    """
    Run HHsearch on the A3M against pdb70 HHM database.
    """
    hhr_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        hhsearch_bin,
        "-i", str(a3m_path),
        "-o", str(hhr_out),
        "-d", str(pdb70_db),
    ]
    log("Running HHsearch")
    run_cmd(cmd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CPU-side MMseqs2 + HHsearch pipeline for OpenFold."
    )

    parser.add_argument(
        "--protein-id",
        required=True,
        help="Protein FASTA filename (e.g. protein1.fasta) under --fasta-root.",
    )
    parser.add_argument(
        "--fasta-root",
        type=Path,
        default=Path("/data/input_fasta_single"),
        help="Root directory containing input FASTA files.",
    )
    parser.add_argument(
        "--precomputed-alignments-dir",
        type=Path,
        default=Path("/data/databases/embeddings_output_dir"),
        help=(
            "Root directory to write A3M/HHR outputs. "
            "Final layout: <root>/<target_id>/ where <target_id> is derived "
            "from the first FASTA header (e.g. '>protein1')."
        ),
    )
    parser.add_argument(
        "--uniref50-db",
        type=Path,
        default=Path("/data/databases/uniref50/uniref50_mmseqs"),
        help="MMseqs2 UniRef50 DB prefix (no extension).",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("/data/tmp"),
        help="Temporary directory for MMseqs2 DBs.",
    )

    # MMseqs2 binary / tuning
    parser.add_argument(
        "--mmseqs-bin",
        type=str,
        default=os.environ.get("MMSEQS_BIN", "/usr/local/bin/mmseqs-latest"),
        help="Path to mmseqs2 binary. Default: /usr/local/bin/mmseqs-latest or $MMSEQS_BIN.",
    )
    parser.add_argument(
        "--mmseqs-threads",
        type=int,
        default=32,
        help="Number of threads for MMseqs2.",
    )
    parser.add_argument(
        "--mmseqs-db-load-mode",
        type=int,
        default=2,
        help="MMseqs2 --db-load-mode (2 = direct read from storage).",
    )
    parser.add_argument(
        "--search-evalue",
        type=float,
        default=1.0,
        help="MMseqs2 search E-value cutoff (passed to -e). Higher = more hits. Default: 1.0",
    )
    parser.add_argument(
        "--min-hit-cov",
        type=float,
        default=0.3,
        help="Minimum query coverage for hits (0–1, passed to -c). Default: 0.3.",
    )
    parser.add_argument(
        "--result2msa-max-seq-id",
        type=float,
        default=0.95,
        help="Maximum seq identity in result2msa (0–1, --max-seq-id). Default: 0.95.",
    )

    # HHsearch / pdb70
    parser.add_argument(
        "--hhsearch-bin",
        type=str,
        default="hhsearch",
        help="Path to hhsearch binary (from hhsuite).",
    )
    parser.add_argument(
        "--pdb70-db",
        type=Path,
        default=Path("/data/databases/pdb70/pdb70"),
        help="pdb70 HHM database prefix (no extension).",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    protein_id = args.protein_id
    fasta_path = args.fasta_root / protein_id

    if not fasta_path.is_file():
        log(f"ERROR: FASTA file not found: {fasta_path}")
        sys.exit(1)

    # Tag used by OpenFold (from FASTA header, e.g. '>protein1')
    tag_for_openfold = get_first_fasta_tag(fasta_path)
    log(f"Derived OpenFold tag from FASTA header: '{tag_for_openfold}'")

    # Final layout for this protein:
    #   <precomputed-alignments-dir>/<tag_for_openfold>/
    #
    # GPU step will pass:
    #   --use_precomputed_alignments <precomputed-alignments-dir>
    #
    # OpenFold then appends '/<tag_for_openfold>' internally to locate these files.
    alignments_root = args.precomputed_alignments_dir
    target_align_dir = alignments_root / tag_for_openfold
    target_align_dir.mkdir(parents=True, exist_ok=True)

    log(f"ALIGNMENTS_ROOT:    {alignments_root}")
    log(f"TARGET_ALIGN_DIR:   {target_align_dir}")

    # Paths for outputs inside target_align_dir
    # (filename isn't critical; directory is what matters to OpenFold)
    a3m_out = target_align_dir / f"{protein_id}.a3m"
    hhr_out = target_align_dir / "pdb70.hhr"

    mmseqs_bin = args.mmseqs_bin

    # If mmseqs-latest is not found, fall back to "mmseqs" in PATH
    if not Path(mmseqs_bin).exists():
        log(f"WARNING: mmseqs_bin '{mmseqs_bin}' not found. Falling back to 'mmseqs' in PATH.")
        mmseqs_bin = "mmseqs"

    # Run MMseqs2 alignment pipeline, writing into target_align_dir
    a3m_path = run_mmseqs(
        fasta_path=fasta_path,
        uniref_db=args.uniref50_db,
        tmp_dir=args.tmp_dir,
        output_dir=target_align_dir,
        mmseqs_bin=mmseqs_bin,
        threads=args.mmseqs_threads,
        db_load_mode=args.mmseqs_db_load_mode,
        search_evalue=args.search_evalue,
        min_hit_cov=args.min_hit_cov,
        result2msa_max_seq_id=args.result2msa_max_seq_id,
    )

    # Run HHsearch using the generated A3M
    run_hhsearch(
        hhsearch_bin=args.hhsearch_bin,
        a3m_path=a3m_path,
        pdb70_db=args.pdb70_db,
        hhr_out=hhr_out,
    )

    log(f"Generated alignments for {protein_id}")
    log(f"A3M: {a3m_path}")
    log(f"HHR: {hhr_out}")
    log("Pipeline completed.")


if __name__ == "__main__":
    main()
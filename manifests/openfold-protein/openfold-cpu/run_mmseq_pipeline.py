#!/usr/bin/env python3
import argparse
import os
import subprocess
import time
import re
from pathlib import Path
from typing import Optional


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [CPU] {msg}", flush=True)


def find_fasta_for_protein(fasta_root: Path, protein: str) -> Path:
    """
    If FASTA_ROOT/protein is a file → use it.
    If it's a directory → pick first *.fasta, *.fa, *.faa.
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


# ------------------ MMSEQS UTILS: DB CHECKS & SIZE LOGGING ------------------ #

def ensure_mmseqs_db_ready(
    mmseqs_bin: str,
    database_path: Path,
    tmpdir: Path,
    threads: int,
    env: dict,
):
    """
    Sanity checks that the MMseqs DB exists, is a valid DB, and is indexed.
    - Uses `mmseqs dbtype` to verify DB.
    - If no .index file exists, runs `mmseqs createindex` (once).
    """
    if not database_path.exists():
        raise FileNotFoundError(f"MMseqs DB '{database_path}' does not exist")

    cmd_dbtype = [mmseqs_bin, "dbtype", str(database_path)]
    log(f"Checking MMseqs DB type: {' '.join(cmd_dbtype)}")
    try:
        result = subprocess.run(
            cmd_dbtype,
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        log(f"ERROR: mmseqs dbtype failed for DB '{database_path}': {e}")
        raise

    dbtype_output = (result.stdout or "").strip()
    if dbtype_output:
        log(f"MMseqs DB type for '{database_path}': {dbtype_output}")

    index_glob = list(database_path.parent.glob(database_path.name + "*.index"))
    if index_glob:
        log(f"DB appears indexed (found index files: {[p.name for p in index_glob]})")
        return

    log("No MMseqs index files found; creating index with `mmseqs createindex`")
    cmd_createindex = [
        mmseqs_bin,
        "createindex",
        str(database_path),
        str(tmpdir),
        "--threads",
        str(threads),
    ]
    log(f"Running createindex: {' '.join(cmd_createindex)}")
    subprocess.run(cmd_createindex, check=True, env=env)
    log("MMseqs DB indexing complete.")


def log_db_size(
    mmseqs_bin: str,
    db_path: Path,
    env: dict,
    label: str = "DB",
) -> int:
    """
    Uses `mmseqs dbsize` to log DB stats and returns a best-effort
    estimate of the number of entries.

    If parsing fails, returns -1.
    """
    cmd_dbsize = [mmseqs_bin, "dbsize", str(db_path)]
    log(f"Inspecting {label} size with: {' '.join(cmd_dbsize)}")
    try:
        result = subprocess.run(
            cmd_dbsize,
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        log(f"WARNING: mmseqs dbsize failed for '{db_path}': {e}")
        return -1

    out = (result.stdout or "").strip()
    if out:
        log(f"{label} size info for '{db_path}':\n{out}")

    m = re.search(r"(\d+)", out)
    if not m:
        return -1

    try:
        entry_count = int(m.group(1))
    except ValueError:
        entry_count = -1

    return entry_count


def remove_mmseqs_tmp_db(
    mmseqs_bin: str,
    db_path: Path,
    env: dict,
):
    """
    Remove a temporary MMseqs DB (and associated files) if it exists.

    Uses `mmseqs rmdb` on db_path if we see any files matching db_path*.
    Only call this for DBs in tmpdir (query/result/msa/a3m), NOT for the
    main target database.
    """
    matches = list(db_path.parent.glob(db_path.name + "*"))
    if not matches:
        return

    log(f"Cleaning up existing MMseqs tmp DB '{db_path}' (files: {[m.name for m in matches]})")
    cmd_rmdb = [mmseqs_bin, "rmdb", str(db_path)]
    try:
        subprocess.run(cmd_rmdb, check=True, env=env)
    except subprocess.CalledProcessError as e:
        log(f"WARNING: mmseqs rmdb failed for '{db_path}': {e}")


# ----------------------------- A3M FALLBACK HELPERS ---------------------------- #

def write_single_sequence_a3m_from_fasta(fasta_path: Path, a3m_out: Path):
    """
    Read the first sequence from `fasta_path` and write a trivial
    single-sequence A3M to `a3m_out`.
    """
    header = None
    seq_lines = []

    with fasta_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is None:
                    header = line[1:].strip() or "query"
                else:
                    # Already captured one sequence; ignore the rest
                    break
            else:
                if header is None:
                    header = "query"
                seq_lines.append(line)

    if header is None or not seq_lines:
        raise RuntimeError(f"Could not extract a sequence from FASTA '{fasta_path}'")

    seq = "".join(seq_lines).replace(" ", "").upper()

    log(f"Writing trivial single-sequence A3M to '{a3m_out}'")
    with a3m_out.open("w") as out_f:
        out_f.write(f">{header}\n")
        out_f.write(seq + "\n")


# ----------------------------- MAIN MMSEQS PIPELINE --------------------------- #

def run_mmseqs(
    mmseqs_bin: str,
    fasta_path: Path,
    database_path: Path,
    out_dir: Path,
    tmpdir: Path,
    threads: int = 8,
    protein_id: Optional[str] = None,
    db_load_mode: int = 2,
):
    """
    CPU-ONLY MMseqs2 alignment pipeline for OpenFold.

    Pipeline:

      1) createdb <query.fasta> <query_db>
      2) search <query_db> <database> <result_db> <tmpdir> --threads N [--db-load-mode M]
      3) result2msa <query_db> <database> <result_db> <msa_db>
      4) convertmsa <msa_db> <msa_a3m_db> --format-output a3m
      5) convert2fasta <msa_a3m_db> <protein>.a3m

    If anything in steps 3–5 fails, fall back to a trivial single-sequence A3M.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    base = protein_id or fasta_path.stem

    query_db = tmpdir / f"{base}_queryDB"
    result_db = tmpdir / f"{base}_resultDB"
    msa_db = tmpdir / f"{base}_msaDB"
    msa_a3m_db = tmpdir / f"{base}_msaA3M_DB"
    a3m_out = out_dir / f"{base}.a3m"

    env = os.environ.copy()
    env["TMPDIR"] = str(tmpdir)

    ensure_mmseqs_db_ready(mmseqs_bin, database_path, tmpdir, threads, env)

    # Clean up any old tmp DBs
    remove_mmseqs_tmp_db(mmseqs_bin, result_db, env)
    remove_mmseqs_tmp_db(mmseqs_bin, msa_db, env)
    remove_mmseqs_tmp_db(mmseqs_bin, msa_a3m_db, env)

    # 1) createdb
    cmd_createdb = [
        mmseqs_bin, "createdb",
        str(fasta_path),
        str(query_db),
    ]
    log("Running mmseqs createdb")
    subprocess.run(cmd_createdb, check=True, env=env)

    # 2) search
    cmd_search = [
        mmseqs_bin, "search",
        str(query_db),
        str(database_path),
        str(result_db),
        str(tmpdir),
        "--threads", str(threads),
        "--db-load-mode", str(db_load_mode),
    ]
    log(f"Running mmseqs search (CPU, db-load-mode={db_load_mode})")
    subprocess.run(cmd_search, check=True, env=env)

    # 3) result2msa – build MSA DB with real multi-sequence depth
    cmd_result2msa = [
        mmseqs_bin, "result2msa",
        str(query_db),
        str(database_path),
        str(result_db),
        str(msa_db),
        "--threads", str(threads),
    ]
    log("Running mmseqs result2msa")
    try:
        subprocess.run(cmd_result2msa, check=True, env=env)
    except subprocess.CalledProcessError as e:
        log(f"ERROR: mmseqs result2msa failed: {e}")
        log("Falling back to trivial single-sequence A3M from FASTA.")
        write_single_sequence_a3m_from_fasta(fasta_path, a3m_out)
        return a3m_out

    # MSA depth sanity/logging
    seq_count = log_db_size(mmseqs_bin, msa_db, env, label="MSA DB")
    if seq_count == 0:
        log("WARNING: MSA DB appears to have 0 sequences (no hits). "
            "A3M will effectively be single-sequence.")
    elif seq_count == 1:
        log("WARNING: MSA DB appears to contain only 1 sequence (likely just the query). "
            "Alignment depth is poor.")
    elif seq_count > 1:
        log(f"MSA sanity check: ~{seq_count} sequences detected in MSA DB.")

    # 4) convertmsa – MSA DB → A3M-encoded DB
    cmd_convertmsa = [
        mmseqs_bin, "convertmsa",
        str(msa_db),
        str(msa_a3m_db),
        "--format-output", "a3m",
    ]
    log("Running mmseqs convertmsa (MSA DB → A3M DB)")
    try:
        subprocess.run(cmd_convertmsa, check=True, env=env)
    except subprocess.CalledProcessError as e:
        log(f"ERROR: mmseqs convertmsa failed: {e}")
        log("Falling back to trivial single-sequence A3M from FASTA.")
        write_single_sequence_a3m_from_fasta(fasta_path, a3m_out)
        return a3m_out

    # 5) convert2fasta – A3M DB → plain text A3M file
    cmd_convert2fasta = [
        mmseqs_bin, "convert2fasta",
        str(msa_a3m_db),
        str(a3m_out),
    ]
    log("Running mmseqs convert2fasta (A3M DB → A3M file)")
    try:
        subprocess.run(cmd_convert2fasta, check=True, env=env)
    except subprocess.CalledProcessError as e:
        log(f"ERROR: mmseqs convert2fasta failed: {e}")
        log("Falling back to trivial single-sequence A3M from FASTA.")
        write_single_sequence_a3m_from_fasta(fasta_path, a3m_out)
        return a3m_out

    return a3m_out


def run_hhsearch(hhsearch_bin: str, a3m_path: Path, pdb70_db: Path, out_dir: Path):
    """Run HHsearch on generated A3M."""
    hhr_path = out_dir / "pdb70.hhr"
    cmd = [
        hhsearch_bin,
        "-i", str(a3m_path),
        "-o", str(hhr_path),
        "-d", str(pdb70_db),
    ]
    log("Running HHsearch")
    subprocess.run(cmd, check=True)
    return hhr_path


def main():
    parser = argparse.ArgumentParser(
        description="CPU MMseqs2 + HHsearch alignment pipeline for OpenFold."
    )
    parser.add_argument(
        "--protein-id",
        required=True,
        help="Protein identifier (e.g. protein1 or protein1.fasta)."
    )
    parser.add_argument(
        "--fasta-root",
        default=os.environ.get("FASTA_ROOT", "/data/input_fasta_single"),
        help="Root directory or parent of FASTA files.",
    )
    parser.add_argument(
        "--precomputed-alignments-dir",
        default=os.environ.get("PRECOMPUTED_ALIGNMENT_DIR", "/data/databases/embeddings_output_dir"),
        help="Output root for precomputed alignments.",
    )
    parser.add_argument(
        "--mmseqs-db",
        default=os.environ.get("MMSEQS_DB", "/data/databases/uniref50/uniref50_mmseqs"),
        help="Path to MMseqs2 target DB (e.g. UniRef50).",
    )
    parser.add_argument(
        "--pdb70-db",
        default=os.environ.get("PDB70_DB", "/data/databases/pdb70/pdb70"),
        help="Path to PDB70 HHsearch DB.",
    )
    parser.add_argument(
        "--mmseqs-bin",
        default=os.environ.get("MMSEQS_BIN", "/opt/conda/envs/cpu-env/bin/mmseqs"),
        help="Path to mmseqs binary.",
    )
    parser.add_argument(
        "--hhsearch-bin",
        default=os.environ.get("HHSEARCH_BIN", "/opt/conda/envs/cpu-env/bin/hhsearch"),
        help="Path to hhsearch binary.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("MMSEQS_THREADS", "32")),
        help="Number of CPU threads for MMseqs2.",
    )
    parser.add_argument(
        "--db-load-mode",
        type=int,
        default=int(os.environ.get("MMSEQS_DB_LOAD_MODE", "2")),
        help="MMseqs --db-load-mode (0:auto, 1:fread, 2:mmap, 3:mmap+touch; default: 2).",
    )
    parser.add_argument(
        "--tmpdir",
        default=os.environ.get("TMPDIR", "/data/tmp"),
        help="Temporary directory.",
    )

    args = parser.parse_args()

    protein_id = args.protein_id
    fasta_root = Path(args.fasta_root)
    mmseqs_db = Path(args.mmseqs_db)
    pdb70_db = Path(args.pdb70_db)
    tmpdir = Path(args.tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    protein_align_dir = Path(args.precomputed_alignments_dir) / protein_id

    log(f"Starting CPU-only MMseqs2 pipeline for {protein_id}")
    start_time = time.time()

    try:
        fasta_path = find_fasta_for_protein(fasta_root, protein_id)
        log(f"Using FASTA: {fasta_path}")

        a3m_path = run_mmseqs(
            args.mmseqs_bin,
            fasta_path,
            mmseqs_db,
            protein_align_dir,
            tmpdir,
            threads=args.threads,
            protein_id=protein_id,
            db_load_mode=args.db_load_mode,
        )

        hhr_path = run_hhsearch(
            args.hhsearch_bin,
            a3m_path,
            pdb70_db,
            protein_align_dir,
        )

        log(f"Generated alignments for {protein_id}")
        log(f"A3M: {a3m_path}")
        log(f"HHR: {hhr_path}")

    except Exception as e:
        log(f"ERROR in pipeline for '{protein_id}': {e}")
        raise

    log(f"MMseqs2 pipeline for '{protein_id}' completed in {time.time() - start_time:.1f} seconds")


if __name__ == "__main__":
    os.environ["LD_LIBRARY_PATH"] = "/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LIBRARY_PATH"] = "/lib/x86_64-linux-gnu:" + os.environ.get("LIBRARY_PATH", "")
    main()
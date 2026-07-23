#!/usr/bin/env python3
"""
Concatenate score_zinc_surrogates.py's per-chunk output
(<output-dir>/chunks/chunk_*.csv.gz) into the single combined file the rest of
the pipeline (build_zinc_holdouts.py, build_zinc_reference.py,
build_zinc_surfactant_subset.py) reads as --scored.

Usage:
    python workflow/combine_zinc_chunks.py \
        --chunks-dir data/ZINC/scored/chunks \
        --out data/ZINC/zinc_scored_9props.csv.gz
"""
import argparse
import glob

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    chunk_paths = sorted(glob.glob(f"{args.chunks_dir}/chunk_*.csv.gz"))
    if not chunk_paths:
        raise SystemExit(f"no chunk files found under {args.chunks_dir}")

    print(f"[combine_zinc_chunks] combining {len(chunk_paths)} chunks...", flush=True)
    frames = []
    n_rows = 0
    for path in chunk_paths:
        df = pd.read_csv(path)
        n_rows += len(df)
        frames.append(df)
        print(f"  {path}: {len(df)} rows (running total {n_rows})", flush=True)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.out, index=False, compression="gzip")
    print(f"[combine_zinc_chunks] wrote {len(combined)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

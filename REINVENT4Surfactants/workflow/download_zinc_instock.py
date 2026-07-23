#!/usr/bin/env python3
"""
Pull the complete ZINC20 "in-stock" chemical space from files.docking.org: every
tranche across the full molecular-weight x logP grid (121 first-level tranches,
AA-KK), restricted to purchasability codes A/B (in stock) and C (in stock via
agent) -- i.e. molecules that are actually available for purchase, not
"make on demand"/"boutique"/"annotated" ones. All reactivity classes are
included, so this is an unbiased random cross-section of real, synthesizable
small molecules (~11.5M compounds, ~370MB on disk), not pre-selected for any
surfactant-like property -- see data/ZINC/README.md for the full rationale.

Usage:
    python workflow/download_zinc_instock.py --out-dir data/ZINC
"""
import argparse
import gzip
import re
import sys
import time
import urllib.request
import concurrent.futures
from pathlib import Path

BASE = "https://files.docking.org/2D/"
REACTIVITY = list("ABCEGI")
PURCHASABILITY = list("ABC")


def fetch(url, retries=6, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                return data if binary else data.decode("utf-8", "replace")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def download_tranches(out_raw: Path, workers: int) -> list[tuple[str, str]]:
    root_html = fetch(BASE)
    tranches = sorted(set(re.findall(r'href="([A-K][A-K])/"', root_html)))
    print(f"found {len(tranches)} first-level tranches", file=sys.stderr)

    row_re = re.compile(r'<a href="([A-K][A-K][ABCEGI][ABC])\.smi">')

    def list_files(tranche):
        html = fetch(BASE + tranche + "/")
        return [f"{n}.smi" for n in row_re.findall(html)]

    all_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for tranche, names in zip(tranches, ex.map(list_files, tranches)):
            for n in names:
                all_files.append((tranche, n))

    print(f"total files to fetch: {len(all_files)}", file=sys.stderr)

    def download_one(item):
        tranche, name = item
        dest = out_raw / (name + ".gz")
        if dest.exists():
            return name, 0
        data = fetch(BASE + tranche + "/" + name, binary=True)
        with gzip.open(dest, "wb") as f:
            f.write(data)
        return name, len(data)

    done, total_bytes, failed = 0, 0, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, item): item for item in all_files}
        for fut in concurrent.futures.as_completed(futures):
            item = futures[fut]
            try:
                _, nbytes = fut.result()
                total_bytes += nbytes
            except Exception as e:
                failed.append((item, str(e)))
            done += 1
            if done % 200 == 0 or done == len(all_files):
                print(f"{done}/{len(all_files)} done, {total_bytes/1024/1024:.1f} MB, {len(failed)} failed", file=sys.stderr)

    if failed:
        print(f"{len(failed)} failures:", failed[:20], file=sys.stderr)
    return failed


def combine(out_raw: Path, out_combined: Path):
    files = sorted(out_raw.glob("*.smi.gz"))
    print(f"combining {len(files)} shards -> {out_combined}", file=sys.stderr)
    n_total = 0
    with gzip.open(out_combined, "wt") as out:
        out.write("smiles,zinc_id,tranche,reactivity,purchasability\n")
        for fp in files:
            code = fp.name.replace(".smi.gz", "")
            tranche, react, purch = code[:2], code[2], code[3]
            with gzip.open(fp, "rt") as f:
                f.readline()  # header
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    parts = line.split(" ")
                    if len(parts) < 2:
                        continue
                    out.write(f"{parts[0]},{parts[1]},{tranche},{react},{purch}\n")
                    n_total += 1
    print(f"total molecules: {n_total}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="e.g. data/ZINC")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_raw = out_dir / "raw"
    out_raw.mkdir(parents=True, exist_ok=True)

    download_tranches(out_raw, args.workers)
    combine(out_raw, out_dir / "zinc_instock_combined.csv.gz")

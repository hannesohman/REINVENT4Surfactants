#!/usr/bin/env python3
"""
One-time (cached) prep for checking generated molecules against ChEMBL:
downloads ChEMBL's "chemical representations" file (canonical SMILES +
standard InChI/InChIKey per compound, ~290MB compressed for release 37 --
far smaller than the full relational database, since only identity lookup is
needed here, not bioactivity data) and caches just the InChIKey set (full,
for exact-compound matching) plus a skeleton-level set (first 14 characters
of the InChIKey, which encode only the connectivity/skeleton -- not
stereochemistry, tautomer, or salt/counterion state -- so this catches
"same molecule modulo stereochemistry or salt form" matches too, which
matters here since generated SMILES don't specify stereochemistry and many
ChEMBL entries are salts of the compound REINVENT would generate as the free
form).

Usage:
    python workflow/build_chembl_reference.py \
        --out-dir data \
        [--chemreps-file data/chembl_37_chemreps.txt.gz]  # skip download if already present
        [--url https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_37_chemreps.txt.gz]
"""
import argparse
import gzip
import json
import os
import urllib.request

DEFAULT_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_37_chemreps.txt.gz"


def download(url: str, out_path: str):
    print(f"downloading {url} -> {out_path} ...", flush=True)
    urllib.request.urlretrieve(url, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"downloaded {size_mb:.1f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--chemreps-file", default=None, help="skip download, use this local chemreps.txt.gz")
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    chemreps_path = args.chemreps_file
    if chemreps_path is None:
        chemreps_path = f"{args.out_dir}/chembl_chemreps.txt.gz"
        if not os.path.exists(chemreps_path):
            download(args.url, chemreps_path)
        else:
            print(f"reusing existing {chemreps_path}", flush=True)

    print("parsing chemreps file...", flush=True)
    inchikeys_full = set()
    inchikeys_skeleton = set()
    # chembl_id -> first InChIKey seen for it (for reporting hits)
    inchikey_to_chembl_id = {}
    n = 0
    with gzip.open(chemreps_path, "rt") as f:
        header = f.readline()  # chembl_id, canonical_smiles, standard_inchi, standard_inchi_key
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            chembl_id, _smiles, _inchi, inchikey = parts[0], parts[1], parts[2], parts[3]
            if not inchikey:
                continue
            inchikeys_full.add(inchikey)
            inchikeys_skeleton.add(inchikey[:14])
            inchikey_to_chembl_id.setdefault(inchikey, chembl_id)
            n += 1
            if n % 500_000 == 0:
                print(f"  ...{n} compounds parsed", flush=True)

    print(f"parsed {n} compounds, {len(inchikeys_full)} unique InChIKeys, "
          f"{len(inchikeys_skeleton)} unique skeletons", flush=True)

    out_path = f"{args.out_dir}/chembl_reference.json.gz"
    with gzip.open(out_path, "wt") as f:
        json.dump({
            "n_compounds": n,
            "inchikeys_full": sorted(inchikeys_full),
            "inchikeys_skeleton": sorted(inchikeys_skeleton),
            "inchikey_to_chembl_id": inchikey_to_chembl_id,
        }, f)
    print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Gather all 24 combination results (from workflow/run_production_combo.py's
final_result.json files) into one comparison table, across exactly the
metrics requested: novelty, internal diversity, validity, renormalized_score,
ZINC top-100 rediscovery rate, SurfPro top-100 rediscovery rate, and
nn_tanimoto_to_train (structural similarity to the SurfPro dataset).

Usage:
    python workflow/gather_production_results.py \
        --combos-dir runs/production --out comparison_table.csv
"""
import argparse
import glob
import json

import pandas as pd

METRICS = [
    "novelty", "internal_diversity", "validity", "renormalized_score",
    "zinc_top100", "surfpro_top100", "nn_tanimoto_to_train",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos-dir", default="runs/production")
    ap.add_argument("--out", default="comparison_table.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(f"{args.combos_dir}/*/final_result.json"))
    if not paths:
        raise FileNotFoundError(f"no final_result.json found under {args.combos_dir}")

    rows = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        combo = data["combo"]
        agg = data["aggregate"]
        row = {
            "zinc": combo["zinc"], "unc_mode": combo["unc_mode"], "pareto_mode": combo["pareto_mode"],
            "lm_enabled": combo["lm_enabled"],
            "best_hpo_value": data["best_hpo_value"],
            "sigma": data["best_hpo_params"].get("sigma"),
            "learning_rate": data["best_hpo_params"].get("learning_rate"),
            "batch_size": data["best_hpo_params"].get("batch_size"),
        }
        for m in METRICS:
            if m in agg:
                row[f"{m}_mean"] = agg[m]["mean"]
                row[f"{m}_std"] = agg[m]["std"]
            else:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["zinc", "unc_mode", "pareto_mode"]).reset_index(drop=True)
    df.to_csv(args.out, index=False)
    print(f"saved -> {args.out} ({len(df)} combinations)")

    print("\n=== SUMMARY ===")
    cols = ["zinc", "unc_mode", "pareto_mode"] + [f"{m}_mean" for m in METRICS]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()

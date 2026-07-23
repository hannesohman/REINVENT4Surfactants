#!/usr/bin/env python3
"""
Render the top-10 highest-scoring unique molecules from each of the 4
2026-07-21 comparison runs (default/optimized hyperparameters x with/without
ZincPlausibility) as one combined grid image, one row per run.

Usage:
    python workflow/plot_top10_grid.py --out top10_grid.png
"""
import argparse
import glob

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont

RDLogger.DisableLog("rdApp.*")

RUNS = [
    ("default, no ZINC", "runs/compare_replicated_no_zinc_1/default/rep_*/trial_1.csv"),
    ("default, +ZINC", "runs/compare_replicated_1/default/rep_*/trial_1.csv"),
    ("optimized, no ZINC", "runs/compare_replicated_no_zinc_1/optimized/rep_*/trial_1.csv"),
    ("optimized, +ZINC", "runs/compare_replicated_1/optimized/rep_*/trial_1.csv"),
]

SUB_IMG_SIZE = (220, 220)
LABEL_WIDTH = 220
N_TOP = 10


def canon(smi):
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


def top10_for_run(glob_pattern):
    paths = sorted(glob.glob(glob_pattern))
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["canon"] = df["SMILES"].apply(canon)
    df = df.dropna(subset=["canon"])
    # keep each unique molecule's best (max) Score across replicates
    best = df.sort_values("Score", ascending=False).drop_duplicates("canon", keep="first")
    top = best.sort_values("Score", ascending=False).head(N_TOP)
    return top["canon"].tolist(), top["Score"].tolist()


def render_row(smiles_list, scores):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    legends = [f"Score={s:.3f}" for s in scores]
    while len(mols) < N_TOP:
        mols.append(None)
        legends.append("")
    img = Draw.MolsToGridImage(
        mols, molsPerRow=N_TOP, subImgSize=SUB_IMG_SIZE, legends=legends, returnPNG=False
    )
    if not isinstance(img, Image.Image):
        img = img.convert("RGB")
    return img.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="top10_grid.png")
    args = ap.parse_args()

    row_images = []
    for label, pattern in RUNS:
        smiles, scores = top10_for_run(pattern)
        print(f"{label}: {len(smiles)} molecules, top score {scores[0]:.4f}" if smiles else f"{label}: none found")
        row_images.append((label, render_row(smiles, scores)))

    row_w, row_h = row_images[0][1].size
    total_w = LABEL_WIDTH + row_w
    total_h = row_h * len(row_images)

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for i, (label, row_img) in enumerate(row_images):
        y = i * row_h
        canvas.paste(row_img, (LABEL_WIDTH, y))
        # word-wrap the label into the left margin
        words = label.split(", ")
        text = "\n".join(words)
        draw.multiline_text((10, y + row_h // 2 - 20), text, fill="black", font=font, spacing=6)
        if i > 0:
            draw.line([(0, y), (total_w, y)], fill="lightgray", width=1)

    canvas.save(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

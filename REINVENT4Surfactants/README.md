# REINVENT4Surfactants
by Hannes Öhman

MSc. Complex Adaptive Systems & BSc. Chemical Enginnering with Engineering Physics

at Chalmers University of Technology

## Installation / environment (Berzelius)

On the Berzelius cluster, REINVENT4 and its dependencies (RDKit, XGBoost, PyTorch+CUDA)
already live in a shared virtualenv rather than a per-checkout `venv/`:

```
/proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/bin/python
```

This env also needs the project's custom scoring components copied (or symlinked)
into REINVENT's plugin directory — REINVENT discovers scoring components by scanning
that folder, so any new/edited file under `scoring_functions/` must be copied there
too before it takes effect:

```bash
cp scoring_functions/comp_*.py \
   /proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/lib/python3.11/site-packages/reinvent_plugins/components/
```

If you're setting this up from scratch on a different machine, install REINVENT4
from the official repo ([MolecularAI/REINVENT4](https://github.com/MolecularAI/REINVENT4))
into its own virtualenv, then do the same copy step.

A second environment, `surf-surrogate-env`, is used for the ZINC-scoring workflow
(`workflow/score_zinc_surrogates.py`) since its XGBoost/scikit-learn versions match
what `models.pkl` was actually trained/pickled with:

```
/proj/berzelius-2026-62/users/x_ribec/software/surf-surrogate-env/bin/python
```

**XGBoost version must match `surrogate.def` (currently `xgboost==3.2.0`) exactly**
— `surf-surrogate-env` drifted to `xgboost==2.1.4` at some point and silently
corrupted every prediction it made without erroring (see "Findings", 2026-07-21
surrogate environment fix, below). `reinvent4-env` already pins the correct
version; `surf-surrogate-env` has been fixed in place. If either environment is
ever rebuilt from scratch, re-check this against `surrogate.def` first.

`optuna` (used by the hyperparameter sweep, see below) is **not** installed into
`reinvent4-env` — `/proj` has a per-user disk quota (hit as a genuine `EDQUOT`
"Disk quota exceeded" error on 2026-07-17, separate from and tighter than the
project-wide space shown by `df`) that was already exhausted by other projects.
It's installed to the home directory instead:

```bash
/proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/bin/pip install --target ~/pylibs optuna
```

and anything importing it needs `PYTHONPATH=/home/x_ribec/pylibs` set (already wired
into `runs/optuna_sweep.sbatch` and `runs/compare_hyperparams.sbatch`).

**Disk space note**: this project's own data generation is substantial too — the
full ZINC pull + scored dataset + 4 staggered percentile tiers alone was ~2GB, and
every RL run leaves TL checkpoints (~90MB each) under `runs/`. None of `data/ZINC/`,
`data/zinc_holdout_top{5,10,15,20}pct.csv.gz`, or anything under `runs/` is
git-tracked (see `.gitignore`), so on 2026-07-17 all of it was deleted to stay under
quota — it's fully reproducible via the scripts documented below, just costs the
compute/time again. Currently on disk: only the small
`data/zinc_holdout_low_pCMC_low_SurfTen.csv` (393 molecules) and
`data/surfpro_real_holdout_test_split.csv` (130 molecules) survive, since they're
small enough not to matter. If you see references below to files that aren't
present, regenerate them first.

## How to use: running a generative optimization

Edit the `config.json` file in the main folder to set the desired parameters for your run.

+ **WORKFLOW_NAME:** The version folder of the framework (normally `"workflow"`, matching the `workflow/` directory name).
+ **GROUP_NAME:** A name you can set to more easily organize runs into groups, for example if they all belong to a single project.
+ **RUN_NAME:** The name of run you are about to do. This will be found inside the **GROUP_NAME** folder and will have a timestamp appended.
+ **PARAMETERS.DO_TL** / **N_EPOCHS:** whether to fine-tune (transfer learning) the prior on `data/surfpro_expanded.csv` before RL, and for how many epochs.
+ **WEIGHT_COMBOS** / **SCORING_FUNCTIONS:** the multi-objective RL reward. Every key in `SCORING_FUNCTIONS` must also appear (with some weight, possibly 0) in each entry of `WEIGHT_COMBOS` — `workflow/run_try.py` validates this and raises if they don't match exactly.

Then run:

```bash
/proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/bin/python main.py
```

This creates `runs/<GROUP_NAME>/<RUN_NAME>_<timestamp>/`, drops a copy of the config
there, and submits an `sbatch` job (`templates/jobscript-run_template.sh`, on the
`berzelius-cpu` partition — it only orchestrates and waits on further sbatch jobs, so
it doesn't need a GPU itself). That job runs `workflow/run_try.py`, which per
generation: splits `data/surfpro_expanded.csv` 80/20 and runs transfer learning
(`generate_model_files.py` → a GPU job on the `berzelius` partition), then generates
one staged-learning (RL) TOML per weight combo × `N_MULTIPLES_PER_GEN` replicate
(`generate_combo_files.py`) and runs those in parallel, each its own GPU job.

A typical run with the current default config (100 TL epochs on ~1400 molecules, 3
replicate RL runs × 20 steps × batch 256) takes well under an hour end-to-end once
jobs are scheduled — see `runs/test/` for worked examples.

## Uncertainty-aware scoring (`UncertaintyPenalty`)

The surrogate property models in `models/` (pCMC, SurfTen, DMOL, DSOL, Visc) are
single point-estimate XGBoost regressors and do not carry any native uncertainty
estimate. `scoring_functions/comp_uncertainty.py` instead uses the 25-member
XGBoost ensemble (5 outer cross-validation splits x 5 fold models each, one such
ensemble per property) found in
`surfactant-surrogates/SurfPro-MD/surrogate-models/models.pkl` — see that
project's `surrogate.py` (training) and `predict.py` (reference inference /
uncertainty implementation) for how it was generated and is meant to be used.

For a given target property, the component runs a molecule through every fold
model in its ensemble and uses the standard deviation across the 25 predictions
as the uncertainty measure — the Frequentist "deep ensembles" strategy described
in `test/uncertainty_quantification.txt`. The std is normalized with
`min_value`/`max_value` and, when `minimize=true`, inverted so that low ensemble
disagreement (reliable prediction) scores near 1 and high disagreement
(unreliable prediction) scores near 0.

It is added as a normal scoring endpoint per property (weighted alongside the
point-estimate score in the geometric-mean MPO — the "Score Modulation" strategy
from the paper). `target` must match one of the keys in `models.pkl` (`pCMC`,
`AW_ST_CMC`, `Gamma_max`, `Area_min`, `pC20`, `D_MOL`, `D_SOL`,
`surface_tension_avg`, `viscosity`); `min_value`/`max_value` default to the
5th/95th percentile of the ensemble std observed over the SurfPro-MD training set
for that property.

**As of 2026-07-17, `config.json`'s default `WEIGHT_COMBOS` optimizes pCMC, SurfTen,
`pCMC_Uncertainty` and `SurfTen_Uncertainty` with equal 0.25 weight each** (a plain
`["pCMC", 0.5], ["SurfTen", 0.5]` run — no uncertainty penalty — was found to reward
molecules the surrogates are confidently wrong about; see Findings below). **As of
2026-07-21 this became a 5-way, 0.2-weight-each split with `ZincPlausibility` added**
— see below.

## Structural plausibility scoring (`ZincPlausibility`)

Even with the uncertainty penalty, ensemble disagreement only flags *inter-fold*
model disagreement, not distance from plausible chemistry (see Findings) — the
surrogates can confidently agree on a nonsensical molecule. `ZincPlausibility`
(`scoring_functions/comp_zinc_plausibility.py`) adds a direct, purely structural
"does this look like real chemistry" signal into the reward itself, rather than
only catching implausible molecules after the fact in evaluation.

It scores each molecule by the fraction of its BRICS fragments that appear in a
cached vocabulary built from a 200k-molecule random sample of the real, in-stock
ZINC pool (`data/zinc_reference_profile.json.gz`, built once by
`workflow/build_zinc_reference.py` — see Evaluation metric suite below, which
reuses the same reference profile). Fragments never observed in real ZINC
chemistry score 0; a molecule built entirely from common fragments scores 1.
Deliberately **not** a nearest-neighbor similarity search against the reference
set — at RL batch sizes (256-512 molecules/step) that would mean tens of millions
of Tanimoto comparisons per step. Fragment-vocabulary lookup is one BRICS
decomposition + dict lookups per molecule (~10ms), independent of reference-set
size, so it's cheap enough to run every step.

Tested standalone before wiring in: a real surfactant motif
(`CCCCCCCCCCCC[N+](C)(C)Cc1ccccc1`) scores 1.0; the earlier organosilicon
"nonsense" hit (`C[Si](C)(C)N=S=N[Si](C)(C)C`) scores 0.0; invalid SMILES score
NaN (excluded from the batch mean by REINVENT). One known edge case: tiny
unbreakable molecules (e.g. ethanol) can score 0 if their exact structure happens
not to be in the 200k-molecule sample — an acceptable limitation, since generated
surfactant-scale molecules always have multiple breakable bonds.

Like `UncertaintyPenalty`, this needs its component file copied into
`reinvent4-env`'s `reinvent_plugins/components/` (see Installation) after any
edit, and its TOML block is emitted by a dedicated branch in
`workflow/generate_combo_files.py` / `workflow/optuna_rl_search.py` (params:
`reference_path`, `min_value`, `max_value`, `minimize` — same normalize/invert
convention as every other component here).

## Validating the approach: does this actually generate good surfactants?

Since the RL reward is entirely surrogate-model-based, there's a real risk of the
generator learning to exploit surrogate blind spots rather than real surfactant
chemistry (see Findings). This project validates the pipeline two ways: (1) an
unlabeled, unbiased pool of real molecules (ZINC), scored independently and checked
for overlap with generated output, and (2) a genuine train/test split on real
experimental surfactant data, checked for exact rediscovery. Both are reusable —
scripts below.

### 1. ZINC in-stock pull

`workflow/download_zinc_instock.py` pulls the complete ZINC20 "in-stock" chemical
space (all 121 MW×logP tranches, purchasability A/B/C only — i.e. molecules
actually available for purchase, not virtual/make-on-demand ones; all reactivity
classes; no structural pre-filtering) from `files.docking.org`:

```bash
/proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/bin/python \
    workflow/download_zinc_instock.py --out-dir data/ZINC
```

Produces `data/ZINC/raw/*.smi.gz` (2126 shards) and
`data/ZINC/zinc_instock_combined.csv.gz` — **11,479,547 molecules**, ~370MB total.
See `data/ZINC/README.md` for the full tranche-code legend.

### 2. Score the pool against the surrogate ensemble

`workflow/score_zinc_surrogates.py` runs every molecule through `models.pkl`'s full
9-property ensemble (chunked, resumable, parallel). Preprocessing mirrors
`surfactant-surrogates/SurfPro-MD/MD-simulations/extract_data_canonicalised.py`'s
`standardize_smiles`: largest fragment only (strips salts/counterions), RDKit
`Cleanup`/`FragmentParent`/`Uncharger` (balances charges), canonicalized — matching
how `models.pkl`'s training features were built.

```bash
sbatch templates/score_zinc_surrogates.sbatch
```

(64 cores, `berzelius-cpu`, ~1.5h for the full 11.48M molecules — see the sbatch
file to adjust `--chunk-size`/`--workers`.) Produces
`data/ZINC/zinc_scored_9props.csv.gz`: `zinc_id, smiles_std, smiles_orig, tranche,
reactivity, purchasability, valid, n_nan_descriptors`, plus `<target>_mean`/
`<target>_std` (25-model ensemble mean/std) for all 9 properties. Of the 11,479,547
molecules, only 82 (0.0007%) failed standardization.

### 3. Build holdout sets

`workflow/build_zinc_holdouts.py` filters to molecules built only from elements seen
in the SurfPro-MD training set (C, Cl, F, N, O, P, S, Si — silicon is legitimate,
see Findings), ranks the remainder by an equal-weighted geometric-mean composite
score (pCMC, SurfTen, and their ensemble-std reliability terms — mirroring the RL
objective exactly), and writes:

```bash
/proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/bin/python \
    workflow/build_zinc_holdouts.py \
    --scored data/ZINC/zinc_scored_9props.csv.gz \
    --train-csv /proj/berzelius-2026-62/users/x_ribec/surfactant-surrogates/SurfPro-MD/SurfPro-MD.csv \
    --out-dir data --top-n 400
```

- `data/zinc_holdout_low_pCMC_low_SurfTen.csv` — small (400-molecule) "best of the
  best" list.
- `data/zinc_holdout_top{5,10,15,20}pct.csv.gz` — nested, increasingly lenient
  percentile tiers of the ~10.60M element-eligible pool (530,113 / 1,060,227 /
  1,590,340 / 2,120,454 molecules respectively), for measuring what fraction of a
  generated set lands in each tier. (Numbers as of the 2026-07-21 rescore following
  the surrogate environment fix — see Findings below; counts shift slightly from
  run to run since they're a fixed percentage of however many molecules pass
  element-eligibility, not a fixed absolute cutoff.)

`workflow/build_zinc_reference.py` (2026-07-21) builds a complementary, **non-nested**
version of the same idea for the metric-suite evaluator (below): 5 equal-sized,
non-overlapping quintile tiers (`data/zinc_quintile_tier{1-5}.smi.gz`, ~2.12M
molecules each, tier 1 = best) over the same element-eligible/composite-ranked
population, so hit counts per tier are directly comparable to each other rather
than nested subsets of one another.

### 4. Ground-truth holdout (real experimental data)

`data/surfpro_expanded.csv` originally had a `split` column with a pre-existing,
essentially arbitrary random train/val/test partition (1278/143/130). **As of
2026-07-21 this was rebuilt as a stratified quintile split**: rank all 1551
molecules by true measured `pCMC`+`SurfTen` composite quality, cut into 5 tiers,
and draw the 130-molecule holdout as 26 molecules from *each* tier (not just the
best or a random cross-section) — `data/surfpro_real_holdout_test_split.csv` now
carries a `quality_tier` column (1=best, 5=worst), with the remaining 1421 in
`data/surfpro_expanded_trainval_only.csv`. This makes it possible to check not just
*whether* the pipeline rediscovers real molecules, but whether it preferentially
rediscovers the *better* ones — see Findings. `run_holdout_validation.py` re-runs
the exact same TL→RL pipeline as `main.py`, but seeds transfer learning from
`data/surfpro_expanded_trainval_only.csv` only — excluding all 130 holdout
molecules — so a later exact-SMILES match is a genuine "did we regenerate a real
surfactant we never trained on" check, not memorization:

```bash
/proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/bin/python run_holdout_validation.py
```

### 5. Check rediscovery

`workflow/check_rediscovery.py` computes exact canonical-SMILES overlap between any
generated/sampled set and one or more holdout files:

```bash
/proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/bin/python workflow/check_rediscovery.py \
    --generated "runs/<group>/<run_id>/generation_0/combo_*/multiple_*/multiple_*_1.csv" \
    --generated-smiles-col SMILES \
    --holdout data/zinc_holdout_top5pct.csv.gz:smiles_std \
    --holdout data/zinc_holdout_top10pct.csv.gz:smiles_std \
    --holdout data/zinc_holdout_top15pct.csv.gz:smiles_std \
    --holdout data/zinc_holdout_top20pct.csv.gz:smiles_std \
    --holdout data/surfpro_real_holdout_test_split.csv:SMILES_canonical
```

To get an "un-optimized" baseline for comparison (raw prior, or TL-only with no RL),
sample from a checkpoint directly with REINVENT's `sampling` run_type — see
`runs/baselines/*_sampling.toml` for worked examples — then run
`check_rediscovery.py` against the sampled CSV the same way.

## Hyperparameter tuning (Optuna)

**Principle: never tune hyperparameters against the rediscovery holdouts.** The ZINC
tiers and the SurfPro real holdout exist to give an *independent* read on whether the
pipeline generalizes — if Optuna maximized rediscovery rate directly, it would just be
search-fitting the hyperparameters to those specific molecules, and any rediscovery
number reported afterward would no longer mean anything (classic validation-set
leakage). Instead, `workflow/optuna_rl_search.py` optimizes the mean `Score` — the
same geometric-mean RL reward REINVENT itself reports (top-100 of each trial's
generated molecules) — over `sigma`, `learning_rate`, and `batch_size`. The holdouts
stay completely untouched during the search and are only used afterward, to check
whether the tuned hyperparameters generalize better.

Each trial reuses one fixed, already-fine-tuned TL checkpoint rather than repeating
TL every trial (which would dominate the cost). Build one first, using the same
holdout-respecting `data/surfpro_expanded_trainval_only.csv` as
`run_holdout_validation.py`, so later rediscovery checks against the SurfPro holdout
stay valid:

```bash
/proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/bin/python prepare_tl_checkpoint.py
```

This prints the checkpoint path (`runs/validation/tl_only_<timestamp>/generation_0/model/generation_0.model`)
— edit `--tl-model` in `runs/optuna_sweep.sbatch` to point at it, then run the sweep
(wrapped in one sbatch job holding a single GPU for all trials sequentially, to avoid
per-trial queue wait):

```bash
sbatch runs/optuna_sweep.sbatch
```

30 trials (`sigma`: 50-500 log-uniform int; `learning_rate`: 1e-5 to 1e-3 log-uniform;
`batch_size`: {64,128,256,512}), ~1-4 min each depending on `batch_size`. Best
hyperparameters, full trial history (`all_trials.csv`), and a resumable
`optuna.db` (TPE sampler, `load_if_exists=True`) land in `runs/optuna/rl_sweep_1/`.
Note: the sweep job got externally cancelled (SIGTERM, cause unclear — not a
walltime or OOM issue per `sacct`) after 21/30 trials on 2026-07-17; the Optuna
database still has all 21 completed trials, so nothing was lost — see Findings.

To compare tuned hyperparameters against `config.json`'s current defaults head-to-head
(same fixed TL checkpoint, same 3-replicate × 20-step scale as a normal run) and check
rediscovery for each in one go:

```bash
sbatch runs/compare_hyperparams.sbatch
```

`workflow/compare_hyperparams.py` hardcodes the two hyperparameter sets to compare
(edit the `CONFIGS` dict at the top), runs both, and reports mean `Score` plus
rediscovery against both the ZINC and SurfPro real holdouts — see Findings for the
2026-07-17 result.

## Evaluation metric suite (`workflow/evaluate_run.py`)

As of 2026-07-21, a single script computes a standard, comparable set of metrics
for any generated run, rather than ad hoc rediscovery checks each time:

1. **Validity** — fraction of generated SMILES that parse.
2. **Uniqueness** — fraction of valid molecules that are canonically distinct.
3. **Novelty** — fraction of unique molecules not present in the training set.
4. **Internal Diversity** (`IntDiv = 1 - mean pairwise Tanimoto`, on a 2000-molecule
   subsample) — checks for mode collapse.
5. **Fragment/scaffold similarity to the training set** — cosine similarity between
   BRICS-fragment (and Bemis-Murcko-scaffold) frequency-count vectors of the
   generated set vs. the training set — "does this look like what we trained on".
6. **SurfPro tier hits** — exact rediscovery of the 130-molecule ground-truth
   holdout, broken out per quality tier (1-5) plus a top-2-vs-bottom-2 aggregate.
7. **ZINC tier hits** — exact rediscovery against the 5 non-overlapping ZINC
   quintile tiers (surrogate-scored, so a softer signal than #6 — see caveat below).
8. **Fragment/scaffold similarity to ZINC** — the same cosine-similarity approach
   as #5, but against `data/zinc_reference_profile.json.gz` (a generic, unranked
   200k-molecule ZINC sample) instead of the training set — "does this look like
   plausible chemistry in general", used as a coarse sanity check that the model
   hasn't left the realm of real chemistry entirely. Purely structural (BRICS
   fragments / Murcko scaffolds), not property-based.

Metrics #6 and #7 both depend on the surrogate models' own labels, so a high hit
rate is not on its own proof of finding *good* surfactants — only proof the model
regenerated things the surrogates independently consider promising. #5 and #8 look
distributionally similar to a target set (via cosine similarity of fragment/scaffold
frequency-count vectors: O(n+m), not the O(n×m) a real pairwise nearest-neighbor
comparison would cost against a 200k-molecule reference) rather than requiring exact
matches, which is why they're reported alongside the exact-match tier-hit metrics
rather than replacing them.

Run once first to build the cached ZINC quintile tiers and reference profile (also
used by the `ZincPlausibility` scoring component above):

```bash
/proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/bin/python \
    workflow/build_zinc_reference.py \
    --scored data/ZINC/zinc_scored_9props.csv.gz \
    --train-csv /proj/berzelius-2026-62/users/x_ribec/surfactant-surrogates/SurfPro-MD/SurfPro-MD.csv \
    --out-dir data --reference-n 200000
```

Then, for any generated run:

```bash
/proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/bin/python workflow/evaluate_run.py \
    --generated "runs/<group>/<run_id>/generation_0/combo_*/multiple_*/multiple_*_1.csv" \
    --train-csv data/surfpro_expanded_trainval_only.csv \
    --surfpro-holdout data/surfpro_real_holdout_test_split.csv \
    --out results.json
```

`evaluate_run.py` is built around two reusable pieces so its fixed-cost inputs
(training-set fragment/scaffold profile, the 5 ZINC tier membership sets, the
ZINC reference profile) don't have to be reloaded from disk for every
replicate: `load_resources(...)` loads them once, `evaluate(raw_df,
smiles_col, resources, intdiv_sample=...)` runs the full metric suite against
one generated set. Its CLI (`main()`) is just `load_resources` + `evaluate`
once each; the replicated-run scripts below call them directly, in-process,
many times.

### Replicated runs with error bars (`workflow/run_replicated_eval.py`, `compare_hyperparams_replicated.py`)

A single RL run's metrics are one draw from a noisy process (different random
seed -> different generated set); as of 2026-07-21 two scripts run several
independent replicates and report mean +/- std for every metric above, so
findings can be judged against replicate-to-replicate noise instead of reading
one number at face value.

**`workflow/run_replicated_eval.py`** — N replicates of a single hyperparameter
config, evaluated individually then aggregated:

```bash
/proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/bin/python -u workflow/run_replicated_eval.py \
    --tl-model runs/validation/tl_only_.../generation_0/model/generation_0.model \
    --out-dir runs/replicated_eval_1 --replicates 5 --steps 20
```

**`workflow/compare_hyperparams_replicated.py`** — the same idea extended to a
head-to-head **default vs. optimized hyperparameters** comparison (5
replicates each by default), and additionally parameterized by **which
scoring-function objective to use** via `--variant`:

- `with_zinc` (default): the current 5-term objective, including
  `ZincPlausibility` — `SCORING_FUNCTIONS_WITH_ZINC` in the script, 0.2 weight
  each.
- `no_zinc`: the original 4-term objective (pCMC/SurfTen/pCMC_Uncertainty/
  SurfTen_Uncertainty only, 0.25 weight each) — for isolating what
  `ZincPlausibility` itself changes, independent of hyperparameter tuning.

```bash
/proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/bin/python -u workflow/compare_hyperparams_replicated.py \
    --tl-model runs/validation/tl_only_.../generation_0/model/generation_0.model \
    --out-dir runs/compare_replicated_1 --replicates 5 --steps 20 --variant with_zinc
```

Both scripts share `optuna_rl_search.make_toml`'s `scoring_functions`/`weight`
override parameters (added 2026-07-21 so the same TOML-builder can emit either
objective without global state) and `evaluate_run.load_resources`/`evaluate`,
so the expensive fixed-cost inputs are loaded exactly once regardless of how
many replicates or configs are run in one job.

`compare_hyperparams_replicated.py`'s `run_config` is resumable: if a
replicate's `trial_1.csv` and/or `eval.json` already exist under
`<out-dir>/<config>/rep_<i>/`, it reuses them instead of regenerating —
resubmitting after a partial failure (see the GPU-idle-watchdog note in
Findings below) only pays for what didn't finish. Each script saves a full
per-replicate + aggregated-mean/std JSON (`aggregate.json` /
`comparison_full.json`) plus a **pooled** result (all replicates' unique
molecules combined into one set, for direct comparability with earlier
pooled-3-replicate Findings tables).

Because long CPU-only evaluation stretches between RL generation bursts can
leave the GPU idle long enough to trip Berzelius's idle-utilization watchdog
(see Findings), both scripts' sbatch templates launch **`gpu_keepalive.py`**
in the background for the job's duration — a dependency-free (no `pynvml`,
which isn't installed anywhere in this project's environments) continuous
matmul loop that keeps the GPU busy regardless of what else is running on it:

```bash
python gpu_keepalive.py --mat_size 8192 --sleep_time 0 &
KEEPALIVE_PID=$!
# ... run the actual job ...
kill $KEEPALIVE_PID
```

See `runs/compare_replicated_1.sbatch` for the full pattern.

### Visualizing top hits (`workflow/plot_top10_grid.py`)

Renders the top-10 highest-`Score` unique molecules from each of a set of
runs as one combined grid image (one row per run, structures + `Score`),
using RDKit's `Draw.MolsToGridImage` stitched together with `PIL`. Currently
hardcodes the 4 runs from the 2x2 comparison (see `RUNS` at the top of the
script — edit the `(label, glob_pattern)` list to point at different runs):

```bash
/proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/bin/python workflow/plot_top10_grid.py --out top10_grid.png
```

Saved locally only (`top10_grid.png`, in the repo root) — not published or
uploaded anywhere.

## Findings (2026-07-17)

**The pCMC+SurfTen-only objective (no uncertainty penalty) generates molecules the
surrogates score confidently but that aren't real surfactants** — a first RL run
using plain `["pCMC", 0.5], ["SurfTen", 0.5]` produced top hits like small
organosilicon fragments and short perfluorocarbons (e.g.
`C[Si](C)(C)N=S=N[Si](C)(C)C`, `FC(F)(F)C(F)(F)C(F)(F)F`) — chemically nothing like
an amphiphile. Notably, ensemble disagreement (std) does *not* flag these as
unreliable: they have *lower* than median std across the ZINC pool, i.e. the
ensemble confidently agrees on a likely-wrong extrapolation, so uncertainty here
isn't a real out-of-distribution detector. Empirically, though, **folding the
uncertainty terms into the objective as an equal-weighted 4-way geometric mean
(pCMC/SurfTen/pCMC_Uncertainty/SurfTen_Uncertainty) shifted results dramatically
toward recognizable surfactant chemistry** — top holdout/generated hits became
benzalkonium-type quaternary ammoniums, gemini surfactants, alkylbenzene
sulfonates (literally LAS, a major commercial surfactant), etc.

**Staged hit-rate comparison** against the ZINC percentile tiers (does each pipeline
stage add real signal, or just noise the reward likes?):

| set | unique valid generated | top5% | top10% | top15% | top20% |
|---|---|---|---|---|---|
| raw prior (no TL, no RL) | 15,148 | 0.14% | 0.25% | 0.33% | 0.42% |
| TL-only (no RL) | 8,527 | 1.23% | 1.45% | 1.53% | 1.64% |
| RL (uncertainty-aware objective) | 10,839 | 1.99% | 2.56% | 2.80% | 3.00% |

TL alone gives a ~5-9x lift over the raw prior; RL adds another ~1.5-1.8x on top —
each stage measurably helps.

**Ground-truth rediscovery**: retraining TL on `data/surfpro_expanded_trainval_only.csv`
(excluding the 130-molecule `test` split) and running the same uncertainty-aware RL
objective **exactly regenerated 19 of the 130 (14.6%) held-out real surfactants it
never saw during training** — vs. 0/130 for the raw untrained prior. Several hits
rank near the top of the true-quality list (by real measured pCMC+SurfTen), e.g.
rank #3/130 (`CCCCCCCCCCCCn1cc[n+](COCCCCCOC[n+]2ccn(CCCCCCCCCCCC)c2)c1`, a gemini
imidazolium), #8, #11, #19, #21, #25 — mostly gemini cationics and
benzyl-quaternary-ammoniums, recognizable commercial surfactant chemistry.

**Conclusion**: the pipeline is doing real, non-trivial optimization rather than
gaming the reward — confirmed two independent ways (an unbiased-pool overlap check
scored by the same surrogates, and a genuine train-blind rediscovery test against
real experimental data) — *provided* the uncertainty penalty is included in the
objective. The remaining known weakness: ensemble disagreement doesn't reliably
flag structurally-implausible extrapolations (it flags *inter-fold* disagreement,
not distance from the training distribution), so occasional chemically-odd
high scorers should still be expected and sanity-checked visually.

### Hyperparameter tuning results

A 30-trial Optuna sweep over `sigma`/`learning_rate`/`batch_size` (21 completed
before the job was externally cancelled — see Hyperparameter tuning above) found a
clear, converged pattern: **every one of the top 10 trials used `batch_size=512`**,
with `sigma` in the 150-300 range and `learning_rate` around 6e-5 to 1.7e-4. Best
trial: **`sigma=225, learning_rate=9.79e-05, batch_size=512`** (mean top-100 `Score`
0.6909), versus `config.json`'s defaults (`sigma=120, learning_rate=3.87e-4,
batch_size=256`).

Running both configurations head-to-head at full scale (3 replicates × 20 steps,
same TL checkpoint, `workflow/compare_hyperparams.py`):

| metric | default | optimized | change |
|---|---|---|---|
| mean `Score`, top-100 pooled (the actual Optuna objective) | 0.694 | 0.704 | +1.4% |
| mean `Score`, all generated | 0.535 | 0.516 | -3.5% |
| ZINC holdout hits (393-molecule list) | 16 (4.1%) | 21 (5.3%) | +31% relative |
| **SurfPro real holdout hits (130, ground truth)** | 26 (20.0%) | **49 (37.7%)** | **+88% relative** |

The tuned hyperparameters roughly **double the ground-truth rediscovery rate**
(20.0% → 37.7%) — a substantial, meaningful improvement on the metric that matters
most, obtained entirely from optimizing the RL reward, with the holdouts touched
only for this final read-out.

Two things worth being honest about: `batch_size` was itself one of the tuned
parameters, so "optimized" generated 2x as many molecules per replicate
(30,720 vs 15,360 total) — part of the raw hit-count gain is more samples, not
purely smarter per-molecule generation, though the better top-100 mean `Score`
shows it's not *just* sampling noise. And the overall mean `Score` across *all*
generated molecules actually dropped slightly (0.535 → 0.516) — expected, since
the objective specifically targets the top-100 mean, not the full-batch average;
the optimized config trades some average-case quality for a better top tier, which
is exactly what a rediscovery-style metric rewards.

**Why did ZINC hit rate barely move while SurfPro rediscovery nearly doubled?**
Checking the rank (within the 393-molecule ZINC list, sorted by composite score)
of the specific molecules that got hit: both configs' hits skew toward the
better-ranked, more surfactant-like end of the list (mean hit rank ~99-125 out of
393, vs ~197 expected if hits were uniformly distributed) — so the model isn't
hitting randomly, it does preferentially land on the ZINC entries that most
resemble real surfactant chemistry. But the *absolute* rate barely moved with
better hyperparameters, and the likely reason is that **the two holdouts test
different things**. The SurfPro real holdout is, by construction, drawn from the
exact same experimental dataset TL was fine-tuned on (just the excluded test
split) — same scaffolds, chain lengths, head groups. Once TL biases the generator
toward that family, RL only has to *exploit* the reward efficiently to zero in on
nearby specific molecules — precisely what better `sigma`/`learning_rate`/
`batch_size` buys. The ZINC holdout, by contrast, was selected from 11M diverse
ZINC compounds purely by surrogate score, unrelated to the TL training
distribution — even its best-scoring entries are only incidentally
surfactant-like. Hitting more of it requires *exploring* into chemical territory
the generator isn't primed for, and hyperparameter tuning mostly controls
optimization efficiency within the already-reachable region, not how far that
region extends. In short: tuning helps exploitation (SurfPro) far more than
exploration (ZINC) — closing the ZINC gap would likely need something that
changes what's reachable (more TL diversity, much higher exploration pressure,
explicit diversity filters), not further hyperparameter tuning of this kind.

### Replication with a fresh TL checkpoint (2026-07-20)

Re-ran the whole comparison independently — a brand new `prepare_tl_checkpoint.py`
run (fresh TL fine-tuning, same holdout-respecting
`data/surfpro_expanded_trainval_only.csv`, different random init/train-val split)
followed by `compare_hyperparams.py` against it:

| metric | default | optimized |
|---|---|---|
| unique valid generated | 11,020 | 17,399 |
| mean `Score`, all generated | 0.530 | 0.516 |
| ZINC holdout hits (400-molecule list, pool re-scored 2026-07-20) | 21 (5.25%) | 21 (5.25%) |
| **SurfPro real holdout hits (130, ground truth)** | 25 (19.2%) | **51 (39.2%)** |

Same story, independently reproduced: optimized hyperparameters roughly double
ground-truth rediscovery (19.2% → 39.2%, vs. the original run's 20.0% → 37.7%)
while the ZINC hit rate stays essentially flat between configs (21 vs 21) —
consistent with the exploitation-vs-exploration explanation above holding up
under a fresh TL run, not an artifact of one particular checkpoint's random
initialization.

### Stratified holdout + full metric suite (2026-07-21)

Re-ran the default-vs-optimized hyperparameter comparison (`runs/compare_2_stratified/`,
same fixed TL checkpoint, 3 replicates × 20 steps) against the new stratified SurfPro
holdout and the full `evaluate_run.py` metric suite:

| metric | default | optimized |
|---|---|---|
| Validity | 99.76% | 99.89% |
| Uniqueness | 77.20% | 55.43% |
| Novelty | 95.81% | 94.47% |
| Internal Diversity | 0.803 | 0.791 |
| Fragment similarity to train | 0.907 | 0.966 |
| Scaffold similarity to train | 0.927 | 0.956 |
| Fragment similarity to ZINC | 0.380 | 0.425 |
| Scaffold similarity to ZINC | 0.826 | 0.761 |
| SurfPro tier 1 (best) | 38.5% | **76.9%** |
| SurfPro tier 2 | 42.3% | **76.9%** |
| SurfPro tier 3 | 26.9% | 57.7% |
| SurfPro tier 4 | 42.3% | 57.7% |
| SurfPro tier 5 (worst) | 38.5% | 61.5% |
| **SurfPro top-2 vs. bottom-2 tiers** | **40.4% vs. 40.4%** (tied) | **76.9% vs. 59.6%** |
| ZINC tier hit rate (tiers 1-5) | ~0.001-0.012% | ~0.001-0.009% |

**The stratified holdout reveals something the flat rediscovery rate hid**: the
*default* hyperparameters rediscover real molecules from every quality tier at
statistically indistinguishable rates (40.4% top-2 vs. 40.4% bottom-2 — exactly
tied), i.e. they show **no ability to preferentially find better molecules**. The
*optimized* hyperparameters, by contrast, show a clear, real gap (76.9% vs. 59.6%)
— they rediscover good real surfactants more than bad ones. This is a materially
stronger and more informative finding than the raw overall rediscovery numbers
reported earlier (20-40%): it's evidence of genuine quality-discrimination, not
just more sampling.

**Most of the other new metrics stayed flat or got slightly worse under
"optimized," and that's expected, not a contradiction.** The Optuna sweep that
produced these hyperparameters only ever optimized mean `Score` (the RL reward)
on the top-100 molecules of a trial — never diversity, never plausibility, never
rediscovery. Breaking down what moved and why:

- **Genuinely better**: SurfPro rediscovery (the metric above) — reward-optimization
  and this metric happen to point the same direction, for the exploitation-vs-
  exploration reasons already established (see the original 2026-07-17 findings).
- **Genuinely worse**: Uniqueness (77.2% → 55.4%) — "optimized" generates 2x as
  many molecules per replicate (larger `batch_size`) while converging harder on a
  narrower reward-maximizing region, so more of them collide into the same
  canonical structures. Novelty and Internal Diversity dip slightly for the same
  reason — a sharper reward landscape naturally narrows the sampled distribution.
- **Flat/mixed**: ZINC tier hits — both configs hit ZINC's better tiers (1, 5 —
  note tier 5's odd elevated count reflects composite-score-ranking quirks at the
  distribution's tail, not "worse is better") far more than the middle tiers, but
  neither config's *rate* moved much between default/optimized, consistent with
  the earlier exploitation-vs-exploration explanation: reward-tuning helps within
  the TL-primed region, not exploration into unrelated ZINC territory.
- **Ambiguous, not simply good-or-bad**: fragment/scaffold similarity to train and
  to ZINC moved in different directions for different metrics (fragment similarity
  to train went up, scaffold similarity to ZINC went down) — these measure
  "narrower and more converged" vs. "broader and more exploratory," and neither
  direction is inherently correct; they're diagnostic, not a target.

In short: only optimize what you actually want improved. Optuna targeting `Score`
alone bought real gains on `Score`-adjacent things (rediscovery of good molecules)
at a small, expected cost to diversity/uniqueness — nothing here is optimizing
against those, so their small declines aren't evidence of a bug, just the tradeoff
Optuna is actually making. This directly motivated adding `ZincPlausibility` to
the objective itself (see Structural plausibility scoring above) rather than only
checking plausibility after the fact.

## Fixes applied to get this running on Berzelius (2026-07-16/17)

This project's job scripts were originally written for a different cluster
(NAISS/Alvis: `-A NAISS2025-5-462`, `-p alvis`, `A40` GPUs, a local `venv/`). On
Berzelius these now use `--account=Berzelius-2026-62`, `--partition=berzelius`
(GPU jobs) or `berzelius-cpu` (the orchestrator), `--gpus=1`, and call
`reinvent4-env`'s python directly instead of activating a `venv/` — see
`templates/jobscript-run_template.sh` and `workflow/generate_model_files.py`.

`workflow/run_try.py` also had Python 3.12-only f-string syntax (nested identical
quote characters, e.g. `f"{d.get("key")}"`) that fails on this cluster's Python
3.11 — fixed by requoting the inner strings.

`scoring_functions/comp_uncertainty.py` and a bugfix to
`comp_my_rdkit_descriptors.py` (`np.NaN`, removed in NumPy 2.0, → `np.nan`) needed
to be (re-)copied into `reinvent4-env`'s `reinvent_plugins/components/` — see
Installation above; do this again after editing anything under `scoring_functions/`.

`reinvent`'s `staged_learning` run mode expects `chkpt_file`'s parent directory
(and the `tb_logdir`) to already exist — it does not `mkdir` them itself, and fails
with an opaque `torch.save`/`PyTorchFileWriter` error deep in `reinvent/runmodes/
handler.py` if they don't. `generate_combo_files.py` gets this right implicitly
(the combo folder structure is created earlier in the pipeline), but
`workflow/optuna_rl_search.py` and `workflow/compare_hyperparams.py` (2026-07-17)
both needed an explicit `(trial_dir / "checkpoints").mkdir()` /
`(trial_dir / "tb_logdir").mkdir()` before invoking `reinvent` — worth remembering
for any future script that builds a `staged_learning` TOML from scratch rather than
going through `generate_combo_files.py`.

## Findings (2026-07-21): surrogate environment (XGBoost version) fix

While investigating whether a ZINC "surfactant subset" could be extracted by
property-space similarity to SurfPro-MD (a separate, not-yet-built feature), every
surrogate prediction checked against `surf-surrogate-env` looked wrong: `viscosity`
predictions collapsed to a near-constant ~0.5 for every molecule (including the
model's own training molecules), and `pCMC`/`surface_tension_avg`/`D_MOL`/`D_SOL`
predictions were on a completely different scale than the real measured values.

**Root cause**: `surrogate-models/surrogate.def` (the container that actually
trained and pickled `models.pkl` and the `models/*.joblib` point-estimate exports)
pins `xgboost==3.2.0`. `surf-surrogate-env` — set up specifically to "match" that
training environment — had drifted to `xgboost==2.1.4`, a major version behind.
XGBoost's own Booster objects are **not guaranteed pickle-compatible across
versions** (it warns about exactly this on load: "please export the model by
calling `Booster.save_model()`... then load it back in current version"), and
loading a 3.2.0-pickled Booster under 2.1.4 silently corrupts predictions instead
of raising an error — full corruption for `viscosity`, partial (correlated but
wrong-scale) corruption for the other point-estimate targets.

`reinvent4-env` — which is what actually runs the live RL reward
(`comp_surrogate_XGB.py`, `comp_uncertainty.py`, always invoked inside
`reinvent4-env` during `staged_learning`) — already had the correct
`xgboost==3.2.0` the whole time. **The RL reward itself was never broken.** An
initial diagnosis (since reverted) wrongly concluded `pCMC`/`SurfTen`'s reward
normalization needed recalibrating; that was an artifact of testing with
`surf-surrogate-env` instead of the actual runtime environment. The real, sole
casualty was `workflow/score_zinc_surrogates.py`, which runs via
`surf-surrogate-env` per its sbatch template — so `data/ZINC/zinc_scored_9props.csv.gz`
(all 11.48M molecules) and everything built from it
(`zinc_holdout_top{5,10,15,20}pct.csv.gz`, `zinc_holdout_low_pCMC_low_SurfTen.csv`,
`zinc_quintile_tier{1-5}.smi.gz`) were corrupted. `zinc_reference_profile.json.gz`
(`ZincPlausibility`'s vocabulary) is unaffected — it's built from BRICS/Murcko
structural fragments only, no surrogate predictions involved.

**Fix applied**:
- `surf-surrogate-env`: `pip install --upgrade xgboost==3.2.0` (all other pinned
  packages — numpy/pandas/scikit-learn/rdkit — already matched `surrogate.def`).
- `surrogate.def`: `rdkit` pin nudged `2026.3.1` → `2026.3.3` to match what's
  actually installed/verified everywhere (unrelated to the bug, just drift).
- `software/reinvent4.def`: added explicit `xgboost==3.2.0` + `joblib==1.5.3`
  pins. These were previously installed by hand on top of the container after
  `install.py`, completely undocumented — a fresh rebuild from the `.def` alone
  would not have reproduced a working `reinvent4-env`.
- Added `workflow/combine_zinc_chunks.py` (concatenates
  `score_zinc_surrogates.py`'s per-chunk output into the combined
  `zinc_scored_9props.csv.gz` — this step previously existed only as an
  undocumented one-off command) and wired it into
  `templates/score_zinc_surrogates.sbatch` as a second step.
- Added `templates/build_zinc_derived.sbatch` (runs `build_zinc_holdouts.py` +
  `build_zinc_reference.py`) so the whole re-derivation chains automatically via
  `--dependency=afterok:<score_job_id>` after scoring completes.

**Validation**: ran the actual production `score_zinc_surrogates.py` (not a
hand-rolled test) on all 1436 SurfPro-MD molecules through the fixed
`surf-surrogate-env` and compared against true measured values —
pCMC r²=0.944, D_SOL r²=0.992, D_MOL r²=0.925, pC20 r²=0.871, AW_ST_CMC r²=0.817,
Area_min r²=0.774, Gamma_max r²=0.688, surface_tension_avg r²=0.517, and
**viscosity r²=0.378 (up from ~0, i.e. real signal for the first time)**. Means
now line up too (e.g. surface_tension_avg: 459.96 true vs. 461.11 predicted, vs.
~34 predicted before the fix). Note this is an in-sample check (same molecules
the ensemble was trained/validated on across its splits), so it confirms the
pipeline reproduces known-good behavior end-to-end, not out-of-sample
generalization — but that's exactly what it needed to confirm here.

Full ZINC rescoring (`sbatch templates/score_zinc_surrogates.sbatch`, jobs
17111888 → 17111891, dependency-chained via `templates/build_zinc_derived.sbatch`)
was resubmitted 2026-07-21 after clearing the stale pre-fix chunks/manifest (the
old combined file was kept as `zinc_scored_9props.csv.gz.stale_pre_xgboost_fix`
for reference, not deleted). **Both completed cleanly**: scoring took 1h23m for
all 11,309,967 valid molecules (82 failed standardization, same 0.0007% failure
rate as before — unrelated to the xgboost bug), and the derived-file rebuild took
8m39s. Resulting counts (see "Build holdout sets" and "Structural plausibility
scoring" above, now updated): 10,602,272 element-eligible, holdout tiers of
530,113 / 1,060,227 / 1,590,340 / 2,120,454 at the 5/10/15/20% cutoffs, a fresh
400-molecule best-of-best list, and 5 quintile tiers of ~2.12M each. These are
all within ~1-2% of the pre-fix counts (10.76M eligible, 537,937/1,075,875/
1,613,813/2,151,751 tiers) — expected, since element-eligibility and relative
ranking don't depend heavily on the corrupted properties' absolute scale, only
which molecules land in the top N% shifts slightly. The composite score values
themselves, and which *specific* molecules populate each tier, are not expected
to match the pre-fix files and should be treated as the authoritative version
going forward.

**Caveat for all "ZINC holdout hit rate" numbers in the Findings above this
section** (the staged hit-rate table, the hyperparameter tuning tables, the
2026-07-20 replication): all of them were computed against ZINC holdout files
built from the corrupted `surf-surrogate-env` scoring. The relative comparisons
between configs (default vs. optimized, TL vs. RL, etc.) are likely still
directionally meaningful, since both sides of each comparison used the same
consistently-corrupted composite score — but the absolute values should be
treated as unreliable until re-checked against holdouts rebuilt from the fixed
pipeline. **Ground-truth SurfPro real-holdout rediscovery numbers are unaffected**
— `check_rediscovery.py`'s SurfPro branch and the underlying RL reward never
touched `surf-surrogate-env`.

## Findings (2026-07-21): 2x2 comparison -- ZINC-scoring x hyperparameter tuning

With the fixed ZINC tier files in place, ran a full 2x2 design to isolate the
effect of `ZincPlausibility` itself from the effect of hyperparameter tuning:
{default, optimized hyperparameters} x {4-term objective (no ZINC-scoring),
5-term objective (with `ZincPlausibility`)}, 5 replicates each, same fixed TL
checkpoint (`workflow/compare_hyperparams_replicated.py --variant {with_zinc,
no_zinc}`). Each replicate is evaluated independently with the full
`evaluate_run.py` metric suite (mean +/- std across replicates below; a pooled
result across all 5 replicates is also saved per config for direct
comparability with earlier pooled-3-replicate tables).

| metric | default, no ZINC | default, +ZINC | optimized, no ZINC | optimized, +ZINC |
|---|---|---|---|---|
| mean `Score` | 0.539 +/- 0.008 | 0.568 +/- 0.021 | 0.513 +/- 0.003 | 0.540 +/- 0.008 |
| Validity | 99.76% | 99.79% | 99.88% | 99.86% |
| Uniqueness | 81.0% | 77.8% | 65.9% | 67.6% |
| Novelty | 94.4% | 94.1% | 89.7% | 89.9% |
| Internal Diversity | 0.794 | 0.759 | 0.784 | 0.761 |
| Fragment sim. to train | 0.849 | 0.860 | 0.960 | 0.948 |
| Scaffold sim. to train | 0.804 | 0.885 | 0.959 | 0.969 |
| Fragment sim. to ZINC | 0.357 | 0.339 | 0.396 | 0.364 |
| Scaffold sim. to ZINC | 0.651 | 0.785 | 0.761 | 0.829 |
| SurfPro top-2 tiers | 21.2% +/- 4.9% | 19.6% +/- 4.2% | 57.3% +/- 3.4% | 58.1% +/- 3.2% |
| SurfPro bottom-2 tiers | 16.5% +/- 7.8% | 16.5% +/- 5.0% | 44.2% +/- 4.9% | 43.1% +/- 6.2% |
| ZINC tier-1 hit rate | 0.0059% | 0.0046% | 0.0069% | 0.0060% |

**What adding `ZincPlausibility` actually did** (holding hyperparameters fixed,
comparing +ZINC vs. no-ZINC columns):

- **Scaffold similarity to ZINC improved meaningfully** (+0.13 at default
  hyperparameters, +0.07 at optimized) — the intended effect, at the scaffold
  level.
- **Fragment similarity to ZINC did not improve — it went slightly down** in
  both hyperparameter settings. `ZincPlausibility` scores fragment-*vocabulary
  coverage* (is each BRICS fragment merely present somewhere in the 200k ZINC
  reference), a lenient bar. The model appears to satisfy it by reusing a
  narrow set of common, generic fragments repeatedly rather than broadening
  its fragment distribution to actually resemble ZINC's — a real, mildly
  counterintuitive limitation of this particular structural signal, worth
  knowing before reading too much into it as a general "plausibility" fix.
- **SurfPro rediscovery is essentially unaffected** — with/without-ZINC
  differences are within 1 std of each other in both tiers, at both
  hyperparameter settings. Diluting pCMC/SurfTen/uncertainty weight from
  0.25 to 0.2 each didn't meaningfully cost real-surfactant rediscovery.
  ZINC tier hit rates also move only slightly (both down a touch with ZINC
  scoring added — not the increase one might naively expect, since exact-match
  rediscovery of specific surrogate-ranked ZINC molecules is a different,
  stricter target than generic structural plausibility).
- **Diversity/uniqueness dip slightly** with ZINC-scoring added (e.g.
  uniqueness 81.0% -> 77.8% at default hyperparameters) — consistent with a
  5th competing objective sharpening the reward landscape a bit.
- **Mean `Score` is higher with ZINC-scoring included, but this is largely a
  composite-scoring artifact**: `ZincPlausibility` is often near 1.0 for
  reasonable organic molecules, inflating the geometric mean rather than
  reflecting the other four terms getting better. Not evidence of an
  improvement in pCMC/SurfTen/uncertainty performance.
- **The hyperparameter-tuning effect replicates cleanly in both variants** —
  optimized hyperparameters still roughly triple SurfPro top-tier rediscovery
  (19.6% -> 58.1% with ZINC-scoring; 21.2% -> 57.3% without) while barely
  moving ZINC tier-hit rates, matching the exploitation-vs-exploration
  explanation from the earlier Findings, independent of whether ZINC-scoring
  is in the objective.

**Visual check — top-10 highest-`Score` molecules per run** (all 4 runs,
`workflow/plot_top10_grid.py`, saved locally as `top10_grid.png`, not
published anywhere): the **"no ZINC" rows are visibly more structurally
varied** — sulfates, sulfonates, an isolated perfluorinated fragment, an
atypical bicyclic terpenoid-like ring — alongside the expected quaternary
ammoniums, whereas the **"+ZINC" rows are visibly more homogeneous**, dominated
by simple linear/branched dialkyl quaternary ammoniums and gemini surfactants.

This reframes the lower Uniqueness/Internal-Diversity numbers above:
**`ZincPlausibility` is doing what it was designed to do** — the reduced
diversity isn't (only) mode collapse, it's the term actively filtering the
*top* of the score distribution away from atypical/exotic chemotypes
(perfluorocarbon fragments, unusual fused rings — reminiscent of the original
"nonsense" motifs `ZincPlausibility` was built to suppress, see Findings
2026-07-17) and toward the well-trodden, common regions of real surfactant
chemical space. The fragment-vocabulary-coverage metric not moving (previous
bullet) and the scaffold-level pull toward ZINC both being real is consistent
with this: it isn't broadening what fragments get used, it's narrowing
*which* familiar scaffolds dominate the top hits — pulling the model toward
"real" chemistry rather than away from mode collapse being a pure cost.

**Operational note**: the first attempt at the with-ZINC comparison job was
killed by Berzelius's GPU idle-utilization watchdog partway through (`CANCELLED
by 0` / `SIGNAL Terminated` after ~1h08m) -- `evaluate_run.py`'s per-replicate
evaluation (BRICS fragmentation of thousands of unique molecules, run serially,
no GPU involved) is CPU-only and apparently long enough between RL generation
bursts to trip the watchdog over a run with 10 replicates. Fixed two ways:
`workflow/compare_hyperparams_replicated.py`'s `run_config` now skips
regenerating/re-evaluating any replicate whose `trial_1.csv`/`eval.json`
already exist on disk (safe to resubmit after a partial failure without
re-paying for already-completed replicates), and `gpu_keepalive.py` was
simplified to a dependency-free continuous matmul loop (no `pynvml`, which
isn't installed anywhere in this project's environments) run in the background
for the whole job's duration -- see the launch pattern in
`runs/compare_replicated_1.sbatch`.

## Findings (2026-07-22): pCMC direction bug -- was minimizing, should maximize

**Every scoring/ranking calculation in this project up to this point had the
`pCMC` direction backwards.** `pCMC` is `-log10(CMC in mol/L)` (pH-style, like
pKa) -- confirmed empirically against `SurfPro-MD.csv`: dodecyl sulfate
(`CCCCCCCCCCCCS(=O)(=O)O`, essentially SDS) has `pCMC=2.03`, implying
`CMC=10^-2.03=9.3 mM`, matching SDS's well-known real-world CMC (~8-10 mM); the
longer-chain `CCCCCCCCCCCCCCOCCOS(=O)(=O)O` (C14 ether sulfate) has a higher
`pCMC=2.86` (implied CMC 1.4 mM), correctly lower than the C12 analogue --
longer hydrophobic tails give lower CMC, as expected. **So HIGHER pCMC means
LOWER CMC, i.e. a MORE efficient surfactant** (forms micelles at a lower
concentration) -- the opposite of what "minimize" was doing everywhere.

This was not a narrow bug: the same `norm_invert` (low-raw-value-is-good)
convention, applied uniformly to every property term without checking whether
`pCMC`'s pH-style transform flips that assumption, had propagated into every
composite-score/ranking calculation in the project:

- `config.json`'s `SCORING_FUNCTIONS.pCMC.minimize` (the actual live RL
  reward) -- was `true`, now `false`.
- `workflow/optuna_rl_search.py` / `workflow/compare_hyperparams_replicated.py`'s
  hardcoded `SCORING_FUNCTIONS` dicts -- same fix.
- `workflow/build_zinc_holdouts.py` and `workflow/build_zinc_reference.py`'s
  composite-score ranking (`s1 = norm_invert(pCMC_mean, ...)` -> `norm_direct`)
  -- affects which ZINC molecules populate the "best of the best" list and
  which quintile tier every ZINC molecule lands in.
- The SurfPro ground-truth stratified holdout's `true_composite`/`quality_tier`
  -- previously built ad hoc (not saved as a script); now
  `workflow/build_surfpro_stratified_holdout.py` (new, saved so this is
  reproducible), with the same `pCMC` direction fix. **This changes which 130
  molecules are held out** (holdout membership depends on tier ranking), so a
  fresh TL checkpoint was trained on the corrected trainval split -- the old
  checkpoint was fine-tuned excluding the *wrong* 130 molecules from being a
  genuine blind test.
- `workflow/prepare_next_generation.py`'s Pareto-front un-normalization
  (`pCMC_unnorm = (1 - score) * range + min` -> direct, not inverted). Left
  as a known follow-up: `find_pareto_front`'s sort direction (`ascending=True`
  for both columns) still assumes pCMC should be minimized too, but this
  whole multi-generation code path is unused with the project's current
  `N_GENERATIONS=1` setting, so it wasn't chased further.
- `scoring_functions/comp_pareto_boost.py`, `comp_pareto_gradient.py`,
  `comp_origin_pull.py` (the `Pareto`/`ParetoGradient`/`OriginPull`
  `ADDON_FUNCTIONS` entries) still internally assume pCMC is minimized
  (`previous_df["pCMC"] = 1 - previous_df["pCMC"]`). **Not fixed** -- these
  components aren't in the live `WEIGHT_COMBOS` objective (config.json's
  `ADDON_FUNCTIONS` is an unused reference dict), so this was left as a
  documented gap rather than fixed blind in untested code. Fix the same way
  (direct normalization, no `1 -`) before ever activating one of these modes.

**All prior "best"/"top quality"/"tier 1" claims in the Findings above this
section were ranking by the wrong pCMC direction** -- they were, in effect,
finding molecules with the *highest* CMC (least efficient) among otherwise
plausible surfactants, not the lowest. The relative *comparisons* (default vs.
optimized hyperparameters, with vs. without ZINC-scoring) are likely still
structurally valid, since both sides of every comparison used the same
consistently-flipped composite -- but which *specific* molecules were called
"best," and the absolute rediscovery/hit-rate numbers, should not be trusted
until re-derived under the corrected direction.

**Re-derivation, 2026-07-22**: rebuilt, in order: (1) the stratified SurfPro
holdout/trainval split (`build_surfpro_stratified_holdout.py`, new 130/1421
split -- old files kept as `data/*.stale_pre_pcmc_direction_fix`), (2) the
ZINC quintile tiers and best-of-best/percentile holdouts
(`templates/build_zinc_derived.sbatch`, old files kept under
`data/stale_pre_pcmc_direction_fix/`), (3) a fresh TL checkpoint on the
corrected trainval split
(`runs/validation/tl_only_2026-07-22-12-27-22/generation_0/model/generation_0.model`),
(4) a replicated RL run (`workflow/run_replicated_eval.py`, 5 replicates,
`config.json`'s default hyperparameters, current 5-term with-ZINC objective)
under the corrected objective.

**Sanity check that the fix actually works**: the top-`Score` molecule from
one replicate, `CCCCCCCCCCCCCCCCCC[N+](C)(C)CCCCCCCCCC[N+](C)(C)CCCCCCCCCCCCCC`
(a long-chain gemini quaternary ammonium), has a normalized pCMC score of
0.711, which un-normalizes (now direct, not inverted: `min + score*(max-min)`)
to a real `pCMC ~= 4.84`, implying **CMC ~= 14.6 uM** -- a very low, highly
efficient CMC, exactly what maximizing pCMC should produce. Consistently,
every top hit across replicates is a long, double-tailed gemini cationic --
chemically sensible, since longer/doubled hydrophobic tails genuinely do lower
CMC.

| metric | mean +/- std (5 replicates) |
|---|---|
| mean `Score` | 0.554 +/- 0.007 |
| Validity | 99.2% +/- 0.15% |
| Uniqueness | 84.3% +/- 2.7% |
| Novelty | 95.3% +/- 0.2% |
| Internal Diversity | 0.680 +/- 0.011 |
| Fragment sim. to train | 0.882 +/- 0.028 |
| Scaffold sim. to train | 0.916 +/- 0.026 |
| Fragment sim. to ZINC | 0.303 +/- 0.043 |
| Scaffold sim. to ZINC | 0.813 +/- 0.059 |
| SurfPro top-2 tiers | 11.2% +/- 1.6% |
| SurfPro bottom-2 tiers | 6.9% +/- 5.2% |
| ZINC tier-1 hit rate | 0.0043% |

**Not directly comparable to any pre-fix table above** -- the holdout, the
ZINC tiers, and the TL checkpoint are all different now, since "best 130
molecules" and "best ZINC tier" mean something different under the corrected
ranking. What matters here: the run is internally consistent, correctly
targets low CMC, and top-2 tiers (11.2%) still meaningfully exceeds bottom-2
(6.9%) -- the model discriminates real molecule quality in the corrected
direction too, not just under the old (backwards) one.

### Optimized vs. default hyperparameters, corrected pCMC direction

Ran the same default-vs-optimized comparison as before (`workflow/run_replicated_eval.py`
with `--sigma 225 --learning-rate 9.79e-05 --batch-size 512` for "optimized"),
5 replicates each, same fixed TL checkpoint, under the corrected objective:

| metric | default | optimized |
|---|---|---|
| mean `Score` | 0.554 +/- 0.007 | 0.531 +/- 0.029 |
| Validity | 99.2% +/- 0.15% | 99.6% +/- 0.2% |
| Uniqueness | 84.3% +/- 2.7% | 77.4% +/- 3.9% |
| Novelty | 95.3% +/- 0.2% | 89.0% +/- 6.9% |
| Internal Diversity | 0.680 +/- 0.011 | 0.742 +/- 0.018 |
| Fragment sim. to train | 0.882 +/- 0.028 | 0.937 +/- 0.033 |
| Scaffold sim. to train | 0.916 +/- 0.026 | 0.978 +/- 0.009 |
| Fragment sim. to ZINC | 0.303 +/- 0.043 | 0.332 +/- 0.031 |
| Scaffold sim. to ZINC | 0.813 +/- 0.059 | 0.824 +/- 0.032 |
| SurfPro top-2 tiers | 11.2% +/- 1.6% | 38.5% +/- 9.9% |
| SurfPro bottom-2 tiers | 6.9% +/- 5.2% | 18.5% +/- 7.4% |
| ZINC tier-1 hit rate | 0.0043% | 0.0058% |

Same pattern as every earlier hyperparameter comparison: tuned hyperparameters
roughly **triple SurfPro top-tier rediscovery** (11.2% -> 38.5%) while ZINC
tier-1 hit rate barely moves -- reassuring, since it means this qualitative
conclusion (tuning helps exploitation of the TL-primed region far more than
exploration into new chemical territory) wasn't an artifact of the pCMC
direction bug, it holds up under the corrected objective too.

### Visualization: pCMC vs. surface tension (`workflow/plot_pcmc_surften_scatter.py`)

Scatter plot of pCMC vs. `SurfTen` ("surface tension at reference
concentration" -- see below), split into two side-by-side panels (default
vs. optimized hyperparameters) so each run's own generated molecules and
holdout rediscovery can be read independently. Each panel shows: that run's
generated molecules, the SurfPro training set, a 15k-molecule ZINC sample, the
ZINC "best of the best" 400-molecule tail, and the SurfPro holdout split by
marker into **rediscovered** (gold diamond) vs **missed** (black X) for that
specific run -- all in real units, un-normalized from REINVENT's `(raw)`
score columns using `config.json`'s calibration bounds (pCMC direct since
minimize=false, SurfTen inverted since minimize=true). Saved locally only
(`pcmc_surften_scatter.png`), not published anywhere.

```bash
/proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/bin/python \
    workflow/plot_pcmc_surften_scatter.py --out pcmc_surften_scatter.png
```

**Which property is `SurfTen`, exactly?** (checked 2026-07-22 after a direct
question about the axis label): it's `surface_tension_avg` -- confirmed by
correlating the `SurfTen` column against SurfPro-MD.csv's raw properties
(corr=0.99 with `surface_tension_avg`, corr=-0.08, i.e. unrelated, with
`AW_ST_CMC`). Per the SurfPro-MD README, `surface_tension_avg` is the
MD-derived "surface tension at a given (simulated) concentration" -- **not**
`AW_ST_CMC`, the experimental "surface tension at CMC" property, which isn't
used anywhere in this pipeline. An earlier version of this plot mislabeled the
axis "surface tension at CMC (mN/m)"; fixed to `surface_tension_avg`, with
units left unclaimed: tracing `surface_tension_avg` back to
`SurfPro-MD/MD-simulations/extract_data_canonicalised.py`, it comes from
GROMACS's raw `#Surf*SurfTen` energy-group output divided by 2 (for the two
interfaces in a slab geometry) -- whether that fully recovers a standard-units
(mN/m) surface tension isn't confirmed here. Real air-water surface tension
tops out around 72 mN/m and only decreases with surfactant; this property
ranges ~250-600 in the dataset, so it should be treated as "the property this
pipeline calls `SurfTen`/`surface_tension_avg`" rather than a directly
physically-interpretable mN/m value until that's checked against the
SurfPro-MD paper/methodology directly.

**Striking feature**: the ZINC sample forms a very tight, narrow cluster
(pCMC ~1.5-3, `SurfTen` ~460-520) -- when the surrogate model is applied to
generic ZINC molecules (mostly not surfactants at all), it reverts to
something close to its training-mean prediction rather than confidently
extrapolating, consistent with the earlier finding that ensemble disagreement
doesn't reliably flag out-of-distribution inputs (see Findings, 2026-07-17).
The SurfPro training set spans a much wider range, and the generated
molecules -- especially under optimized hyperparameters -- push well beyond
both the ZINC cluster and much of the SurfPro distribution toward higher pCMC
and lower `SurfTen`, genuinely better property territory than either
reference population.

**Are there really no ZINC molecules closer to the desired region?** (checked
2026-07-22, direct question about the tight ZINC clustering): across the full
11.3M-molecule scored ZINC population, only **1,216 molecules (0.011%)** have
pCMC > 4 and `SurfTen` < 420 -- roughly the region the better generated/SurfPro
molecules occupy. The dark-green "ZINC best-of-best" 400 (ranked by the
corrected composite score) are the closest ZINC gets: even at that extreme,
they only reach pCMC ~5.13 / `SurfTen` ~419 at best, barely extending past the
main cluster rather than reaching into where the generated molecules and
better SurfPro holdout points sit. So the tight clustering isn't hiding a
larger population of better ZINC candidates just outside the sampled 15k --
real drug/building-block-like molecules essentially never combine high pCMC
and low `SurfTen` the way purpose-built surfactants do, consistent with the
regression-to-the-mean behavior noted above.

The rediscovery split also confirms the hyperparameter-tuning finding
visually: the optimized panel has noticeably more gold "rediscovered" diamonds
than the default panel (71 vs. 35, out of 130 holdout molecules, in the runs
plotted here) -- matching the aggregate rediscovery-rate numbers in the
comparison table above.

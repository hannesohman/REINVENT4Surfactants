# REINVENT4Surfactants
by Hannes Öhman 

 MSc. Complex Adaptive Systems & BSc. Chemical Enginnering with Engineering Physics 

at Chalmers University of Technology

## Installation:
1. Start by installing REINVENT4 in a seperate folder from this project.

Reinvent can installed from the official github:
[REINVENT4](https://github.com/MolecularAI/REINVENT4)

2. Clone this repository into your desired folder.
3. Create a virtual environment for this project.
4. Install REINVENT into that virutal environment using the steps described in the github.
5. Copy (or create symbolic links for) the scoring functions into the components folder of REINVENT in your virtual environment.

    Example:
    ```
    venv/lib/python3.13/site-packages/reinvent_plugins/components
    ```

## How to use:
Edit the `config.json` file contained in the main folder. Here you can set the desired parameters for your run.

+ **WORKFLOW_NAME:** The version folder of the framework.
+ **GROUP_NAME:** A name you can set to more easily organize runs into groups, for example if they all belong to a single project.
+ **RUN_NAME:** The name of run you are about to do. This will be found inside the **GROUP_NAME** folder and will have a timestamp at the end of the folder name.

Then run `python main.py`, this will create the file structure described in the config and make a copy of the config to place there.

It will then start an `sbatch` job.

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
from the paper) — see the `*_Uncertainty` entries (`pCMC_Uncertainty`,
`SurfTen_Uncertainty`, `DMOL_Uncertainty`, `DSOL_Uncertainty`, `Visc_Uncertainty`)
under `ADDON_FUNCTIONS` in `config.json` for the parameters (`model_path`,
`target`, `min_value`, `max_value`) and `workflow/generate_combo_files.py` for how
they are wired into the generated TOML files. `target` must match one of the keys
in `models.pkl` (`pCMC`, `AW_ST_CMC`, `Gamma_max`, `Area_min`, `pC20`, `D_MOL`,
`D_SOL`, `surface_tension_avg`, `viscosity`); `min_value`/`max_value` default to
the 5th/95th percentile of the ensemble std observed over the SurfPro-MD training
set for that property.
import os
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from .component_results import ComponentResults
from .add_tag import add_tag


# ============================================================
# Ensemble-uncertainty scoring component
# ============================================================
#
# The surrogate property models in models/ (pCMC, SurfTen, DMOL, DSOL, Visc) are
# single point-estimate XGBoost regressors and carry no native uncertainty.
#
# surfactant-surrogates/SurfPro-MD/surrogate-models/models.pkl instead stores, for
# each property, a 25-member XGBoost ensemble (5 outer cross-validation splits x 5
# fold models each). Following the Frequentist "deep ensembles" strategy described
# in test/uncertainty_quantification.txt (uncertainty estimated from the
# variability across independently trained models), this component runs a molecule
# through every fold model for a given target property and uses the standard
# deviation of the 25 predictions as the uncertainty measure.
#
# The raw std is normalized to [0, 1] with min_value/max_value (as for the other
# surrogate endpoints) and, when minimize=true, inverted so that low ensemble
# disagreement (reliable prediction) scores close to 1 and high disagreement
# (unreliable prediction) scores close to 0. This score can be combined with the
# other property scoring components in the geometric-mean MPO, implementing the
# Score Modulation (SM) strategy from the paper: the agent is rewarded not only
# for optimizing the predicted properties, but also for staying inside the
# ensemble's region of agreement.
#
# See surfactant-surrogates/SurfPro-MD/surrogate-models/surrogate.py (training)
# and predict.py (reference inference/uncertainty implementation) for how
# models.pkl was generated and is meant to be used.


_DESCRIPTOR_NAMES = [name for name, _ in Descriptors._descList]
_DESCRIPTOR_CALCULATOR = MoleculeDescriptors.MolecularDescriptorCalculator(_DESCRIPTOR_NAMES)


def _compute_rdkit_descriptors(mol) -> dict:
    values = _DESCRIPTOR_CALCULATOR.CalcDescriptors(mol)
    return {f"rdkit-{name}": val for name, val in zip(_DESCRIPTOR_NAMES, values)}


def normalize(values: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    return (values - min_value) / (max_value - min_value)


@add_tag("__parameters")
@dataclass
class Parameters:
    model_path: list[str]
    target: list[str]

    min_value: list[float]
    max_value: list[float]
    minimize: list[bool] = field(default_factory=lambda: [True])


@add_tag("__component")
class UncertaintyPenalty:
    """
    Ensemble-uncertainty scoring component.

    Expects a pickle file (SurfPro-MD's models.pkl) containing:
        {
            target_name: {
                split_idx: {
                    "fold_models": [
                        {"model": xgboost.core.Booster, "scaler": StandardScaler, "features": [...]},
                        ...
                    ],
                    "feature_schema": [...],
                    ...
                },
                ...
            },
            ...
        }
    """

    def __init__(self, parameters: Parameters):
        self.model_path = parameters.model_path[0]
        self.target = parameters.target[0]
        self.min_value = parameters.min_value[0]
        self.max_value = parameters.max_value[0]
        self.minimize = parameters.minimize[0]

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        with open(self.model_path, "rb") as f:
            all_models = pickle.load(f)

        if self.target not in all_models:
            raise KeyError(
                f"Target '{self.target}' not found in {self.model_path}. "
                f"Available targets: {list(all_models.keys())}"
            )

        target_splits = all_models[self.target]

        # Flatten the 5 outer-split x 5 fold ensemble into a single list of models
        self.fold_models = [
            fold_model
            for split_info in target_splits.values()
            for fold_model in split_info["fold_models"]
        ]
        self.feature_schema = next(iter(target_splits.values()))["feature_schema"]

        print(
            f"[UncertaintyPenalty] Loaded {len(self.fold_models)}-model ensemble "
            f"for target '{self.target}' from: {self.model_path}"
        )

    def __call__(self, smiles: List[str]) -> ComponentResults:
        valid_mask = np.zeros(len(smiles), dtype=bool)
        feature_rows = []

        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                print(f"[UncertaintyPenalty] Invalid SMILES: {smi}")
                feature_rows.append([np.nan] * len(self.feature_schema))
                continue

            descriptors = _compute_rdkit_descriptors(mol)
            row = pd.Series(descriptors).reindex(self.feature_schema).to_numpy(dtype=np.float64)
            feature_rows.append(row)
            valid_mask[len(feature_rows) - 1] = True

        X = np.array(feature_rows, dtype=np.float64)

        ensemble_predictions = np.stack(
            [
                fold_model["model"].inplace_predict(fold_model["scaler"].transform(X))
                for fold_model in self.fold_models
            ],
            axis=0,
        )

        ensemble_std = ensemble_predictions.std(axis=0).astype(np.float32)

        scores = np.where(valid_mask, ensemble_std, np.nan).astype(np.float32)
        scores = normalize(scores, self.min_value, self.max_value)

        if self.minimize:
            scores = 1.0 - scores

        return ComponentResults([scores])


if __name__ == "__main__":
    model_path = "/proj/berzelius-2026-62/users/x_ribec/surfactant-surrogates/SurfPro-MD/surrogate-models/models.pkl"

    smiles_list = [
        "CCCCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]",
        "CCCCCCCCCCCCCCCCCC[N+](C)(C)Cc1cccc(CN2CCCCC2)c1",
        "not a smiles",
    ]

    uncertainty = UncertaintyPenalty(Parameters(
        model_path=[model_path],
        target=["pCMC"],
        min_value=[0.0],
        max_value=[1.0],
        minimize=[True],
    ))

    results = uncertainty(smiles_list)
    print(results)

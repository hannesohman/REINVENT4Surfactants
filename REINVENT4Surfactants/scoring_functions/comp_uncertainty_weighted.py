import os
import pickle
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from .component_results import ComponentResults
from .add_tag import add_tag


# ============================================================
# Uncertainty-Weighted Optimization (UWO) scoring component
# ============================================================
#
# Implements the UWO combination rule from Coste et al. 2024 ("Reward Model
# Ensembles Help Mitigate Overoptimization", ICLR 2024): R_UWO = mean - lambda*Var,
# i.e. subtract an uncertainty penalty directly from a property's own score,
# rather than treating the point estimate and the uncertainty as two separate
# terms in the outer geometric-mean MPO (the previous "Score Modulation"
# approach: SurrogateModel + a separate UncertaintyPenalty endpoint, combined
# only via the outer geometric mean -- see comp_uncertainty.py, now superseded
# by this component for pCMC/SurfTen as of 2026-07-22).
#
# Point estimate: from the existing single joblib model (e.g.
# pcmc_model.joblib), normalized/inverted into [0,1] exactly as SurrogateModel
# does (higher = better, regardless of the property's own minimize direction).
#
# Uncertainty: from the 25-member XGBoost ensemble in SurfPro-MD's models.pkl
# (5 outer CV splits x 5 fold models), same source as UncertaintyPenalty -- the
# standard deviation across the 25 predictions, normalized to [0,1] (NOT
# inverted here: 0 = low disagreement/certain, 1 = high disagreement/uncertain).
#
# Combined score = clip(point_score - lambda_weight * uncertainty_score, 0, 1).
# This directly penalizes molecules the ensemble disagrees on, in proportion to
# lambda_weight -- the paper found results fairly robust to lambda's exact
# value (0.05-1.0 all performed reasonably; 0.5 used as this project's default).


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
    # point-estimate model (e.g. models/pcmc_model.joblib)
    model_path: list[str]
    min_value: list[float]
    max_value: list[float]
    minimize: list[bool]

    # uncertainty ensemble (SurfPro-MD's models.pkl)
    uncertainty_model_path: list[str]
    uncertainty_target: list[str]
    uncertainty_min_value: list[float]
    uncertainty_max_value: list[float]

    # UWO penalty weight (lambda in R_UWO = mean - lambda * Var)
    lambda_weight: list[float]


@add_tag("__component")
class UncertaintyWeightedScore:
    def __init__(self, parameters: Parameters):
        self.model_path = parameters.model_path[0]
        self.min_value = parameters.min_value[0]
        self.max_value = parameters.max_value[0]
        self.minimize = parameters.minimize[0]

        self.uncertainty_model_path = parameters.uncertainty_model_path[0]
        self.uncertainty_target = parameters.uncertainty_target[0]
        self.uncertainty_min_value = parameters.uncertainty_min_value[0]
        self.uncertainty_max_value = parameters.uncertainty_max_value[0]

        self.lambda_weight = parameters.lambda_weight[0]

        # --- point-estimate model ---
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        model_package = joblib.load(self.model_path)
        self.point_model = model_package["model"]
        self.point_scaler = model_package.get("scaler", None)
        self.point_features = model_package["features"]
        point_rdkit_names = [f.replace("rdkit-", "") for f in self.point_features]
        self.point_descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
            point_rdkit_names
        )

        # --- uncertainty ensemble ---
        if not os.path.exists(self.uncertainty_model_path):
            raise FileNotFoundError(f"Model file not found: {self.uncertainty_model_path}")
        with open(self.uncertainty_model_path, "rb") as f:
            all_models = pickle.load(f)
        if self.uncertainty_target not in all_models:
            raise KeyError(
                f"Target '{self.uncertainty_target}' not found in {self.uncertainty_model_path}. "
                f"Available targets: {list(all_models.keys())}"
            )
        target_splits = all_models[self.uncertainty_target]
        self.fold_models = [
            fold_model
            for split_info in target_splits.values()
            for fold_model in split_info["fold_models"]
        ]
        self.ensemble_feature_schema = next(iter(target_splits.values()))["feature_schema"]

        print(
            f"[UncertaintyWeightedScore] point model: {self.model_path} | "
            f"uncertainty ensemble: {len(self.fold_models)} models for "
            f"'{self.uncertainty_target}' from {self.uncertainty_model_path} | "
            f"lambda={self.lambda_weight}"
        )

    def __call__(self, smiles: List[str]) -> ComponentResults:
        valid_mask = np.zeros(len(smiles), dtype=bool)
        point_rows = []
        ensemble_rows = []

        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                point_rows.append([0.0] * len(self.point_features))
                ensemble_rows.append([np.nan] * len(self.ensemble_feature_schema))
                continue

            valid_mask[len(point_rows)] = True

            point_values = self.point_descriptor_calculator.CalcDescriptors(mol)
            point_rows.append(point_values)

            descriptors = _compute_rdkit_descriptors(mol)
            row = pd.Series(descriptors).reindex(self.ensemble_feature_schema).to_numpy(dtype=np.float64)
            ensemble_rows.append(row)

        # --- point estimate ---
        X_point = np.array(point_rows, dtype=np.float32)
        bad_mask = ~np.isfinite(X_point)
        if np.any(bad_mask):
            X_point = np.nan_to_num(X_point, nan=0.0, posinf=2e9, neginf=-2e9)
        X_point = np.clip(X_point, -2e9, 2e9)
        if self.point_scaler is not None:
            X_point = self.point_scaler.transform(X_point)
        point_predictions = np.array(self.point_model.predict(X_point), dtype=np.float32).flatten()

        point_score = normalize(point_predictions, self.min_value, self.max_value)
        if self.minimize:
            point_score = 1.0 - point_score

        # --- uncertainty ---
        X_ens = np.array(ensemble_rows, dtype=np.float64)
        ensemble_predictions = np.stack(
            [
                fold_model["model"].inplace_predict(fold_model["scaler"].transform(X_ens))
                for fold_model in self.fold_models
            ],
            axis=0,
        )
        ensemble_std = ensemble_predictions.std(axis=0).astype(np.float32)
        uncertainty_score = normalize(ensemble_std, self.uncertainty_min_value, self.uncertainty_max_value)
        uncertainty_score = np.clip(uncertainty_score, 0.0, 1.0)

        # --- UWO combination: R_UWO = mean - lambda * uncertainty ---
        combined = point_score - self.lambda_weight * uncertainty_score
        combined = np.clip(combined, 0.0, 1.0)
        combined = np.where(valid_mask, combined, np.nan).astype(np.float32)

        return ComponentResults([combined])


if __name__ == "__main__":
    comp = UncertaintyWeightedScore(Parameters(
        model_path=["/proj/berzelius-2026-62/users/x_ribec/surfactant_generation/REINVENT4Surfactants/models/pcmc_model.joblib"],
        min_value=[0.0089955596692448], max_value=[6.79588001734408], minimize=[False],
        uncertainty_model_path=[
            "/proj/berzelius-2026-62/users/x_ribec/surfactant-surrogates/SurfPro-MD/surrogate-models/models.pkl"
        ],
        uncertainty_target=["pCMC"],
        uncertainty_min_value=[0.0476], uncertainty_max_value=[0.6120],
        lambda_weight=[0.5],
    ))
    smiles_list = [
        "CCCCCCCCCCCCCCCCCC[N+](C)(C)CCCCCCCCCC[N+](C)(C)CCCCCCCCCCCCCC",
        "C[Si](C)(C)N=S=N[Si](C)(C)C",
        "not a smiles",
    ]
    print(comp(smiles_list))

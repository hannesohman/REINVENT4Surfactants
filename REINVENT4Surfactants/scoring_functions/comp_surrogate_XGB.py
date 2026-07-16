import os
import joblib
import numpy as np
import xgboost
from dataclasses import dataclass
from typing import List

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors

from .component_results import ComponentResults
from .add_tag import add_tag
# from reinvent_plugins.components.component import Component


# ============================================================
# Parameter dataclass (only ONE allowed per file)
# ============================================================

@add_tag("__parameters")
@dataclass
class SurrogateParameters:
    model_path: list[str]

    min_value: list[float]
    max_value: list[float]
    minimize: list[bool] 


# ============================================================
# XGBoost Surrogate Component
# ============================================================

@add_tag("__component")
class SurrogateModel:
    """
    Generic XGBoost surrogate model scoring component.

    Expects a joblib .pkl file containing:
        {
            "model": trained_xgb_model,
            "scaler": fitted_scaler (optional),
            "features": list_of_rdkit_descriptor_names
        }
    """

    def __init__(self, parameters: SurrogateParameters):
        # super().__init__(parameters)

        self.model_path = parameters.model_path[0]
        self.min_value = parameters.min_value[0]
        self.max_value = parameters.max_value[0]
        self.minimize = parameters.minimize[0]


        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load model package
        model_package = joblib.load(self.model_path)

        self.model = model_package["model"]
        self.scaler = model_package.get("scaler", None)
        self.feature_names = model_package["features"]
        # print(f"[Surrogate] Feature names: {self.feature_names}")

        # Pre-build RDKit descriptor calculator
        self.rdkit_feature_names = [
            f.replace("rdkit-", "") for f in self.feature_names
        ]
        
        self.descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
            self.rdkit_feature_names
        )

        print(f"[Surrogate] Loaded model from: {self.model_path}")
        print(f"[Surrogate] Using {len(self.feature_names)} descriptors")

    # ============================================================
    # Core scoring method
    # ============================================================

    def __call__(self, smiles: List[str]) -> np.ndarray:

        descriptors = []
        valid_mask = []

        for smi in smiles:
            # print(smi)
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                print(f"[Surrogate] Invalid SMILES: {smi}")
                descriptors.append([0.0] * len(self.feature_names))
                valid_mask.append(False)
            else:
                # print(f"[Surrogate] Calculating descriptors for: {smi}")
                values = self.descriptor_calculator.CalcDescriptors(mol)
                descriptors.append(values)
                valid_mask.append(True)

                # print(values)

        X = np.array(descriptors, dtype=np.float32)
        # print(f"[Surrogate] {X.shape} descriptor matrix calculated")

        # detect bad rows to prevent infinite values from breaking XGBoost
        bad_mask = ~np.isfinite(X)

        if np.any(bad_mask):
            print("[Surrogate] Non-finite descriptor values detected")

            rows = np.where(np.any(bad_mask, axis=1))[0]

            for r in rows[:10]:
                print(f"Bad molecule row {r}")
                print("Values:", X[r])

            # Option 1: replace with zeros
            X = np.nan_to_num(
                X,
                nan=0.0,
                posinf=2e9,
                neginf=-2e9
            )

        # optional clipping
        X = np.clip(X, -2e9, 2e9)

        # Apply scaler if present
        if self.scaler is not None:
            # print("[Surrogate] Applying scaler to descriptors")
            X = self.scaler.transform(X)

        # Predict using XGBoost
        predictions = self.model.predict(X)
        
        # Convert to float32 flat array (one score per molecule)
        predictions = np.array(predictions, dtype=np.float32).flatten()

        # Set invalid molecules to NaN (REINVENT uses NaN to indicate failure)
        predictions[~np.array(valid_mask)] = float("nan")

        # Normalize predictions to [0, 1] based on provided min/max values
        predictions = (predictions - self.min_value) / (self.max_value - self.min_value)

        if self.minimize:
            predictions = 1 - predictions
    
        # print(f"[Surrogate] Predictions shape: {predictions.shape}")

        # with open("debug_log.csv", "w") as f:
        #     for smi, pred in zip(smiles, predictions):
        #         f.write(f"{smi},{pred:.4f}\n")

        # ComponentResults expects a list of 1D arrays, one per endpoint
        return ComponentResults([predictions])

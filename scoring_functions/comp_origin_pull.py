import os
import joblib
import numpy as np
import xgboost
from dataclasses import dataclass, field
from typing import List

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors

from .component_results import ComponentResults
from .add_tag import add_tag

from .run_prediction import run_prediction


# ============================================================
# Parameter dataclass (only ONE allowed per file)
# ============================================================

@add_tag("__parameters")
@dataclass
class Parameters:
    SurfTen_model_path: list[str]
    SurfTen_min_value: list[float]
    SurfTen_max_value: list[float]

    pCMC_model_path: list[str]
    pCMC_min_value: list[float]
    pCMC_max_value: list[float]

    distance_exponent: list[float] = field(default_factory=lambda: [1.0])
    max_score: list[float] = field(default_factory=lambda: [1.0])

# ============================================================
# XGBoost Surrogate Component
# ============================================================

@add_tag("__component")
class OriginPull:
    """
    Generic XGBoost surrogate model scoring component.

    Expects a joblib .pkl file containing:
        {
            "model": trained_xgb_model,
            "scaler": fitted_scaler (optional),
            "features": list_of_rdkit_descriptor_names
        }
    """

    def __init__(self, parameters: Parameters):
        self.SurfTen_model_path = parameters.SurfTen_model_path[0]
        self.SurfTen_min_value = parameters.SurfTen_min_value[0]
        self.SurfTen_max_value = parameters.SurfTen_max_value[0]

        self.pCMC_model_path = parameters.pCMC_model_path[0]
        self.pCMC_min_value = parameters.pCMC_min_value[0]
        self.pCMC_max_value = parameters.pCMC_max_value[0]

        self.distance_exponent = parameters.distance_exponent[0]
        self.max_score = parameters.max_score[0]

    def __call__(self, smiles: List[str]) -> np.ndarray:
        print(f"[OriginPull] Running predictions for {len(smiles)} molecules...")
        
        SurfTen_predictions = run_prediction(smiles, self.SurfTen_model_path, normalize=True, min_value=self.SurfTen_min_value, max_value=self.SurfTen_max_value)
        pCMC_predictions = run_prediction(smiles, self.pCMC_model_path, normalize=True, min_value=self.pCMC_min_value, max_value=self.pCMC_max_value)

        # print(f"[OriginPull] SurfTen predictions (normalized): {SurfTen_predictions}")
        # print(f"[OriginPull] pCMC predictions (normalized): {pCMC_predictions}")

        # Calculate straight distance to origin (0, 0) in the normalized property space
        distances = np.sqrt(SurfTen_predictions**2 + pCMC_predictions**2)

        score = self.max_score - distances**self.distance_exponent

        return ComponentResults([score])

if __name__ == "__main__":
    # Example usage
    SurfTen_model_path = "models/final_model_surface_tension_avg.joblib"
    SurfTen_min_value = 173.98984
    SurfTen_max_value = 594.85364

    pCMC_model_path = "models/pcmc_model.joblib"
    pCMC_min_value = 0.0089955596692448
    pCMC_max_value = 6.79588001734408

    distance_exponent = 1.0
    max_score = 1.0

    smiles_list = [
        "CCCCCCCCCc1ccc(OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)OS(=O)[O-])cc1",  
        "CCCCCCCCCCC(CCCC)c1ccc(S(=O)(=O)[O-])cc1",                                             
        "CCCCCCCCOC(=O)C[n+]1cccc(N=Cc2ccc(O)c(OC)c2)c1"                                        
        ]
        # pCMC: 4.79588001734408 SurfTen: 446.00748 pCMC_norm: 0.7053139754 SurfTen_norm: 0.6463317586 -> dist: 0.9566674166 -> score: 0.04333258336
        # pCMC: 3.04000516167158 SurfTen: 267.03305 pCMC_norm: 0.446598085  SurfTen_norm: 0.2210767712 -> dist: 0.4983219725 -> score: 0.5016780275
        # pCMC: 2.60205999132796 SurfTen: 491.67413 pCMC_norm: 0.3820699244 SurfTen_norm: 0.754838715  -> dist: 0.8460253618 -> score: 0.1539746382

    origin_pull = OriginPull(Parameters(
        SurfTen_model_path=[SurfTen_model_path],
        SurfTen_min_value=[SurfTen_min_value],
        SurfTen_max_value=[SurfTen_max_value],
        pCMC_model_path=[pCMC_model_path],
        pCMC_min_value=[pCMC_min_value],
        pCMC_max_value=[pCMC_max_value],
        distance_exponent=[distance_exponent],
        max_score=[max_score]
    ))

    results = origin_pull(smiles_list)
    print(results)
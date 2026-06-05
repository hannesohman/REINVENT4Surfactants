import os
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
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
class Parameters:
    diversity_threshold: list[float] = field(default_factory=lambda: [0.03])
    minimize: list[bool] = field(default_factory=lambda: [False])


# ============================================================
# DiversityBoost component implementation
# ============================================================

@add_tag("__component")
class DiversityBoost:

    def __init__(self, parameters: Parameters):

        self.diversity_threshold = parameters.diversity_threshold[0]

        self.minimize = parameters.minimize[0]

    # ============================================================
    # Core scoring method
    # ============================================================

    def __call__(self, smiles: List[str]) -> np.ndarray:

        mols = [Chem.MolFromSmiles(smi) for smi in smiles]

        n_carbons        = pd.Series([sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C") if mol is not None else None for mol in mols])
        n_not_carbons    = pd.Series([sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() != "C") if mol is not None else None for mol in mols])
        n_atoms          = pd.Series([mol.GetNumAtoms() if mol is not None else None for mol in mols])
        diversity_ratio  = n_not_carbons / n_atoms


        if self.minimize:
            divsersity_mask = (diversity_ratio < self.diversity_threshold)
        else:
            divsersity_mask = (diversity_ratio > self.diversity_threshold)

        predictions = divsersity_mask.values.astype(float)

        return ComponentResults([predictions])


if __name__ == "__main__":
    # Example usage
    parameters = Parameters(diversity_threshold=[0.03], minimize=[False])
    component = DiversityBoost(parameters)

    smiles_list = ["CCO", "CCCC", "CCN", "CCCl", "C1CCCCC1"]
    results = component(smiles_list)
    print(results)
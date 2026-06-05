"""Compute a desired list of RDKit descriptors up to a total of 210"""

__all__ = ["RDKitDescriptors"]
from typing import List
import logging

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
import numpy as np
from pydantic.dataclasses import dataclass
from dataclasses import field

from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag

logger = logging.getLogger("reinvent")

KNOWN_DESCRIPTORS = {d.lower(): d for d, _ in Descriptors._descList}


@add_tag("__parameters")
@dataclass
class Parameters:
    descriptor: list[str]
    min_value: list[float]
    max_value: list[float]
    inverse: list[bool] = field(default_factory=lambda: [False])  # whether to minimize (True) or maximize (False) the descriptor


@add_tag("__component")
class MyRDKitDescriptors:
    def __init__(self, params: Parameters):
        # collect descriptor from all endpoints: only one descriptor per endpoint!
        self.descriptor = params.descriptor[0]
        self.min_value = params.min_value[0]
        self.max_value = params.max_value[0]
        self.inverse = params.inverse[0]

        self.descriptor = self.descriptor.replace("rdkit-", "")

        self.calc = MolecularDescriptorCalculator([self.descriptor]).CalcDescriptors


    def __call__(self, smiles: list[str]) -> ComponentResults:
        print(f"[MyRDKitDescriptors] Computing {self.descriptor} for {len(smiles)} molecules...")
        scores = []

        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            result = self.calc(mol, missingVal=np.NaN)
            scores.append(np.array(result))

        scores = np.array(scores).transpose()

        # Normalize scores
        norm_scores = (scores - self.min_value) / (self.max_value - self.min_value)

        if self.inverse:
            norm_scores = 1.0 - norm_scores

        return ComponentResults(list(norm_scores))


if __name__ == "__main__":
    # Example usage
    min_value = -6.347099999999987
    max_value = 21.657000000000007

    smiles_list = [
        "CCCCCCCCCc1ccc(OCC(C)OCC(C)OCC(C)OS(=O)[O-])cc1",  
        "CCCCCCCCCCCC(CCC)c1ccc(S(=O)(=O)[O-])cc1",                                             
        "CCCCCCCCCC[N+](CCC)(CCC)CCCC[N+](CCC)(CCC)CCCCCCCCCC"                                        
        ]
        # MolLogP: 5.367900000000005    Norm: 0.4183316014 -> score: 0.5816683986
        # MolLogP: 3.4843000000000024   Norm: 0.3510700219 -> score: 0.6489299781
        # MolLogP: 11.321599999999991   Norm: 0.6309326134 -> score: 0.3690673866

    my_rdkit_descriptors = MyRDKitDescriptors(Parameters(
        descriptor=["MolLogP"],
        min_value=[min_value],
        max_value=[max_value],
        inverse=[True]
    ))

    results = my_rdkit_descriptors(smiles_list)
    print(results)
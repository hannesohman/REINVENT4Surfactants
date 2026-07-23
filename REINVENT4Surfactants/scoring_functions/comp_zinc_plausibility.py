import gzip
import json
from dataclasses import dataclass
from typing import List

from rdkit import Chem
from rdkit.Chem import BRICS

from .component_results import ComponentResults
from .add_tag import add_tag


# ============================================================
# ZINC fragment-vocabulary plausibility component
# ============================================================
#
# A cheap, purely structural "is this plausible chemistry" gate -- NOT a
# surfactant-quality signal. Scores each molecule by the fraction of its
# BRICS fragments that appear in a cached ZINC reference vocabulary (built
# once by workflow/build_zinc_reference.py from a 200k-molecule random
# sample of the real, purchasable ZINC in-stock pool -- see
# data/zinc_reference_profile.json.gz).
#
# Deliberately avoids per-molecule nearest-neighbor similarity search
# against the ZINC pool (or even the 200k reference sample): at RL batch
# sizes (256-512 molecules/step) that would mean tens of millions of
# Tanimoto comparisons per step, dominating training time. Fragment-
# vocabulary lookup is a single BRICS decomposition + dict lookups per
# molecule (~10ms), independent of reference-set size.
#
# Score = fraction of a molecule's BRICS fragments found in the reference
# vocabulary (fragments never observed in real ZINC chemistry count as 0).
# A molecule with no breakable bonds (BRICS returns the whole molecule as
# one "fragment") is scored on whether that exact structure is itself
# in-vocabulary.


@add_tag("__parameters")
@dataclass
class Parameters:
    reference_path: list[str]
    min_value: list[float]
    max_value: list[float]
    minimize: list[bool]


@add_tag("__component")
class ZincPlausibility:
    def __init__(self, parameters: Parameters):
        self.reference_path = parameters.reference_path[0]
        self.min_value = parameters.min_value[0]
        self.max_value = parameters.max_value[0]
        self.minimize = parameters.minimize[0]

        with gzip.open(self.reference_path, "rt") as f:
            profile = json.load(f)
        self.vocab = set(profile["fragment_counts"].keys())

        print(
            f"[ZincPlausibility] Loaded ZINC fragment vocabulary "
            f"({len(self.vocab)} distinct fragments, from {profile['n_reference']} "
            f"reference molecules): {self.reference_path}"
        )

    def _coverage(self, smi: str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        try:
            frags = list(BRICS.BRICSDecompose(mol))
        except Exception:
            return None
        if not frags:
            return None
        n_in_vocab = sum(1 for frag in frags if frag in self.vocab)
        return n_in_vocab / len(frags)

    def __call__(self, smiles: List[str]) -> ComponentResults:
        import numpy as np

        scores = []
        for smi in smiles:
            cov = self._coverage(smi)
            scores.append(float("nan") if cov is None else cov)

        scores = np.array(scores, dtype=np.float32)
        scores = (scores - self.min_value) / (self.max_value - self.min_value)
        scores = np.clip(scores, 0.0, 1.0)

        if self.minimize:
            scores = 1.0 - scores

        return ComponentResults([scores])


if __name__ == "__main__":
    smiles_list = [
        "CCCCCCCCCCCC[N+](C)(C)Cc1ccccc1",  # real, ZINC-plausible surfactant motif
        "C[Si](C)(C)N=S=N[Si](C)(C)C",       # the earlier "nonsense" hit
        "not a smiles",
    ]
    comp = ZincPlausibility(Parameters(
        reference_path=["data/zinc_reference_profile.json.gz"],
        min_value=[0.0], max_value=[1.0], minimize=[False],
    ))
    print(comp(smiles_list))

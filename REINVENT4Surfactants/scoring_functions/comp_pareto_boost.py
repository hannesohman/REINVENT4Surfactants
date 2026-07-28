import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
import time
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from .component_results import ComponentResults
from .add_tag import add_tag

from .run_prediction import run_prediction


def find_pareto_front(
        df: pd.DataFrame, 
        x_col: str = "SurfTen", 
        y_col: str = "pCMC", 
        tol: float = 1e-12,
        loops: int = 1
        ) -> pd.DataFrame:
    
    if df.empty:
        return df.copy()

    pareto_chunks: list[pd.DataFrame] = []
    
    pareto_data = df.copy()

    for _ in range(loops):
        pareto_data.sort_values([x_col, y_col], ascending=[True, True], inplace=True)

        x = pareto_data[x_col].to_numpy(dtype=float)
        y = pareto_data[y_col].to_numpy(dtype=float)

        valid = np.isfinite(x) & np.isfinite(y)
        pareto_data = pareto_data.loc[valid].copy()
        y = y[valid]
        if pareto_data.empty:
            return pareto_data

        running_min_prev = np.minimum.accumulate(np.r_[np.inf, y[:-1]])
        is_pareto = y < (running_min_prev - tol)

        pareto_chunks.append(pareto_data.loc[is_pareto].copy())

        pareto_data = pareto_data.loc[~is_pareto].copy()

    if not pareto_chunks:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(pareto_chunks, ignore_index=True)

def normalize(
        values: pd.Series,
        min_value: float,
        max_value: float,
        unnormalize: bool = False,
    ) -> np.ndarray:

    if unnormalize:
        return values * (max_value - min_value) + min_value
    else:
        return (values - min_value) / (max_value - min_value)



@add_tag("__parameters")
@dataclass
class Parameters:
    data_path: list[str]

    SurfTen_model_path: list[str]
    SurfTen_min_value: list[float]
    SurfTen_max_value: list[float]

    pCMC_model_path: list[str]
    pCMC_min_value: list[float]
    pCMC_max_value: list[float]

    base_score: list[float] = field(default_factory=lambda: [1.0])
    boost_factor: list[float] = field(default_factory=lambda: [1.0])
    skip: list[int] = field(default_factory=lambda: [0])


@add_tag("__component")
class ParetoBoost:

    def __init__(self, parameters: Parameters):
        self.data_path = parameters.data_path[0]

        self.SurfTen_model_path = parameters.SurfTen_model_path[0]
        self.SurfTen_min_value = parameters.SurfTen_min_value[0]
        self.SurfTen_max_value = parameters.SurfTen_max_value[0]

        self.pCMC_model_path = parameters.pCMC_model_path[0]
        self.pCMC_min_value = parameters.pCMC_min_value[0]
        self.pCMC_max_value = parameters.pCMC_max_value[0]

        self.base_score = parameters.base_score[0]
        self.boost_factor = parameters.boost_factor[0]
        self.skip = parameters.skip[0]

    def __call__(self, smiles: list[str]) -> ComponentResults:
        print(f"[ParetoBoost] Loading previous data from {self.data_path}")

        if not Path(self.data_path).exists() or os.path.getsize(self.data_path) == 0:
            print(f"[ParetoBoost] No previous data found at {self.data_path}, returning default zero scores")
            score = np.zeros(len(smiles))
            return ComponentResults([score])

        previous_df = pd.read_csv(self.data_path)
        print(f"[ParetoBoost] Loaded previous data with {len(previous_df)} entries")

        # SurfTen and pCMC are inverted in the previous data since they are used for scoring, 
        # so we need to invert them back to their original normalized values before finding the pareto front
        previous_df["SurfTen"] = 1 - previous_df["SurfTen"]
        previous_df["pCMC"] = 1 - previous_df["pCMC"]


        # Find the points on the preto front from the previous data
        pareto_df = find_pareto_front(previous_df)
        print(f"Length of pareto front: {len(pareto_df)}")


        # Save the pareto front points to a new csv file for visualization
        time_id = str(int(time.time()))[-6:]
        multiple_folder = Path(self.data_path).parent
        pareto_folder = multiple_folder / "pareto"
        pareto_folder.mkdir(exist_ok=True)

        pareto_file = pareto_folder / f"pareto_front_{time_id}.csv"
        # print(pareto_file.name)
        pareto_df.to_csv(pareto_file, index=False)


        # Predict the scores for the new SMILES
        surften_predictions = run_prediction(smiles, self.SurfTen_model_path)
        surften_predictions = normalize(surften_predictions, self.SurfTen_min_value, self.SurfTen_max_value)

        pCMC_predictions = run_prediction(smiles, self.pCMC_model_path)
        pCMC_predictions = normalize(pCMC_predictions, self.pCMC_min_value, self.pCMC_max_value)
        # pCMC is maximized (higher pCMC = lower CMC = better; see README), but
        # the Pareto front below and previous_df's un-inverted values are both
        # in a "lower = better" frame (matching find_pareto_front's minimize-
        # both-axes assumption). Invert here so fresh candidates use the same
        # frame -- confirmed 2026-07-27 this was the one place still missing
        # the pCMC direction fix; previous_df's own un-inversion needs no
        # change since REINVENT's reported scores are always "higher=better"
        # regardless of a property's own minimize setting.
        pCMC_predictions = 1 - pCMC_predictions

        smiles_df = pd.DataFrame({
            "SMILES": smiles,
            "SurfTen": surften_predictions,
            "pCMC": pCMC_predictions,
            "boost": None
        })

        #Create a mask for all points that are lower and more left than the pareto front, and set their boost to True

        pareto_coords = pareto_df[["SurfTen", "pCMC"]].to_numpy()
        smiles_coords = smiles_df[["SurfTen", "pCMC"]].to_numpy()

        left_mask = smiles_coords[:, 0][:, None] <= pareto_coords[0][0]
        below_mask = smiles_coords[:, 1][:, None] <= pareto_coords[-1][1]
        boost_mask = left_mask | below_mask

        for idx in range(len(pareto_df)-2):
            a_coords = pareto_coords[idx]
            b_coords = pareto_coords[idx+1]

            mask = ((smiles_coords[:, 1][:, None] <= a_coords[1]) & (smiles_coords[:, 0][:, None] <= b_coords[0]))
            boost_mask = boost_mask | mask


        smiles_df["boost"] = boost_mask.any(axis=1)
        print(smiles_df)

        n_to_boost = smiles_df["boost"].sum()
        print(f"[ParetoBoost] Boosting {n_to_boost} molecules")

        score = np.array(smiles_df["boost"]).astype(float)
        score = score * self.boost_factor + self.base_score

        return ComponentResults([score])



if __name__ == "__main__":

    import matplotlib.pyplot as plt

    smiles = [
        "CCCCCCCCCCCCCCCC[N+](C)(C)CCCCCCCCCCCCCCCC[N+](=O)CCCC[N+](C)(CCCC)CCCC[N+](C)(C)CCCCCCCCCCCCCC",
        "CCCCCCCCCCC[N+](C)(C)CCN=Cc1ccc(OC)cc1",
        "CCCCCCCCCCCCCCC[N+](C)(C)CCCCCCC[N+](C)(C)CC[N+](C)CCCCCCCCCCCCCC",
        "CCCCCCCCCCC(C)c1ccc(S(=O)(=O)O)cc1",
        "CCCCCCCCCCCCCCCCCC[N+](C)(C)Cc1cccc(CN2CCCCC2)c1",
        "CCCCCCCCCCCC[N+](C)(C)Cc1ccccc1CO",
        "CCCCCCCCCCCCCC(=O)NCCC[N+](C)(C)CCCCCCCC",
        "CCCCCCCCCCCCCCC[N+](C)(C)CC[N+](C)(C)CCCCCC[N+](C)(C)CCCCCCCCCCCCCC",
        "CCCCCCCCCCCCCCC[N+](C)(C)CCNC(=O)COc1ccccc1",
        "CCCCCCCCCCCCC[N+](C)(C)CCCC[N+](=O)CC[N+](C)(C)CCCCCCCCCCCCCCCCCC[N+](C)(C)C",
        "CCCCCCCCCCCCC[N+](C)(C)CCCCNC(=O)CC[N+](C)(C)CCCCCCCCCCCC",
        "CCCCCCCCCCCCCCCCCCC[N+](C)(C)CCC[N+](C)(C)CCCCCCCCCCCCCCCC",
        "CCCCCCCCCCCCC[N+](C)(C)Cc1ccc(C[N+](C)(C)CCCNC(=O)CCCCCCCCCCC)cc1",
        "CCCCCCCCCCCCC[N+](C)(C)CCSCC[N+](C)=O",
        "CCCCCCCCCCCCCSC(O)C1OC(=O)C(O)C1O",
        "CCCCCCCCCCC[n+]1ccccc1OOCCOCCOCCO",
        "CCCCCCCCCCCCCCC[N+](C)(C)CCCCCCCCCCCCCCCC[N+](C)(C)CCCC[N+](C)(C)CCCCCCCCCCCCCCCC",
        "CCCCCCCCCCCCCCC[N+](C)(C)CCCC[N+](=O)CCCCCCCCCCCC[N+](C)(C)CCCC[N+](C)(C)CCCCCCCCCCCCCCCC",
        "CC(=O)NC1C(OCCCCCCCCCCCCCCCCC(F)(F)C(F)(F)CCCOCCOCCO)OC(CO)C(O)C1O",
        "CCCCCCCCCCCCCCC[N+](C)(C)CCC[N+](C)(C)CCCCCCCCCCCC",
        "CCCCCCCCCCCCCCC[N+](C)(C)cccccc[N+](C)(C)cccccc[N+](C)(C)cccccc"
    ]

    data_path = "kladd/pareto_line_distance/pareto_front_normalized.csv"

    SurfTen_model_path = "models/final_model_surface_tension_avg.joblib"
    SurfTen_min_value = 173.98984
    SurfTen_max_value = 594.85364

    pCMC_model_path = "models/pcmc_model.joblib"
    pCMC_min_value = 0.0089955596692448
    pCMC_max_value = 6.79588001734408


    pareto_boost = ParetoBoost(Parameters(
        data_path=[data_path],

        SurfTen_model_path=[SurfTen_model_path],
        SurfTen_min_value=[SurfTen_min_value],
        SurfTen_max_value=[SurfTen_max_value],

        pCMC_model_path=[pCMC_model_path],
        pCMC_min_value=[pCMC_min_value],
        pCMC_max_value=[pCMC_max_value]
    ))

    scores = pareto_boost(smiles)
    print(scores)
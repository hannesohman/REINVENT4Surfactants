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



def normalize(
    series: pd.Series,
    min_val: float,
    max_val: float,
    unnorm: bool = False,
) -> pd.Series:
    if unnorm:
        return series * (max_val - min_val) + min_val
    else:
        return (series - min_val) / (max_val - min_val)

    





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










def find_closest_pareto_point(
    points: pd.DataFrame, 
    pareto_df: pd.DataFrame,
    x_col: str = "SurfTen",
    y_col: str = "pCMC",
) -> pd.DataFrame:
    point_coords = points[[x_col, y_col]].to_numpy()
    pareto_coords = pareto_df[[x_col, y_col]].to_numpy()

    if len(pareto_coords) == 0:
        raise ValueError("pareto_df must contain at least one point.")

    distances = np.linalg.norm(point_coords[:, None, :] - pareto_coords[None, :, :], axis=2)

    closest_indices = distances.argmin(axis=1)

    points["closest_idx"] = pareto_df.index.to_numpy().astype(int)[closest_indices]
    points["closest_distance"] = distances[np.arange(len(points)), closest_indices]

    return points






def find_closest_point_on_line(
    points: pd.DataFrame,
    pareto_df: pd.DataFrame,
    x_col: str = "SurfTen",
    y_col: str = "pCMC",
    out_x_col: str = "closest_line_SurfTen",
    out_y_col: str = "closest_line_pCMC",
    out_dist_col: str = "closest_line_distance",
    restrict_to_segment: bool = True,
) -> pd.DataFrame:
    """
    For each row in `points` (which must include a `closest_idx` column), compute the nearest
    point on two candidate lines defined by the Pareto front:
      - line A: (closest_idx - 1) -> (closest_idx)
      - line B: (closest_idx) -> (closest_idx + 1)
    If `restrict_to_segment` is True, the projection is clamped to the line segment endpoints.
    Otherwise, the closest point is allowed to lie anywhere on the infinite line.
    The function chooses the closer of the two candidates and returns unnormalized coordinates.
    """

    points = points.copy()



    if len(pareto_df) < 3 or "closest_idx" not in points:
        raise ValueError("pareto_df must contain at least 3 points and points must have 'closest_idx' column.")


    # closest_idx = points["closest_idx"].to_numpy()
    
    left_edgecase_mask = (points["closest_idx"] == 0)
    right_edgecase_mask = (points["closest_idx"] == len(pareto_df) - 1)
    middle_mask = ~(left_edgecase_mask | right_edgecase_mask)

    ## HANDLE LEFT EDGE CASES
    left_edgecase_idx = points.loc[left_edgecase_mask, "closest_idx"].to_numpy().astype(int)
    left_edgecase_coords = points.loc[left_edgecase_mask, [x_col, y_col]].to_numpy()
    b_coords = pareto_df.iloc[left_edgecase_idx][[x_col, y_col]].to_numpy()
    c_coords = pareto_df.iloc[left_edgecase_idx + 1][[x_col, y_col]].to_numpy()

    # Compute the projection of point p onto line BC
    bc = c_coords - b_coords
    bp = left_edgecase_coords - b_coords
    t = np.sum(bp * bc, axis=1) / np.sum(bc * bc, axis=1)
    t = np.clip(t, 0.0, 1.0) if restrict_to_segment else t
    proj_bc = b_coords + t[:, None] * bc
    dist_bc = np.linalg.norm(left_edgecase_coords - proj_bc, axis=1)

    points.loc[left_edgecase_mask, out_x_col] = proj_bc[:, 0]
    points.loc[left_edgecase_mask, out_y_col] = proj_bc[:, 1]
    points.loc[left_edgecase_mask, out_dist_col] = dist_bc



    ## HANDLE RIGHT EDGE CASES
    right_edgecase_idx = points.loc[right_edgecase_mask, "closest_idx"].to_numpy().astype(int)
    right_edgecase_coords = points.loc[right_edgecase_mask, [x_col, y_col]].to_numpy()
    a_coords = pareto_df.iloc[right_edgecase_idx - 1][[x_col, y_col]].to_numpy()
    b_coords = pareto_df.iloc[right_edgecase_idx][[x_col, y_col]].to_numpy()

    # Compute the projection of point p onto line AB
    ab = b_coords - a_coords
    ap = right_edgecase_coords - a_coords
    t = np.sum(ap * ab, axis=1) / np.sum(ab * ab, axis=1)
    t = np.clip(t, 0.0, 1.0) if restrict_to_segment else t
    proj_ab = a_coords + t[:, None] * ab
    dist_ab = np.linalg.norm(right_edgecase_coords - proj_ab, axis=1)

    points.loc[right_edgecase_mask, out_x_col] = proj_ab[:, 0]
    points.loc[right_edgecase_mask, out_y_col] = proj_ab[:, 1]
    points.loc[right_edgecase_mask, out_dist_col] = dist_ab


    ## HANDLE MIDDLE CASES
    middle_idx = points.loc[middle_mask, "closest_idx"].to_numpy().astype(int)
    middle_coords = points.loc[middle_mask, [x_col, y_col]].to_numpy()

    a_coords = pareto_df.iloc[middle_idx - 1][[x_col, y_col]].to_numpy()
    b_coords = pareto_df.iloc[middle_idx][[x_col, y_col]].to_numpy()
    c_coords = pareto_df.iloc[middle_idx + 1][[x_col, y_col]].to_numpy()

    # Compute the projection of point p onto line AB
    ab = b_coords - a_coords
    ap = middle_coords - a_coords
    t = np.sum(ap * ab, axis=1) / np.sum(ab * ab, axis=1)
    t = np.clip(t, 0.0, 1.0) if restrict_to_segment else t
    proj_ab = a_coords + t[:, None] * ab
    dist_ab = np.linalg.norm(middle_coords - proj_ab, axis=1)

    # Compute the projection of point p onto line BC
    bc = c_coords - b_coords
    bp = middle_coords - b_coords
    t = np.sum(bp * bc, axis=1) / np.sum(bc * bc, axis=1)
    t = np.clip(t, 0.0, 1.0) if restrict_to_segment else t
    proj_bc = b_coords + t[:, None] * bc
    dist_bc = np.linalg.norm(middle_coords - proj_bc, axis=1)


    # Choose the closer projection
    use_ab = dist_ab < dist_bc
    final_proj = np.where(use_ab[:, None], proj_ab, proj_bc)

    points.loc[middle_mask, out_x_col] = final_proj[:, 0]
    points.loc[middle_mask, out_y_col] = final_proj[:, 1]
    points.loc[middle_mask, out_dist_col] = np.where(use_ab[:, None], np.array([dist_ab]).T, np.array([dist_bc]).T)

    return points













def score_points(
    points: pd.DataFrame,
    pareto_df: pd.DataFrame,
    x_col: str = "SurfTen",
    y_col: str = "pCMC",
    closest_line_x_col: str = "closest_line_SurfTen",
    closest_line_y_col: str = "closest_line_pCMC",
    dist_col: str = "closest_line_distance",
    exponent: float = 1.0,
) -> pd.DataFrame:
    points = points.copy()
    if dist_col not in points:
        raise ValueError(f"{dist_col} must be computed in points before scoring.")

    # Check if any point on the pareto front is up and to the right for each individual point. If so, assign a score of 1 to that point.
    pareto_coords = pareto_df[[x_col, y_col]].to_numpy()
    point_coords = points[[x_col, y_col]].to_numpy()

    left_past_pareto_mask = point_coords[:, 0] < pareto_coords[0, 0]
    underneath_pareto_mask = points[y_col] < points[closest_line_y_col]
    # underneath_pareto_mask = np.any((point_coords[:, 0][:, None] <= pareto_coords[:, 0]) & (point_coords[:, 1][:, None] <= pareto_coords[:, 1]), axis=1)
    right_past_pareto_mask = point_coords[:, 1] < pareto_coords[-1, 1]

    combined_past_mask = left_past_pareto_mask | underneath_pareto_mask | right_past_pareto_mask

    points.loc[combined_past_mask, "score"] = 1.0

    points.loc[~combined_past_mask, "score"] = (1 - points[dist_col])**exponent
    return points







@add_tag("__parameters")
@dataclass
class Parameters:

    data_path: list[str]

    SurfTen_model_path: list[str] = field(default_factory=list)
    SurfTen_min_value: list[float] = field(default_factory=list)
    SurfTen_max_value: list[float] = field(default_factory=list)

    pCMC_model_path: list[str] = field(default_factory=list)
    pCMC_min_value: list[float] = field(default_factory=list)
    pCMC_max_value: list[float] = field(default_factory=list)

    distance_exponent: list[float] = field(default_factory=lambda: [1.0])
    max_score: list[float] = field(default_factory=lambda: [1.0])


@add_tag("__component")
class ParetoGradient:

    def __init__(self, parameters: Parameters):
        self.data_path = parameters.data_path[0]

        self.SurfTen_model_path = parameters.SurfTen_model_path[0]
        self.SurfTen_min_value = parameters.SurfTen_min_value[0]
        self.SurfTen_max_value = parameters.SurfTen_max_value[0]

        self.pCMC_model_path = parameters.pCMC_model_path[0]
        self.pCMC_min_value = parameters.pCMC_min_value[0]
        self.pCMC_max_value = parameters.pCMC_max_value[0]

        self.max_score = parameters.max_score[0]
        self.distance_exponent = parameters.distance_exponent[0]


    def __call__(self, smiles: list[str]) -> ComponentResults:
        print(f"[ParetoGradient] Loading previous data from {self.data_path}")

        if not Path(self.data_path).is_file() or os.stat(self.data_path).st_size == 0:
            print(f"[ParetoGradient] No previous data found at {self.data_path}, returning max score of {self.max_score} for all molecules.")
            score = np.full(len(smiles), self.max_score)
            return ComponentResults([score])
        
        previous_df = pd.read_csv(self.data_path)

        # The values in the SurfTen and pCMC columns are inverted because they were used for scoring.
        # We need to un-invert them to compute the correct Pareto front.
        previous_df["SurfTen"] = 1 - previous_df["SurfTen"]
        previous_df["pCMC"] = 1 - previous_df["pCMC"]

        pareto_df = find_pareto_front(previous_df)

        if len(pareto_df) < 3:
            # find_closest_point_on_line needs at least 3 Pareto points to
            # define line segments (left/middle/right edge cases) -- with few
            # RL steps/small batches the front can still be this sparse.
            # Falls back the same way as "no previous data yet" (2026-07-27,
            # first hit while smoke-testing this never-before-used component).
            print(
                f"[ParetoGradient] Pareto front has only {len(pareto_df)} point(s), "
                f"need >=3; returning max score of {self.max_score} for all molecules."
            )
            score = np.full(len(smiles), self.max_score)
            return ComponentResults([score])

        surften_predictions = run_prediction(smiles, self.SurfTen_model_path)
        surften_predictions = normalize(surften_predictions, self.SurfTen_min_value, self.SurfTen_max_value)

        pcmc_predictions = run_prediction(smiles, self.pCMC_model_path)
        pcmc_predictions = normalize(pcmc_predictions, self.pCMC_min_value, self.pCMC_max_value)
        # pCMC is maximized (higher pCMC = lower CMC = better; see README), but
        # the Pareto front below and previous_df's un-inverted values are both
        # in a "lower = better" frame (matching find_pareto_front's minimize-
        # both-axes assumption). Invert here so fresh candidates use the same
        # frame -- confirmed 2026-07-27 this was the one place still missing
        # the pCMC direction fix; previous_df's own un-inversion needs no
        # change since REINVENT's reported scores are always "higher=better"
        # regardless of a property's own minimize setting.
        pcmc_predictions = 1 - pcmc_predictions

        smiles_df = pd.DataFrame({
            "SMILES": smiles,
            "SurfTen": surften_predictions,
            "pCMC": pcmc_predictions
        })

        smiles_df = find_closest_pareto_point(smiles_df, pareto_df)
        smiles_df = find_closest_point_on_line(smiles_df, pareto_df)
        smiles_df = score_points(smiles_df, pareto_df, exponent=self.distance_exponent)

        score = smiles_df["score"].to_numpy()

        return ComponentResults([score])



if __name__ == "__main__":

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

    data_path = "pareto_line_distance/pareto_front_normalized.csv"

    SurfTen_model_path = "models/final_model_surface_tension_avg.joblib"
    SurfTen_min_value = 173.98984
    SurfTen_max_value = 594.85364

    pCMC_model_path = "models/pcmc_model.joblib"
    pCMC_min_value = 0.0089955596692448
    pCMC_max_value = 6.79588001734408

    distance_exponent = 1.0
    max_score = 1.0


    pareto_gradient = ParetoGradient(Parameters(
        data_path=[data_path],

        SurfTen_model_path=[SurfTen_model_path],
        SurfTen_min_value=[SurfTen_min_value],
        SurfTen_max_value=[SurfTen_max_value],

        pCMC_model_path=[pCMC_model_path],
        pCMC_min_value=[pCMC_min_value],
        pCMC_max_value=[pCMC_max_value],

        distance_exponent=[distance_exponent],
        max_score=[max_score]
    ))

    scores = pareto_gradient(smiles)
    print(scores)
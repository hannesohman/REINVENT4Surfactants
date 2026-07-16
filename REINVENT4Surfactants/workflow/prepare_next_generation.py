import pandas as pd
from pathlib import Path
import numpy as np

def find_pareto_front(
        df: pd.DataFrame, 
        x_col: str, 
        y_col: str, 
        tol: float = 1e-12,
        loops: int = 4
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


def _coalesce_columns(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    """Move values from source_col into target_col without creating duplicate column names."""
    if source_col not in df.columns:
        return df

    if target_col in df.columns:
        df[target_col] = df[target_col].where(df[target_col].notna(), df[source_col])
        df = df.drop(columns=[source_col])
    else:
        df = df.rename(columns={source_col: target_col})

    return df


def prepare_data(
        meta_config: dict,
        generation: int, 
        n_last_entries: int = None
        ) -> None:
    
    group_name = meta_config.get("GROUP_NAME")
    run_name = meta_config.get("RUN_NAME")
    run_id = meta_config.get("RUN_ID")
    

    cwd = Path.cwd()
    try_path = cwd / "runs" / group_name / run_id

    generation_folder = try_path / f"generation_{generation}"
    combo_folders = list(generation_folder.glob("combo_*"))

    print(f"[{run_name}] (GEN {generation}) {len(combo_folders)} combo folders found...")

    all_data = pd.DataFrame()

    for folder in combo_folders:
        csv_files = list(folder.rglob("multiple_*.csv"))

        print(f"[{run_name}] (GEN {generation})  {folder.name}")

        if not csv_files:
            print(f"[{run_name}] (GEN {generation})   {folder.name}: No CSV files found, skipping.")
            continue
        
        for file in csv_files:
            # print(f"[{try_name}] (GEN {generation})        {file.name}")
            try:
                df = pd.read_csv(file)

                if n_last_entries is not None:
                    df = df.tail(n_last_entries)

                all_data = pd.concat([all_data, df], ignore_index=True)
            except Exception as e:
                print(f"[{run_name}] (GEN {generation})   {file.name}: Error reading CSV - {e}")
                continue

    # print(all_data.head())

    if "pCMC" in all_data.columns:
        all_data["pCMC_unnorm"] = (1 - all_data["pCMC"]) * (6.79588001734408 - 0.0089955596692448) + 0.0089955596692448
    if "SurfTen" in all_data.columns:
        all_data["SurfTen_unnorm"] = (1 - all_data["SurfTen"]) * (594.85364 - 173.98984) + 173.98984

    pareto_df = find_pareto_front(all_data, x_col="SurfTen_unnorm", y_col="pCMC_unnorm")

    print(f"[{run_name}] (GEN {generation}) Pareto front has {len(pareto_df)} entries")

    save_pareto_path = try_path / f"generation_{generation}" / "data" / "pareto_front.csv"
    pareto_df.to_csv(save_pareto_path, index=False)

    last_gen_data = pd.read_csv(try_path / f"generation_{generation}" / "data" / "all_data.csv")
    last_gen_data = _coalesce_columns(last_gen_data, source_col="pCMC", target_col="pCMC_unnorm")
    last_gen_data = _coalesce_columns(last_gen_data, source_col="pred_surface_tension_avg_norm", target_col="SurfTen_unnorm")

    # Guard against accidental duplicate headers in CSV inputs before concat.
    last_gen_data = last_gen_data.loc[:, ~last_gen_data.columns.duplicated()]
    pareto_df = pareto_df.loc[:, ~pareto_df.columns.duplicated()]

    last_gen_data.reset_index(drop=True, inplace=True)
    pareto_df.reset_index(drop=True, inplace=True)

    last_gen_data = pd.concat([last_gen_data, pareto_df], ignore_index=True)
    last_gen_data.drop_duplicates(subset="SMILES", inplace=True)
    last_gen_data.reset_index(drop=True, inplace=True)

    return last_gen_data

def create_next_generation(
        meta_config: dict,
        generation: int
        ) -> None:
    
    group_name = meta_config.get("GROUP_NAME")
    run_name = meta_config.get("RUN_NAME")
    run_id = meta_config.get("RUN_ID")

    print(f"[{run_name}] (GEN {generation}) Creating next generation data based on staged learning results...")

    all_data = prepare_data(meta_config, generation)

    cwd = Path.cwd()
    run_path = cwd / "runs" / group_name / run_id

    next_generation_folder = run_path / f"generation_{generation + 1}"
    data_folder = next_generation_folder / "data"
    data_folder.mkdir(parents=True, exist_ok=True)

    all_data.to_csv(data_folder / "all_data.csv", index=False)


if __name__ == "__main__":
    meta_config = {
        "WORKFLOW_NAME": "synthetic_data",
        "VERSION_FOLDER_NAME": "version_1",
        "TRY_NAME": "try_2"
    }
    create_next_generation(meta_config, 2)
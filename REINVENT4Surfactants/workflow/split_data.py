from pathlib import Path
import pandas as pd


def split_data(
        meta_config: dict,
        generation: int,
        training_fraction: float = 0.8
        ) -> None:
    
    group_name = meta_config.get("GROUP_NAME")
    run_name = meta_config.get("RUN_NAME")
    run_id = meta_config.get("RUN_ID")

    print(f"[{run_name}] (GEN {generation}) Splitting data for generation {training_fraction*100:.0f}/{(1-training_fraction)*100:.0f} train/validation")

    cwd = Path.cwd()
    run_path = cwd / "runs" / group_name / run_id

    generation_folder = run_path / f"generation_{generation}"
    data_folder = generation_folder / "data"

    all_data = pd.read_csv(data_folder / "all_data.csv")

    training_amount = int(len(all_data) * training_fraction)
    shuffled_data = all_data.sample(frac=1).reset_index(drop=True)

    training_data = shuffled_data[:training_amount]
    validation_data = shuffled_data[training_amount:]

    training_data.to_csv(data_folder / "training_data.csv", index=False)
    validation_data.to_csv(data_folder / "validation_data.csv", index=False)

    with open(data_folder / "training_smiles.smi", "w") as f:
        for smi in training_data["SMILES"]:
            f.write(f"{smi}\n")

    with open(data_folder / "validation_smiles.smi", "w") as f:
        for smi in validation_data["SMILES"]:
            f.write(f"{smi}\n")

if __name__ == "__main__":
    GROUP_NAME = "group_1"
    RUN_NAME = "run_1"
    GENERATION = 0

    meta_config = {
        "GROUP_NAME": GROUP_NAME,
        "RUN_NAME": RUN_NAME
    }
    split_data(meta_config, GENERATION)


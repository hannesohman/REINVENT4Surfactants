from pathlib import Path


def surrogate_model_component(
        name: str, 
        weight: float, 
        scoring_package: dict
    ) -> str:
    return f"""
[[stage.scoring.component]]
[stage.scoring.component.SurrogateModel]
[[stage.scoring.component.SurrogateModel.endpoint]]
name = "{name}"
weight = {weight:.2f}
params.model_path = "models/{scoring_package['pkl']}"
params.min_value = {scoring_package['min_value']}
params.max_value = {scoring_package['max_value']}
params.minimize = {str(scoring_package['minimize']).lower()}
"""


def make_multiple_toml(
        parameter_config: dict,
        try_path: Path, 
        generation: int, 
        combo_folder: str, 
        scoring_functions: dict, 
        weight_combo: list,
        idx: int = 0
        ) -> str:
    
    sigma = parameter_config.get("SIGMA")
    max_steps = parameter_config.get("N_STEPS_PER_MULT")

    batch_size = parameter_config.get("BATCH_SIZE", 128)
    learning_rate = parameter_config.get("LEARNING_RATE", 0.0001)

    set_seed = parameter_config.get("SET_SEED", False)
    if set_seed:
        seed_idx = idx
    else:
        seed_idx = "null"

    if parameter_config.get("DO_TL"):
        agent_file_path = f"{try_path}/generation_{generation}/model/generation_{generation}.model"
    else:
        # Fall back on the prior if not doing TL
        agent_file_path = "models/pubchem20250312.prior.12.chkpt"

    
    weight_dictionary = {name: weight for name, weight in weight_combo}

    scoring_components = """"""

    for name, scoring_package in scoring_functions.items():
        weight = weight_dictionary.get(name, 0.0)
    
        if name == "MolLogP":
            scoring_components += f"""
[[stage.scoring.component]]
[stage.scoring.component.MyRDKitDescriptors]
[[stage.scoring.component.MyRDKitDescriptors.endpoint]]
name = "{name}"
weight = {weight:.2f}
params.descriptor = "MolLogP"
params.min_value = {scoring_package['min_value']}
params.max_value = {scoring_package['max_value']}
params.inverse = {str(scoring_package['minimize']).lower()}
"""




        elif name == "Pareto":
            scoring_components += f"""
[[stage.scoring.component]]
[stage.scoring.component.ParetoBoost]
[[stage.scoring.component.ParetoBoost.endpoint]]
name = "{name}"
weight = {weight:.2f}
params.data_path = "{try_path}/generation_{generation}/{combo_folder}/multiple_{idx}/multiple_{idx}_1.csv"
params.skip = {scoring_package.get("skip", 0)}
params.SurfTen_model_path = "models/{scoring_package['SurfTen']['pkl']}"
params.SurfTen_min_value = {scoring_package['SurfTen']['min_value']}
params.SurfTen_max_value = {scoring_package['SurfTen']['max_value']}
params.pCMC_model_path = "models/{scoring_package['pCMC']['pkl']}"
params.pCMC_min_value = {scoring_package['pCMC']['min_value']}
params.pCMC_max_value = {scoring_package['pCMC']['max_value']}
params.base_score = {scoring_package.get("base_score", 1.0)}
params.boost_factor = {scoring_package.get("boost_factor", 1.0)}
"""


        elif name == "OriginPull":
            scoring_components += f"""
[[stage.scoring.component]]
[stage.scoring.component.OriginPull]
[[stage.scoring.component.OriginPull.endpoint]]
name = "{name}"
weight = {weight:.2f}
params.distance_exponent = {scoring_package.get("distance_exponent", 1.0)}
params.max_score = {scoring_package.get("max_score", 1.0)}

params.SurfTen_model_path = "models/{scoring_package['SurfTen']['pkl']}"
params.SurfTen_min_value = {scoring_package['SurfTen']['min_value']}
params.SurfTen_max_value = {scoring_package['SurfTen']['max_value']}
params.pCMC_model_path = "models/{scoring_package['pCMC']['pkl']}"
params.pCMC_min_value = {scoring_package['pCMC']['min_value']}
params.pCMC_max_value = {scoring_package['pCMC']['max_value']}
"""

        elif name == "ParetoGradient":
            scoring_components += f"""
[[stage.scoring.component]]
[stage.scoring.component.ParetoGradient]
[[stage.scoring.component.ParetoGradient.endpoint]]
name = "{name}"
weight = {weight:.2f}
params.data_path = "{try_path}/generation_{generation}/{combo_folder}/multiple_{idx}/multiple_{idx}_1.csv"

params.distance_exponent = {scoring_package.get("distance_exponent", 1.0)}
params.max_score = {scoring_package.get("max_score", 1.0)}

params.SurfTen_model_path = "models/{scoring_package['SurfTen']['pkl']}"
params.SurfTen_min_value = {scoring_package['SurfTen']['min_value']}
params.SurfTen_max_value = {scoring_package['SurfTen']['max_value']}

params.pCMC_model_path = "models/{scoring_package['pCMC']['pkl']}"
params.pCMC_min_value = {scoring_package['pCMC']['min_value']}
params.pCMC_max_value = {scoring_package['pCMC']['max_value']}
"""



        elif name.endswith("_Uncertainty"):
            scoring_components += f"""
[[stage.scoring.component]]
[stage.scoring.component.UncertaintyPenalty]
[[stage.scoring.component.UncertaintyPenalty.endpoint]]
name = "{name}"
weight = {weight:.2f}
params.model_path = "{scoring_package['model_path']}"
params.target = "{scoring_package['target']}"
params.min_value = {scoring_package['min_value']}
params.max_value = {scoring_package['max_value']}
params.minimize = {str(scoring_package.get("minimize", True)).lower()}
"""

        elif name == "ZincPlausibility":
            scoring_components += f"""
[[stage.scoring.component]]
[stage.scoring.component.ZincPlausibility]
[[stage.scoring.component.ZincPlausibility.endpoint]]
name = "{name}"
weight = {weight:.2f}
params.reference_path = "{scoring_package['reference_path']}"
params.min_value = {scoring_package['min_value']}
params.max_value = {scoring_package['max_value']}
params.minimize = {str(scoring_package['minimize']).lower()}
"""

        else:
            scoring_components += f"""
[[stage.scoring.component]]
[stage.scoring.component.SurrogateModel]
[[stage.scoring.component.SurrogateModel.endpoint]]
name = "{name}"
weight = {weight:.2f}
params.model_path = "models/{scoring_package['pkl']}"
params.min_value = {scoring_package['min_value']}
params.max_value = {scoring_package['max_value']}
params.minimize = {str(scoring_package['minimize']).lower()}
"""


## FINAL CONFIG TEXT:


    config_text = f"""
run_type = "staged_learning"
device = "cuda:0"

seed = {seed_idx}

tb_logdir = "{try_path}/generation_{generation}/{combo_folder}/multiple_{idx}/tb_logdir"
json_out_config = "{try_path}/generation_{generation}/{combo_folder}/multiple_{idx}/out_config.json"

[parameters]

prior_file = "models/pubchem20250312.prior.12.chkpt"
agent_file = "{agent_file_path}"
summary_csv_prefix = "{try_path}/generation_{generation}/{combo_folder}/multiple_{idx}/multiple_{idx}"

batch_size = {batch_size}

use_checkpoint = false

[learning_strategy]

type = "dap"
sigma = {sigma}
rate = {learning_rate}

[[stage]]

max_score = 1
max_steps = {max_steps}

chkpt_file = "{try_path}/generation_{generation}/{combo_folder}/multiple_{idx}/checkpoints/multiple_{idx}.chkpt"

[stage.scoring]
type = "geometric_mean"
{scoring_components}
"""


    return f"{config_text}"


def generate_combo_files(
        meta_config: dict,
        parameter_config: dict,
        generation: int, 

        scoring_functions: dict[str, dict], 
        weight_combos: dict[str, list[float]],
        ) -> None:
    
    group_name = meta_config.get("GROUP_NAME")
    run_name = meta_config.get("RUN_NAME")
    run_id = meta_config.get("RUN_ID")

    multiples = parameter_config.get("N_MULTIPLES_PER_GEN")
    
    print(f"[{run_name}] (GEN {generation}) Generating combo TOML files for {multiples} multiples...")
    
    # Determine the path to the try folder based on the current working directory
    cwd = Path.cwd()
    try_path = cwd / "runs" / group_name / run_id

    # Determine the path to the generation folder
    generation_folder = try_path / f"generation_{generation}"

    # Initiate lists to store the generated TOML file paths and combo folder paths
    toml_files = []
    combo_folders = []

    # Loop through each weight combination and generate the corresponding TOML files
    for weight_combo in weight_combos:
        
        # Define and create the folder for the current weight combination
        file_name = "combo_" + "_".join([f"{weight:.2f}" for name, weight in weight_combo])        

        combo_folder = generation_folder / file_name
        combo_folder.mkdir(parents=True, exist_ok=True)

        combo_folders.append(combo_folder)

        # For each weight combination, create the specified number of multiples and generate TOML files for each multiple
        for idx in range(multiples):
            multiple_folder = combo_folder /f"multiple_{idx}"
            multiple_folder.mkdir(parents=True, exist_ok=True)
            
            multiple_toml_file = multiple_folder /f"multiple_{idx}.toml"
            
            config_text = make_multiple_toml(parameter_config, try_path, generation, combo_folder.name, scoring_functions, weight_combo, idx=idx)
            with open(multiple_toml_file, "w") as f:
                f.write(config_text)

            toml_files.append(multiple_toml_file)

    print(f"[{run_name}] (GEN {generation}) Generated {len(toml_files)} combo TOML files")

    return combo_folders

if __name__ == "__main__":
    VERSION_FOLDER_NAME = "synthetic_data_scoring"
    TRY_NAME = "try_1"
    GENERATION = 0
    N_MULTIPLES = 4

    SCORING_FUNCTIONS = {
        "pCMC": { "minimize": False, "pkl": "pcmc_model.joblib",  # higher pCMC = lower CMC = better
            "min_value": 0.0089955596692448, "max_value": 6.79588001734408
        },
        "SurfTen": { "minimize": True, "pkl": "final_model_surface_tension_avg.joblib",
            "min_value": 173.98984, "max_value": 594.85364
        },
        "MolLogP": {"minimize": True,
            "min_value": -6.347099999999987, "max_value": 21.657000000000007
        }
    }

    generate_combo_files(VERSION_FOLDER_NAME, TRY_NAME, GENERATION, N_MULTIPLES, SCORING_FUNCTIONS)


        

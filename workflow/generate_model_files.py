from pathlib import Path


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def make_tl_toml(
    run_path: Path,
    generation: int,
    *,
    input_model_file: str = "models/pubchem20250312.prior.12.chkpt",
    num_epochs: int = 500,
    batch_size: int = 128,
    sample_batch_size: int = 2000,
    device: str = "cuda:0"
) -> str:
    return f"""run_type = "transfer_learning"
device = "{device}"
tb_logdir = "{run_path}/generation_{generation}/model/tb_logdir"

[parameters]
num_epochs = {num_epochs}
save_every_n_epochs = 50
batch_size = {batch_size}
sample_batch_size = {sample_batch_size}

input_model_file = "{input_model_file}"
output_model_file = "{run_path}/generation_{generation}/model/generation_{generation}.model"
smiles_file = "{run_path}/generation_{generation}/data/training_smiles.smi"
validation_smiles_file = "{run_path}/generation_{generation}/data/validation_smiles.smi"

standardize_smiles = true
randomize_smiles = true
randomize_all_smiles = true
shuffle_each_epoch = true
internal_diversity = true

[scheduler]
lr = 1e-4
min = 1e-7
gamma = 0.95
step = 10
"""


def generate_tl_toml(
    meta_config: dict,
    parameter_config: dict,
    generation: int, 
    ) -> Path:
    
    group_name = meta_config.get("GROUP_NAME")
    run_name = meta_config.get("RUN_NAME")
    run_id = meta_config.get("RUN_ID")

    num_epochs = parameter_config.get("N_EPOCHS")

    print(f"[{run_name}] (GEN {generation}) Generating TL TOML file...", flush=True)

    cwd = Path.cwd()
    try_path = cwd / "runs" / group_name / run_id

    model_folder = try_path / f"generation_{generation}" / "model"
    model_folder.mkdir(parents=True, exist_ok=True)

    toml_path = model_folder / f"TL_generation_{generation}.toml"
    return _write_text(toml_path, make_tl_toml(try_path, generation, num_epochs=num_epochs))


def make_tl_jobscript(
    run_path: Path,
    generation: int,
    walltime: str = "02:00:00",
    gpus_per_node: str = "A40:1"
    ) -> str:
    return f"""#!/bin/bash
#SBATCH -A NAISS2025-5-462
#SBATCH -p alvis
#SBATCH -t {walltime}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH -J TL_{run_path.name}_g{generation}
#SBATCH -o {run_path}/generation_{generation}/model/TL_generation_{generation}.out

module purge
module load virtualenv/20.32.0-GCCcore-14.3.0 SciPy-bundle/2025.07-gfbf-2025b
source venv/bin/activate

python -u -m reinvent -l {run_path}/generation_{generation}/model/TL_generation_{generation}.log \
    {run_path}/generation_{generation}/model/TL_generation_{generation}.toml
"""


def generate_tl_jobscript(
        meta_config: dict,
        generation: int, 
        walltime: str = "02:30:00"
        ) -> Path:

    group_name = meta_config.get("GROUP_NAME")
    run_name = meta_config.get("RUN_NAME")
    run_id = meta_config.get("RUN_ID")

    print(f"[{run_name}] (GEN {generation}) Generating TL jobscript...", flush=True)

    cwd = Path.cwd()
    run_path = cwd / "runs" / group_name / run_id

    script_path = run_path / f"generation_{generation}" / "model" / f"jobscript-TL_generation_{generation}.sh"
    return _write_text(script_path, make_tl_jobscript(run_path, generation, walltime=walltime))


def make_sl_jobscript(
        meta_config: dict,
        run_path: Path, 
        generation: int,  
        walltime: str = "04:00:00",
        gpus_per_node: str = "A40:1"
        ) -> str:
    
    workflow_name = meta_config.get("WORKFLOW_NAME")

    return f"""#!/bin/bash
#SBATCH -A NAISS2025-5-462
#SBATCH -p alvis
#SBATCH -t {walltime}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH -J {run_path.name}_G{generation}
#SBATCH -o {run_path}/generation_{generation}/generation_{generation}.out

module purge
module load virtualenv/20.32.0-GCCcore-14.3.0 SciPy-bundle/2025.07-gfbf-2025b
source venv/bin/activate

# Start low-priority filler
nice -n 19 python -u gpu_keepalive.py &
FILLER_PID=$!
trap "kill $FILLER_PID 2>/dev/null" EXIT

python -u {workflow_name}/run_synthetic_data.py --folder="{run_path}/generation_{generation}"
"""

def generate_sl_jobscript(
        meta_config: dict,
        generation: int, 
        walltime: str = "04:00:00"
        ) -> Path:
    
    group_name = meta_config.get("GROUP_NAME")
    run_name = meta_config.get("RUN_NAME")
    run_id = meta_config.get("RUN_ID")
    
    print(f"[{run_name}] (GEN {generation}) Generating SL jobscript...", flush=True)

    cwd = Path.cwd()
    run_path = cwd / "runs" / group_name / run_id

    script_path = run_path / f"generation_{generation}" / f"jobscript-generation_{generation}.sh"
    return _write_text(script_path, make_sl_jobscript(meta_config, run_path, generation, walltime=walltime))






def make_sl_jobscript_for_combo(
        meta_config: dict,
        run_path: Path, 
        generation: int, 
        combo_path: Path,
        walltime: str = "01:30:00",
        gpus_per_node: str = "A40:1"
        ) -> str:
    
    workflow_name = meta_config.get("WORKFLOW_NAME")

    return f"""#!/bin/bash
#SBATCH -A NAISS2025-5-462
#SBATCH -p alvis
#SBATCH -t {walltime}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH -J {run_path.name}_G{generation}_{combo_path.name}
#SBATCH -o {combo_path}/G_{generation}_{combo_path.name}.out

module purge
module load virtualenv/20.32.0-GCCcore-14.3.0 SciPy-bundle/2025.07-gfbf-2025b
source venv/bin/activate


# Start low-priority filler
nice -n 19 python -u gpu_keepalive.py &
FILLER_PID=$!
trap "kill $FILLER_PID 2>/dev/null" EXIT

python -u {workflow_name}/run_synthetic_data.py --folder="{combo_path}"
"""



def generate_sl_jobscript_for_combo(
        meta_config: dict,
        parameter_config: dict,
        generation: int, 
        combo_path: Path,
        ) -> Path:
    
    group_name = meta_config.get("GROUP_NAME")
    run_name = meta_config.get("RUN_NAME")
    run_id = meta_config.get("RUN_ID")

    walltime = parameter_config.get("COMBO_TIME")

    print(f"[{run_name}] (GEN {generation}) Generating SL jobscript for combo folder: {combo_path.name}...", flush=True)

    cwd = Path.cwd()
    try_path = cwd / "runs" / group_name / run_id

    script_path = combo_path / f"jobscript-{combo_path.name}.sh"
    return _write_text(
        script_path, 
        make_sl_jobscript_for_combo(
            meta_config,
            try_path, 
            generation,  
            combo_path, 
            walltime=walltime
            )
        )


if __name__ == "__main__":
    VERSION_FOLDER_NAME = "synthetic_data_scoring"
    TRY_NAME = "try_1"
    GENERATION = 0
    
    generate_tl_toml(VERSION_FOLDER_NAME, TRY_NAME, GENERATION)
    generate_tl_jobscript(VERSION_FOLDER_NAME, TRY_NAME, GENERATION)

    generate_sl_jobscript(VERSION_FOLDER_NAME, TRY_NAME, GENERATION)
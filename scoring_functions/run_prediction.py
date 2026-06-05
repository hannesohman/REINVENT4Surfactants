import os
import joblib
import numpy as np

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors



def run_prediction(
    smiles: list[str], 
    model_path: str,

    normalize: bool = False,
    min_value: float = None,
    max_value: float = None
) -> np.ndarray:
    """
    Generic XGBoost surrogate model scoring component.

    Expects a joblib .pkl file containing:
        {
            "model": trained_xgb_model,
            "scaler": fitted_scaler (optional),
            "features": list_of_rdkit_descriptor_names
        }
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model package
    model_package = joblib.load(model_path)

    # Initialize attributes
    model = model_package["model"]
    scaler = model_package.get("scaler", None)
    feature_names = model_package["features"]
    # print(f"[Surrogate] Feature names: {feature_names}")

    # Pre-build RDKit descriptor calculator
    rdkit_feature_names = [
        f.replace("rdkit-", "") for f in feature_names
    ]
    
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
        rdkit_feature_names
    )

    print(f"[RunPrediction] Loaded model from: {model_path}")
    print(f"[RunPrediction] Using {len(feature_names)} descriptors")



    descriptors = []
    valid_mask = []

    for smi in smiles:
        # print(smi)
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            print(f"[Surrogate] Invalid SMILES: {smi}")
            descriptors.append([0.0] * len(feature_names))
            valid_mask.append(False)
        else:
            # print(f"[Surrogate] Calculating descriptors for: {smi}")
            values = descriptor_calculator.CalcDescriptors(mol)
            descriptors.append(values)
            valid_mask.append(True)

            # print(values)

    X = np.array(descriptors, dtype=np.float32)
    # print(f"[Surrogate] {X.shape} descriptor matrix calculated")

    # Apply scaler if present
    if scaler is not None:
        # print("[Surrogate] Applying scaler to descriptors")
        X = scaler.transform(X)

    # Predict using XGBoost
    predictions = model.predict(X)
    
    # Convert to float32 flat array (one score per molecule)
    predictions = np.array(predictions, dtype=np.float32).flatten()

    # Set invalid molecules to NaN (REINVENT uses NaN to indicate failure)
    predictions[~np.array(valid_mask)] = float("nan")

    if normalize:
        predictions = (predictions - min_value) / (max_value - min_value)

    return predictions

if __name__ == "__main__":
    # Example usage
    model_path = "models/pcmc_model.joblib"
    smiles_list = [
        "CCCCCCCCCc1ccc(OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)OS(=O)[O-])cc1", #4.79588001734408
        "CCCCCCCCCCC(CCCC)c1ccc(S(=O)(=O)[O-])cc1", #3.04000516167158
        "CCCCCCCCOC(=O)C[n+]1cccc(N=Cc2ccc(O)c(OC)c2)c1" #2.60205999132796
        ]  # Example SMILES
    predictions = run_prediction(smiles_list, model_path)
    print(predictions)
# API FUNCTION
# pka_predictor/predict.py

import os
import tempfile
import pandas as pd
from .GNN.train_pKa_predictor import inferring
from .GNN.argParser import argsParser


# main API function
def predict(
    input_data,
    pH=7.4,
    verbose=0,
    model_dir=None,
    model_name=None,
    batch_size=None,
    **kwargs
):
    """
    Predict pKa values and major protonated species at given pH.

    Parameters
    ----------
    input_data : str, pd.DataFrame, or SMILES string
        Path to CSV, pandas DataFrame, or single SMILES string.
    pH : float, optional
        pH at which to predict (default: 7.4)
    verbose : int, optional
        Verbosity level (default: 0)
    model_dir, model_name, batch_size : various, optional
        Model settings (optional, uses defaults if not specified)
    kwargs : dict
        Any other arguments supported by the model.

    Returns
    -------
    results : pd.DataFrame
        DataFrame with columns:
            - mol_number: molecule index in the input
            - input_smiles: input SMILES
            - predicted_pKa: predicted pKa value(s)
            - ionization_center: index of ionizable atom(s)
            - protonated_smiles: major protonated species (may differ from input_smiles)
            - label: true label if available (else NaN)
    """
    # 1. Normalize input
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    elif isinstance(input_data, str):
        if os.path.exists(input_data):
            df = pd.read_csv(input_data)
        else:
            # Assume input is a single SMILES string
            df = pd.DataFrame([{"Smiles": input_data}])
    else:
        raise ValueError("input_data must be a DataFrame, path to a CSV, or a SMILES string.")

    # 2. Ensure 'Smiles' column exists and is correctly capitalized - some libraries use uppercase 'SMILES'
    for col in df.columns:
        if col.lower() == "smiles":
            df.rename(columns={col: "Smiles"}, inplace=True)

    # 3. Set up args (defaults from argParser, override as needed)
    args = argsParser()
    args.pH = pH
    args.verbose = verbose
    if model_dir: args.model_dir = model_dir
    if model_name: args.model_name = model_name
    if batch_size: args.batch_size = batch_size
    # You can handle atom_indices as needed for your use-case

    # Override any other defaults with kwargs
    for k, v in kwargs.items():
        setattr(args, k, v)

    # 4. Write input to a temp CSV and run inferring
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_csv = os.path.join(tmpdir, "input.csv")
        df.to_csv(tmp_csv, index=False)
        args.input = "input.csv"
        args.data_path = tmpdir + "/"

        # inferring() returns multiple lists, as per your latest function
        (
            predicts,
            labels,
            protonated_smiles,
            mol_nums,
            centers,
            proposed_centers,
            ionization_states
        ) = inferring(args)

    # 5. Prepare output DataFrame
    # Protonated SMILES: usually the output SMILES after inference, but you can adjust this as needed.
    results = pd.DataFrame({
        "mol_number": mol_nums,
        "input_smiles": df["Smiles"].values,
        "predicted_pKa": predicts,
        "ionization_center": centers,
        "protonated_smiles": protonated_smiles,
        "label": labels if labels is not None else [float('nan')] * len(mol_nums)
    })

    # Optionally add more fields:
    results["proposed_center"] = proposed_centers
    results["ionization_states"] = ionization_states

    # If your input had "Name" or other ID columns, you could merge those back in here if desired.

    return results


# potentially look into the get_protonation_sites() and get_major_protonated_state()
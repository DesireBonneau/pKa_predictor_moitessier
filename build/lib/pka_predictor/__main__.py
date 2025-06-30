# CLI ENTRY POINT

import argparse
import sys
import pandas as pd
from .predict import predict

def main():
    parser = argparse.ArgumentParser(
        prog="pka-predictor",
        description="Predict pKa for SMILES string(s) or CSV"
    )
    # Haven't set up a '--mode' argument since I have only made the 'infer' mode accessible.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--smiles", help="Single SMILES string to predict")
    group.add_argument("-i", "--input", help="Path to input CSV with a column named 'Smiles' or 'SMILES'")

    parser.add_argument("-p", "--pH", type=float, default=7.4, help="pH at which to predict (default: 7.4)")
    parser.add_argument("-v", "--verbose", type=int, choices=[0,1,2], default=0, help="Verbosity level (0, 1, or 2)")
    parser.add_argument("-d", "--model-dir", type=str, default=None, help="Directory containing model weights")
    parser.add_argument("-m", "--model-name", type=str, default=None, help="Model checkpoint filename")
    parser.add_argument("-b", "--batch-size", type=int, default=None, help="Batch size for inference")
    parser.add_argument("-o", "--output", help="Path to output CSV file (default: stdout)")

    args = parser.parse_args()

    # Determine the input type
    if args.smiles:
        input_data = args.smiles
    elif args.input:
        input_data = args.input
    else:
        parser.error("You must provide either --smiles or --input")

    # Call your API
    results = predict(
        input_data,
        pH=args.pH,
        verbose=args.verbose,
        model_dir=args.model_dir,
        model_name=args.model_name,
        batch_size=args.batch_size
    )

    # Output
    if args.output:
        results.to_csv(args.output, index=False)
    else:
        results.to_csv(sys.stdout, index=False)

if __name__ == "__main__":
    main()

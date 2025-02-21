import argparse
import json

import pandas as pd

from raid.evaluate import run_evaluation, run_my_evaluation, run_evaluation_for_hybrid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--results_path", type=str, required=True, help="Path to the detection result JSON to evaluate"
    )
    parser.add_argument(
        "-d", "--data_path", type=str, required=True, help="Path to the dataset to evaluate for the results"
    )
    parser.add_argument(
        "-th", "--threshold_path", type=str, required=True, help="Path to the threshold JSON to evaluate"
    ) 
    parser.add_argument(
        "-o", "--output_path", type=str, default="results.json", help="Path to the output JSON to write scores"
    )
    parser.add_argument(
        "-t", "--target_fpr", type=float, default=0.05, help="Target false positive rate to evaluate detectors at"
    )
    args = parser.parse_args()

    print(f"Reading dataset at {args.data_path}...")
    df = pd.read_csv(args.data_path)

    print(f"Reading detection result at {args.results_path}...")
    with open(args.results_path) as f:
        d = json.load(f)

    print(f"Reading threshold at {args.threshold_path}...")
    with open(args.threshold_path) as f:
        th = json.load(f)
    acc_threshold = th["threshold"]

    print(f"Running evaluation...")
    evaluation_result = run_evaluation_for_hybrid(d, df, acc_threshold)

    with open(args.output_path, "w") as f:
        json.dump(evaluation_result, f, indent=4) 
    
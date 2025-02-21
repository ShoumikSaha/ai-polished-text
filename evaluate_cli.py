import argparse
import json

import pandas as pd

from raid.evaluate import run_evaluation, run_my_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--results_path", type=str, required=True, help="Path to the detection result JSON to evaluate"
    )
    parser.add_argument(
        "-d", "--data_path", type=str, required=True, help="Path to the dataset to evaluate for the results"
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

    print(f"Running evaluation...")
    evaluation_result_fpr = run_evaluation(d, df, args.target_fpr, per_domain_tuning=False)
    evaluation_result_acc = run_my_evaluation(d, df)

    output_path_fpr = args.output_path.replace(".json", "_fpr.json")
    output_path_acc = args.output_path.replace(".json", "_acc.json")
    with open(output_path_fpr, "w") as f:
        json.dump(evaluation_result_fpr, f, indent=4)
    with open(output_path_acc, "w") as f:
        json.dump(evaluation_result_acc, f, indent=4)
    print(f"Done! Writing evaluation result to output path: {args.output_path}")

#!/bin/bash

model="gptzero"

python detect_cli.py -m "$model" -d "data/merged_mgt_hwt_data.csv" -o "results/${model}/mgt_hwt_predictions.json"

python evaluate_cli.py -r "results/${model}/mgt_hwt_predictions.json" -d "data/merged_mgt_hwt_data.csv" -o "results/${model}/mgt_hwt_results.json"
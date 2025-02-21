#!/bin/bash

model="detectgpt"
#polish_pctg="10"
polish_type_list=("extreme_minor" "minor" "slight_major" "major")
polisher_models=("llama70b")

for polisher_model in "${polisher_models[@]}"
do
    for polish_type in "${polish_type_list[@]}"
    do
        echo "Running ${model} for polished data with ${polish_type} polish by ${polisher_model}"
        python detect_cli.py -m "$model" -d "data/polished/polished_texts_${polish_type}_${polisher_model}.csv" -o "results/${model}/polished_${polish_type}_${polisher_model}_predictions.json"
        python evaluate_for_hybrid.py -r "results/${model}/polished_${polish_type}_${polisher_model}_predictions.json" -d "data/polished/polished_texts_${polish_type}_${polisher_model}.csv" -th "results/${model}/mgt_hwt_results_acc.json" -o "results/${model}/polished_${polish_type}_${polisher_model}_results.json"
    done
done

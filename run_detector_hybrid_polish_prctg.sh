#!/bin/bash

model="gltr"
polish_pctg_list=("1" "5" "10" "20" "35" "50" "75")
polisher_models=("gpt" "llama70b" "llama" "llama2")

for polisher_type in "${polisher_models[@]}"
do
    for polish_pctg in "${polish_pctg_list[@]}"
    do
        echo "Running ${model} for polished data with ${polish_pctg}% polish by polisher ${polisher_type}"
        python detect_cli.py -m "$model" -d "data/polished/polished_texts_${polish_pctg}_${polisher_type}.csv" -o "results/${model}/polished${polish_pctg}_${polisher_type}_predictions.json"
        python evaluate_for_hybrid.py -r "results/${model}/polished${polish_pctg}_${polisher_type}_predictions.json" -d "data/polished/polished_texts_${polish_pctg}_${polisher_type}.csv" -th "results/${model}/mgt_hwt_results_acc.json" -o "results/${model}/polished${polish_pctg}_${polisher_type}_results.json"
    done
done
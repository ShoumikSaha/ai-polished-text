# APT-Eval 

## Dataset Overview

The **APT-Eval** dataset includes over **11.7K**  AI-polished-text samples. They can be found in `data/polished` directory. The overview of our dataset is given below --

| **Polish Type**                           | **GPT-4o** | **Llama3.1-70B** | **Llama3-8B** | **Llama2-7B** | **Total** |
|-------------------------------------------|------------|------------------|---------------|---------------|-----------|
| **no-polish / pure HWT**                  | -          | -                | -             | -             | 300       |
| **Degree-based**                          | 1152       | 1085             | 1125          | 744           | 4406      |
| **Percentage-based**                      | 2072       | 2048             | 1977          | 1282          | 7379      |
| **Total**                                 | 3224       | 3133             | 3102          | 2026          | **11785** |


APT-Eval is the only dataset that covers AI-polishing of different degrees. For more details, check out our paper. To access all samples (unfiltered) with distance and similarity metrics, see the `data/polished_json` directory.

## Installation

### Using Conda
To setup, run this --

```
conda env create -f environment.yml
```

## Detectors

### Included Detectors
1. **Model-based:** RADAR, RoBERTa-Base (ChatGPT), RoBERTa-Base (GPT2), and RoBERTa-Large (GPT2).

2. **Metric-based:** GLTR, DetectGPT, FastDetectGPT, LLMDet, Binoculars.

3. **Commercial:** [ZeroGPT](https://www.zerogpt.com/), [GPTZero](https://gptzero.me/).


### Validating Detectors
Above-mentioned detectors are already validated on 600 samples of HWT and AI-texts. The result files are included in the `results` directory as well. So, you can skip this step if you are using the same detector and same dataset.

However, if you want to validate a new detector or on a different dataset, use the `run_detector_basic.sh`:

```
model="model_name"

$ python detect_cli.py 
    -m "$model" 
    -d "data/merged_mgt_hwt_data.csv" # Replace with your own dataset file
    -o "results/${model}/mgt_hwt_predictions.json" # Path to write the predictions JSON file

$ python evaluate_cli.py 
    -r "results/${model}/mgt_hwt_predictions.json" # Use the previous predictions JSON file path
    -d "data/merged_mgt_hwt_data.csv" # Replace with your own dataset file
    -o "results/${model}/mgt_hwt_results.json" # Path to write the final results JSON file
```
This will result into 3 different JSON file -- 
- predictions.json: contains predictions score for each sample.
- results_acc.json: contains the best accuracy with its threshold (including confusion matrix).
- results_fpr.json: contains results for 5% FPR (or the second lowest) for all combinations.


### Evaluating Detectors on APT-Eval



*This codebase was built upon the [RAID](https://github.com/liamdugan/raid) repo.  

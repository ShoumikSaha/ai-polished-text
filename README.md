# APT-Eval 
This repository is the implementation of the paper ['Almost AI, Almost Human: The Challenge of Detecting AI-Polished Writing'](https://arxiv.org/abs/2502.15666).

## Dataset Overview

The **APT-Eval** dataset includes over **15K**  AI-polished-text samples. They can be found in `data/polished` directory. The overview of our dataset is given below --

| **Polish Type**                           | **GPT-4o** | **Llama3.1-70B** | **Llama3-8B** | **Llama2-7B** | **DeepSeek-V3** | **Total** |
|-------------------------------------------|------------|------------------|---------------|---------------|-- |-----------|
| **no-polish / pure HWT**                  | -          | -                | -             | -             | - | 300       |
| **Degree-based**                          | 1152       | 1085             | 1125          | 744           | 1141 | 4406      |
| **Percentage-based**                      | 2072       | 2048             | 1977          | 1282          | 2078 | 7379      |
| **Total**                                 | 3224       | 3133             | 3102          | 2026          | 3219 | **15004** |


APT-Eval is the only dataset that covers AI-polishing of different degrees. For more details, check out our paper. To access all samples (unfiltered) with distance and similarity metrics, see the `data/polished_json` directory.

You can also load it from [huggingface](https://huggingface.co/datasets/smksaha/apt-eval) --

First, install the library `datasets` with `pip install datasets`. Then,

```
from datasets import load_dataset
apt_eval_dataset = load_dataset("smksaha/apt-eval")
```

If you also want to access the original human written text samples, use this
```
from datasets import load_dataset
dataset = load_dataset("smksaha/apt-eval", data_files={
    "test": "merged_apt_eval_dataset.csv",
    "original": "original.csv"
})
``` 

## Installation

### Using Conda
To setup, run this --

```
conda env create -f environment.yml
```

Setup the api-keys, by running --
```
source set_api_keys.sh
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
$ python detect_cli.py 
    -m, --model           The name of the detector model you wish to run
    -d, --data_path       The path to the csv file with the dataset
    -o, --output_path     The path to write the predictions JSON file
```

```
$ python evaluate_cli.py
    -r, --results_path    The path to the detection predictions JSON file
    -d, --data_path       The path to the csv file with the dataset
    -o, --output_path     The path to write the final results JSON file
```
Example:
```
#!/bin/bash

model="radar"

python detect_cli.py -m "$model" -d "data/merged_mgt_hwt_data.csv" -o "results/${model}/mgt_hwt_predictions.json"

python evaluate_cli.py -r "results/${model}/mgt_hwt_predictions.json" -d "data/merged_mgt_hwt_data.csv" -o "results/${model}/mgt_hwt_results.json"
```

This will result into 3 different JSON file -- 
- predictions.json: contains predictions score for each sample.
- results_acc.json: contains the best accuracy with its threshold (including confusion matrix).
- results_fpr.json: contains results for 5% FPR (or the second lowest) for all combinations.


### Running Detectors on APT-Eval

To evaluate detectors on our APT-Eval dataset, run the following:
```
$ python detect_cli.py 
    -m, --model           The name of the detector model you wish to run
    -d, --data_path       The path to the csv file for the polished-dataset ('data/polished')
    -o, --output_path     The path to write the predictions JSON file
```

```
$ python evaluate_for_hybrid.py
    -r, --results_path    The path to the detection predictions JSON file
    -d, --data_path       The path to the csv file with the dataset
    -th, --threshold_path The path to the results_acc JSON file
    -o, --output_path     The path to write the final results JSON file
```

Example:
```
#!/bin/bash

model="radar"
polish_type="extreme_minor" # Replace with your desired polish-type
polisher_model="gpt" # Replace with your desired polisher-type


python detect_cli.py 
    -m "$model" 
    -d "data/polished/polished_texts_${polish_type}_${polisher_model}.csv" 
    -o "results/${model}/polished_${polish_type}_${polisher_model}_predictions.json"

python evaluate_for_hybrid.py 
    -r "results/${model}/polished_${polish_type}_${polisher_model}_predictions.json" 
    -d "data/polished/polished_texts_${polish_type}_${polisher_model}.csv" 
    -th "results/${model}/mgt_hwt_results_acc.json" 
    -o "results/${model}/polished_${polish_type}_${polisher_model}_results.json"
```

If you want to automate the running for all polish-type (degree-based) with all polishers, use the script `run_detector_hybrid_polish_type.sh`. For percentage-based polishing, use the script `run_detector_hybrid_polish_prctg.sh`.

### Results

Running the above mentioned commands will generate result files under the `results/model_name` directory. However, we have already provided all the results (from our paper) there. Feel free to use/extend them, and extract more insights out of them. 


*This codebase was built upon the [RAID](https://github.com/liamdugan/raid) repo.  

## Citation

If you use our code or findings in your research, please cite us as:

```
@article{saha2025almost,
  title={Almost AI, Almost Human: The Challenge of Detecting AI-Polished Writing},
  author={Saha, Shoumik and Feizi, Soheil},
  journal={arXiv preprint arXiv:2502.15666},
  year={2025}
}
```

import json
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)

def merge_csv_files(input_folder, output_file):
    """
    Merges all CSV files from the specified folder into a single CSV file.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing the CSV files to be merged.
    output_file : str
        Path (including filename) for the merged CSV output.
    """

    # Use glob to find all CSV files in the specified folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    # Read each CSV and add to a list
    df_list = []
    for file in csv_files:
        if 'polish_type_results' in file:
            continue
        df = pd.read_csv(file)
        df_list.append(df)
    
    # Concatenate all dataframes
    merged_df = pd.concat(df_list, ignore_index=True)
    
    # Write out the merged dataframe
    merged_df.to_csv(output_file, index=False)
    print(f"Merged {len(df_list)} CSV files into {output_file}.")

def plot_polish_ratio_vs_accuracy(polish_ratios, accuracies, xlabel="Polish Ratio", ylabel="APTs predicted as AI-text", title="Polish Ratio vs. MGT Prediction Rate", save_path=None):
    """
    Plots the relationship between polish ratio and accuracy.
    
    Parameters:
        polish_ratios (list or array): List of polish ratios.
        accuracies (list or array): Corresponding accuracy values.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(polish_ratios, accuracies, marker='o', linestyle='-', markersize=8, linewidth=2)

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    #plt.title(title, fontsize=16)
    #plt.gca().invert_xaxis()  # Assuming higher polish ratio decreases accuracy
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


def plot_multiple_models_polish_ratio(polish_ratios_dict, xlabel="Polish Ratio", ylabel="APTs predicted as AI-text", title="Polish Ratio vs. MGT Prediction Rate", save_path=None):
    """
    Plots the relationship between polish ratio and accuracy for multiple models.
    
    Parameters:
        polish_ratios_dict (dict): Dictionary where keys are model names and values are tuples
                                   containing (polish_ratios, accuracies).
    """
    plt.figure(figsize=(8, 6))

    for model, (polish_ratios, accuracies) in polish_ratios_dict.items():
        plt.plot(polish_ratios, accuracies, marker='o', linestyle='-', markersize=3, linewidth=1.5, label=model)

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    #plt.title(title, fontsize=16)
    # plt.gca().invert_xaxis()  # Assuming higher polish ratio decreases accuracy
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(1.05, 1.0))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

from utils import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def sort_df_by_polish_type(df, custom_order=None, key=None):
    """
    Sorts a DataFrame by polish_type in the order: extreme_minor, minor, slight_major, major.
    """
    df[key] = pd.Categorical(
        df[key],
        categories=custom_order,
        ordered=True
    )
    return df

def rename_values_in_column(df, column, old_values, new_values):
    """
    Renames values in a column of a DataFrame.
    """
    df[column].replace(old_values, new_values, inplace=True)
    return df

def drop_rows_with_values(df, column, value):
    """
    Remove rows in a DataFrame where a specified column contains certain values.
    """
    #df_cleaned = df[df[column] != value]
    #df.drop({column: value}, inplace=True)
    df = df[df.column != value]
    return df
    

def plot_model_accuracies(data, show_bar_val=True, x_col='polish_type', y_col='ai_rate', hue_col='polisher', title='MGT Prediction Rate for Different Polisher', x_label="Polish type", y_label="APTs predicted as AI-text", save_path=None):
    """
    Plots the ai_rate of multiple polisher across different polish-types in a single figure.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the data to plot. Must include columns for subdomain, model, and accuracy.
    x_col : str, optional
        Column name in `data` for the x-axis (default is 'polish_type').
    y_col : str, optional
        Column name in `data` for the y-axis (default is 'ai_rate').
    hue_col : str, optional
        Column name in `data` that differentiates models (default is 'polisher').
    """

    # Set a theme or style (optional)
    sns.set_theme(style="whitegrid")

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Create the barplot (grouped by the hue column) and retrieve the Axes
    ax = sns.barplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette="Set2",   # Choose a color palette
        edgecolor="black",
        errorbar=None
    )

    # Labeling the axes
    plt.xlabel(x_col.capitalize(), fontsize=18)
    plt.ylabel(y_label, fontsize=18)

    # Optionally set a plot title
    # plt.title(title)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, fontsize=15)
    plt.yticks(fontsize=15)
    # Fix the range of y-axis
    plt.ylim(0, 0.75)

    # Adjust legend
    plt.legend(title=hue_col.capitalize(), loc='upper center')

    # Use bar_label to show bar values (Matplotlib 3.4+)
    if show_bar_val:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='edge')  # or 'center', etc.

    # Adjust layout (especially important when x-labels are rotated)
    plt.tight_layout()

    # Save or show the figure
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def generate_combined_csv():
    input_folder = "domain_results_ratio"
    output_file = "polish_type_results.csv"
    merge_csv_files(input_folder, f"{input_folder}/{output_file}")

def main():
    df = pd.read_csv("domain_results/polish_type_results.csv")

    #df = df[df["polish_type"] != "major"]

    custom_order = ["extreme_minor", "minor", "slight_major", "major", "average"]
    #custom_order = ["1", "5", "10", "20", "35", "50", "75", "average"]
    df = sort_df_by_polish_type(df, custom_order, "polish_type")

    polisher_order = ["gpt", "llama70b", "llama", "llama2"]
    df = sort_df_by_polish_type(df, polisher_order, "polisher")
    df = rename_values_in_column(df, "polisher", ["gpt", "llama70b", "llama", "llama2"], ["GPT-4o", "Llama3.1-70B", "Llama3-8B", "Llama2-7B"])

    # Drop rows with polish_type "major"
    #df_cleaned = drop_rows_with_values(df, "polish_type", "major")
    #df_cleaned = df[df["polish_type"] != "major"]
    # df_cleaned = df.drop(df.index[df['polish_type'] == 'major'], inplace = False)
    # print(df_cleaned)


    plot_model_accuracies(df, show_bar_val=False, x_label="Polish Type", save_path="domain_results/polisher_plots/polish_type_results_wmajor.pdf")
    #plot_model_accuracies_catplot(df, save_path="domain_results/polisher_plots/polish_type_results_catplot.png")

if __name__ == "__main__":
    main()
    #generate_combined_csv()
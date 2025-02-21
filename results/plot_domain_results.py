import pandas as pd
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns

domain_list = ["blog", "email_content", "game_review", "news", "paper_abstract", "speech"]


def plot_detector_accuracy_for_domains(df, detector_name, save_path=None):
    """
    Plots the accuracy of a specific detector across different domains.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing columns for 'domain', 'detector', and 'accuracy'.
    detector_name : str
        The name of the detector to filter on and plot.
    """

    # Filter the DataFrame for the specified detector
    df_filtered = df[df['model'] == detector_name]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a bar plot of accuracy by domain for the chosen detector
    sns.barplot(
        data=df_filtered,
        x='domain',
        y='ai_rate',
        color='skyblue',
        edgecolor='black',
        ax=ax
    )

    # Annotate each bar with its value
    for p in ax.patches:
        # Get the height (value) of each bar
        height = p.get_height()
        # Annotate the bar at its center (x-position) and just above its top (y-position)
        ax.annotate(
            f"{height:.2f}",         # Format to 1 decimal place (or remove formatting if not needed)
            xy=(p.get_x() + p.get_width() / 2, height),  # Position at the bar's center and top
            xytext=(0, 5),          # Offset the text a bit above the bar top
            textcoords="offset points",
            ha="center", va="bottom", fontsize=14
        )

    # Set plot labels and title
    # plt.title(f"MGT Predictions for '{detector_name}' Across Domains", fontsize=14)
    ax.set_xlabel("Domain", fontsize=18)
    ax.set_ylabel('APTs predicted as AI-text', fontsize=18)

    # Rotate x-ticks if needed (helpful for long domain names)
    plt.xticks(rotation=45, ha="right", fontsize=15)
    plt.yticks(fontsize=15)

    # Make the layout tight so labels are not cut off
    plt.tight_layout()

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path)
    else:
        # Show the plot
        plt.show()

def plot_overall_accuracy_for_domains(df, save_path=None):
    """
    Plots the mean ai_rate (with standard deviation as error bars) 
    for each domain, aggregated across all models.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing columns for at least 'domain' and 'ai_rate'.
        It can also include 'model' (which is ignored for the grouping here).
    save_path : str, optional
        If provided, the plot will be saved to this path. Otherwise, it will be displayed.
    """
    
    # Option 1 (manual groupby and then plotting) -------------------------
    # 1. Group by domain and aggregate
    domain_stats = df.groupby('domain')['ai_rate'].agg(['mean', 'std']).reset_index()

    # 2. Create the figure
    plt.figure(figsize=(8, 6))

    # 3. Plot a bar chart of mean ai_rate with std as error bars
    ax = sns.barplot(
        data=domain_stats, 
        x='domain', 
        y='mean', 
        yerr=None, 
        color='skyblue',
        edgecolor='black'
    )

    for i, bar in enumerate(ax.patches):
        height = bar.get_height()
        # domain_stats['mean'][i] should match the bar's height
        ax.annotate(f'{height:.2f}',    # Format to 2 decimals
                    xy=(bar.get_x() + bar.get_width() / 2, height), 
                    xytext=(0, 3),     # offset the text a little
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14)

    # 4. Set labels and title
    #plt.title("Average MGT Prediction Rate Across All Models by Domain", fontsize=14)
    plt.xlabel("Domain", fontsize = 18)
    plt.ylabel('APTs predicted as AI-text', fontsize = 18)
    plt.xticks(rotation=45, ha="right", fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()

    # 5. Save or show
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def create_combined_dataframe(polish_type=1, polisher_type="llama70b"):
    model_list = ['binoculars', 'chatgpt-roberta', 'detectgpt', 'fastdetectgpt', 'gltr', 'gpt2-base', 'gpt2-large', 'llmdet', 'radar', 'zerogpt']
    combined_list = []

    # polish_type = "average"
    # polisher_type = "llama"

    for domain in domain_list:
        for model_name in model_list:
            try:
                json_content = read_json(f"domain_results_ratio/{domain}/{model_name}_{polisher_type}_{domain}_results.json")
                print(json_content)
                for content in json_content:
                    if "polish_type" in content and content["polish_type"] == polish_type:
                        preds = content["prediction_labels"]
                    elif "polish_percent" in content and content["polish_percent"] == polish_type:
                        preds = content["prediction_labels"]

                temp_dict = {
                    "domain": domain,
                    "model": model_name,
                    "polisher": polisher_type,
                    "polish_type": str(polish_type),
                    "ai_rate": preds["AI"]["rate"],
                    "human_rate": preds["human"]["rate"]
                }
                combined_list.append(temp_dict)
            except:
                print(f"Error for {domain} and {model_name} and {polish_type} and {polisher_type}")
                continue
    
    #print(combined_list)
    df = pd.DataFrame(combined_list)
    print(df)
    df.to_csv(f"domain_results_ratio/{polisher_type}_domain_{polish_type}.csv", index=False)
    plot_overall_accuracy_for_domains(df, f"domain_results_ratio/domain_plots/{polish_type}_domain_accuracy_{polisher_type}.pdf")
    for model_name in model_list:
        plot_detector_accuracy_for_domains(df, model_name, f"domain_results_ratio/domain_plots/{model_name}_{polisher_type}_domain_{polish_type}.pdf")

def main():
    #polish_types = ["average", "extreme_minor", "minor", "slight_major", "major"]
    polish_types = ["average", 1, 5, 10, 20, 35, 50, 75]
    polisher_type = "llama2"

    for polish_type in polish_types:
        create_combined_dataframe(polish_type, polisher_type)

if __name__ == "__main__":
    main()
    #create_combined_dataframe()
    # plot_detector_accuracy(df, "detectgpt")
    # plot_detector_accuracy(df, "llmdet")
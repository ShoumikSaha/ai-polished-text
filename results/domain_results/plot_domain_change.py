import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_ai_rate_for_polisher(df, domain_name, save_path):
    # Define the order for polish_type
    polish_order = ['extreme_minor', 'minor', 'slight_major', 'major']
    
    # Filter out 'average' polish_type and select only the specified domain
    df_filtered = df[(df['polish_type'] != 'average') & (df['domain'] == domain_name)]
    
    # Group by 'polisher' and 'polish_type' and take the mean of 'ai_rate'
    df_grouped = df_filtered.groupby(['polisher', 'polish_type'])['ai_rate'].mean().unstack()
    
    # Reorder columns based on predefined order
    df_grouped = df_grouped[polish_order]
    
    # Set Seaborn style
    sns.set_theme(style="whitegrid")
    
    # Plot
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("husl", len(df_grouped.index))
    
    for (polisher, color) in zip(df_grouped.index, palette):
        sns.lineplot(x=df_grouped.columns, y=df_grouped.loc[polisher], marker='o', linestyle='-', label=polisher, color=color, linewidth=2, markersize=8)
    
    plt.xlabel('Polish Type', fontsize=22)
    plt.ylabel('APTs predicted as AI-text ', fontsize=22)
    #plt.title(f'AI Rate vs Polish Type for Fixed Domain: {domain_name}', fontsize=14)
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(title='Polisher', fontsize=18, title_fontsize=20)
    plt.tight_layout()
    
    # Save plot to specified path
    plt.savefig(save_path)
    #plt.show()

def main():
    # Load the CSV file
    file_path = "polish_type_results.csv"
    df = pd.read_csv(file_path)
    df = df[df["model"] != "zerogpt"]
    polisher_order = ["gpt", "llama70b", "llama", "llama2"]
    df = sort_df_by_polish_type(df, polisher_order, "polisher")
    df = rename_values_in_column(df, "polisher", ["gpt", "llama70b", "llama", "llama2"], ["GPT-4o", "Llama3.1-70B", "Llama3-8B", "Llama2-7B"])

    domain_list = ["blog", "email_content", "game_review", "news", "paper_abstract", "speech"]
    for domain_name in domain_list:
        #domain_name = 'speech'  # Change to the desired domain
        save_path = f'domain_change_plots/{domain_name}.pdf'  # Change to the desired save location
        plot_ai_rate_for_polisher(df, domain_name, save_path)

if __name__ == "__main__":
    main()
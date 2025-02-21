
import json
import pandas as pd

def json_to_csv(json_path, csv_path):
    """
    Converts a JSON file to a CSV file.
    """
    with open(json_path) as f:
        d = json.load(f)
    df = pd.DataFrame(d)
    df.to_csv(csv_path, index=False)
    return df

def add_column_to_csv(csv_path, column_name, column_data):
    """
    Adds a column to a CSV file.
    """
    df = pd.read_csv(csv_path)
    df[column_name] = column_data
    df.to_csv(csv_path, index=False)

def rename_column_in_csv(csv_path, old_column_name, new_column_name):
    """
    Renames a column in a CSV file.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={old_column_name: new_column_name})
    df.to_csv(csv_path, index=False)

def reorder_columns_in_csv(csv_path, new_column_order):
    """
    Reorders columns in a CSV file.
    """
    df = pd.read_csv(csv_path)
    df = df.reindex(columns=new_column_order)
    df.to_csv(csv_path, index=False)

def merge_csv_files(csv_path1, csv_path2, merged_csv_path, modify_id=True):
    """
    Merges two CSV files into a single CSV file.
    """
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)
    if modify_id:
        df2["id"] += len(df1)
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv(merged_csv_path, index=False)

def remove_row_from_csv(column_name, threshold_value, csv_path):
    """
    Removes rows from a CSV file based on a threshold value.
    """
    df = pd.read_csv(csv_path)
    df = df[df[column_name] >= threshold_value]
    df.to_csv(csv_path, index=False)
    return df

def main_hwt(editing_type, polish_percent, polish_type, model_type, model_name):

    # Define the model and polish type
    # polish_percent = "35"
    # polish_type = "major"

    # # Editing type 1 for polish percent, 2 for polish type
    # editing_type = 1

    # model_type = "gpt"
    # model_name = "gpt"

    # Load the JSON file
    if editing_type == 1:
        json_path = f"polished_json/polished_texts_{polish_percent}_{model_type}.json"
        csv_path = f"polished/polished_texts_{polish_percent}_{model_type}.csv"
    elif editing_type == 2:
        json_path = f"polished_json/polished_texts_{polish_type}_{model_type}.json"
        csv_path = f"polished/polished_texts_{polish_type}_{model_type}.csv"
    df = json_to_csv(json_path, csv_path)

    # Remove rows with semantic similarity less than 0.85
    df = remove_row_from_csv("sem_similarity", 0.85, csv_path)
    
    # Add 'id' column if it does not exist
    if "id" not in df.columns:
        #df["id"] = range(len(df))
        add_column_to_csv(csv_path, "id", range(len(df)))
        
    add_column_to_csv(csv_path, "model", ["hybrid"] * len(df))
    if editing_type == 1:
        add_column_to_csv(csv_path, "attack", [f"polish_{polish_percent}"] * len(df))
    elif editing_type == 2:
        add_column_to_csv(csv_path, "attack", [f"polish_{polish_type}"] * len(df))
    add_column_to_csv(csv_path, "decoding", ["none"] * len(df))
    add_column_to_csv(csv_path, "repetition_penalty", ["none"] * len(df))
    add_column_to_csv(csv_path, "polisher", [model_name] * len(df))
    rename_column_in_csv(csv_path, "polished", "generation")
    #rename_column_in_csv(csv_path, "category", "domain")

    new_column_order = ["id", "model", "decoding", "repetition_penalty", "attack", "domain", "generation"]
    reorder_columns_in_csv(csv_path, new_column_order)

    print(f"Converted JSON file {json_path} to CSV file {csv_path}.")

def main_mgt():
    # Load the JSON file
    json_path = "selected_pure_data/MGT_original_data.json"
    csv_path = "selected_pure_data/MGT_original_data.csv"
    df = json_to_csv(json_path, csv_path)
    add_column_to_csv(csv_path, "attack", ["none"] * len(df))
    add_column_to_csv(csv_path, "decoding", ["none"] * len(df))
    add_column_to_csv(csv_path, "repetition_penalty", ["none"] * len(df))
    rename_column_in_csv(csv_path, "MGT_sentence", "generation")
    rename_column_in_csv(csv_path, "category", "domain")

    new_column_order = ["id", "model", "decoding", "repetition_penalty", "attack", "domain", "generation"]
    reorder_columns_in_csv(csv_path, new_column_order)

    print(f"Converted JSON file {json_path} to CSV file {csv_path}.")

def main_merge():
    mgt_csv_path = "selected_pure_data/MGT_original_data.csv"
    hwt_csv_path = "selected_pure_data/HWT_original_data.csv"
    merged_csv_path = "selected_pure_data/merged_mgt_hwt_data.csv"
    merge_csv_files(mgt_csv_path, hwt_csv_path, merged_csv_path)

def main():
    editing_type = 2
    

    model_type = "llama70b"
    model_name = "llama3-70b"

    polish_types = ["extreme_minor", "minor", "slight_major", "major"]
    for polish_type in polish_types:
        main_hwt(editing_type, None, polish_type, model_type, model_name)

    # polish_percent_list = ["1", "5", "10", "20", "35", "50", "75"]
    # for polish_percent in polish_percent_list:
    #     main_hwt(editing_type, polish_percent, None, model_type, model_name)



if __name__ == "__main__":
    #main_hwt()
    main()
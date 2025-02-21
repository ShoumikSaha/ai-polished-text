from utils import *
from extract_results import *
import os
import re
import pandas as pd

polish_percent_list = ["1", "5", "10", "20", "50", "75"]
polish_type_list = ["extreme_minor", "minor", "slight_major", "major"]

def get_id_for_domain(polish_text_file, domain):
    """
    Extracts the ID for a specific domain from the csv file.
    Return the ID as a list.
    """
    df = pd.read_csv(polish_text_file)
    domain_id = df[df["domain"] == domain]["id"]
    # convert the df to a list
    domain_id = domain_id.tolist()
    return domain_id

def get_polished_prediction_files(file_list, editing_type, polisher_type):
    """
    Extracts the polished prediction files for a model.
    """
    polished_files = []
    for file in file_list:
        temp_dict = {}
        if editing_type == 1:
            if file.startswith("polished") and not file.startswith('polished_') and file.split(".")[0].endswith("predictions") and contains_digit(file) and polisher_type in file and file.split("_")[1] == polisher_type:
                polish_percent = int(file.split("_")[0].split("polished")[1])
                temp_dict["polish_percent"] = polish_percent
                temp_dict["file_name"] = file
                polished_files.append(temp_dict)
        elif editing_type == 2:
            if file.startswith("polished_") and file.split(".")[0].endswith("predictions") and polisher_type in file:
                if contains_digit(file) and not contains_digit(polisher_type):    continue                    
                temp_file_name = file.split(f"_{polisher_type}")[0]
                polish_type = " ".join(temp_file_name.split("polished_")[1:])
                temp_dict["polish_type"] = polish_type
                temp_dict["file_name"] = file
                polished_files.append(temp_dict)
                
    return polished_files

def get_threshold(model_name):
    """
    Extracts the threshold for a model.
    """
    threshold_file = f"{model_name}/mgt_hwt_results_acc.json"
    return read_json(threshold_file)["threshold"]

def filter_json_file_with_id(file, domain_id):
    """
    Filters the json file with the domain ID.
    """
    file_content = read_json(file)
    filtered_content = [content for content in file_content if content["id"] in domain_id]
    return filtered_content

def get_pred_labels(predictions, threshold):
    total_samples = len(predictions)
    MGT_count = 0
    HWT_count = 0
    for prediction in predictions:
        if prediction["score"] >= threshold:
            MGT_count += 1
        else:
            HWT_count += 1
    return {
        "AI": {'count': MGT_count, 'rate': MGT_count/total_samples},
        "human": {'count': HWT_count, 'rate': HWT_count/total_samples}
    }


def get_domain_specific_results(directory, model_name, editing_type, polisher_type, domain):
    """
    Extracts the domain specific results for a model.
    """
    # Extract the polished result files
    file_list = extract_files_from_directory(directory, ".json")
    polished_files = get_polished_prediction_files(file_list, editing_type, polisher_type)
    #print(polished_files)

    # Extract the ID for the domain
    if editing_type == 1:
        for polish_file in polished_files:
            polish_text_dataset_file = f"../data/polished/polished_texts_{polish_file['polish_percent']}_{polisher_type}.csv"
            domain_id = get_id_for_domain(polish_text_dataset_file, domain)
            polish_file["domain"] = domain_id
    elif editing_type == 2:
        for polish_file in polished_files:
            polish_text_dataset_file = f"../data/polished/polished_texts_{polish_file['polish_type']}_{polisher_type}.csv"
            domain_id = get_id_for_domain(polish_text_dataset_file, domain)
            polish_file["domain"] = domain_id
    #print(polished_files)
    threshold = get_threshold(model_name)
    #print(threshold)
    for polished_file in polished_files:
        filtered_predictions = filter_json_file_with_id(f"{directory}/{polished_file['file_name']}", polished_file["domain"])
        #print(filtered_predictions)
        pred_labels = get_pred_labels(filtered_predictions, threshold)
        polished_file["prediction_labels"] = pred_labels
    #print(polished_files)

    custom_order = {'no_polish': 0, 'extreme_minor': 1, 'minor': 2, 'slight_major': 3, 'major': 4}
    #sort the list depending on polish percent
    if editing_type == 1:
        polished_files.sort(key=lambda x: x["polish_percent"])
    elif editing_type == 2:
        polished_files = sorted(polished_files, key=lambda x: custom_order.get(x['polish_type'], float('inf')))

    return polished_files


def main(domain):
    model_list = ['binoculars', 'chatgpt-roberta', 'detectgpt', 'fastdetectgpt', 'gltr', 'gpt2-base', 'gpt2-large', 'llmdet', 'radar', 'zerogpt']

    for model_name in model_list:
        editing_type = 1
        polisher_type = "llama70b"
        #domain = 'speech'
        directory = model_name

        polished_files = get_domain_specific_results(directory, model_name, editing_type, polisher_type, domain)

        avg_ai_rate = sum([file["prediction_labels"]["AI"]["rate"] for file in polished_files])/len(polished_files)
        avg_human_rate = sum([file["prediction_labels"]["human"]["rate"] for file in polished_files])/len(polished_files)
        polished_files.append({"polish_type": "average", "prediction_labels": {"AI": {"rate": avg_ai_rate}, "human": {"rate": avg_human_rate}}})
        for file in polished_files:
            #drop a key from a list of dictionaries
            if "domain" in file:
                file.pop("domain")
        print(polished_files)

        if not os.path.exists(f"domain_results_ratio/{domain}"):
            os.makedirs(f"domain_results_ratio/{domain}")
        with open(f"domain_results_ratio/{domain}/{model_name}_{polisher_type}_{domain}_results.json", "w") as f:
            json.dump(polished_files, f, indent=4)

if __name__ == "__main__":
    domain_list = ['blog', 'email_content', 'game_review', 'news', 'paper_abstract', 'speech']
    for domain in domain_list:
        main(domain)
    #main()

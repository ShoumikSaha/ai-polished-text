from utils import *
import os
import re

def contains_digit(s):
    return bool(re.search(r'\d', s))

def extract_files_from_directory(directory, extension):
    """
    Extracts files with a specific extension from a directory.
    """
    files = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            files.append(file)
    return files


def get_polished_result_files(file_list, editing_type, polisher_type):
    """
    Extracts the polished result files from a list of files.
    """
    polished_files = []
    for file in file_list:
        temp_dict = {}
        if editing_type == 1:
            if file.startswith("polished") and not file.startswith("polished_") and file.split(".")[0].endswith("results") and contains_digit(file) and polisher_type == file.split("_")[1]:
                polish_percent = int(file.split("_")[0].split("polished")[1])
                temp_dict["polish_percent"] = polish_percent
                temp_dict["file_name"] = file
                polished_files.append(temp_dict)
        elif editing_type == 2:
            if file.startswith("polished_") and file.split(".")[0].endswith("results") and polisher_type in file:
                if contains_digit(file) and not contains_digit(polisher_type):    continue                    
                temp_file_name = file.split(f"_{polisher_type}")[0]
                polish_type = " ".join(temp_file_name.split("polished_")[1:])
                temp_dict["polish_type"] = polish_type
                temp_dict["file_name"] = file
                polished_files.append(temp_dict)
                
    return polished_files

def extract_ai_rate_for_hwt(directory, file):
    """
    Extracts the positive rate from the file.
    File is a dict with keys "file_name" and "polish_percent".
    """
    file_content = read_json(f"{directory}/{file['file_name']}")
    if "polished" in file['file_name']:
        pos_rate = file_content['prediction_labels']['AI']['rate']
    else:
        pos_rate = file_content['false_positive']['rate']
    return pos_rate
    

def plot_for_model(model_name, editing_type, polisher_type):
    # Extract the polished result files
    directory = model_name
    extension = ".json"
    custom_order = {'no_polish': 0, 'extreme_minor': 1, 'minor': 2, 'slight_major': 3, 'major': 4}


    file_list = extract_files_from_directory(directory, extension)
    #print(file_list)

    
    polished_files = get_polished_result_files(file_list, editing_type, polisher_type)
    #print(polished_files)

    if 'mgt_hwt_results_acc.json' in file_list:
        if editing_type == 1:
            polished_files.append({"polish_percent": 0, "file_name": "mgt_hwt_results_acc.json"})
        elif editing_type == 2:
            polished_files.append({"polish_type": "no_polish", "file_name": "mgt_hwt_results_acc.json"})

    for file in polished_files:
        #print(f"Polish percent: {file['polish_percent']}, File name: {file['file_name']}")
        pos_rate = extract_ai_rate_for_hwt(directory, file)
        file['pos_rate'] = pos_rate
        #print(f"Positive rate: {pos_rate}")

    #sort the list depending on polish percent
    if editing_type == 1:
        polished_files.sort(key=lambda x: x["polish_percent"])
    elif editing_type == 2:
        polished_files = sorted(polished_files, key=lambda x: custom_order.get(x['polish_type'], float('inf')))

    print(polished_files)

    #Plot the positive rate vs polish percent / polish type
    if editing_type == 1:
        polish_percent = [file['polish_percent'] for file in polished_files]
        pos_rate = [file['pos_rate'] for file in polished_files]
        plot_polish_ratio_vs_accuracy(polish_percent, pos_rate, title=f'Polish Ratio vs. MGT Prediction Rate ({directory})', save_path=f"plots/{directory}_polish_rate_vs_mgt_{polisher_type}.pdf")
    elif editing_type == 2:
        polish_type = [file['polish_type'] for file in polished_files]
        pos_rate = [file['pos_rate'] for file in polished_files]
        plot_polish_ratio_vs_accuracy(polish_type, pos_rate, xlabel="Polish Type", title=f'Polish Type vs. MGT Prediction Rate ({directory})', save_path=f"plots/{directory}_polish_type_vs_mgt_{polisher_type}.pdf")

    return polished_files

def main(editing_type, polisher_type):
    
    model_list = ['binoculars', 'chatgpt-roberta', 'detectgpt', 'fastdetectgpt', 'gltr', 'gpt2-base', 'gpt2-large', 'llmdet', 'radar', 'zerogpt']
    
    model_results = []
    combined_plot_dict = {}

    for model in model_list:
        file_with_result = plot_for_model(model, editing_type, polisher_type)
        model_results.append({'model': model, 'results': file_with_result})
        if editing_type == 1:
            polish_prct_list = [file['polish_percent'] for file in file_with_result]
        elif editing_type == 2:
            polish_prct_list = [file['polish_type'] for file in file_with_result]
        #polish_prct_list = [file['polish_percent'] for file in file_with_result]
        pos_rate_list = [file['pos_rate'] for file in file_with_result]
        combined_plot_dict[model] = (polish_prct_list, pos_rate_list)
    
    with open('combined_model_results.json', 'w') as f:
        json.dump(model_results, f, indent=4)

    if editing_type == 1:
        plot_multiple_models_polish_ratio(combined_plot_dict, save_path=f"plots/combined_model_polish_rate_vs_mgt_{polisher_type}.pdf")
    elif editing_type == 2:
        plot_multiple_models_polish_ratio(combined_plot_dict, xlabel='Polish Type', title='Polish Type vs. MGT Prediction Rate', save_path=f"plots/combined_model_polish_type_vs_mgt_{polisher_type}.pdf")


if __name__ == "__main__":
    editing_type = 1
    polisher_type_list = ['gpt', 'llama', 'llama70b', 'llama2']
    for polisher_type in polisher_type_list:
        main(editing_type, polisher_type)
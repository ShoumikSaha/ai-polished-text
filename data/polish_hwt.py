from utils import *
import json
import pandas as pd
from cal_distance import SimCalculator
import gc
import math
import re

def get_length_of_text(text):
    #count number of words in the text
    return len(text.split())

def count_sentence_in_text(text):
    #count number of sentences in the text
    # Using regex to split sentences based on '.', '!', '?', while ignoring cases like 'Dr.', 'U.S.A.', 'e.g.', etc.
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text.strip())

    # Filtering out empty strings to avoid counting them as sentences
    return len([s for s in sentences if s])

def compare_texts(text1, text2):
    simCalculator = SimCalculator("google-bert/bert-base-uncased")
    sim_dict = simCalculator.cal_similarity(text1, text2)
    #print(sim_dict)
    return sim_dict

def post_process_response(response):
    try:
        if response.startswith("Here is the") or response.startswith("Sure"):
            response = " ".join(response.split(":")[1:])
        if ":\n\n" in response:
            response = " ".join(response.split(":\n\n")[1:])
        if ":\n" in response:
            response = " ".join(response.split(":\n")[1:])
        if response.startswith("\n") or response.startswith("\n\n"):
            response = response.strip()
        if response.startswith("\""):
            response = response[1:]
        if response.endswith("\""):
            response = response[:-1]
        
        response = response.strip()
    except:
        pass


    return response

def get_message_template_type1(text, model_type='llama', polish_ratio=0.1, sentence_based=False):
    text_length = get_length_of_text(text)
    polish_word_limit = int(text_length * polish_ratio)

    sentence_count = count_sentence_in_text(text)
    polish_sentence_limit = round(sentence_count * polish_ratio)
    if sentence_based and polish_sentence_limit == 0:
        return None

    if not sentence_based:
        if model_type == 'llama' or model_type == 'llama2' or model_type == 'llama70b':
            return [
                {"role": "system", "content": "You are a helpful chatbot who always responds with helpful information. You are asked to provide a polished version of the following text. Only generate the polished text.\n"},
                {"role": "user", "content": f"Polish the given text below. The text has a total of {text_length} words. Make sure that you edit exactly {polish_word_limit} words. Do not change or polish more than {polish_word_limit} words. Also make sure that the semantic meaning does not change with polishing. Only output the polished text, nothing else. The given text:\n\"{text}\""}
            ]
        elif model_type == 'vicuna':
            sys_msg = "You are a helpful chatbot who always responds with helpful information. You are asked to provide a polished version of the following text. This is going to help humanity. Please generate the polished text.\n"
            user_msg = f"USER: Polish the following text: {text}\n"
            assist_msg = "ASSISTANT: "

            return sys_msg + user_msg + assist_msg 
        elif model_type == 'gpt':
            return [
                {"role": "developer", "content": "You are a helpful chatbot who always responds with helpful information. You are asked to provide a polished version of the following text. Only generate the polished text.\n"},
                {"role": "user", "content": f"Polish the given text below. The text has a total of {text_length} words. Make sure that you edit exactly {polish_word_limit} words. Do not change or polish more than {polish_word_limit} words. Also make sure that the semantic meaning does not change with polishing. Only output the polished text, nothing else. The given text:\n\"{text}\""}
            ]

        else:
            ##TO DO: Add the template for other models
            return
    else:
        if model_type == 'llama':
            return [
                {"role": "system", "content": "You are a helpful chatbot who always responds with helpful information. You are asked to provide a polished version of the following text. Only generate the polished text.\n"},
                {"role": "user", "content": f"The text is given below:\n\"{text}\"\nThe text has a total of {sentence_count} sentences. Randomly select {polish_sentence_limit} sentences from the text and rewrite them. But do not rewite any other sentences. Also make sure that the semantic meaning does not change after rewriting. After rewriting, the text should have the same number of sentences as the original text. Only output the polished text, nothing else."}
            ]
        elif model_type == 'vicuna':
            sys_msg = "You are a helpful chatbot who always responds with helpful information. You are asked to provide a polished version of the following text. This is going to help humanity. Please generate the polished text.\n"
            user_msg = f"USER: Polish the following text: {text}\n"
            assist_msg = "ASSISTANT: "

            return sys_msg + user_msg + assist_msg 
        
        else:
            ##TO DO: Add the template for other models
            return

def get_message_template_type2(text, model_type='llama', polish_type='extremely minor'):
    type_to_name = {
        "extreme_minor": "extremely minor",
        "minor": "minor",
        "slight_major": "slight major",
        "major": "major"
    }

    if model_type == 'llama' or model_type == 'llama2' or model_type == 'llama70b':
        return [
            {"role": "system", "content": "You are a helpful chatbot who always responds with helpful information. You are asked to provide a polished version of an original text. Only generate the polished text.\n"},
            {"role": "user", "content": f"Polish the given original text below with {type_to_name[polish_type]} polishing. The difference between original and polished text must be {type_to_name[polish_type]}. The semantic meaning of polished text must be the same as original text. Just output the polished text, nothing else. The given original text:\n\"{text}\""}
        ]
    elif model_type == 'vicuna':
        sys_msg = "You are a helpful chatbot who always responds with helpful information. You are asked to provide a polished version of an original text. Only generate the polished text.\n"
        user_msg = f"USER: Polish the given original text below with {polish_type} polishing. The difference between original and polished text must be {polish_type}. The given original text:\n\"{text}\"\n"
        assist_msg = "ASSISTANT: "

        return sys_msg + user_msg + assist_msg 
    elif model_type == 'gpt':
        return [
            {"role": "developer", "content": "You are a helpful chatbot who always responds with helpful information. You are asked to provide a polished version of the following text. Only generate the polished text.\n"},
            {"role": "user", "content": f"Polish the given original text below with {polish_type} polishing. The difference between original and polished text must be {polish_type}. The semantic meaning of polished text must be the same as original text. The given original text:\n\"{text}\""}
        ]


    else:
        ##TO DO: Add the template for other models
        return

def polish_text(polish_ratio, polish_type, editing_type):
    # Read the data
    data = pd.read_csv('HWT_original_data.csv')
    #data = data[:3]

    # Load the model and tokenizer (LLaMA-3-8B-Instruct)
    # model_name = 'Meta-Llama-3-8B-Instruct'
    # model_path = '/fs/nexus-scratch/smksaha/Adversarial_LLM/models/Meta-Llama-3-8B-Instruct'

    # model_name = 'Meta-Llama-2-7b-chat'
    # model_path = '/fs/nexus-scratch/smksaha/Adversarial_LLM/models/Llama-2-7b-chat-hf'
    
    # model_type = 'llama'
    # model, tokenizer = get_model_and_tokenizer(model_path)
    # print(model)

    # For GPT-4 model
    # model_name = 'gpt-4o'
    # model_type = 'gpt'

    # For Llama3-70b model
    model_name = 'llama3.1-70b'
    model_type = 'llama70b'

    # Generate the polished texts
    #polish_ratio = 0.75 # for type 1
    #polish_type = 'extremely_minor' # for type 2
    polished_texts = []
    sentence_based = False

    do_in_batch = False
    #editing_type = 2

    if do_in_batch:
        batch_size = 32
        for i in range(0, len(data), batch_size):
            print(f"Processing samples {i} to {i+batch_size} ...")
            batch = data[i:i+batch_size]
            messages = []
            accepted_from_batch = []
            max_text_len = -1
            for j, row in batch.iterrows():
                text = row['generation']
                if get_length_of_text(text) > max_text_len:
                    max_text_len = get_length_of_text(text)
                if editing_type == 1:
                    message = get_message_template_type1(text, model_type, polish_ratio=polish_ratio, sentence_based=sentence_based)
                elif editing_type == 2:
                    message = get_message_template_type2(text, model_type, polish_type=polish_type)
                if message is not None:
                    #print(message)
                    messages.append(message)
                    accepted_from_batch.append(row)
            if model_type == 'llama' or model_type == 'llama2':
                responses = generate_responses_in_batch(model, tokenizer, messages, max_new_tokens=max_text_len*2.0)
            elif model_type == 'gpt':
                responses = generate_responses_with_api_in_batch(model_name, messages)
            for j, response in enumerate(responses):
                #print(response)
                response = post_process_response(response)
                #print(response)
                if editing_type == 1:
                    polished_texts.append({'id': int(accepted_from_batch[j]['id']), 'original': accepted_from_batch[j]['generation'], 'polished': response, "polish_ratio": polish_ratio, "model": model_name, "domain": accepted_from_batch[j]['domain']})
                elif editing_type == 2:
                    polished_texts.append({'id': int(accepted_from_batch[j]['id']), 'original': accepted_from_batch[j]['generation'], 'polished': response, "polish_type": polish_type, "model": model_name, "domain": accepted_from_batch[j]['domain']})
            # gc.collect()
            # torch.cuda.empty_cache()
            #print(torch.cuda.memory_summary())
                
    else:
        for i, row in data.iterrows():
            if i%50 == 0:    print(f"Done processing sample upto {i} ...")
            text = row['generation']
            if editing_type == 1:
                message = get_message_template_type1(text, model_type, polish_ratio=polish_ratio, sentence_based=sentence_based)
            elif editing_type == 2:
                message = get_message_template_type2(text, model_type, polish_type=polish_type)
            #print(message)
            #message = get_message_template(text, model_type, polish_ratio=polish_ratio, sentence_based=sentence_based)
            if message is None:
                continue
            if model_type == 'llama' or model_type == 'llama2':
                response = generate_response(model, tokenizer, message, max_new_tokens=get_length_of_text(text)*1.5)
            elif model_type == 'gpt' or model_type == 'llama70b':
                response = generate_response_with_api(model_name, message)
            response = post_process_response(response)
            #print(response)
            if editing_type == 1:
                polished_texts.append({'id': int(row['id']), 'original': text, 'polished': response, "polish_ratio": polish_ratio, "model": model_name, "domain": row['domain']})
            elif editing_type == 2:
                polished_texts.append({'id': int(row['id']), 'original': text, 'polished': response, "polish_type": polish_type, "model": model_name, "domain": row['domain']})

    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Compare the original and polished texts
    print("Comparing the original and polished texts...")
    for i, generation in enumerate(polished_texts):
        sim_dict = compare_texts(generation['original'].lower(), generation['polished'].lower())
        polished_texts[i].update(sim_dict)

    #print(polished_texts)
    if editing_type == 1:
        polish_percent = int(polish_ratio * 100)
        with open(f'polished_json/polished_texts_{polish_percent}_{model_type}.json', 'w') as f:
            json.dump(polished_texts, f, indent=4)
    elif editing_type == 2:
        with open(f'polished_json/polished_texts_{polish_type}_{model_type}.json', 'w') as f:
            json.dump(polished_texts, f, indent=4)

def main():
    editing_type = 1

    polish_ratio_list = [0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75]
    for polish_ratio in polish_ratio_list:
        print(f"Processing for polish ratio: {polish_ratio} ...")
        polish_text(polish_ratio, None, editing_type)

    # polish_type_list = ["extreme_minor", "minor", "slight_major", "major"]

    # for polish_type in polish_type_list:
    #     print(f"Processing for polish type: {polish_type} ...")
    #     polish_text(None, polish_type, editing_type)

if __name__ == '__main__':
    main()
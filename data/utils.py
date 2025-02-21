import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from os import environ
from openai import OpenAI

openai_api_key = environ['OPENAI_API_KEY']
llama_api_key = environ['LLAMA_API_KEY']

def read_csv(file_path):
    return pd.read_csv(file_path)

def write_csv(df, file_path):
    df.to_csv(file_path, index=False)

def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def get_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    return model, tokenizer

def generate_response(model, tokenizer, message, max_new_tokens=100):
    """
    Generate a response to a given message using the model and tokenizer.
    message: a dict with 'role' and 'content' keys so that chat templates can be applied.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    try:
        # Attempt to use `apply_chat_template` (preferred for chat models)
        encodings = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,  # Ensure uniform input size
            truncation=True
        )

        if isinstance(encodings, dict):
            input_ids = encodings["input_ids"].to(model.device)
            attention_mask = encodings["attention_mask"].to(model.device)
        else:
            input_ids = encodings.to(model.device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(model.device)  # Fallback mask

    except:
        input_ids = tokenizer.encode(message, return_tensors="pt").to(model.device)

    # Handle LLaMA and other models' stop tokens
    if 'llama' in model.config.model_type:
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = tokenizer.eos_token_id
    
    if attention_mask is not None:
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=pad_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
    else:
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    response = outputs[0][input_ids.shape[-1]:]
    response_text = tokenizer.decode(response, skip_special_tokens=True)
    return response_text

def generate_responses_in_batch(model, tokenizer, messages, max_new_tokens=100):
    """
    Generate responses for multiple messages using the model and tokenizer.
    
    messages: a list of dicts, each with 'role' and 'content' keys for chat templates.
    Returns a list of response texts corresponding to each input message.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    try:
        # Attempt to use `apply_chat_template` (preferred for chat models)
        encodings = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,  # Ensure uniform input size
            truncation=True
        )

        if isinstance(encodings, dict):
            input_ids = encodings["input_ids"].to(model.device)
            attention_mask = encodings["attention_mask"].to(model.device)
        else:
            input_ids = encodings.to(model.device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(model.device)  # Fallback mask

    except:
        # Fallback to batch encoding for non-chat models
        encodings = tokenizer.batch_encode_plus(
            [msg["content"] for msg in messages],  # Extract message text
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        input_ids = encodings["input_ids"].to(model.device)
        attention_mask = encodings["attention_mask"].to(model.device)

    # Handle LLaMA and other models' stop tokens
    if 'llama' in model.config.model_type:
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = tokenizer.eos_token_id

    # Explicitly set pad_token_id to avoid warnings
    pad_token_id = tokenizer.pad_token_id

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        pad_token_id=pad_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    responses = [
        tokenizer.decode(output[input_ids.shape[-1]:], skip_special_tokens=True) 
        for output in outputs
    ]

    return responses

def generate_response_with_api(model_name, message):
    """
    Generate a response to a given message using the model and tokenizer.
    message: a dict with 'role' and 'content' keys so that chat templates can be applied.
    """
    if 'gpt' in model_name:
        client = OpenAI(api_key=openai_api_key)

    elif 'llama' in model_name:
        client = OpenAI(
                api_key = llama_api_key,
                base_url = "https://api.llama-api.com"
                )

    completion = client.chat.completions.create(
        model=model_name,
        messages=message
    )
    #print(completion)
    return completion.choices[0].message.content

def generate_responses_with_api_in_batch(model_name, messages_list):
    """
    Generate responses for multiple messages using the GPT-4o model.
    
    Args:
        model_name (str): The name of the OpenAI model to use.
        messages_list (list): A list where each item is a list of message dictionaries with 'role' and 'content' keys.
        openai_api_key (str): Your OpenAI API key.
    
    Returns:
        list: A list of responses corresponding to each message input.
    """
    client = OpenAI(api_key=openai_api_key)
    
    responses = []
    for message in messages_list:
        #print(message)
        completion = client.chat.completions.create(
            model=model_name,
            messages=message
        )
        #print(completion.choices[0])
        responses.append(completion.choices[0].message.content)

    return responses
import json
import os

import requests
from tqdm import tqdm


class Pangram:
    def __init__(self):
        self.api_key = os.environ["PANGRAM_API_KEY"]
        if self.api_key == "":
            print("Warning: Pangram API key is not set. Add API key to api_keys.py and run the script.")
            exit(-1)

    def inference(self, texts: list) -> list:
        url = "https://text.api.pangramlabs.com"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        predictions = []
        
        for text in tqdm(texts):
            payload = json.dumps({"text": text})
            response = requests.request("POST", url, headers=headers, data=payload)
            
            if response.status_code == 200:
                response_json = response.json()
                ai_score = response_json.get("ai_likelihood", -1)  # -1 as placeholder for error
                predictions.append(ai_score)
            else:
                print(f"Pangram returned a status code {response.status_code} error: {response}\n")
                predictions.append(-1)  # Using -1 as placeholder for error
                
        return predictions

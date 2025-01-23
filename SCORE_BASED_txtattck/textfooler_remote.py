import requests
import torch
import numpy as np
import transformers
import textattack
# from textattack.attack_recipes import TextFoolerJin2019

import requests
import transformers
import torch
import time

class RemoteModelWrapper():
    def __init__(self, api_url):
        self.api_url = api_url
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")

    def __call__(self, text_input_list):
        predictions = []
        for text in text_input_list:
            params = dict()
            params["text"] = text
            retries = 3  # Number of retries for each API call
            delay = 5  # Seconds to wait before retrying
            
            for attempt in range(retries):
                try:
                    response = requests.post(self.api_url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        # Assuming the API returns probabilities for positive and negative
                        predictions.append([result["negative"], result["positive"]])
                        break
                    elif response.status_code == 403:
                        print(f"403 Forbidden: {response.text}. Skipping this input.")
                        predictions.append([0.5, 0.5])  # Assign neutral probabilities
                        break
                    else:
                        print(f"Unexpected status code {response.status_code}: {response.text}")
                        raise ValueError(f"API call failed with status {response.status_code}")
                
                except (requests.exceptions.RequestException, ValueError) as e:
                    print(f"Error: {e}. Attempt {attempt + 1} of {retries}")
                    if attempt < retries - 1:
                        time.sleep(delay)  # Wait before retrying
                    else:
                        print("Max retries reached. Assigning default prediction.")
                        predictions.append([0.5, 0.5])  # Assign neutral probabilities if retries fail
                        break
        
        return torch.tensor(predictions)

# Define the remote model API endpoint and tokenizer
api_url = "https://<model>/predict"
model_wrapper = RemoteModelWrapper(api_url)

# Build the attack
attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)

# Define dataset and attack arguments
dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")

attack_args = textattack.AttackArgs(
    num_examples=100,
    log_to_csv="/Users/<user>/Documents/adversarial-ai/textfooler.csv",
    checkpoint_interval=5,
    checkpoint_dir="checkpoints", 
    disable_stdout=True
)

# Run the attack
attacker = textattack.Attacker(attack, dataset, attack_args)
attacker.attack_dataset()

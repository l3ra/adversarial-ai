import requests
import numpy as np
from textattack.models.wrappers import PyTorchModelWrapper

class RemoteModelWrapper(PyTorchModelWrapper):
    def __init__(self, api_url, tokenizer):
        """
        Args:
            api_url (str): The URL of the remote model API endpoint.
            tokenizer: A tokenizer compatible with the inputs expected by the remote model.
        """
        self.api_url = api_url
        self.tokenizer = tokenizer

    @property
    def model(self):
        """
        Dummy property to comply with TextAttack's requirements.
        TextAttack expects a 'model' attribute, but we only have an endpoint.
        """
        return None  # There is no local model, so we return None.

    def __call__(self, text_input_list):
        """
        Sends inputs to the remote model API and retrieves predictions.

        Args:
            text_input_list (list[str]): List of input texts.

        Returns:
            np.ndarray: Model's logits for each input.
        """
        # Tokenize input text
        inputs = self.tokenizer(
            text_input_list,
            truncation=True,
            padding=True,
            return_tensors="json",  # Use "json" format for API compatibility
        )

        # Make the HTTP request to the remote model
        response = requests.post(self.api_url, json=inputs)

        # Check for errors
        if response.status_code != 200:
            raise ValueError(f"API call failed with status code {response.status_code}: {response.text}")

        # Extract logits from API response
        logits = response.json().get("logits")
        if logits is None:
            raise ValueError("No 'logits' found in API response")

        return np.array(logits)

    def get_grad(self, text_input):
        """
        Gradient computation is not supported for remote models.
        """
        raise NotImplementedError("Gradient computation is not supported for remote models.")


from transformers import AutoTokenizer
import textattack

# Define API URL and tokenizer
api_url = "https://leraleonteva.com/predict"
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")

# Wrap the remote model
model_wrapper = RemoteModelWrapper(api_url, tokenizer)

# Set up TextAttack
dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")
attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
attack_args = textattack.AttackArgs(num_examples=20, log_to_csv="log.csv", checkpoint_interval=5)

attacker = textattack.Attacker(attack, dataset, attack_args)
attacker.attack_dataset()

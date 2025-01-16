from textattack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    MaxModificationRate,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import SBERT
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapEmbedding, WordSwapMaskedLM
import textattack
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack.datasets import HuggingFaceDataset

# Define custom attack recipe
def custom_a2t_recipe(model_wrapper):
    # Define constraints
    constraints = [
        WordEmbeddingDistance(min_cos_sim=0.8),  # Specify a valid parameter
        StopwordModification(),
        PartOfSpeech(allow_verb_noun_swap=False),
    ]

    # Transformation
    transformation = WordSwapEmbedding(max_candidates=20)

    # Goal function
    goal_function = UntargetedClassification(model_wrapper)

    # Search method
    search_method = GreedyWordSwapWIR(wir_method="gradient")

    # Create and return attack
    return Attack(goal_function, constraints, transformation, search_method)

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Load dataset
dataset = HuggingFaceDataset("imdb", split="test")
print(f"Dataset loaded: {len(dataset)} samples")

# Ensure the dataset has the correct structure
print(f"Sample data structure: {dataset[0]}")  # Inspect the first sample

# Build attack using the custom recipe
attack = custom_a2t_recipe(model_wrapper)

# Process and attack first 10 samples
for i in range(10):
    sample = dataset[i]
    
    # Check if sample is a tuple or a dictionary and handle accordingly
    if isinstance(sample, tuple):
        original_text = sample[0]  # Assuming the text is the first element
        label = sample[1]  # Assuming the label is the second element
    else:
        original_text = sample['text']  # If it's a dictionary, access by key
        label = sample['label']  # Access label
    
    print(f"Original Text: {original_text}")
    
    # Pass both the text and the label to the attack
    attack_input = (original_text, label)

    # Perform the attack
    result = attack.attack(attack_input[0], attack_input[1])  # Separate the text and label
    
    # Output perturbed text
    print(f"Attacked Text: {result.perturbed_text}")

import typing as t
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import dataclasses
from collections import defaultdict
import random

PROJECT = "llm-pgd"
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant."
)

@dataclasses.dataclass
class Config:
    """
    Configuration for PGD-based attack.
    """
    model_name: str = "textattack/bert-base-uncased-imdb"
    precision: t.Optional[str] = None
    num_samples: int = 100  # Number of examples to evaluate
    attack_threshold: float = 0.5  # Threshold for attack success
    max_queries: int = 100  # Max number of queries for an attack
    random_seed: int = 42  # Random seed for reproducibility


def initialize_model_and_tokenizer(config: Config):
    """
    Initialize the model and tokenizer using the transformers library.
    """
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, clean_up_tokenization_spaces=False)
    return model, tokenizer


def evaluate_attacks(config: Config, model, tokenizer):
    """
    Perform a simulated attack evaluation.
    """
    # Set random seed
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # Simulated data
    dataset = [
        ("This movie was fantastic!", 1),  # Positive
        ("The film was a disaster.", 0),  # Negative
        # Add more samples for realistic testing
    ] * (config.num_samples // 2)

    metrics = defaultdict(int)
    total_queries = 0
    total_perturbed_words = 0
    total_words = 0

    for text, true_label in dataset:
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits).item()

        if predicted_label != true_label:
            metrics["skipped_attacks"] += 1
            continue

        # Simulate an attack
        num_queries = random.randint(1, config.max_queries)
        perturbed_words = random.randint(1, len(text.split()))  # Randomly simulate perturbation
        is_attack_successful = random.random() > config.attack_threshold

        total_queries += num_queries
        total_perturbed_words += perturbed_words
        total_words += len(text.split())

        if is_attack_successful:
            metrics["successful_attacks"] += 1
        else:
            metrics["failed_attacks"] += 1

    metrics["original_accuracy"] = (len(dataset) - metrics["skipped_attacks"]) / len(dataset)
    metrics["accuracy_under_attack"] = metrics["failed_attacks"] / len(dataset)
    metrics["attack_success_rate"] = metrics["successful_attacks"] / len(dataset)
    metrics["avg_perturbed_word_pct"] = (total_perturbed_words / total_words) * 100
    metrics["avg_num_words_per_input"] = total_words / len(dataset)
    metrics["avg_num_queries"] = total_queries / len(dataset)

    return metrics


def main():
    # Configuration
    config = Config()

    # Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer(config)

    # Evaluate attacks
    metrics = evaluate_attacks(config, model, tokenizer)

    # Output results
    print(f"Number of successful attacks: {metrics['successful_attacks']}")
    print(f"Number of failed attacks: {metrics['failed_attacks']}")
    print(f"Number of skipped attacks: {metrics['skipped_attacks']}")
    print(f"Original accuracy: {metrics['original_accuracy']:.2%}")
    print(f"Accuracy under attack: {metrics['accuracy_under_attack']:.2%}")
    print(f"Attack success rate: {metrics['attack_success_rate']:.2%}")
    print(f"Average perturbed word %: {metrics['avg_perturbed_word_pct']:.2f}%")
    print(f"Average number of words per input: {metrics['avg_num_words_per_input']:.2f}")
    print(f"Average number of queries: {metrics['avg_num_queries']:.2f}")


if __name__ == "__main__":
    main()

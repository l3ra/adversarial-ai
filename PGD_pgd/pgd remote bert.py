import typing as t
import requests
import dataclasses
from collections import defaultdict
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("attack_log.txt"),  # Log to a file
        logging.StreamHandler()  # Also log to console
    ]
)

PROJECT = "llm-pgd"
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant."
)

@dataclasses.dataclass
class Config:
    """
    Configuration for PGD-based attack.
    """
    api_url: str = "https://leraleonteva.com/predict"  # URL for the model's API
    num_samples: int = 100  # Number of examples to evaluate
    attack_threshold: float = 0.5  # Threshold for attack success
    max_queries: int = 100  # Max number of queries for an attack
    random_seed: int = 42  # Random seed for reproducibility


def query_model(api_url: str, text: str) -> t.Tuple[float, int]:
    """
    Query the remote model via an API.

    Args:
        api_url (str): The URL for the model API.
        text (str): The input text for the model.

    Returns:
        (float, int): The confidence score for the positive class and the predicted label.
    """
    try:
        url = f"{api_url.rstrip('/')}/?text={requests.utils.quote(text)}"
        response = requests.post(url)
        response.raise_for_status()
        result = response.json()
        positive_score = result["positive"]
        negative_score = result["negative"]
        predicted_label = 1 if positive_score > negative_score else 0
        return positive_score, predicted_label
    except requests.RequestException as e:
        logging.error(f"HTTP Request failed: {e}")
        raise
    except KeyError:
        logging.error(f"Unexpected response format: {response.text}")
        raise


def evaluate_attacks(config: Config):
    """
    Perform a simulated attack evaluation.
    """
    # Set random seed
    random.seed(config.random_seed)

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

    for idx, (text, true_label) in enumerate(dataset):
        try:
            positive_score, predicted_label = query_model(config.api_url, text)
        except Exception as e:
            logging.warning(f"Error querying model for text: {text}. Skipping. Error: {e}")
            metrics["skipped_attacks"] += 1
            continue

        if predicted_label != true_label:
            logging.info(f"Skipped attack on sample {idx} (text: {text}). Predicted label {predicted_label} != True label {true_label}")
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
            logging.info(
                f"Attack successful on sample {idx}: "
                f"Text: {text}, Queries: {num_queries}, Perturbed words: {perturbed_words}, "
                f"Positive score: {positive_score:.4f}"
            )
        else:
            metrics["failed_attacks"] += 1
            logging.info(
                f"Attack failed on sample {idx}: "
                f"Text: {text}, Queries: {num_queries}, Perturbed words: {perturbed_words}, "
                f"Positive score: {positive_score:.4f}"
            )

    metrics["original_accuracy"] = (len(dataset) - metrics["skipped_attacks"]) / len(dataset)
    metrics["accuracy_under_attack"] = metrics["failed_attacks"] / len(dataset)
    metrics["attack_success_rate"] = metrics["successful_attacks"] / len(dataset)
    metrics["avg_perturbed_word_pct"] = (total_perturbed_words / total_words) * 100
    metrics["avg_num_words_per_input"] = total_words / len(dataset)
    metrics["avg_num_queries"] = total_queries / len(dataset)

    logging.info("Final metrics:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value}")

    return metrics


def main():
    # Configuration
    config = Config(api_url="https://leraleonteva.com/predict")

    # Evaluate attacks
    metrics = evaluate_attacks(config)

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

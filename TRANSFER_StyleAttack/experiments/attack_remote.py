import argparse
import requests
from style_paraphrase.inference_utils import GPT2Generator
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_name')
parser.add_argument('--orig_file_path')
parser.add_argument('--model_dir')
parser.add_argument('--output_file_path')
parser.add_argument('--p_val', default=0.6, type=float)
parser.add_argument('--iter_epochs', default=10, type=int)
parser.add_argument('--orig_label', default=None, type=int)
parser.add_argument('--bert_type', default='bert-base-uncased')
parser.add_argument('--output_nums', default=2, type=int)
params = parser.parse_args()

def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    return data

def write_data(attack_data):
    with open(params.output_file_path, 'w') as f:
        print('p_val', '\t', 'orig_sent', '\t', 'adv_sent', '\t', 'original_class', '\t', 'adversarial_class', file=f)
        for p_val, orig_sent, adv_sent, label, predict in attack_data:
            print(p_val, '\t', orig_sent, '\t', adv_sent, '\t', label, '\t', predict, file=f)

def get_predict_label(sent):
    """
    Sends the sentence to the deployed BERT model for inference and returns the predicted label.
    """
    try:
        response = requests.post(f"https://leraleonteva.com/predict/?text={sent}")
        response.raise_for_status()  # Raise an HTTPError for bad responses
        sentiment_scores = response.json()  # Parse the JSON response
        positive = sentiment_scores.get("positive", 0)
        negative = sentiment_scores.get("negative", 0)
        
        # Determine the label based on the highest score
        return 1 if positive > negative else 0
    except requests.RequestException as e:
        print(f"Error during inference: {e}")
        return None  # Return None if the prediction fails

if __name__ == '__main__':
    orig_data = read_data(params.orig_file_path)
    paraphraser = GPT2Generator(params.model_dir, upper_length="same_5")

    mis = 0
    total = 0
    attack_data = []
    paraphraser.modify_p(params.p_val)

    for sent, label in tqdm(orig_data):
        # Skip sentences with mismatched original labels or predictions
        if params.orig_label is not None:
            predicted_label = get_predict_label(sent)
            if predicted_label is None or label != params.orig_label or predicted_label != params.orig_label:
                continue

        if label != get_predict_label(sent):
            continue

        flag = False
        generated_sent = [sent for _ in range(params.iter_epochs)]
        paraphrase_sentences_list = paraphraser.generate_batch(generated_sent)[0]

        for paraphrase_sent in paraphrase_sentences_list:
            predict = get_predict_label(paraphrase_sent)
            if predict is not None and predict != label:
                attack_data.append((1, sent, paraphrase_sent, label, predict))
                flag = True
                mis += 1
                break

        if not flag:
            attack_data.append((-1, sent, sent, label, label))

        total += 1

    write_data(attack_data)

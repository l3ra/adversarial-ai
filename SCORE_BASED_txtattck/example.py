import transformers
import textattack


model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")

tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")

model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")

attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)

attack_args = textattack.AttackArgs(
    num_examples=20, 
    log_to_csv="/Users/leraleonteva/Documents/adversarial-ai/log.csv", 
    checkpoint_interval=5, 
    checkpoint_dir="checkpoints", 
    disable_stdout=True)

attacker = textattack.Attacker(attack, dataset, attack_args)

attacker.attack_dataset()
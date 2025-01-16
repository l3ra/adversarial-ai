# This repo uses a sample of possible adversarial attacks designed to provoke unintended behaviour in machine learning models

Key samples are taken from Projected Gradient Descent, Backdoor, Transfer and Score Based Attacks are used to test the output metrics of remotely deployed language based models.

# Example CLI command to run a remote score based attack:
```
python remote_model.py
```

# Example CLI command to run a remote transfer attack:
```
CUDA_VISIBLE_DEVICES=0 python attack_remote.py --model_name  textattack/bert-base-uncased-SST-2 --orig_file_path ../data/clean/sst-2/test.tsv --model_dir bible_model --output_file_path record.log cd
```
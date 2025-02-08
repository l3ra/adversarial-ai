## This repo is part of the publication titled 'Evaluating Adversarial Attacks against Artificial Intelligence Systems in Application Deployments

### It uses a sample of possible adversarial attacks designed to provoke unintended behaviour in machine learning models

Key samples are taken from Projected Gradient Descent, Backdoor, Transfer and Score Based Attacks are used to test the output metrics of remotely deployed language based models.

# Example CLI command to run a remote score based attack:
```
python remote_model.py
```
###Location of original attack script:
`SCORE_BASED_txtattck/example.py`
###Location of remote attack script:
`SCORE_BASED_txtattck/remote_model.py`

# Example CLI command to run a remote transfer attack:
```
CUDA_VISIBLE_DEVICES=0 python attack_remote.py --model_name  textattack/bert-base-uncased-SST-2 --orig_file_path ../data/clean/sst-2/test.tsv --model_dir bible_model --output_file_path record.log cd
```
###Location of original attack script:
`TRANSFER_StyleAttack/experiments/attack.py`
###Location of remote attack script:
`TRANSFER_StyleAttack/experiments/attack_remote.py`

# Example CLI command to run a remote PGD attack:
```
python ../pgd/pgd remote bert.py
```
###Location of original attack script:
`PGD_pgd/pgd local%20llama 1B.py`
###Location of remote attack script:
`PGD_pgd/pgd remote attack llama 1B.py`

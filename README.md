
## Example CLI command to run a remote score based attack:
CUDA_VISIBLE_DEVICES=0 python attack_remote.py --model_name  textattack/bert-base-uncased-SST-2 --orig_file_path ../data/clean/sst-2/test.tsv --model_dir bible_model --output_file_path record.log

## Example CLI command to run a remote transfer attack:
CUDA_VISIBLE_DEVICES=0 python attack_remote.py --model_name  textattack/bert-wbase-uncased-SST-2 --orig_file_path ../data/clean/sst-2/test.tsv --model_dir bible_model --output_file_path record.log cd
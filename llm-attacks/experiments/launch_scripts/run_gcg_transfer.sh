#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export n=25
export model=$1 # llama2 or vicuna or vicuna_guanaco
export num_train_models=$2

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

export PYTHONPATH=$(dirname $(dirname $(pwd))):$PYTHONPATH
python3 -u ../main.py \
    --config="../configs/transfer_${model}.py" \
    --config.attack=gcg \
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.result_prefix="../results/transfer_${model}_gcg_${n}_progressive" \
    --config.progressive_goals=True \
    --config.stop_on_success=True \
    --config.allow_non_ascii=False \
    --config.num_train_models=$num_train_models \
    --config.n_train_data=$n \
    --config.n_test_data=$n \
    --config.n_steps=1 \
    --config.test_steps=1 \
    --config.batch_size=512

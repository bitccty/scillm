#! /bin/bash

# ============================= alpaca ================================

CUDA_VISIBLE_DEVICES=0 python ./playground/alpaca-baseline.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --result_path ./playground/alpaca_qa_scimrc.txt \
    --data_type natural \
    --task qa

CUDA_VISIBLE_DEVICES=1 python ./playground/alpaca-baseline.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --result_path ./playground/alpaca_qa_scimrc_recall.txt \
    --data_type natural \
    --task qa \
    --qa_type recall

CUDA_VISIBLE_DEVICES=2 python ./playground/alpaca-baseline.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --result_path ./playground/alpaca_qa_scimrc_bias.txt \
    --data_type natural \
    --task qa \
    --qa_type bias

CUDA_VISIBLE_DEVICES=2 python ./playground/alpaca-baseline.py \
    --data_path ../data/sft/processed_pubmed_test_set.json \
    --result_path ./playground/alpaca_summary_pubmed.txt \
    --data_type natural \
    --task summary 


# ============================= openalpaca ================================

CUDA_VISIBLE_DEVICES=3 python ./playground/alpaca-baseline.py \
    --model_path openllmplayground/openalpaca_7b_700bt_preview \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --result_path ./playground/openalpaca_qa_scimrc.txt \
    --data_type natural \
    --task qa

CUDA_VISIBLE_DEVICES=4 python ./playground/alpaca-baseline.py \
    --model_path openllmplayground/openalpaca_7b_700bt_preview \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --result_path ./playground/openalpaca_qa_scimrc_recall.txt \
    --data_type natural \
    --task qa \
    --qa_type recall

CUDA_VISIBLE_DEVICES=5 python ./playground/alpaca-baseline.py \
    --model_path openllmplayground/openalpaca_7b_700bt_preview \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --result_path ./playground/openalpaca_qa_scimrc_bias.txt \
    --data_type natural \
    --task qa \
    --qa_type bias

CUDA_VISIBLE_DEVICES=3 python ./playground/alpaca-baseline.py \
    --model_path openllmplayground/openalpaca_7b_700bt_preview \
    --data_path ../data/sft/processed_pubmed_test_set.json \
    --result_path ./playground/openalpaca_summary_pubmed.txt \
    --data_type natural \
    --task summary 

# ============================= vicuna ================================ 

CUDA_VISIBLE_DEVICES=6 python ./playground/vicuna-baseline.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --result_path ./playground/vicuna_qa_scimrc.txt \
    --data_type natural \
    --task qa

CUDA_VISIBLE_DEVICES=7 python ./playground/vicuna-baseline.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --result_path ./playground/vicuna_qa_scimrc_recall.txt \
    --data_type natural \
    --task qa \
    --qa_type recall

CUDA_VISIBLE_DEVICES=0 python ./playground/vicuna-baseline.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --result_path ./playground/vicuna_qa_scimrc_bias.txt \
    --data_type natural \
    --task qa \
    --qa_type bias

CUDA_VISIBLE_DEVICES=4 python ./playground/vicuna-baseline.py \
    --data_path ../data/sft/processed_pubmed_test_set.json \
    --result_path ./playground/vicuna_summary_pubmed.txt \
    --data_type natural \
    --task summary 
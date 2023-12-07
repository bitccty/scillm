#!/bin/bash

# ========================= qasper =======================

CUDA_VISIBLE_DEVICES=0 python test_generation.py \
    --data_path ../data/sft/processed_qasper_test_set.json \
    --delta_model_path  ./ckpt/scillm-structure-base-delta-19/18\
    --result_path ./result/scillm-structure-base-qa-qasper.txt \
    --data_type structure \
    --task qa 

CUDA_VISIBLE_DEVICES=1 python test_generation.py \
    --data_path ../data/sft/processed_qasper_test_set.json \
    --delta_model_path  ./ckpt/scillm-structure-delta-18/18\
    --result_path ./result/scillm-structure-raw-qa-qasper.txt \
    --data_type raw_structure \
    --task qa 

CUDA_VISIBLE_DEVICES=2 python test_generation.py \
    --data_path ../data/sft/processed_qasper_test_set.json \
    --delta_model_path  ./ckpt/scillm-structure-delta-18/18\
    --result_path ./result/scillm-structure-raw-qa-qasper-recall.txt \
    --data_type raw_structure \
    --task qa \
    --qa_type recall

CUDA_VISIBLE_DEVICES=3 python test_generation.py \
    --data_path ../data/sft/processed_qasper_test_set.json \
    --delta_model_path  ./ckpt/scillm-structure-delta-18/18\
    --result_path ./result/scillm-structure-raw-qa-qasper-bias.txt \
    --data_type raw_structure \
    --task qa \
    --qa_type bias


CUDA_VISIBLE_DEVICES=1 python test_generation.py \
    --data_path ../data/sft/processed_qasper_test_set.json \
    --delta_model_path  ./ckpt/scillm-structure-nounified-delta-18/18\
    --result_path ./result/scillm-structure-nounified-qa-qasper.txt \
    --data_type structure \
    --task qa 



# ========================= scimrc =======================

CUDA_VISIBLE_DEVICES=0 python test_generation.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --delta_model_path  ./ckpt/scillm-structure-delta-18/18\
    --result_path ./result/scillm-structure-qa-scimrc.txt \
    --data_type structure \
    --task qa 

CUDA_VISIBLE_DEVICES=3 python test_generation.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --delta_model_path  ./ckpt/scillm-natural-delta-19/18\
    --result_path ./result/scillm-natural-qa-scimrc.txt \
    --data_type natural \
    --task qa 

CUDA_VISIBLE_DEVICES=4 python test_generation.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --delta_model_path  ./ckpt/scillm-latex-delta-19/18\
    --result_path ./result/scillm-latex-qa-scimrc.txt \
    --data_type natural \
    --task qa 

CUDA_VISIBLE_DEVICES=5 python test_generation.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --delta_model_path  ./ckpt/llama-new/18\
    --result_path ./result/scillm-llama-qa-scimrc.txt \
    --data_type natural \
    --task qa 

CUDA_VISIBLE_DEVICES=7 python test_generation.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --delta_model_path  None\
    --result_path ./result/llama-qa-scimrc.txt \
    --data_type natural \
    --task qa 

# ========================= scimrc recall =======================

CUDA_VISIBLE_DEVICES=1 python test_generation.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --delta_model_path  ./ckpt/scillm-structure-delta-18/18\
    --result_path ./result/scillm-structure-qa-scimrc-recall.txt \
    --data_type structure \
    --task qa \
    --qa_type recall

CUDA_VISIBLE_DEVICES=2 python test_generation.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --delta_model_path  ./ckpt/scillm-natural-delta-19/18\
    --result_path ./result/scillm-natural-qa-scimrc-recall.txt \
    --data_type natural \
    --task qa \
    --qa_type recall

CUDA_VISIBLE_DEVICES=3 python test_generation.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --delta_model_path  ./ckpt/scillm-latex-delta-19/18\
    --result_path ./result/scillm-latex-qa-scimrc-recall.txt \
    --data_type natural \
    --task qa \
    --qa_type recall

CUDA_VISIBLE_DEVICES=4 python test_generation.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --delta_model_path  ./ckpt/llama-new/18\
    --result_path ./result/scillm-llama-qa-scimrc-recall.txt \
    --data_type natural \
    --task qa \
    --qa_type recall

CUDA_VISIBLE_DEVICES=5 python test_generation.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --delta_model_path  None\
    --result_path ./result/llama-qa-scimrc-recall.txt \
    --data_type natural \
    --task qa \
    --qa_type recall

# ========================= scimrc bias =======================

CUDA_VISIBLE_DEVICES=0 python test_generation.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --delta_model_path  ./ckpt/scillm-structure-delta-18/18\
    --result_path ./result/scillm-structure-qa-scimrc-bias.txt \
    --data_type structure \
    --task qa \
    --qa_type bias

CUDA_VISIBLE_DEVICES=3 python test_generation.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --delta_model_path  ./ckpt/scillm-natural-delta-19/18\
    --result_path ./result/scillm-natural-qa-scimrc-bias.txt \
    --data_type natural \
    --task qa \
    --qa_type bias

CUDA_VISIBLE_DEVICES=4 python test_generation.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --delta_model_path  ./ckpt/scillm-latex-delta-19/18\
    --result_path ./result/scillm-latex-qa-scimrc-bias.txt \
    --data_type natural \
    --task qa \
    --qa_type bias

CUDA_VISIBLE_DEVICES=5 python test_generation.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --delta_model_path  ./ckpt/llama-new/18\
    --result_path ./result/scillm-llama-qa-scimrc-bias.txt \
    --data_type natural \
    --task qa \
    --qa_type bias

CUDA_VISIBLE_DEVICES=6 python test_generation.py \
    --data_path ../data/sft/processed_scimrc_test_set.json \
    --delta_model_path  None\
    --result_path ./result/llama-qa-scimrc-bias.txt \
    --data_type natural \
    --task qa \
    --qa_type bias

# ========================= summary pubmed =======================

CUDA_VISIBLE_DEVICES=0 python test_generation.py \
    --data_path ../data/sft/processed_pubmed_test_set.json \
    --delta_model_path  None\
    --result_path ./result/llama-summary-pubmed.txt \
    --data_type natural \
    --task summary

CUDA_VISIBLE_DEVICES=3 python test_generation.py \
    --data_path ../data/sft/processed_pubmed_test_set.json \
    --delta_model_path  ./ckpt/llama-new/18 \
    --result_path ./result/scillm-llama-summary-pubmed.txt \
    --data_type natural \
    --task summary

CUDA_VISIBLE_DEVICES=5 python test_generation.py \
    --data_path ../data/sft/processed_pubmed_test_set.json \
    --delta_model_path  ./ckpt/llama-latex-new/18\
    --result_path ./result/scillm-latex-summary-pubmed.txt \
    --data_type natural \
    --task summary

CUDA_VISIBLE_DEVICES=4 python test_generation.py \
    --data_path ../data/sft/processed_pubmed_test_set.json \
    --delta_model_path  ./ckpt/llama-natural-new/18 \
    --result_path ./result/scillm-natural-summary-pubmed.txt \
    --data_type natural \
    --task summary

CUDA_VISIBLE_DEVICES=1 python test_generation.py \
    --data_path ../data/sft/processed_pubmed_test_set.json \
    --delta_model_path  ./ckpt/scillm-structure-delta-18/18\
    --result_path ./result/scillm-structure-summary-pubmed.txt \
    --data_type structure \
    --task summary
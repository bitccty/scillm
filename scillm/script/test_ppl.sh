#!/bin/bash

# for num in {0..19}
# do
#     CUDA_VISIBLE_DEVICES=6 python test_ppl.py \
#         --model scillm\
#         --model_path decapoda-research/llama-7b-hf\
#         --delta_model_path ./ckpt/scillm_llama_natural/$num\
#         --data_path ../data/pretrain/train_NL/base/c4_tokens.txt\
#         --result_path ./result/scillm_llama_natural
# done

for num in {0..17}
do
    CUDA_VISIBLE_DEVICES=5 python test_ppl.py \
        --model scillm\
        --model_path decapoda-research/llama-7b-hf\
        --delta_model_path ./ckpt/scillm_llama_ST_base_a/$num\
        --data_path ../data/pretrain/train_NL/base/c4_tokens.txt\
        --result_path ./result/scillm_llama_structure
done

# CUDA_VISIBLE_DEVICES=7 python test_ppl.py \
#     --model scillm\
#     --model_path decapoda-research/llama-7b-hf\
#     --data_path ../data/pretrain/train_NL/base/c4_tokens.txt\
#     --result_path ./result/llama
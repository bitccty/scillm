#! /bin/bash


deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_addr 127.0.0.1 --master_port 28457 train_sft.py \
    --model scillm-sft\
    --model_path decapoda-research/llama-7b-hf\
    --delta_model_path ./ckpt/scillm_llama_natural/19\
    --train_data_path ../data/sft/train.json\
    --save_path ./ckpt/scillm-natural-delta-19 \
    --log_path ./rest/scillm-natural-delta-19 \
    --data_type natural
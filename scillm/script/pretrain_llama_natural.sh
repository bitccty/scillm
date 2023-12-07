#!/bin/bash

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_addr 127.0.0.1 --master_port 28463 pretrain.py \
    --model scillm\
    --model_path decapoda-research/llama-7b-hf\
    --train_data_path ../data/pretrain/train_NL/base \
    --save_path ./ckpt/scillm_llama_natural  \
    --log_path ./rest/scillm_llama_natural

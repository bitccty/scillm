#! /bin/bash

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_addr 127.0.0.1 --master_port 28456 train_sft.py \
    --model scillm-sft\
    --model_path decapoda-research/llama-7b-hf\
    --delta_model_path None\
    --train_data_path ../data/sft/train.json\
    --save_path ./ckpt/scillm-llama-sft \
    --log_path ./rest/scillm-llama-sft \
    --data_type evidence
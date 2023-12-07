#!/bin/bash

source ~/miniconda3/bin/activate scillm
root_path=./pretrain
for p in train_LT train_NL/base train_NL/base_f train_ST/base train_ST/base_a train_ST/base_f train_ST/base_p train_ST/base_p_a train_ST/base_pf
do
    path=$root_path/$p
    echo Start process $path
    # combination
    cat $path/arxiv_tokens.txt $path/c4_tokens.txt > $path/train.txt
    # shuffle
    shuf $path/train.txt -o $path/train_shuffle.txt
    echo Finish shuffle from $path/train_tokens.txt to $path/train_shuffle.txt
    # split
    line_count=$(wc -l < "$path/train_shuffle.txt")
    split_count=$(($line_count/8))
    echo $line_count, $split_count
    split -l $split_count $path/train_shuffle.txt -d $path/split_
    echo Success
done

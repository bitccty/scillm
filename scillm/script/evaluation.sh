#! /bin/bash

# for file in `ls ./result | grep ^scillm`
# do
#     echo Processing ./result/$file
#     python evaluation.py -p ./result/$file
# done

# for file in `ls ./result | grep ^llama`
# do
#     echo Processing ./result/$file
#     python evaluation.py -p ./result/$file
# done


for file in `ls ./playground | grep ^alpaca_`
do
    echo Processing ./playground/$file
    python evaluation.py -p ./playground/$file
done
for file in `ls ./playground | grep ^openalpaca_`
do
    echo Processing ./playground/$file
    python evaluation.py -p ./playground/$file
done
for file in `ls ./playground | grep ^vicuna_`
do
    echo Processing ./playground/$file
    python evaluation.py -p ./playground/$file
done
# Model

## 1. Pretrain

Taking pretraining the LLaMA with Data Augmentation Method for example, run the command:
```bash
bash ./script/pretrain_llama_structure.sh
```
The key arguments of the training script are as follows:

- model: The model name listed in config/base.json.
- model_path: The checkpoint for large-scale langauge models decapoda-research/llama-7b-hf for LLaMA-7B.
- train_data_path: The path saves the pretraining corpus.
- log_path: The directory that saves the pre-trained log in tensorboard format. This directory will be automatically created.
- save_path: The directory which saves the trained QLoRA delta weights. This directory will be automatically created.

Note that the total training steps can be set in the `total_step` argument at [./config/base.yaml](./config/base.yaml) file. Set this argument carefully to make sure all the tokens will be used during training.


## 2. Fine-tune

To fine-tune pretrained model on a mixture of (kp20, arxiv-suammrizarion, qasper), Taking USSC for example, run the command:
```bash
bash ./script/train_sft_structure.sh
```

- model: The model name listed in config/base.json.
- model_path: The checkpoint for large-scale langauge models decapoda-research/llama-7b-hf for LLaMA-7B.
- delta_model_path: The LoRA checkpoint weighted pre-traing in Section 1. The SFT process will continue optimize these LoRA weights for paper-ground question answering task.
- train_data_path: The path saves the pretraining corpus.
- log_path: The directory that saves the pre-trained log in tensorboard format. This directory will be automatically created.
- save_path: The directory which saves the trained QLoRA delta weights. This directory will be automatically created.
- data_type: The data representation used to fine-tune

## 3. Evaluation

To generate and evaluate the models, run the command:
```bash
bash ./script/test_generation.sh
```

- model: The model name listed in config/base.json.
- model_path: The checkpoint for large-scale langauge models decapoda-research/llama-7b-hf for LLaMA-7B.
- delta_model_path: The LoRA checkpoint weighted pre-traing in Section 1. The SFT process will continue optimize these LoRA weights for paper-ground question answering task.
- data_path: The path saves the test set.
- result_path: The directory that saves the generation results.
- data_type: The data representation used to fine-tune
- task: The chosen task, including qa, keyword and summary
- qa_type: The evidence type used to represent context, including ground_truth, recall and bias

As the baselines, run the command:
```bash
bash ./script/test_baseline.sh
```

If you want to evaluate metrics by the generation results, run the command:
```bash
bash ./evaluation.py -p ./result/llama-qa.txt
```
- p: the generation result file

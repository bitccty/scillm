import copy
import os
import json
from tqdm import tqdm
import ipdb
import random
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: str,
    eos_token_id
) -> Dict:
    input_ids, labels, slens = [], [], []
    for s, t in zip(sources, targets):
        s_tokens = tokenizer.encode(s, add_special_tokens=False)
        t_tokens = tokenizer.encode(t, add_special_tokens=False)
        inpt = s_tokens + t_tokens + [2]
        label = [-100] * len(s_tokens) + t_tokens + [2]
        inpt = inpt[-max_length:]
        label = label[-max_length:]
        slen = max(len(inpt) - len(t_tokens) - 1, 0)
        input_ids.append(torch.LongTensor(inpt))
        labels.append(torch.LongTensor(label))
        slens.append(slen)
    return dict(input_ids=input_ids, labels=labels, slens=slens)


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, **args):
        super(SFTDataset, self).__init__()
        self.args = args
        list_data_dict = json.load(open(args['train_data_path']))
        # list_data_dict = json.load(open(args['data_path']))
        self.tokenizer = args['tokenizer']

        if args["data_type"] == "evidence":
            prompt_input = '### Evidence:\n{evidence}\n\n### Instruction:\n{question}\n\n### Response:\n'
        elif args["data_type"] == "natural":
            prompt_input = '### Context:\n{natural_context}\n\n### Instruction:\n{question}\n\n### Response:\n'
        else:
            prompt_input = '{structure_context}\n\n### Instruction:\n{question}\n\n### Response:\n'
        sources = [prompt_input.format_map(example) for example in tqdm(list_data_dict)]
        targets = [example['answer'] for example in list_data_dict]
        # ipdb.set_trace()
        data_dict = preprocess(sources, targets, self.tokenizer, self.args['max_seq_length'], self.tokenizer.eos_token_id)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.slens = data_dict["slens"]
        print(f'[!] collect {len(self.input_ids)} samples for training')

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], slens=self.slens[i])

    def collate(self, instances):
        input_ids, labels, slens = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "slens"))
        lengths = [len(item) for item in input_ids]
        max_length = max(lengths)
        attention_mask = torch.LongTensor(
           [[1] * length + [0] * (max_length - length) for length in lengths]
        )
        attention_mask = attention_mask.to(torch.bool)
        position_mask = torch.LongTensor(
            [[[1] * max_length] * slen + [[0] * slen + [1] * (max_length-slen)] * (max_length-slen) for slen in slens]
        )
        position_mask = position_mask.to(torch.bool)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_mask=position_mask
        )


class QASPERTestDataset(Dataset):
    
    def __init__(self, **args):
        super(QASPERTestDataset, self).__init__()
        self.args = args
        self.tokenizer = args['tokenizer'] 
        # cached dataset
        with open(self.args['data_path']) as f:
            data = json.load(f)

        if args["data_type"] == "evidence":
            demonstrations = f'## Description: Please refer to the evidence to answer the question.\n\n\n### Evidence:\nTable TABREF19 and TABREF26 report zero-shot results on Europarl and Multi-UN evaluation sets, respectively. We compare our approaches with related approaches of pivoting, multilingual NMT (MNMT) BIBREF19, and cross-lingual transfer without pretraining BIBREF16. The results show that our approaches consistently outperform other approaches across languages and datasets, especially surpass pivoting, which is a strong baseline in the zero-shot scenario that multilingual NMT systems often fail to beat BIBREF19, BIBREF20, BIBREF23. Pivoting translates source to pivot then to target in two steps, causing inefficient translation process. Our approaches use one encoder-decoder model to translate between any zero-shot directions, which is more efficient than pivoting. Regarding the comparison between transfer approaches, our cross-lingual pretraining based transfer outperforms transfer method that does not use pretraining by a large margin.\n### Question:\nwhich multilingual approaches do they compare with?\n### Answer:\nBIBREF19\n\n\n'
            demonstrations += f'### Evidence:For MultiUN corpus, we use four languages: English (En) is set as the pivot language, which has parallel data with other three languages which do not have parallel data between each other. The three languages are Arabic (Ar), Spanish (Es), and Russian (Ru), and mutual translation between themselves constitutes six zero-shot translation direction for evaluation. We use 80K BPE splits as the vocabulary. Note that all sentences are tokenized by the tokenize.perl script, and we lowercase all data to avoid a large vocabulary for the MultiUN corpus.\n### Question:\nwhat language pairs are explored?### Answer:De-En, En-Fr, Fr-En, En-Es, Ro-En, En-De, Ar-En, En-Ru\n\n\n'
            demonstrations += '### Evidence:\n{}\n'
            query = '### Instruction:\n{}\n\n### Response:\n'
        elif args["data_type"] == "natural_context":
            demonstrations = '### Context:\n{}\n\n'
            query = '### Instruction:\n{}\n\n### Response:\n'
        else: # structure_context
            demonstrations = '{}\n\n'
            query = '### Instruction:\n{}\n\n### Response:\n'
        self.data = []
        self.labels = []
        for sample in tqdm(data):
            # prompt = deepcopy(prompt_base)
            prompt_context = demonstrations.format(sample[args["data_type"]])
            prompt_query = query.format(sample["question"])
            tokens_context = self.tokenizer.encode(prompt_context, add_special_tokens=False)
            tokens_query = self.tokenizer.encode(prompt_query, add_special_tokens=False)
            if len(tokens_context) + len(tokens_query) > 4096:
                tokens = tokens_context[:4096-len(tokens_query)] + tokens_query
            else:
                tokens = tokens_context + tokens_query
            self.data.append(tokens)
            tokens_answer = self.tokenizer.encode(sample['answer'][0], add_special_tokens=False)
            self.labels.append(tokens_answer)
        print(f'[!] collect {len(self.data)} multiple choices samples for testing')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return torch.LongTensor(self.data[i]), self.labels[i]

    def collate(self, batch):
        instances = [i for i, j in batch]
        labels = [j for i, j in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            instances,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels
        )

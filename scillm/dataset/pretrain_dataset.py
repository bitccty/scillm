import copy
import linecache
from itertools import chain,islice
from copy import deepcopy
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


def iter_count(file_name):
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)


class PretrainDataset(Dataset):
    def __init__(self, **args):
        super(PretrainDataset, self).__init__()
        self.args = args
        self.tokenizer = args['tokenizer'] 
        # cached dataset
        self.cache_f_reader = open(self.args['data_path'])
        self.cache_tokens = []
        self.instance_num = iter_count(args['data_path'])
        print(f'[!] collect {self.instance_num} samples from {args["data_path"]}')

    def __len__(self):
        # useless thing
        return self.instance_num

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if len(self.cache_tokens) < self.args['max_seq_length']:
            cache = []
            for _ in range(self.args['max_dataset_cache_size']):
                line = self.cache_f_reader.readline().strip()
                if line:
                    cache.append(json.loads(line))
                else:
                    ipdb.set_trace()
                    print(f'[!] read out of the file, reload ...')
                    self.cache_f_reader = open(self.args['data_path'])
                    line = self.cache_f_reader.readline().strip()
                    cache.append(json.loads(line))
            random.shuffle(cache)
            # concatentate
            self.cache_tokens = []
            for item in cache:
                self.cache_tokens += item + [self.tokenizer.eos_token_id]
        tokens = deepcopy(self.cache_tokens[:self.args['max_seq_length']])
        del self.cache_tokens[:self.args['max_seq_length']]
        return torch.LongTensor(tokens)

    def collate(self, instances):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            instances,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(instances, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class PretrainTestDataset(Dataset):
    def __init__(self, **args):
        super(PretrainTestDataset, self).__init__()
        self.args = args
        self.tokenizer = args['tokenizer'] 
        # cached dataset
        self.cache_f_reader = open(self.args['data_path'])
        self.cache = list(islice(self.cache_f_reader, 0, 100))
        self.cache = [json.loads(i) for i in self.cache]
        self.cache_tokens = []
        for item in tqdm(self.cache):
            self.cache_tokens += item + [self.tokenizer.eos_token_id]
        self.tokens = [self.cache_tokens[i:i+self.args['test_max_seq_length']]for i in range(0, len(self.cache_tokens), self.args['test_max_seq_length'])]
        self.tokens = self.tokens[:100]
        print(f'[!] load {len(self.tokens)} samples for testing')

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return torch.LongTensor(self.tokens[i])

    def collate(self, instances):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            instances,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(instances, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class PretrainQASPERTestDataset(Dataset):
    
    def __init__(self, **args):
        super(PretrainQASPERTestDataset, self).__init__()
        self.args = args
        self.tokenizer = args['tokenizer'] 
        # cached dataset
        with open(self.args['data_path']) as f:
            data = json.load(f)

        if args["data_type"] == "evidence":
            demonstrations = f'## Description: Please refer to the evidence to answer the question.\n\n\n### Evidence:\nTable TABREF19 and TABREF26 report zero-shot results on Europarl and Multi-UN evaluation sets, respectively. We compare our approaches with related approaches of pivoting, multilingual NMT (MNMT) BIBREF19, and cross-lingual transfer without pretraining BIBREF16. The results show that our approaches consistently outperform other approaches across languages and datasets, especially surpass pivoting, which is a strong baseline in the zero-shot scenario that multilingual NMT systems often fail to beat BIBREF19, BIBREF20, BIBREF23. Pivoting translates source to pivot then to target in two steps, causing inefficient translation process. Our approaches use one encoder-decoder model to translate between any zero-shot directions, which is more efficient than pivoting. Regarding the comparison between transfer approaches, our cross-lingual pretraining based transfer outperforms transfer method that does not use pretraining by a large margin.\n### Question:\nwhich multilingual approaches do they compare with?\n### Answer:\nBIBREF19\n\n\n'
            demonstrations += f'### Evidence:For MultiUN corpus, we use four languages: English (En) is set as the pivot language, which has parallel data with other three languages which do not have parallel data between each other. The three languages are Arabic (Ar), Spanish (Es), and Russian (Ru), and mutual translation between themselves constitutes six zero-shot translation direction for evaluation. We use 80K BPE splits as the vocabulary. Note that all sentences are tokenized by the tokenize.perl script, and we lowercase all data to avoid a large vocabulary for the MultiUN corpus.\n### Question:\nwhat language pairs are explored?### Answer:De-En, En-Fr, Fr-En, En-Es, Ro-En, En-De, Ar-En, En-Ru\n\n\n'
            demonstrations += '### Evidence:\n{}\n'
            query = '### Question:\n{}\n### Answer:\n'
        elif args["data_type"] == "natural_context":
            demonstrations = '{}\nAbove are evidences. Please answer to the following question based on it.\n'
            query = '{}\n'
        else: # structure_context
            demonstrations = '{}\nAbove are evidences. Please answer to the following question based on it.\n'
            query = '{}\n'
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


if __name__ == "__main__":
    from transformers import LlamaTokenizer
    from tqdm import tqdm
    args = {'max_seq_length': 4096, 'tokenizer': LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf'), 'max_dataset_cache_size': 1, 'data_path': '../data/pretrain/train_ST/base/split_01'}
    dataset = PretrainDataset(**args)
    for i in tqdm(range(50000)):
        dataset[i]


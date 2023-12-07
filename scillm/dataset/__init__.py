from header import *
from .pretrain_dataset import *
from .sft_dataset import *

def load_dataset(args):
    # load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args["model_path"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    args["tokenizer"] = tokenizer
    
    if args['mode'] == 'test':
        dataset_name = args['models'][args['model']]['test_dataset']
    else:
        dataset_name = args['models'][args['model']]['dataset']
    data = globals()[dataset_name](**args)
    if args['mode'] == 'test':
        sampler = torch.utils.data.SequentialSampler(data)
        batch_size = 1
    else:
        sampler = torch.utils.data.DistributedSampler(data)
        batch_size = args['dschf'].config['train_micro_batch_size_per_gpu']
    args["batch_size"] = batch_size
    iter_ = DataLoader(
        data, 
        batch_size=batch_size,
        collate_fn=data.collate,
        sampler=sampler
    )
        
    return data, iter_, sampler


if __name__ == "__main__":
    args = {
        "base_model_name": "llama",
        "model": "scillm",
        "model_path": "/home/cty/pretrained_models/llama_7b_hf",
        # "data_path": "/home/cty/scillm/data/pretrain/train_ST/base/split_00",
        "data_path": "/home/cty/scillm/data/pretrain/test_tokens.txt",
        "local_rank": "0",
        "mode": "train",
        "max_seq_length": 4096,
        "max_dataset_cache_size": 1
    }
    load_dataset(args)
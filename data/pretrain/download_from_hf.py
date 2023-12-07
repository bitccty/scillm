from datasets import load_dataset
import json
import os
from tqdm import tqdm
from transformers import LlamaTokenizer

def download():
    dataset_arxiv = load_dataset('togethercomputer/RedPajama-Data-1T', 'arxiv', split='train', streaming=True)
    dataset_c4 = load_dataset('togethercomputer/RedPajama-Data-1T', 'c4', split='train', streaming=True)
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    print(f'[!] initialize the redpajama dataset over')

    root_path = "./pretrain"
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    f_arxiv = open(os.path.join(root_path, 'redpajama_arxiv.json'), 'w')
    f_c4 = open(os.path.join(root_path, 'redpajama_c4.json'), 'w')
    f_error = open(os.path.join(root_path, 'error.json'), 'w')

    # c4 vs arxiv with 1:19
    max_arxiv_tokens, max_c4_tokens = int(28e9), int(2e9)
    current_arxiv_item, current_c4_item = 0, 0
    total_arxiv_tokens, total_c4_tokens = 0, 0

    pbar_arxiv = tqdm(total=max_arxiv_tokens)
    for item in dataset_arxiv:
        try:
            assert item['red_pajama_subset'] == 'arxiv', item['red_pajama_subset']
            meta = json.loads(item["meta"].replace("\'", "\""))
            if not meta["url"]:
                raise Exception()
            temp = {
                'text': item['text'],
                'language': meta["language"],
                'url': meta["url"],
                'red_pajama_subset': item['red_pajama_subset'],
                'index_in_raw_file': current_arxiv_item
            }
            f_arxiv.write(json.dumps(temp) + '\n')
            f_arxiv.flush()
            tokens = tokenizer.encode(item["text"], add_special_tokens=False)
            total_arxiv_tokens += len(tokens)
            current_arxiv_item += 1
            pbar_arxiv.update(len(tokens))
        except Exception as e:
            f_error.write(json.dumps(item) + '\n')
            f_error.flush()
    
    pbar_c4 = tqdm(total=max_c4_tokens)
    for item in dataset_c4:
        try:
            assert item['red_pajama_subset'] == 'c4', item['red_pajama_subset']
            temp = {
                'text': item["text"],
                'red_pajama_subset': item['red_pajama_subset'],
                'index_in_raw_file': current_c4_item
            }
            f_c4.write(json.dumps(temp) + '\n')
            f_c4.flush()
            tokens = tokenizer.encode(item["text"], add_special_tokens=False)
            total_c4_tokens += len(tokens)
            current_c4_item += 1
            pbar_c4.update(len(tokens))
        except Exception as e:
            f_error.write(json.dumps(item) + '\n')
            f_error.flush()
            
    print(f"Count Arxiv {current_arxiv_item} - CommonCrawl {current_c4_item}")
    print(f"Tokens Arxiv {total_arxiv_tokens} - CommonCrawl {total_c4_tokens}")


def eval():
    arxiv_tokens, c4_tokens = 0, 0
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    f_arxiv = open('./data/pretrain/redpajama_arxiv.json', 'r')
    for line in tqdm(f_arxiv.readlines()):
        item = json.loads(line)
        tokens = tokenizer.encode(item["text"], add_special_tokens=False)
        arxiv_tokens += len(tokens)
        if arxiv_tokens >= 28e:
            xxx = 1
    
    f_c4 = open('./data/pretrain/redpajama_c4.json', 'r')
    for line in tqdm(f_c4.readlines()):
        item = json.loads(line)
        tokens = tokenizer.encode(item["text"], add_special_tokens=False)
        c4_tokens += len(tokens)
    
    print(f"Tokens Arxiv {arxiv_tokens} - CommonCrawl {c4_tokens}")
    

if __name__ == "__main__":
    download()
    # eval()
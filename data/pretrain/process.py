import json
import os
import argparse
import ipdb
from tqdm import tqdm
from loguru import logger
from transformers import LlamaTokenizer
from utils import *

prompt_base = '''You are reading millions of papers and learn to have the ability to understand, summarize and think.
You know that all papers are structured. The title and abstract are an overview of the full text. The content of each small chapter is closely related to the content of its parent chapter.
Next, The Section is represented by special token <<< >>> and <<</ />>>. 

Example:
<<<Title>>> Attention is all you need . <<<Abstract>>> ... <<<1 Introduction>>> ... <<</1 Introduction/>>> <<</Abstract/>>> <<</Title/>>>

Pay attention to understand the hierarchical relationship between chapters, especially the logical relationship between parent and child chapters.

'''
def build_directory():
    path = [
        "./pretrain/train_NL/base",
        "./pretrain/train_NL/base_f",
        "./pretrain/train_ST/base",
        "./pretrain/train_ST/base_p",
        "./pretrain/train_ST/base_f",
        "./pretrain/train_ST/base_pf",
        "./pretrain/train_ST/base_a",
        "./pretrain/train_ST/base_p_a",
    ]
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)

# save path
path_map = {
    "f_nl": "./pretrain/train_NL/base",
    "f_nl_f": "./pretrain/train_NL/base_f",
    "f_st": "./pretrain/train_ST/base",
    "f_st_p": "./pretrain/train_ST/base_p",
    "f_st_f": "./pretrain/train_ST/base_f",
    "f_st_pf": "./pretrain/train_ST/base_pf",
    "f_st_a": "./pretrain/train_ST/base_a",
    "f_st_p_a": "./pretrain/train_ST/base_p_a",
    "f_lt": "./pretrain/train_LT"
}


# arxiv:c4=99:1
def sample_c4(arxiv_tokens: dict, tokenizer: LlamaTokenizer, data_path="./pretrain/redpajama_c4.json", delta=99):
    c4_tokens = {}
    total_tokens = {}
    for key, value in arxiv_tokens.items():
        max_token_count = value / delta
        c4_tokens[key] = 0
        f_r = open(data_path, "r")
        f_w = open(os.path.join(path_map[key], "c4.txt"), "w")
        ft_w = open(os.path.join(path_map[key], "c4_tokens.txt"), "w")
        pbar = tqdm(total=max_token_count, desc=key)
        for line in f_r.readlines():
            data = json.loads(line)
            tokens = tokenizer.encode(data["text"], add_special_tokens=False)
            c4_tokens[key] += len(tokens)
            if c4_tokens[key] > max_token_count:
                break
            else:
                pbar.update(len(tokens))
                f_w.write(json.dumps(data["text"]) + "\n")
                ft_w.write(json.dumps(tokens)+ "\n")
        total_tokens[key] = value + c4_tokens[key]
        f_r.close()
        f_w.close()
        ft_w.close()
    return c4_tokens, total_tokens


def save(f_text, f_token, prompt, tokenizer):
    prompt = specialize(prompt)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    f_text.write(json.dumps(prompt) + "\n")
    f_token.write(json.dumps(tokens) + "\n")
    f_text.flush()
    f_token.flush()
    return len(tokens)


def main(**args):
    logger.add(args["log_path"], level="INFO")
    # clean data
    if args["do_clean"]:
        data_path = clean_data(args["data_path"])
    else:
        data_path = args["data_path"]
    # build directory
    build_directory()
    # load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    # special_tokens = ["<<<", ">>>", "<<</", "/>>>"]
    # tokenizer.add_tokens(special_tokens)
    args["tokenizer"] = tokenizer
    # save
    f_nl = open(os.path.join(path_map["f_nl"], "arxiv.txt"), "w")
    f_nl_f = open(os.path.join(path_map["f_nl_f"], "arxiv.txt"), "w")
    
    f_st = open(os.path.join(path_map["f_st"], "arxiv.txt"), "w")
    f_st_p = open(os.path.join(path_map["f_st_p"], "arxiv.txt"), "w")
    f_st_f = open(os.path.join(path_map["f_st_f"], "arxiv.txt"), "w")
    f_st_pf = open(os.path.join(path_map["f_st_pf"], "arxiv.txt"), "w")
    f_st_a = open(os.path.join(path_map["f_st_a"], "arxiv.txt"), "w")
    f_st_p_a = open(os.path.join(path_map["f_st_p_a"], "arxiv.txt"), "w")
    
    t_f_nl = open(os.path.join(path_map["f_nl"], "arxiv_tokens.txt"), "w")
    t_f_nl_f = open(os.path.join(path_map["f_nl_f"], "arxiv_tokens.txt"), "w")
    
    t_f_st = open(os.path.join(path_map["f_st"], "arxiv_tokens.txt"), "w")
    t_f_st_p = open(os.path.join(path_map["f_st_p"], "arxiv_tokens.txt"), "w")
    t_f_st_f = open(os.path.join(path_map["f_st_f"], "arxiv_tokens.txt"), "w")
    t_f_st_pf = open(os.path.join(path_map["f_st_pf"], "arxiv_tokens.txt"), "w")
    t_f_st_a = open(os.path.join(path_map["f_st_a"], "arxiv_tokens.txt"), "w")
    t_f_st_p_a = open(os.path.join(path_map["f_st_p_a"], "arxiv_tokens.txt"), "w")

    # load data
    f_read = open(data_path, "r")
    total_tokens = {"f_nl": 0, "f_nl_f": 0, "f_st": 0, "f_st_p": 0, "f_st_f": 0, "f_st_pf": 0, "f_st_a": 0, "f_st_p_a": 0}
    for line in tqdm(f_read.readlines()):
        item = json.loads(line)
        # parse body first
        body_nl, body_st, body_st_a = parse_body(item["body"])
        # generate natural language
        title = "The title of the paper is '{}'. ".format(item["title"])
        abs = "The abstract of the paper is '{}'. ".format("\n".join(item["abstract"][0]))
        body = "The content of the paper is '{}'".format(body_nl)
        prompt = title + abs + body
        
        total_tokens["f_nl"] += save(f_nl, t_f_nl, prompt, args["tokenizer"])
        # full_data
        if parse_publication(item["publication"]):
            publication = "The paper is {}. ".format(parse_publication(item["publication"]))
        else:
            publication = ""
        if parse_author(item["author"]):
            author = "The authors of the paper are '{}.'".format(parse_author(item["author"]))
        else:
            author = ""
        if parse_reference(item["reference"]):
            reference = "The references of the paper are ''.".format(parse_reference(item["reference"]))
        else:
            reference = ""
        prompt = title + publication + author + abs + body + reference
        total_tokens["f_nl_f"] += save(f_nl_f, t_f_nl_f, prompt, args["tokenizer"])
        
        # title, abstract, body
        prompt_title = item["title"]
        prompt_abs = "\n".join(item["abstract"][0])
        prompt_body = body_st
        prompt_item = f"<<<Title>>> {prompt_title} <<<Abstract>>> {prompt_abs} {prompt_body} <<</Abstract/>>> <<</Title/>>>"
        total_tokens["f_st"] += save(f_st, t_f_st, prompt_item, args["tokenizer"])
        total_tokens["f_st_p"] += save(f_st_p, t_f_st_p, prompt_base + prompt_item, args["tokenizer"])
        # Data Argumentation
        total_tokens["f_st_a"] += save(f_st_a, t_f_st_a, prompt_item, args["tokenizer"])
        total_tokens["f_st_p_a"] += save(f_st_p_a, t_f_st_p_a, prompt_base + prompt_item, args["tokenizer"])
        for sta in body_st_a:
            prompt_sta = f"<<<Title>>> {prompt_title} <<<Abstract>>> {prompt_abs} {sta} <<</Abstract/>>> <<</Title/>>>"
            total_tokens["f_st_a"] += save(f_st_a, t_f_st_a, prompt_sta, args["tokenizer"])
            total_tokens["f_st_p_a"] += save(f_st_p_a, t_f_st_p_a, prompt_base + prompt_sta, args["tokenizer"])
        # full data
        prompt_full = "<<<Title>>> " + item["title"]
        if parse_publication(item["publication"]):
            prompt_pub = " <<<Publication>>> " + parse_publication(item["publication"]) + " <<</Publication/>>>"
        else:
            prompt_pub = ""
        if parse_author(item["author"]):
            prompt_author = " <<<Author>>> " + parse_author(item["author"]) + " <<</Author/>>>"
        else:
            prompt_author = ""
        if parse_reference(item["reference"]):
            prompt_ref = " <<<Reference>>> " + parse_reference(item["reference"]) + " <<</Reference/>>>"
        else:
            prompt_ref = ""
        prompt_full = f"<<<Title>>> {prompt_title} {prompt_pub} {prompt_author} <<<Abstract>>> {prompt_abs} {prompt_body} <<</Abstract/>>> {prompt_ref} <<</Title/>>>"
        total_tokens["f_st_f"] += save(f_st_f, t_f_st_f, prompt_full, args["tokenizer"])
        total_tokens["f_st_pf"] += save(f_st_pf, t_f_st_pf, prompt_base + prompt_full, args["tokenizer"])
    
    logger.info(total_tokens)
    c4_token_counts, total_token_counts = sample_c4(total_tokens, args["tokenizer"])
    logger.info(c4_token_counts)
    logger.info(total_token_counts)


def parse_body(body):
    # initial
    temp = [{"section": {"index":"", "name": ""}, "p": []}] + body
    # build tree
    Tree = build_tree(temp)
    
    prompt_nl = Tree.pre_traverse()
    prompt_st = repr(Tree).replace("<<< >>>","").replace("<<</ />>>", "")
    prompt_st_a = [i.replace("<<< >>>","").replace("<<</ />>>", "") for i in Tree.deep_traverse()]
    
    return prompt_nl, prompt_st, prompt_st_a
    

def parse_publication(publication):
    prompt = "Published"
    if publication.get("publication"):
        prompt += " by " + publication["publisher"]
    if publication.get("date"):
        prompt += " in " + publication["date"]
    
    if publication == "Published":
        return None
    else:
        return prompt


def parse_author(authors):
    temp = []
    for author in authors:
        if author["name"] and author["email"]:
            temp.append(f" {author['name']}({author['email']})")
    return "".join(temp)


def parse_reference(references):
    temp = []
    for refer in references:
        if refer["index"] and refer["title"]:
            temp.append(f" {refer['index']}. {refer['title']}")
    return "\n".join(temp)


def clean_data(data_path):
    f_read = open(data_path, "r")
    fw = open("./pretrain/clean_data.txt", "w")
    title_set = []
    raw_count, clean_count = 0, 0
    for line in f_read.readlines():
        raw_count += 1
        item = json.loads(line)
        if not (item["title"] and item["abstract"] and item["abstract"][0] and item["body"]): # remove empty on title or abstract or body
            continue
        elif item["title"] in title_set: # dumplicate
            continue
        else:
            clean_count += 1
            fw.write(json.dumps(item) + "\n")
            fw.flush()
    
    logger.info(f"Raw size: {raw_count} - Clean size: {clean_count}")
    return "./pretrain/clean_data.txt"


def save_latex(file="./pretrain/redpajama_arxiv.json", start_pos=0, length=45418):
    f = open(file, "r")
    total_tokens = {"f_lt": 0}
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    if not os.path.exists("./pretrain/train_LT"):
        os.makedirs("./pretrain/train_LT")
    fw = open("./pretrain/train_LT/arxiv.txt", "w")
    ftw = open("./pretrain/train_LT/arxiv_tokens.txt", "w")
    for line in f.readlines()[start_pos:start_pos+length]:
        data = json.loads(line)["text"]
        total_tokens["f_lt"] += save(fw, ftw, data, tokenizer)
    logger.info(total_tokens)
    c4_token_counts, total_token_counts = sample_c4(total_tokens, tokenizer)
    logger.info(c4_token_counts)
    logger.info(total_token_counts)


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--do_clean', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)


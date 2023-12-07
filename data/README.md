# Data Processing

## 1. Process Scientific Pre-training Dataset

```bash
cd ./pretrain
```

### 1.1 Download Arxiv Urls from Huggingface

```bash
python download_from_hf.py
```

save to `./pretrain/redpajama_arxiv.json` and `./pretrain/redpajama_c4.json`

### 1.2 Download PDF file and Extract to Json Format 

First install the [Grobid](https://github.com/kermitt2/grobid). You can also use the docker install, see [detail](https://hub.docker.com/r/lfoppiano/grobid).

```bash
docker pull lfoppiano/grobid
```

Second, run the Grobid on default port 8070

```bash
docker run -d -t --rm -p 8070:8070 lfoppiano/grobid:0.7.1
```

Finally, start data process.

```bash
python extract.py --batch_size=10000 --start_pos=0
```

- batch_size: the cache size of pdf and xml directories
- start_pos: the index to start process in arxiv urls, supporting **Resume Breakpoint**

save to `./pretrain/output` 

### 1.3 Generate Structured Scientific Dataset

```bash
python process.py --data_path=./pretrain/output/arxiv_structure_0.json --log_path=./pretrain/rest/generate.log --do_clean
```
- data_path: merge all files in `./pretrain/output` to `arxiv_structure_0.json`
- log_path: the log file
- do_clean: data dumplication
  
save to `./train_LT`, `./train_NL` and `./train_ST`
- train_LT: the LaTeX format
- train_NL: the plain text format
- train_ST: the USSC format
  - base_a: utilize data augmentation


### 1.4 combine, shuffle and split

```bash
./post_process.sh
```

### 1.5 Statistics

| Source | Item Count | Token Count |
| ---------------- | ----------------- | --------------------------------- |
| latex | 61,961 | 0.89 Billion |
| plain-text | 47,183 | 0.21 Billion |
| USSC | 50,832 | 0.40 Billion |
| USSC_da | 493,487 | 0.99 Billion  |


## 2. Process Multi-task Fine-tuning Dataset

```bash
cd ./sft
```

### 2.1 Download and Prepocess Datasets

- kp20k: `https://huggingface.co/datasets/midas/kp20k`
- arxiv-summary: `https://huggingface.co/datasets/ccdv/arxiv-summarization`
- pubmed-summary: `https://huggingface.co/datasets/ccdv/pubmed-summarization`
- QASPER: `https://huggingface.co/datasets/allenai/qasper`
- SciMRC: `https://huggingface.co/datasets/DataHammer/scimrc`

**prepocess** 

```bash
cd kp20k
python collect.py
```

which is the same as others.

### 2.2 Combine the train set

```bash
python combine.py
```
save to `train.json`

### 2.3 Retrieve the relevant evidences

SciDPR question encoder: [https://huggingface.co/DataHammer/scidpr-question-encoder](https://huggingface.co/DataHammer/scidpr-question-encoder)
SciDPR context encoder: [https://huggingface.co/DataHammer/scidpr-ctx-encoder](https://huggingface.co/DataHammer/scidpr-ctx-encoder)

```bash
python scidpr.py
```

save to `processed_qasper_test_set.json` and `processed_scimrc_test_set.json`





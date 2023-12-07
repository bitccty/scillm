# An Unsupervised Structural Data Augmentation Method on Scientific Papers for Retrieval-Augmented Large Language Model


<span id='introduction'/>

### 1. Introduction

Scillm is a novel start-of-art LLM on Scientific domain, which utilizes the Unsupervised Structural Data Augmentation Method. It not only can capture the structure information of context but aligns with natural language well. 

<span id='environment'/>

### 2. Environment Installation

To install the required environment, please run
```
pip install -r requirements.txt
```

Then install the Pytorch package with the correct cuda version, for example
```
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 3. Data Process

`cd ./data` and see the [README](./data/README.md)

### 4. Train Model

`cd ./scillm` and see the [README](./scillm/README.md)

### 5. Evaluation

`cd ./scillm` and see the [README](./scillm/README.md)
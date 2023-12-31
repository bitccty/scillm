import json
import ipdb
import random

random.seed(0)

with open('qasper/qasper_train_sft.json') as f:
    data = json.load(f)
    print(f'[!] samples {len(data)}')
    number = len(data)

with open('kp20k/train.json') as f:
    data.extend(json.load(f))
    print(f'[!] samples {len(data)-number}')
    number = len(data)

with open('arxiv-summary/train.json') as f:
    data.extend(json.load(f))
    print(f'[!] samples {len(data)-number}')
    number = len(data)


random.shuffle(data)

with open('train.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
    print(f'[!] collect {len(data)} samples')
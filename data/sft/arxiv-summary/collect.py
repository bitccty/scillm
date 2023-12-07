import json
import random
import ipdb

f = open("train.txt", "r")
dataset = []
for line in f.readlines():
    item = json.loads(line)
    evidence = ''.join(item['article_text'])
    question = "Summarize the abstract on the paper full context."
    answer = ''.join(item["abstract_text"]).replace('<S>', '').replace('</S>','')
    natural_context = f"The full context of this paper is '{evidence}'"
    structure_context = ""
    for i in range(len(item['section_names'])):
        structure_context += "<<<{0}>>> {1} <<<{0}>>> ".format(item['section_names'][i], ''.join(item['sections'][i]))
    dataset.append({
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "natural_context": natural_context,
        "structure_context": structure_context,
        "yes_no": False,
        "paper_id": item['article_id']
    })

random.shuffle(dataset)
dataset = dataset[:1000]

with open("train.json", "w") as fw:
    json.dump(dataset, fw, indent=4)


f = open("test.txt", "r")
dataset = []
for line in f.readlines():
    item = json.loads(line)
    evidence = ''.join(item['article_text'])
    question = "Summarize the abstract on the paper full context."
    answer = ''.join(item["abstract_text"]).replace('<S>', '').replace('</S>','')
    natural_context = f"The full context of this paper is '{evidence}'"
    structure_context = ""
    for i in range(len(item['section_names'])):
        structure_context += "<<<{0}>>> {1} <<<{0}>>> ".format(item['section_names'][i], ''.join(item['sections'][i]))
    dataset.append({
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "natural_context": natural_context,
        "structure_context": structure_context,
        "yes_no": False,
        "paper_id": item['article_id']
    })

with open("../processed_arxiv_test_set.json", "w") as fw:
    json.dump(dataset[:1000], fw, indent=4)
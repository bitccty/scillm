import json
import random
import ipdb

f = open("kp20k_training.json", "r")
dataset = []
for line in f.readlines():
    item = json.loads(line)
    evidence = item["title"] + "\n" + item["abstract"]
    question = "Extract the keywords on the context."
    answer = item["keyword"]
    natural_context = f"The title of this paper is '{item['title']}'\nThe abstract of this paper is '{item['abstract']}'"
    structure_context = f"<<<Title>>> {item['title']} <<<Abstract>>> {item['abstract']} <<</Abstract/>>> <<</Title/>>>"
    dataset.append({
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "natural_context": natural_context,
        "structure_context": structure_context,
        "yes_no": False,
        "paper_id": 0
    })

random.shuffle(dataset)
dataset = dataset[:1000]

with open("train.json", "w") as fw:
    json.dump(dataset, fw, indent=4)


f = open("kp20k_testing.json", "r")
dataset = []
for line in f.readlines():
    item = json.loads(line)
    evidence = item["title"] + "\n" + item["abstract"]
    question = "Extract the keywords on the context."
    answer = item["keyword"]
    natural_context = f"The title of this paper is '{item['title']}'\nThe abstract of this paper is '{item['abstract']}'"
    structure_context = f"<<<Title>>> {item['title']} <<<Abstract>>> {item['abstract']} <<</Abstract/>>> <<</Title/>>>"
    dataset.append({
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "natural_context": natural_context,
        "structure_context": structure_context,
        "yes_no": False,
        "paper_id": 0
    })
with open("../processed_kp20k_test_set.json", "w") as fw:
    json.dump(dataset[:1000], fw, indent=4)
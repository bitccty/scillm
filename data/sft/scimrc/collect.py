import json
from tqdm import tqdm
import ipdb

def remove_duplicate(d):
    question_set = set()
    dataset = {}
    counter = 0
    for sample in d:
        if sample['question'] not in question_set:
            question_set.add(sample['question'])
            sample['answer'] = [sample['answer']]
            dataset[sample['question']] = sample
        else:
            dataset[sample['question']]['answer'].append(sample['answer'])
            counter += 1
    print(f'[!] remove {counter} samples and save {len(dataset)} samples')
    dataset = [dataset[key] for key in dataset]
    return dataset

def generate_context(paper, evidences):
    natural_context = ""
    structure_context = ""
    structure_end = ""
    if "title" in paper and paper["title"]:
        natural_context += "The title of this paper is '{}'\n\n".format(paper["title"])
        structure_context += "<<<Title>>> {} ".format(paper["title"])
        structure_end = " <<</Title/>>>" + structure_end
    if "abstract" in paper and paper["abstract"]:
        natural_context += "The abstract of this paper is '{}'\n\n".format(paper["abstract"])
        structure_context += "<<<Abstract>>> {} ".format(paper["abstract"])
        structure_end = " <<</Abstract/>>>" + structure_end
    if "full_text" in paper and paper["full_text"]:
        natural_context += "The main full text of this paper is as followed:\n"
        for section in paper["full_text"]:
            pp = []
            for pa in section["paragraphs"]:
                if pa in evidences:
                    natural_context += "{}: {}\n".format(section["section_name"], pa)
                    pp.append(pa)
            if pp:
                structure_context += "<<<{0}>>> {1} <<</{0}/>>>".format(section["section_name"], "\n".join(pp))
    structure_context += structure_end
    return natural_context, structure_context

if __name__ == "__main__":
    yes_no_datasets, datasets = [], []
    for mode in ['train', 'dev', 'test']:
        with open(f'smrc_{mode}.jsonl') as f:
            data = [json.loads(line) for line in f.readlines()]
        # ipdb.set_trace()
        yes_no_dataset, dataset = [], []
        for index, paper in tqdm(enumerate(data)):
            for qa in paper['qas']:
                question = qa['question']
                for a in qa['answers']:
                    a = a['answer']
                    if a['unanswerable'] is False:
                        evidence = a['evidence'] + a['highlighted_evidence']
                        natural_context, structure_context = generate_context(paper, evidence)
                        if a['free_form_answer']:
                            answer = a['free_form_answer']
                            dataset.append({
                                'question': question,
                                'answer': answer,
                                'evidence': '\n'.join(evidence),
                                "natural_context": natural_context,
                                "structure_context": structure_context,
                                'yes_no': False,
                                'paper_id': index
                            })
                        elif a['extractive_spans']:
                            answer = ''.join([f'* {i}\n' for idx, i in enumerate(a['extractive_spans'])])
                            answer = 'The answers are shown as follows:\n' + answer
                            dataset.append({
                                'question': question,
                                'answer': answer,
                                'evidence': '\n'.join(evidence),
                                "natural_context": natural_context,
                                "structure_context": structure_context,
                                'yes_no': False,
                                'paper_id': index
                            })
                        else:
                            if a['yes_no'] is not None and evidence:
                                if a['yes_no'] is True:
                                    answer = 1
                                else:
                                    answer = 0
                                yes_no_dataset.append({
                                    'question': question,
                                    'answer': answer,
                                    'evidence': '\n'.join(evidence),
                                    "natural_context": natural_context,
                                    "structure_context": structure_context,
                                    'yes_no': True,
                                    'paper_id': index
                                })
                                dataset.append({
                                    'question': question,
                                    'answer': 'Yes.' if answer else 'No.',
                                    'evidence': '\n'.join(evidence),
                                    "natural_context": natural_context,
                                    "structure_context": structure_context,
                                    'yes_no': True,
                                    'paper_id': index
                                })
                            else:
                                # raise Exception(f'[!] something wrong')
                                pass

        print(f'[!] collect {len(dataset)} samples')
        datasets.append(dataset)
        yes_no_datasets.append(yes_no_dataset)
    # train_dataset = datasets[0] + datasets[1]
    train_dataset = datasets[0]
    test_dataset = datasets[2]
    json.dump(train_dataset, open(f'scimrc_train_sft.json', 'w'), indent=4, ensure_ascii=False)
    test_dataset = remove_duplicate(test_dataset)
    json.dump(test_dataset, open(f'scimrc_test_sft.json', 'w'), indent=4, ensure_ascii=False)
    json.dump(datasets[1], open(f'scimrc_dev_sft.json', 'w'), indent=4, ensure_ascii=False)

    train_dataset = yes_no_datasets[0] + yes_no_datasets[1]
    test_dataset = yes_no_datasets[2]
    json.dump(train_dataset, open(f'scimrc_yes_no_train_sft.json', 'w'), indent=4, ensure_ascii=False)
    json.dump(test_dataset, open(f'scimrc_yes_no_test_sft.json', 'w'), indent=4, ensure_ascii=False)
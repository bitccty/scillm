from header import *

### QASPER Answer F1
def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " the ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1_score(prediction, ground_truth):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def evaluate_answer_f1(path, yes_or_no):
    with open(path) as f:
        results = [json.loads(line) for line in f.readlines()]
        # results = [item for idx, item in enumerate(results) if idx not in yes_or_no]

    max_f1, max_precision, max_recall = [], [], []
    for sample in results:
        pause_f1, pause_precision, pause_recall = [], [], []
        if type(sample["answer"]) is str:
            sample["answer"] = [sample["answer"]]
        for reference in sample['answer']:
            p, r, f1 = token_f1_score(sample['generation'].strip(), reference)
            pause_recall.append(r)
            pause_f1.append(f1)
            pause_precision.append(p)
        max_f1.append(max(pause_f1))
        max_precision.append(max(pause_precision))
        max_recall.append(max(pause_recall))
    print(f'[!] answer F1: {round(np.mean(max_f1), 4)}')
    print(f'[!] answer Precision: {round(np.mean(max_precision), 4)}')
    print(f'[!] answer Recall: {round(np.mean(max_recall), 4)}')
    return round(np.mean(max_f1), 4), round(np.mean(max_precision), 4), round(np.mean(max_recall), 4)

def evaluate_rouge(path, rouge, yes_or_no):
    with open(path) as f:
        results = [json.loads(line) for line in f.readlines()]
        # results = [item for idx, item in enumerate(results) if idx not in yes_or_no]
    scores = []
    r = rouge.compute(
        predictions=[sample['generation'].strip() for sample in results], 
        references=[sample['answer'] if type(sample["answer"]) is str else sample['answer'][0] for sample in results],
    )
    print(r)
    return r


def evaluate_bertscore(path, bertscore, yes_or_no):
    with open(path) as f:
        results = [json.loads(line) for line in f.readlines()]
        # results = [item for idx, item in enumerate(results) if idx not in yes_or_no]
    scores = []
    r = bertscore.compute(
        predictions=[sample['generation'].strip() for sample in results], 
        references=[sample['answer'] if type(sample["answer"]) is str else sample['answer'][0] for sample in results],
        lang='en'
    )
    r = round(np.mean(r['f1']), 4)
    print('BERTScore:', r)
    return r

def evaluate_bleu(path, yes_or_no):
    with open(path) as f:
        results = [json.loads(line) for line in f.readlines()]
        # results = [item for idx, item in enumerate(results) if idx not in yes_or_no]
    scores = []
    for sample in tqdm(results):
        if type(sample["answer"]) is str:
            sample["answer"] = [sample["answer"]]
        reference = [i.split() for i in sample['answer']]
        generation = sample['generation'].strip().split()
        s = sentence_bleu(reference, generation)
        scores.append(s)
    print('BLEU:', round(np.mean(scores), 4))
    return round(np.mean(scores), 4)


def evaluate_mauve(path):
    with open(path) as f:
        results = [json.loads(line) for line in f.readlines()]
    scores = []
    ipdb.set_trace()
    answers = [sample["answer"] if type(sample["answer"]) is str else sample["answer"][0] for sample in tqdm(results)]
    generation = [sample["generation"] for sample in tqdm(results)]
    out = mauve.compute_mauve(p_text=answers, q_text=generation, device_id=0, max_text_length=128, verbose=False, batch_size=64)
    print('MAUVE:', round(out.mauve, 4))
    return round(np.mean(out.mauve), 4)


def evaluate_qa(file_path):
    with open(file_path) as f:
        yes_or_no = []
        for idx, line in enumerate(f.readlines()):
            if json.loads(line)['yes_no']:
                yes_or_no.append(idx)
        yes_or_no = set(yes_or_no)
    
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')
    print(f'[!] load rouge and bertscore over')
    f1, p, r = evaluate_answer_f1(file_path, yes_or_no)
    print("f1:", f1)
    blue = evaluate_bleu(file_path, yes_or_no)
    rouge = evaluate_rouge(file_path, rouge, yes_or_no)
    bertscore = evaluate_bertscore(file_path, bertscore, yes_or_no)
    # mauves = evaluate_mauve(file_path)
    result = {
        "answer_p": p,
        "answer_r": r,
        "answer_f1": f1,
        "blue": blue,
        "bertscore": bertscore,
        # "mauve": mauves
    }
    result.update(rouge)
    fs = file_path.split("/")
    fs[-1] = "result_" + fs[-1]
    save_path = "/".join(fs)
    print(f"[!]save to {save_path}")
    with open(save_path, "w") as fw:
        json.dump(result, fw, indent=4)


def evaluate_keyword(file_path, split_word=";"):
    def normalize_answer(s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " the ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def remove_plural(text):
            if len(text)>0 and text[-1]=="s":
                return text[:-1]
            return text

        def lower(text):
            return text.lower()

        return remove_plural(white_space_fix(remove_articles(remove_punc(lower(s)))))
    
    with open(file_path) as f:
        results = [json.loads(line) for line in f.readlines()]
    max_f1, max_precision, max_recall = [], [], []
    bertscore = evaluate.load('bertscore')
    for sample in results:
        prediction_tokens = [normalize_answer(i) for i in list(set(sample["generation"].split(split_word)))]
        ground_truth_tokens = [normalize_answer(i) for i in sample["answer"].split(";")]
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            precision=recall=f1=0
        else:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
        max_precision.append(precision)
        max_recall.append(recall)
        max_f1.append(f1)
    f1,p,r = round(np.mean(max_f1), 4), round(np.mean(max_precision), 4), round(np.mean(max_recall), 4) 
    print(f"[!]Precies {p} - Recall {r} - F1 {f1}")
    bertscore = evaluate_bertscore(file_path, bertscore, set())
    result = {
        "answer_p": p,
        "answer_r": r,
        "answer_f1": f1,
        "bert_score": bertscore
    }
    fs = file_path.split("/")
    fs[-1] = "result_" + fs[-1]
    save_path = "/".join(fs)
    print(f"[!]save to {save_path}")
    with open(save_path, "w") as fw:
        json.dump(result, fw, indent=4)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('-p', type=str, default="./result/llama-qa.txt")
    args = parser.parse_args()
    file_path = args.p
    if "keyword" in file_path:
        evaluate_qa(file_path)
        # evaluate_keyword(file_path)
    else:
        evaluate_qa(file_path)

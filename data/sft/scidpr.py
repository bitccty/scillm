import torch
import json
import argparse
import ipdb
from tqdm import tqdm
from itertools import chain
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    DPRQuestionEncoder,
    DPRContextEncoder,
    DPRQuestionEncoderTokenizer, 
    DPRContextEncoderTokenizer
)

class SciDPR:
    def __init__(self, **args):
        super(SciDPR, self).__init__()
        self.args = args

        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.args['question_model'])
        self.answer_tokenizer = DPRContextEncoderTokenizer.from_pretrained(self.args['answer_model'])
        self.question_encoder = DPRQuestionEncoder.from_pretrained(self.args['question_model'])
        self.answer_encoder = DPRContextEncoder.from_pretrained(self.args['answer_model'])

        if torch.cuda.is_available():
            self.question_encoder.cuda()
            self.answer_encoder.cuda()


    @torch.no_grad()
    def get_embedding(self, texts, question=True, inner_bsz=128):
        self.question_encoder.eval()
        self.answer_encoder.eval()

        max_len = self.args['q_max_len'] if question else self.args['a_max_len']
        pad_token_id = self.question_tokenizer.pad_token_id if question else self.answer_tokenizer.pad_token_id

        ids = []
        for text in texts:
            if question:
                ids_ = [self.question_tokenizer.cls_token_id] + self.question_tokenizer.encode(text, add_special_tokens=False)[:max_len-2] + [self.question_tokenizer.sep_token_id]
            else:
                ids_ = [self.answer_tokenizer.cls_token_id] + self.answer_tokenizer.encode(text, add_special_tokens=False)[:max_len-2] + [self.answer_tokenizer.sep_token_id]
            ids.append(ids_)

        reps = []
        for i in tqdm(range(0, len(ids), inner_bsz)):
            sub_ids = [torch.LongTensor(i) for i in ids[i:i+inner_bsz]]
            sub_ids = pad_sequence(sub_ids, batch_first=True, padding_value=pad_token_id)
            mask = generate_mask(sub_ids, pad_token_idx=pad_token_id)
            sub_ids, mask = sub_ids.cuda(), mask.cuda()
            if question:
                rep = self.question_encoder(input_ids=sub_ids, attention_mask=mask).pooler_output
            else:
                rep = self.answer_encoder(input_ids=sub_ids, attention_mask=mask).pooler_output
            reps.append(rep)
        reps = torch.cat(reps)    # [B, E]
        return reps.cpu().numpy()
    
def generate_mask(ids, pad_token_idx=0):
    mask = torch.ones_like(ids)
    mask[ids == pad_token_idx] == 0
    return mask

def generate_context(paper, evidences):
    natural_context = ""
    structure_context = ""
    structure_end = ""
    raw_structure_context = ""
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
                raw_structure_context += "<<<{0}>>> {1} <<</{0}/>>>".format(section["section_name"], "\n".join(pp))
    structure_context += structure_end
    return natural_context, structure_context, raw_structure_context

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper_data_file', default='./qasper/qasper-test-v0.3.json', type=str)
    parser.add_argument('--data_file', default='./qasper/qasper_test_sft.json', type=str)
    parser.add_argument('--output_file', default='processed_qasper_test_set.json', type=str)
    # parser.add_argument('--paper_data_file', default='./scimrc/smrc_test.jsonl', type=str)
    # parser.add_argument('--data_file', default='./scimrc/scimrc_test_sft.json', type=str)
    # parser.add_argument('--output_file', default='processed_scimrc_test_set.json', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    # load args
    args = parser_args()
    
    with open(args.paper_data_file) as f:
        if "qasper" in args.paper_data_file:
            paper_data = json.load(f)
        else: # scimrc
            paper_data = [json.loads(line) for line in f.readlines()]
    with open(args.data_file) as f:
        data = json.load(f)
    model_args = {
        'question_model': '/home/cty/scidpr/question_encoder',
        'answer_model': '/home/cty/scidpr/answer_encoder',
        'q_max_len': 128,
        'a_max_len': 256,
        'mode': 'test',
        'topk': 2
    }
    model = SciDPR(**model_args)
    print(f'[!] ========== load the model over ==========')

    # begin to process
    results = []
    for sample in tqdm(data):
        # ipdb.set_trace()
        paper = paper_data[sample['paper_id']]
        # process all the sentences in paper
        sentences = [para['paragraphs'] for para in paper['full_text']]
        sentences = list(chain(*sentences))
        embeddings = model.get_embedding(sentences, question=False, inner_bsz=64)
        query_embedding = model.get_embedding([sample['question']])
        embeddings = torch.from_numpy(embeddings)
        query_embedding = torch.from_numpy(query_embedding)
        matrix = torch.matmul(query_embedding, embeddings.T)    # [1, N]
        index = matrix.topk(model_args['topk'], dim=-1)[1].tolist()[0]
        selected_sentences = [sentences[i] for i in index]

        sample['recall_evidence'] = selected_sentences
        sample['recall_natural_context'], sample['recall_structure_context'], sample['recall_raw_structure_context'] = generate_context(paper, selected_sentences)
        
        bias_sentences = [sentence for sentence in selected_sentences if not sentence in sample["evidence"]]
        sample['bias_sentences'] = bias_sentences
        sample['bias_natural_context'], sample['bias_structure_context'], sample['bias_raw_structure_context'] = generate_context(paper, bias_sentences)
        
        results.append(sample)

    with open(args.output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
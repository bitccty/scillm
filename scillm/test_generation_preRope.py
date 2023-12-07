from header import *
from config import *
from model import *
from dataset import *
from evaluation import evaluate_qa, evaluate_keyword

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', default='scillm-sft', type=str)
    parser.add_argument('--model_path', default='decapoda-research/llama-7b-hf', type=str)
    parser.add_argument('--data_path', default='../data/sft/processed_qasper_test_set.json', type=str)
    parser.add_argument('--delta_model_path', default='ckpt/scillm/pytorch_model.bin', type=str)
    parser.add_argument('--result_path', default='decapoda-research/llama-7b-hf', type=str)
    parser.add_argument('--data_type', default='structure', type=str)
    parser.add_argument('--preRope', action="store_true")
    parser.add_argument('--task', default='qa', type=str)
    parser.add_argument('--qa_type', default='ground_truth', type=str)
    return parser.parse_args()


def main(args):
    args.update({
        'lora_r': 64,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'mode': 'test',
    })
    args.update(load_config(args))
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args['model_path'],
        load_in_4bit=True,
        max_memory={i: '24576MB' for i in range(torch.cuda.device_count())},
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    )
    tokenizer = LlamaTokenizer.from_pretrained(args['model_path'])

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=args['lora_r'],
        lora_alpha=args['lora_alpha'],
        lora_dropout=args['lora_dropout'],
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
    )
    
    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, args['delta_model_path'])
    ppl_criterion = nn.CrossEntropyLoss(reduction='none')
    print(f'[!] load model and tokenizer over')

    # load test dataset
    with open(args['data_path']) as f:
        data = json.load(f)

    with torch.no_grad():
        results = []
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2], encounters=1)])
        f = open(args['result_path'], 'w')
        for item in tqdm(data):
            q = item['question']
            if args["task"] != "qa" or args["qa_type"] == "ground_truth":
                evidence = item['evidence']
                natural_evidence = item["natural_context"]
                structure_context = item["structure_context"]
            elif args["qa_type"] == "recall":
                evidence = "\n".join(item["recall_evidence"])
                natural_evidence = item["recall_natural_context"]
                structure_context = item["recall_structure_context"]
            elif args["qa_type"] == "bias":
                evidence = "\n".join(item["bias_sentences"])
                natural_evidence = item["bias_natural_context"]
                structure_context = item["bias_structure_context"]
            else:
                raise Exception(f'[!] Unknown qa_type: {args["qa_type"]}')
            
            if args["data_type"] == "evidence":
                context = f'### Evidence:\n{evidence}\n\n'
                prompt = f'### Instruction:\n{q}\n\n### Response:\n'
            elif args["data_type"] == "natural":
                context = f'### Context:\n{natural_evidence}\n\n'
                prompt = f'### Instruction:\n{q}\n\n### Response:\n'
            elif args["data_type"] == "structure":
                context = f'{structure_context}\n\n'
                prompt = f'### Instruction:\n{q}\n\n### Response:\n'
            else:
                raise Exception(f'[!] Unknown data_type: {args["data_type"]}')
            # ipdb.set_trace()
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            length = args['test_max_seq_length'] - len(tokens)
            tokens = tokenizer.encode(context, add_special_tokens=False)[-length:] + tokens
            
            tokens = torch.LongTensor(tokens).cuda()
            length = len(tokens)
            tokens = tokens.unsqueeze(0)

            # greedy search
            # outputs = model.generate(
            #     input_ids=tokens,
            #     max_new_tokens=128,
            #     return_dict_in_generate=True,
            #     stopping_criteria=stopping_criteria,
            # )
            outputs = generate(
                model,
                input_ids=tokens.cuda(),
                max_new_tokens=256,
                return_dict_in_generate=True,
                stopping_criteria=stopping_criteria,
                preRope=args["preRope"]
            )
            generation = tokenizer.decode(outputs['sequences'][0, length:], skip_special_tokens=True)
            result_item = deepcopy(item)
            result_item['generation'] = generation
            f.write(json.dumps(result_item) + '\n')
            f.flush()
        
        if args["task"] == "qa" or args["task"] == "summary":
            evaluate_qa(args['result_path'])
        elif args["task"] == "keyword":
            evaluate_keyword(args["result_path"])
        else:
            raise Exception(f'[!] Unknown task: {args["task"]}')
        
def generate(model, input_ids, max_new_tokens, return_dict_in_generate, stopping_criteria, preRope=False):
    new_tokens = 0
    slen = input_ids.shape[1]
    with torch.no_grad():
        while((new_tokens < max_new_tokens) and not stopping_criteria(input_ids, None)):
            max_length = input_ids.shape[1]
            position_mask = torch.LongTensor(
                [[[1] * max_length] * slen + [[0] * slen + [1] * (max_length-slen)] * (max_length-slen)]
            ).to(input_ids.device)
            if preRope:
                output = model.model(
                    input_ids=input_ids, 
                    position_mask=position_mask,
                )
            else:
                output = model.model(
                    input_ids=input_ids,
                )
            token = output.logits[:,-1,:].argmax(dim=-1).unsqueeze(0)
            input_ids = torch.cat((input_ids, token), dim=-1)
            new_tokens += 1
    
    if return_dict_in_generate:
        return dict(sequences=input_ids)
    else:
        return input_ids


if __name__ == "__main__":
    args = vars(parser_args())
    main(args)
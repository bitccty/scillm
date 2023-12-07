from header import *
from config import *
from model import *
from dataset import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', default='scillm', type=str)
    parser.add_argument('--model_path', default='decapoda-research/llama-7b-hf', type=str)
    parser.add_argument('--data_path', default='../data/pretrain/train_ST/split_00', type=str)
    parser.add_argument('--delta_model_path', default='./ckpt/scillm/pytorch_model.bin', type=str)
    parser.add_argument('--result_path', default='./result/scillm_llama_natural/', type=str)
    parser.add_argument('--rewrite', action="store_true")
    return parser.parse_args()


def main(args):
    args.update({
        'lora_r': 72,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'mode': 'test'
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
    
    # build directory
    if not os.path.exists(args["result_path"]):
        os.makedirs(args["result_path"])
    result_file = os.path.join(args["result_path"], "test_ppl.txt")
    if args["rewrite"]:
        fw = open(result_file, "w")
    else:
        fw = open(result_file, "a")
    
    # load test dataset
    with torch.no_grad():
        args['mode'] = 'test'
        _, test_iter, _ = load_dataset(args)
        losses = []
        for batch in tqdm(test_iter):
            outputs = model(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
            )
            logits = outputs.logits[:, :-1, :]
            loss = ppl_criterion(logits.reshape(-1, logits.size(-1)), batch['labels'].cuda()[:, 1:].reshape(-1)).tolist()
            losses.extend(loss)
        ppl = np.exp(np.mean(losses))
        print(f'[!] ppl: {round(ppl, 4)}')
        fw.write(f'model: {args["delta_model_path"]} - data: {args["data_path"]} - ppl: {round(ppl, 4)}\n')


if __name__ == "__main__":
    args = vars(parser_args())
    main(args)
from header import *
from .utils import *

class scillm(nn.Module):
    def __init__(self, **args):
        super(scillm, self).__init__()
        self.args = args
        # vocab_size = args["vocab_size"]
        # self.config = LlamaConfig.from_pretrained(args['model_path'])
        # self.config.vocab_size = vocab_size

        # TODO: replace_llama_attn_with_flash_attn()
        # model loading
        self.model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args['model_path'],
            # config=self.config,
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

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False,
            r=self.args['lora_r'], 
            lora_alpha=self.args['lora_alpha'], 
            lora_dropout=self.args['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
        )

        # model initial
        # state_dict = self.model.state_dict()
        # ipdb.set_trace()
        # trainable_rows = nn.Parameter(torch.zeros(4, 4096), requires_grad=True).cuda()
        # self.model.model.embed_tokens = torch.cat((self.model.model.embed_tokens.weight, trainable_rows), dim=0)
        
        # new_tokens_num = vocab_size - state_dict["model.embed_tokens.weight"].shape[0]
        # word_embeddings = state_dict.pop("model.embed_tokens.weight")
        # state_dict["model.embed_tokens.weight"] = F.pad(word_embeddings, (0, 0, 0, new_tokens_num))
        # output_embeddings = state_dict.pop("lm_head.weight")
        # state_dict["lm_head.weight"] = F.pad(output_embeddings, (0, 0, 0, new_tokens_num))
        # self.model.model.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size, self.config.pad_token_id)
        # self.model.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # self.model.load_state_dict(state_dict)

        # peft preparation
        self.model = prepare_model_for_kbit_training(self.model)
        
        self.model = get_peft_model(self.model, peft_config)
        # add training parameters
        # self.model.base_model.model.model.embed_tokens.weight[-new_tokens_num:].requires_grad = True
        # self.model.base_model.model.lm_head.weight[-new_tokens_num:].requires_grad = True
        self.model.print_trainable_parameters()
        self.ppl_criterion = nn.CrossEntropyLoss(reduction=None)
    
    @torch.no_grad()
    def calculate_ppl(self, inputs):
        outputs = self.model(
            input_ids=inputs['input_ids'].cuda(),
            attention_mask=inputs['attention_mask'].cuda(),
        )
        logits = outputs.logits[:, :-1, :]
        loss = self.ppl_criterion(logits.reshape(-1, logits.size(-1)), inputs['labels'].cuda()[:, 1:].reshape(-1))
        return loss.tolist()

    def forward(self, inputs):
        outputs = self.model(
            input_ids=inputs['input_ids'].to(f"cuda:{self.args['local_rank']}"),
            attention_mask=inputs['attention_mask'].to(f"cuda:{self.args['local_rank']}"),
            # position_ids=inputs['position_ids'].to(f"cuda:{self.args['local_rank']}"),
            # level_ids=inputs['level_ids'].to(f"cuda:{self.args['local_rank']}"),
            labels=inputs['labels'].to(f"cuda:{self.args['local_rank']}")
        )
        loss = outputs.loss
        # trigger = list(self.model.base_model.model.model.layers[0].named_parameters())
        
        # monitor token accuracy
        logits = outputs.logits[:, :-1, :]
        labels = inputs['labels'][:, 1:].to(f"cuda:{self.args['local_rank']}")
        token_acc = monitor_token_acc(logits, labels)
        return loss, token_acc
        
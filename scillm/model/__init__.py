from header import *
from .agent import DeepSpeedAgent
from .scillm import scillm
from .scillm_sft import SciSFTLLM

def load_model(args):
    agent_name = args['models'][args['model']]['agent_name']
    model_name = args['models'][args['model']]['model_name']
    model = globals()[model_name](**args)
    agent = globals()[agent_name](model, args)
    # ipdb.set_trace()
    return agent
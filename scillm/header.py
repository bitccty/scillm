import torch
import datetime
import types
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
from torch.utils.tensorboard import SummaryWriter
import transformers
import numpy as np
from collections import OrderedDict, Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from torch.nn import DataParallel
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import re
import math
import random
import json
import time
from copy import deepcopy
import ipdb
from loguru import logger
import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, GenerationConfig
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from flask import Flask, request, jsonify, make_response, session
from typing import Optional, Union, Tuple
import string
from multiprocessing import Pool
from functools import partial
from nltk.translate.bleu_score import sentence_bleu
import evaluate
import mauve
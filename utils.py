import random

import torch
import numpy as np

from transformers import (
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
)
from tokenization_kobert import KoBertTokenizer

MODEL_CLASSES = {
    'kobert': (BertConfig, BertForSequenceClassification, KoBertTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'kobert-lm': (BertConfig, BertForSequenceClassification, KoBertTokenizer),
}

MODEL_PATH_MAP = {
    'kobert': 'monologg/kobert',
    'bert': 'bert-base-multilingual-cased',
    'kobert-lm': 'monologg/kobert-lm',
}


def get_label(args):
    return [0, 1]


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

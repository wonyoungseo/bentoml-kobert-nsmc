# https://docs.bentoml.org/en/latest/frameworks.html#bentoml.frameworks.transformers.TransformersModelArtifact

import torch
import numpy as np
from torch.utils.data import TensorDataset
from utils import load_tokenizer
from typing import List
from argparse import Namespace

import pprint

import bentoml
from bentoml.frameworks.transformers import TransformersModelArtifact
from bentoml.adapters import JsonInput
from bentoml.types import JsonSerializable



@bentoml.env(pip_packages=['torch', 'transformers', 'numpy', 'sentencepiece'])
@bentoml.artifacts([TransformersModelArtifact('model')])
class KobertSentiClassifier(bentoml.BentoService):

    def get_args(self):
        args = {'adam_epsilon': 1e-08,
                             'data_dir': './data',
                             'do_eval': True,
                             'do_train': True,
                             'eval_batch_size': 64,
                             'gradient_accumulation_steps': 1,
                             'learning_rate': 5e-05,
                             'logging_steps': 2000,
                             'max_grad_norm': 1.0,
                             'max_seq_len': 50,
                             'max_steps': -1,
                             'model_dir': './model',
                             'model_name_or_path': 'monologg/kobert',
                             'model_type': 'kobert',
                             'no_cuda': False,
                             'num_train_epochs': 5.0,
                             'save_steps': 2000,
                             'seed': 42,
                             'task': 'nsmc',
                             'test_file': 'ratings_test.txt',
                             'train_batch_size': 32,
                             'train_file': 'ratings_train.txt',
                             'warmup_steps': 0,
                             'weight_decay': 0.0}
        self.model_args = Namespace(**args) 
        return self.model_args           

    def set_device(self, device):
        torch.device(device)
        self.device = device

    def set_prediction_classes(self):
        self.classes = ['negative', 'positive']

    def convert_text_to_tensor(self, 
                                string_text,
                               cls_token_segment_id=0,
                               pad_token_segment_id=0,
                               sequence_a_segment_id=0,
                               mask_padding_with_zero=True):
        tokenizer = load_tokenizer(self.model_args)
        # tokenizer = self.artifacts.tokenizer
        # Setting based on the current model type
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        pad_token_id = tokenizer.pad_token_id
        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []

        tokens = tokenizer.tokenize(string_text)
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > self.model_args.max_seq_len - special_tokens_count:
            tokens = tokens[:(self.model_args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.model_args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)

        # Change to Tensor
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
        return dataset

    @bentoml.api(input=JsonInput())
    def predict(self, json_input):
        device = self.set_device('cpu')
        self.set_prediction_classes()
        self.get_args()

        model = self.artifacts.model['model']
        dataset = self.convert_text_to_tensor(json_input['text'])
        batch = tuple(t.to(self.device) for t in dataset.tensors)



        model.to('cpu')
        model.eval()
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}
            outputs = model(**inputs)
            logits = outputs[0]

        preds = logits.detach().cpu().numpy()
        idx = np.argmax(preds, axis=1)[0]
        return self.classes[idx]

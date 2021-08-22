import torch
from transformers import BertForSequenceClassification, BertTokenizer
from KobertSentiClassificationService import KobertSentiClassifier
import pprint


def get_model_args():
    return torch.load('model/training_args.bin')

def saveToBento():

    bento_service = KobertSentiClassifier()
    # model_args = get_model_args()
    model_args = bento_service.get_args()
    
    model = BertForSequenceClassification.from_pretrained(model_args.model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path)
    artifact = {
        "model": model,
        "tokenizer": tokenizer
        }
    bento_service.pack("model", artifact)

    saved_path = bento_service.save()
    print('Bento Service Saved in ', saved_path)

if __name__ == "__main__":
    saveToBento()
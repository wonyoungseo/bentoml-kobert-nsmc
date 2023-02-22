import bentoml
from transformers import pipeline
from transformers import ElectraForSequenceClassification, ElectraTokenizer

MODEL_NAME = 'monologg/koelectra-base-finetuned-nsmc'

if __name__ == "__main__":

    model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)

    pipe_clf = pipeline('text-classification', model=model, tokenizer=tokenizer)
    bentoml.transformers.save_model(name="koelectra_nsmc_clf", pipeline=pipe_clf)
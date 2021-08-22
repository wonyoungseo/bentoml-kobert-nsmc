
# KoBert Sentiment Classifier served with BentoML

>Korean sentiment classifier trained on KoBert & NSMC dataset.

## Train KoBert Sentiment Analyzer

- Refer to `train-kobert-sentiment-clf-model.ipynb`

## Serve model with BentoML

### Save model

```bash
python saveToBento.py
```

### Serve model

```bash
bentoml serve KobertSentiClassifier:latest
```

### Request model inference

```bash
curl -X POST "http://127.0.0.1:52621/predict" -H "accept: */*" -H "Content-Type: application/json" -d "{\"text\":\"이 영화는 진짜 엉망이네\"}"
```

## Todos

- Deploy to GCP
- Develop streamlit app and deploy to heroku



## Reference

- [KoBERT-nsmc](https://github.com/monologg/KoBERT-nsmc)
- [BentoML Documentation](https://docs.bentoml.org/en/latest/)

# Sentiment Classification Model Serving (example)

> Korean text sentiment classification API
> with Huggingface, BentoML

## 1. Save model to BentoML

- Save fine-tuned model to Bento

```bash
$ python saveToBento.py
```

## 2. Serve model

- 3 different approaches of serving model

### 2-1. Serve model directly

- serve directly via python script

```bash
$ bentoml serve service:svc
```

### 2-2. Serve model after building `Bento`

1. build `bento` based on `bentofile.yaml`
2. serve directly

```bash
$ bentoml build
$ bentoml serve {BENTOML_SERVICE_NAME}
```


### 2-3. Serve model after containerization with Docker

1. build `bento` based on `bentofile.yaml`
2. containerize as docker image
3. run docker container

```bash
$ bentoml build
$ bentoml containerize {BENTOML_SERVICE_NAME}:latest -t {BENTOML_SERVICE_NAME}:latest 
$ docker run -it --rm -p 3000:3000 {BENTOML_SERVICE_NAME}:latest
```

## 3. Sent prediction request to API

```bash
$ curl -X 'POST' 'http://127.0.0.1:3000/predict' -d '저 영화감독은 진짜 천재인거 같아. 어떻게 저런 소재랑 배우를 가지고 영화를 망칠 수가 있지?'
```
```text
[{"label":"negative","score":0.9831924438476562}]
```
```bash
$ curl -X 'POST' 'http://127.0.0.1:3000/predict' -d '저 영화감독은 진짜 천재인거 같아. 예산이 적은데도 저런 작품성 있는 영화를 만들었네'
```
```text
[{"label":"positive","score":0.9637786746025085}]
```


## Future work

- [ ] Fine-tune pretrained LM with NSMC dataset
- [ ] Model distilation (DistilBERT)
- [ ] Save fine-tuned model to BentoML with Huggingface custom pipeline
- [ ] Deploy with container // Deploy using cloud service
- [ ] Build prototype application and apply CI/CD

## Reference

- [BentoML Documentation](https://docs.bentoml.org/en/latest/)
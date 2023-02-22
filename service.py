import bentoml
from bentoml.io import Text, JSON

runner = bentoml.transformers.get("koelectra_nsmc:latest").to_runner()

svc = bentoml.Service("koelectra_nsmc_clf_service", runners=[runner])

@svc.api(input=Text(), output=JSON())
async def predict(input_text: str) -> list:
    return await runner.async_run(input_text)
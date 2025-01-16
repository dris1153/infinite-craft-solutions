from fastapi import FastAPI
import nltk # type: ignore
from .classes.ElementCombiner import ElementCombiner
from pydantic import BaseModel

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('words', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

app = FastAPI()


@app.get("/get-list-models")
async def root():
    try:
        available_models = ElementCombiner.list_available_models()
        data = []
        for i, model_name in enumerate(available_models, 1):
            data.append({
                "model_name": model_name,
                "id": i
            })
        return {"success": True, "data": data, "code": 200}
    except Exception as e:
        return {"success": False, "error": str(e), "code": 500}
    
class TrainRequest(BaseModel):
    model_name: str
    first_element: str
    second_element: str
@app.post("/train")
async def train_model(request: TrainRequest):
    try:
        combiner = ElementCombiner.load_state(request.model_name)
        model = ElementCombiner.load_model(request.model_name)
        # combiner.train(request.first_element, request.second_element)
        # return {"success": True, "message": "Model trained successfully", "code": 200}
    except Exception as e:
        return {"success": False, "error": str(e), "code": 500}

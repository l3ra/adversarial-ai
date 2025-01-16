from fastapi import FastAPI, HTTPException
from transformers import AutoModelForSequenceClassification, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import torch
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()

@app.get("/")
def success():
    return "Success", 200

@app.get("/health")
def healthy():
    return "Healthy", 200

### Textfooler against Bert Model
bert_model_name = "textattack/bert-base-uncased-imdb"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name)
@app.post("/predict")
def predict(text: str):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return {"positive": float(probs[0][1]), "negative": float(probs[0][0])}


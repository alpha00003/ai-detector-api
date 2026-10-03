"""Earlybird-fast AI-text detector served as a tiny HTTP API."""
import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import pipeline

MODEL_ID = "noumenon-labs/Earlybird-fast"
API_KEY = os.environ.get("API_KEY", "")

clf = pipeline("text-classification", model=MODEL_ID, top_k=None, truncation=True, max_length=512)

app = FastAPI(title="Earlybird detector")

class Item(BaseModel):
    text: str

@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_ID}

@app.post("/predict")
def predict(item: Item, x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="bad api key")
    text = item.text.strip()
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="text too short")

    out = clf(text[:6000])
    rows = out[0] if isinstance(out[0], list) else out
    return {"scores": {r["label"]: float(r["score"]) for r in rows}}

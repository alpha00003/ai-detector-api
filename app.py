"""Multi-model AI-text detector served as an HTTP API on Render."""
import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch
import numpy as np

# --- Model IDs ---
MODEL_EARLYBIRD = "noumenon-labs/Earlybird-fast"
MODEL_ROBERTA = "coai/roberta-ai-detector-v2"
MODEL_CHATGPT = "Hello-SimpleAI/chatgpt-detector-roberta"

API_KEY = os.environ.get("API_KEY", "")

# --- Load all models on startup ---
print("Loading models...")
clf_earlybird = pipeline("text-classification", model=MODEL_EARLYBIRD, top_k=None, truncation=True, max_length=512)
clf_roberta = pipeline("text-classification", model=MODEL_ROBERTA, truncation=True, max_length=512)
clf_chatgpt = pipeline("text-classification", model=MODEL_CHATGPT, truncation=True, max_length=512)
print("All models loaded.")

app = FastAPI(title="FS AI Detector - Multi-Model")

class Item(BaseModel):
    text: str

def get_ai_score(pipe_output, model_key='roberta'):
    """Extracts the AI probability score from pipeline output."""
    if model_key == 'earlybird':
        # Earlybird returns a list of dicts with 'label' and 'score'
        if isinstance(pipe_output, list) and len(pipe_output) > 0:
            for item in pipe_output[0]:
                if item['label'].lower() in ['ai', 'label_1', 'machine', 'generated']:
                    return float(item['score'])
        return 0.0
    else:
        # Other models return a single dict with 'label' and 'score'
        if isinstance(pipe_output, list):
            result = pipe_output[0]
        else:
            result = pipe_output
        # 'LABEL_1' is typically the AI class for these models
        if result.get('label') == 'LABEL_1':
            return float(result.get('score', 0.0))
        return 1.0 - float(result.get('score', 0.0))

@app.get("/")
def health():
    return {"status": "ok", "models": [MODEL_EARLYBIRD, MODEL_ROBERTA, MODEL_CHATGPT]}

@app.post("/predict")
def predict(item: Item, x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="bad api key")
    
    text = item.text.strip()
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="text too short")

    # --- Run all models ---
    out_earlybird = clf_earlybird(text[:6000])
    out_roberta = clf_roberta(text[:512])
    out_chatgpt = clf_chatgpt(text[:512])

    # --- Extract AI scores ---
    score_e = get_ai_score(out_earlybird, 'earlybird')
    score_r = get_ai_score(out_roberta, 'roberta')
    score_c = get_ai_score(out_chatgpt, 'chatgpt')

    # --- Ensemble: Weighted Average ---
    # Giving more weight to the more accurate RoBERTa model
    final_score = (score_e * 0.25) + (score_r * 0.50) + (score_c * 0.25)
    
    # --- Zero-Tolerance Threshold ---
    # If the final score is above 30%, classify as AI.
    final_label = "AI" if final_score > 0.30 else "Human"

    return {
        "scores": {
            "Human": round(1.0 - final_score, 4),
            "AI": round(final_score, 4)
        },
        "final_label": final_label,
        "individual_scores": {
            "earlybird": round(score_e, 4),
            "roberta": round(score_r, 4),
            "chatgpt": round(score_c, 4)
        }
    }

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import os

app = FastAPI(title="AI Text Detector API")

print("Loading model...")
classifier = pipeline(
    "text-classification",
    model="desklib/ai-text-detector-v1.01",
    device=-1
)
print("Model loaded!")

class TextInput(BaseModel):
    text: str

@app.post("/detect")
async def detect_ai(input_data: TextInput):
    text = input_data.text
    result = classifier(text, truncation=True, max_length=512)[0]
    label = result['label']
    score = result['score']
    
    if label == 'LABEL_1' or label == 'AI':
        ai_percent = round(score * 100)
        human_percent = 100 - ai_percent
    else:
        human_percent = round(score * 100)
        ai_percent = 100 - human_percent
    
    return {
        "aiPercent": ai_percent,
        "humanPercent": human_percent,
        "label": label,
        "confidence": score
    }

@app.get("/")
def read_root():
    return {"status": "AI Detector API is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

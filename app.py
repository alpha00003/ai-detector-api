import os, re, json, httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant" # Groq ka sab se fast model
MAX_CHARS = 15000

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class D(BaseModel):
    text: str

class H(BaseModel):
    text: str
    keyword: str = ""
    tone: str = "natural"

@app.post("/api/detect")
async def detect(req: D):
    text = (req.text or "").strip()
    if len(text.split()) < 30:
        return JSONResponse({"error": "Min 30 words required"}, status_code=400)

    text = text[:MAX_CHARS]
    if not GROQ_API_KEY:
        return JSONResponse({"error": "Groq API key missing"}, status_code=500)

    system = """You are an elite AI text detector (like GPTZero, Originality.ai). 
Analyze the provided text and return ONLY a valid JSON object.
Look for these AI patterns:
- Overly perfect grammar and predictable sentence structures.
- Low burstiness (sentences are all similar length).
- Repetitive transition words (Furthermore, Moreover).
- Lack of genuine human emotion, slang, or imperfection.

Return EXACTLY this JSON format:
{"ai_percent": 85, "verdict": "AI-Generated"}
Strictly valid JSON only. No markdown, no extra text."""

    try:
        async with httpx.AsyncClient(timeout=25.0) as c:
            r = await c.post(GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role":"system","content":system},{"role":"user","content":text}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.1
                })
        
        if r.status_code != 200:
            raise Exception(f"Groq {r.status_code}")
        
        result = json.loads(r.json()["choices"][0]["message"]["content"])
        pct = int(result.get("ai_percent", 50))
        
        return {
            "verdict": result.get("verdict", "Uncertain"),
            "aiPercent": pct,
            "humanPercent": 100 - pct,
            "source": "Groq AI Engine"
        }
    except Exception as e:
        print("[DETECT ERROR]", e)
        return JSONResponse({"error": "Detection failed"}, status_code=503)

@app.post("/api/humanize")
async def humanize(req: H):
    text = (req.text or "").strip()
    if len(text.split()) < 20:
        return JSONResponse({"error": "Min 20 words"}, status_code=400)

    text = text[:MAX_CHARS]
    tone = req.tone if req.tone in ("natural","casual","formal","blog") else "natural"
    keyword = (req.keyword or "").strip()
    kw_prompt = f"- Include keyword '{keyword}' naturally." if keyword else ""
    
    system = f"""You rewrite AI text to sound 100% human. 
Rules:
1. Keep all facts the same.
2. Mix short (3-5 words) and long (15-20 words) sentences.
3. Use contractions (don't, it's, won't).
4. NO em-dashes (—), use commas.
5. Output ONLY the rewritten text, no introductory remarks.
Tone: {tone}
{kw_prompt}"""

    try:
        async with httpx.AsyncClient(timeout=40.0) as c:
            r = await c.post(GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role":"system","content":system},{"role":"user","content":text}],
                    "max_tokens": 4000,
                    "temperature": 0.85
                })
        
        if r.status_code != 200:
            raise Exception(f"Groq {r.status_code}")
            
        final = r.json()["choices"][0]["message"]["content"].strip()
        final = re.sub(r'^["\u201C\u201D\']+|["\u201C\u201D\']+$', "", final).strip()
        return {"humanizedText": final, "engine": "groq"}
    except Exception as e:
        print("[HUMANIZE ERROR]", e)
        return JSONResponse({"error": "Humanizer failed"}, status_code=503)

"""
FS AI Tool — Detector + Humanizer + SEO
Model: rasbt/ai-text-detector-distilbert
Groq: openai/gpt-oss-120b
"""
import os, re, time, httpx
from typing import Dict
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

MODEL_ID = os.environ.get("MODEL_ID", "rasbt/ai-text-detector-distilbert")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"
MAX_CHARS = 8000

print("[BOOT] Loading:", MODEL_ID)
clf = pipeline("text-classification", model=MODEL_ID, top_k=None, truncation=True, max_length=512)
print("[BOOT] Ready")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_rate: Dict[str, dict] = {}
def ok(ip):
    t = time.time()
    r = _rate.get(ip)
    if not r or t - r["t"] > 60:
        _rate[ip] = {"c": 1, "t": t}
        return True
    if r["c"] < 20:
        r["c"] += 1
        return True
    return False

class D(BaseModel):
    text: str

class H(BaseModel):
    text: str
    keyword: str = ""
    tone: str = "natural"


@app.post("/api/detect")
async def detect(req: D, request: Request):
    ip = request.client.host if request.client else "x"
    if not ok(ip):
        return JSONResponse({"error": "Rate limit"}, status_code=429)

    text = (req.text or "").strip()
    if len(text.split()) < 50:
        return JSONResponse({"error": "Min 50 words"}, status_code=400)

    text = text[:MAX_CHARS]
    pct = None
    try:
        out = clf(text[:6000])
        rows = out[0] if isinstance(out[0], list) else out
        scores = {r["label"]: float(r["score"]) for r in rows}
        ai_key = next((k for k in scores if re.search(r"^(ai|label_1|machine|generated|fake)", k, re.I)), None)
        if ai_key:
            pct = int(round(scores[ai_key] * 100))
    except Exception as e:
        print("[DETECT] error:", e)

    if pct is None:
        return JSONResponse({"error": "Model unavailable"}, status_code=503)

    if pct >= 60:
        verdict = "AI-Generated"
    elif pct >= 35:
        verdict = "Mixed / Uncertain"
    else:
        verdict = "Human-Written"

    return {
        "verdict": verdict,
        "aiPercent": pct,
        "scores": {"aiPercent": pct, "humanPercent": 100 - pct},
        "source": MODEL_ID,
        "wordCount": len(text.split()),
    }


CLICHES = [
    (r"\bfurthermore\b","also"),(r"\bmoreover\b","also"),(r"\bin conclusion\b","overall"),
    (r"\bdelve\b","explore"),(r"\btestament to\b","shows"),
    (r"\butilize\b","use"),(r"\badditionally\b","also"),(r"\bconsequently\b","so"),
    (r"\bsubsequently\b","then"),(r"\bit is important to note that\b","keep in mind"),
    (r"\bseamlessly\b","smoothly"),(r"\bmeticulously\b","carefully"),(r"\bmultifaceted\b","complex"),
    (r"\bparamount\b","key"),(r"\bpivotal\b","key"),(r"\bcrucial\b","important"),
    (r"\bthus\b","so"),(r"\bhence\b","so"),(r"\bleverage\b","use"),
    (r"\bfacilitate\b","help"),(r"\bcommence\b","start"),(r"\bever-evolving\b","changing"),
    (r"\bstrive to\b","try to"),(r"\bmyriad of\b","many"),(r"\bplethora of\b","many"),
    (r"\bstreamline\b","simplify"),(r"\belevate\b","lift"),(r"\brevolutionize\b","change"),
    (r"\bversatile\b","flexible"),(r"\beffortless\b","easy"),
]

def clean(t, tone):
    o = t
    for p, r in CLICHES:
        o = re.sub(p, r, o, flags=re.I)
    o = re.sub(r"[ \t]+", " ", o)
    o = re.sub(r"\n{3,}", "\n\n", o).strip()
    o = re.sub(r"\s+([,.!?;:])", r"\1", o)
    o = re.sub(r"([,.!?;:])([A-Za-z])", r"\1 \2", o)
    o = re.sub(r"\b(\w+)\s+\1\b", r"\1", o, flags=re.I)
    o = re.sub(r"  +", " ", o)
    if tone == "formal":
        o = re.sub(r"\bcan't\b","cannot",o,flags=re.I)
        o = re.sub(r"\bwon't\b","will not",o,flags=re.I)
    return o.strip()

def strip_pre(s):
    s = re.sub(r"^(here(?:'s| is)[^\n]{0,120}(?:rewrite|rewritten|version|text)[^\n]{0,60}:\s*)", "", s, flags=re.I)
    s = re.sub(r"^(sure[,!]?\s*|certainly[,!]?\s*|of course[,!]?\s*)", "", s, flags=re.I)
    return re.sub(r'^["\u201C\u201D\']+|["\u201C\u201D\']+$', "", s).strip()


@app.post("/api/humanize")
async def humanize(req: H):
    text = (req.text or "").strip()
    if len(text.split()) < 20:
        return JSONResponse({"error": "Min 20 words"}, status_code=400)

    keyword = re.sub(r'["\n\r]', " ", (req.keyword or "")[:100]).strip()
    tone = req.tone if req.tone in ("natural","casual","formal","blog") else "natural"
    text = text[:MAX_CHARS]

    if not GROQ_API_KEY:
        final = clean(text, tone)
        return {
            "humanizedText": final, "originalWords": len(text.split()),
            "newWords": len(final.split()), "keyword": keyword,
            "tone": tone, "engine": "local-fallback"
        }

    tone_map = {
        "formal": "Professional. Clear. No buzzwords.",
        "casual": "Like texting a smart friend. Short sentences.",
        "blog": "Seasoned blogger. Punchy. Mix short and long.",
        "natural": "Fluent speaker. Vary rhythm."
    }
    kw = f'- Weave "{keyword}" in naturally 2-3 times.' if keyword else ""
    system = f"""You rewrite AI text to sound human. You MUST return a rewritten version.

RULES:
1. Preserve EVERY fact. Never invent.
2. Vary sentence length: mix 3-6 word sentences with 15-25 word ones.
3. Vary openings.
4. Use contractions: don't, it's, you'll, we're, can't, that's.
5. Kill AI clichés: furthermore, moreover, delve, tapestry, crucial, leverage, foster, underscore, holistic, seamlessly, realm, beacon, paramount, streamline, elevate, revolutionize, versatile, effortless, maximize, optimize, cutting-edge, unleash, empower, transformative, enhance, harness, tailored, intricate, plethora, myriad, innovative, sleek, genuinely, ultimately, fundamentally.
6. Kill: "not only X but also Y", "in today's world", "gone are the days", "enter the", "if there is one".
7. No preamble. No quotes. No headings.
8. Keep names, numbers, URLs exactly.
9. Prose only. Fragments fine. NO typos.

Tone: {tone_map[tone]}
{kw}

Output ONLY rewritten text."""

    try:
        async with httpx.AsyncClient(timeout=50.0) as c:
            r = await c.post(GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role":"system","content":system},{"role":"user","content":f"Rewrite:\n\n{text}"}],
                    "max_tokens": min(4500, max(900, len(text.split()) * 3)),
                    "temperature": 0.9, "top_p": 0.95
                })
        if r.status_code != 200:
            raise Exception(f"Groq {r.status_code}: {r.text[:150]}")
        raw = r.json()["choices"][0]["message"]["content"].strip()
        final = clean(strip_pre(raw), tone)
        engine = "groq"
    except Exception as e:
        print("[HUMANIZE] Groq failed:", e)
        final = clean(text, tone)
        engine = "local-fallback"

    return {
        "humanizedText": final, "originalWords": len(text.split()),
        "newWords": len(final.split()), "keyword": keyword,
        "tone": tone, "engine": engine
    }


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_ID, "groq": bool(GROQ_API_KEY), "groq_model": GROQ_MODEL}


@app.post("/predict")
def legacy(item: D):
    if len(item.text.strip()) < 20:
        raise HTTPException(400, "too short")
    out = clf(item.text[:6000])
    rows = out[0] if isinstance(out[0], list) else out
    return {"scores": {r["label"]: float(r["score"]) for r in rows}}

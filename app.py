"""
FS AI Tool — Detector + Humanizer + SEO (Render backend)
"""
import os
import re
import math
import time
import httpx
from typing import Dict

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

MODEL_ID = os.environ.get("MODEL_ID", "noumenon-labs/Earlybird-fast")
API_KEY = os.environ.get("API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

MIN_DETECT_WORDS = 50
MIN_HUMANIZE_WORDS = 20
MAX_CHARS = 8000

print("[BOOT] Loading detector:", MODEL_ID)
clf = pipeline("text-classification", model=MODEL_ID, top_k=None, truncation=True, max_length=512)
print("[BOOT] Detector ready.")

app = FastAPI(title="FS AI Tool")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_rate_store: Dict[str, dict] = {}
RATE_MAX = 20
RATE_WINDOW = 60


def check_rate(ip: str) -> bool:
    now = time.time()
    rec = _rate_store.get(ip)
    if not rec or now - rec["start"] > RATE_WINDOW:
        _rate_store[ip] = {"count": 1, "start": now}
        return True
    if rec["count"] < RATE_MAX:
        rec["count"] += 1
        return True
    return False


class DetectReq(BaseModel):
    text: str


class HumanizeReq(BaseModel):
    text: str
    keyword: str = ""
    tone: str = "natural"


# ============================================================
# DETECTOR DATA — expanded for aggressive detection
# ============================================================
AI_VOCAB = [
    "furthermore", "moreover", "in conclusion", "delve", "tapestry",
    "testament", "crucial", "paramount", "pivotal", "subsequently",
    "nonetheless", "unwavering", "fostering", "beacon", "realm",
    "vital role", "it is important to note", "in today's world",
    "a broad range of", "shed light on", "play a crucial role",
    "ever-evolving", "multifaceted", "interplay", "underscore",
    "holistic", "indispensable", "imperative", "testament to",
    "strive to", "navigate", "leverage", "foster",
    "revolutionize", "revolutionized", "revolutionary", "revolutionizing",
    "streamline", "streamlined", "streamlining",
    "seamless", "seamlessly", "seamlessness",
    "elevate", "elevated", "elevating", "elevation",
    "versatile", "versatility",
    "effortless", "effortlessly", "effortlessness",
    "maximize", "maximized", "maximizing",
    "optimize", "optimized", "optimizing",
    "robust", "cutting-edge", "state-of-the-art",
    "game-changer", "game changer", "game-changing",
    "unleash", "unleashing", "unparalleled",
    "empower", "empowering", "empowered",
    "transformative", "transformation",
    "enhance", "enhanced", "enhancing", "enhancement",
    "boast", "boasts", "boasting",
    "harness", "harnessing", "harnessed",
    "cultivate", "cultivating",
    "tailored", "bespoke", "curated",
    "intricate", "intricacies",
    "plethora", "myriad",
    "innovative", "innovation", "innovate",
    "sleek", "multipurpose", "multi-purpose",
    "high-powered", "high-performance",
    "genuinely",
    "ultimately", "fundamentally", "essentially", "in essence",
    "when it comes to", "at the end of the day",
    "gimmicks", "gimmick",
]

AI_PHRASES = [
    r"\bwhether you (are|'re) (a|an)\b",
    r"\bgone are the days\b",
    r"\benter the\b",
    r"\blook for (models|ones|options|versions|units) with\b",
    r"\bkeep these .{0,60} in mind\b",
    r"\bbefore hitting\b",
    r"\bif there (is|are) one\b",
    r"\bhere are the (essential|best|top|key|must-have)\b",
    r"\bdesigned to (streamline|enhance|maximize|help|make|keep|transform|provide|deliver)\b",
    r"\bthis (versatile|innovative|powerful|sleek|compact|multipurpose) (unit|device|tool|gadget|appliance)\b",
    r"\bin the (modern|digital|today's) (world|age|era|kitchen|landscape)\b",
    r"\bnot only .{0,80}? but also\b",
    r"\bwith the (touch|push|click) of a button\b",
    r"\bhas (revolutionized|transformed|changed|elevated|reshaped)\b",
    r"\beliminates? the need for\b",
    r"\bdesigned with .{0,60}? in mind\b",
    r"\btakes? .{0,60}? to the next level\b",
    r"\bdeserves? a permanent spot\b",
    r"\bthe modern (kitchen|world|era|approach|landscape)\b",
    r"\btransforms? from a chore\b",
    r"\ba chore into a\b",
    r"\bpermanent spot in your\b",
    r"\bgame of (guesswork|chance)\b",
    r"\binvesting in the right\b",
    r"\btoday's (smart|modern|advanced|innovative)\b",
    r"\bdesigned for (those|people|users) who\b",
    r"\bif you (are|'re) looking (to|for)\b",
    r"\bwhen you (are|'re) (ready|looking)\b",
    r"\bit's (not|isn't) about .{0,60}? it's about\b",
    r"\bremoving friction from\b",
    r"\bdelicious (home-cooked|home cooked) meals\b",
    r"\bseamless (daily|everyday|experience|integration|flow|delight)\b",
    r"\btransforms? .{0,40}? into a\b",
    r"\bwith a few (smart|simple|easy)\b",
    r"\bat your disposal\b",
    r"\bfrom a chore\b",
]

HUMANIZED_PATTERNS = [
    r"\b(basically|honestly|literally|frankly)\b.*?\b(furthermore|moreover|crucial)\b",
    r"[,;]\s*(and|but|or|so)\s*[,;]",
    r"\b(you know|i mean|look|here is the thing)\b",
]


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def local_metrics(text: str) -> dict:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    lens = [len(re.findall(r"\S+", s)) for s in sentences] or [0]
    n = len(lens)
    avg = sum(lens) / n
    variance = sum((x - avg) ** 2 for x in lens) / n
    stddev = math.sqrt(variance)

    if stddev < 3.2:
        burst_label, burst_score = "Low (AI Symmetrical)", 30
    elif stddev > 6.5:
        burst_label, burst_score = "High (Human Irregular)", -20
    else:
        burst_label, burst_score = "Normal", 0

    lower = text.lower()
    detected = [w for w in AI_VOCAB if w in lower]
    phrase_hits = [p for p in AI_PHRASES if re.search(p, text, re.IGNORECASE | re.DOTALL)]

    vocab_score = min(60, len(detected) * 6)
    phrase_score = min(60, len(phrase_hits) * 10)

    total_signals = len(detected) + len(phrase_hits)

    humanized_triggers = sum(1 for p in HUMANIZED_PATTERNS if re.search(p, text, re.IGNORECASE))

    base = min(100, max(0, 20 + burst_score + vocab_score + phrase_score))

    # Aggressive floors
    if total_signals >= 10:
        base = max(base, 95)
    elif total_signals >= 6:
        base = max(base, 88)
    elif total_signals >= 4:
        base = max(base, 78)
    elif total_signals >= 2:
        base = max(base, 65)
    elif total_signals >= 1:
        base = max(base, 50)

    return {
        "burstiness": burst_label,
        "perplexity": "Low" if stddev < 4 else "High",
        "variance": round(stddev, 1),
        "detected": detected,
        "phrases_detected": phrase_hits,
        "total_signals": total_signals,
        "humanized_triggers": humanized_triggers,
        "base_score": base,
    }


def earlybird_predict(text: str) -> Dict[str, float]:
    out = clf(text[:6000])
    rows = out[0] if isinstance(out[0], list) else out
    return {r["label"]: float(r["score"]) for r in rows}


CLICHE_MAP = [
    (r"\bfurthermore\b", "also"), (r"\bmoreover\b", "also"),
    (r"\bin conclusion\b", "overall"), (r"\bdelve into\b", "look at"),
    (r"\bdelve\b", "explore"), (r"\btapestry\b", "mix"),
    (r"\btestament to\b", "shows"), (r"\btestament\b", "proof"),
    (r"\butilize\b", "use"), (r"\butilization\b", "use"),
    (r"\badditionally\b", "also"), (r"\bconsequently\b", "so"),
    (r"\bsubsequently\b", "then"), (r"\bin today's world\b", "today"),
    (r"\bit is important to note that\b", "keep in mind"),
    (r"\bit is worth noting that\b", "note that"),
    (r"\bseamlessly\b", "smoothly"), (r"\bmeticulously\b", "carefully"),
    (r"\bmultifaceted\b", "complex"), (r"\boverarching\b", "main"),
    (r"\bspearhead\b", "lead"), (r"\bbeacon\b", "guide"),
    (r"\bparamount\b", "key"), (r"\bpivotal\b", "key"),
    (r"\bcrucial\b", "important"), (r"\bundoubtedly\b", "clearly"),
    (r"\bnotably\b", "especially"), (r"\bthus\b", "so"),
    (r"\bhence\b", "so"), (r"\bleverage\b", "use"),
    (r"\bfacilitate\b", "help"), (r"\bcommence\b", "start"),
    (r"\bterminate\b", "end"), (r"\bever-evolving\b", "changing"),
    (r"\bholistic approach\b", "full approach"),
    (r"\bstrive to\b", "try to"), (r"\bfostering\b", "building"),
    (r"\bnavigate the\b", "handle the"),
    (r"\bin order to\b", "to"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bat this point in time\b", "now"),
    (r"\bin the event that\b", "if"),
    (r"\bprior to\b", "before"),
    (r"\bsubsequent to\b", "after"),
    (r"\ba plethora of\b", "many"),
    (r"\bmyriad of\b", "many"),
    (r"\bin the realm of\b", "in"),
    (r"\bplays a significant role\b", "matters"),
    (r"\bit should be noted that\b", "note"),
    (r"\bwith regard to\b", "about"),
    (r"\bwith respect to\b", "about"),
]


def local_cleanup(text: str, tone: str) -> str:
    out = text
    for pat, rep in CLICHE_MAP:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)

    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)
    out = re.sub(r"([,.!?;:])([A-Za-z])", r"\1 \2", out)
    out = re.sub(r"!{2,}", "!", out)
    out = re.sub(r"\?{2,}", "?", out)
    out = re.sub(r",{2,}", ",", out)
    out = re.sub(r"\b(\w+)\s+\1\b", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"  +", " ", out)

    if tone == "formal":
        out = re.sub(r"\bcan't\b", "cannot", out, flags=re.IGNORECASE)
        out = re.sub(r"\bwon't\b", "will not", out, flags=re.IGNORECASE)
        out = re.sub(r"\bdon't\b", "do not", out, flags=re.IGNORECASE)
    return out.strip()


def strip_preamble(s: str) -> str:
    s = re.sub(r"^(here(?:'s| is)[^\n]{0,120}(?:rewrite|rewritten|version|text|attempt|output)[^\n]{0,60}:\s*)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(sure[,!]?\s*(here[^\n]*:)?\s*)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(certainly[,!]?\s*)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(of course[,!]?\s*)", "", s, flags=re.IGNORECASE)
    s = re.sub(r'^["“”\']+|["“”\']+$', "", s)
    return s.strip()


async def groq_rewrite(text: str, keyword: str, tone: str, wc: int) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("no groq key")

    tone_map = {
        "formal": "Write like a knowledgeable professional. Clear, precise, natural. No corporate buzzwords.",
        "casual": "Write like you are texting a smart friend. Short sentences. Real talk. No filler.",
        "blog": "Write like a seasoned blogger. Punchy openings. Mix short and long. Small real-world comparisons.",
        "natural": "Write like a fluent speaker explaining something they actually care about. Vary rhythm constantly.",
    }
    keyword_rule = f'- Weave the phrase "{keyword}" in naturally 2-3 times. Never keyword-stuff.' if keyword else ""

    system_prompt = f"""You are a human writer editing AI text. Make it sound like a real person wrote it.

CRITICAL RULES:

1. PRESERVE meaning and every fact. Never invent stats, quotes, names, or stories.
2. VARY SENTENCE LENGTH dramatically. Mix 3-6 word sentences with 15-25 word ones.
3. VARY SENTENCE OPENINGS. Do not start consecutive sentences the same way.
4. USE CONTRACTIONS: don't, it's, you'll, we're, that's, can't.
5. CUT AI CLICHÉS: furthermore, moreover, delve, tapestry, testament, crucial, pivotal, leverage, navigate, foster, underscore, holistic, multifaceted, seamlessly, realm, beacon, paramount, streamline, elevate, revolutionize, versatile, effortless.
6. KILL these: "not only X but also Y", "it is important to note", "in today's world", "in the realm of", "plays a vital role", "a testament to", "gone are the days", "enter the".
7. NO preamble. No "Here is the rewritten text". No quotes around output.
8. Do NOT add headings, bold, or bullets unless the original had them.
9. Keep names, numbers, URLs, code, and technical terms exactly.
10. Paragraphs must flow as prose, not a list.
11. ADD natural touches: a short aside, a rhetorical question, a dash — like this — a semicolon.
12. NO fake cheerfulness. NO "Let's dive in". NO "In conclusion".
13. Fragment sentences are fine.
14. Do NOT deliberately add typos.

Tone guide: {tone_map[tone]}
{keyword_rule}

Output ONLY the rewritten text."""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Rewrite this naturally:\n\n{text}"},
        ],
        "max_tokens": min(4500, max(900, int(wc * 3.2))),
        "temperature": 0.88,
        "top_p": 0.95,
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=45.0) as client:
        r = await client.post(GROQ_URL, headers=headers, json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"Groq {r.status_code}: {r.text[:150]}")
        return r.json()["choices"][0]["message"]["content"].strip()


def inject_burstiness(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) < 3:
        return text

    for i in range(1, len(sentences) - 1):
        a = len(sentences[i - 1].split())
        b = len(sentences[i].split())
        c = len(sentences[i + 1].split())
        if abs(a - b) < 3 and abs(b - c) < 3 and b > 10:
            trimmed = re.sub(r"^(However|Therefore|Additionally|Furthermore|Moreover|Thus),?\s*", "", sentences[i])
            trimmed = re.sub(r"\b(in order)\s+", "", trimmed)
            trimmed = re.sub(r"\b(very|really|quite|extremely)\s+", "", trimmed)
            sentences[i] = trimmed

    return " ".join(sentences)


def local_humanize_fallback(text: str, tone: str) -> str:
    cleaned = local_cleanup(text, tone)
    cleaned = inject_burstiness(cleaned)
    return cleaned.strip()


def seo_extras(text: str, keyword: str) -> dict:
    plain = re.sub(r"[#*_>`]", "", text).strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", plain) if s]

    meta = ""
    for s in sentences:
        if len(meta + " " + s) > 155:
            break
        meta = (meta + " " + s).strip()
    if not meta and sentences:
        meta = sentences[0][:155]
    if len(meta) > 160:
        meta = meta[:157] + "..."

    title = (keyword.strip().title() + ": " + (sentences[0][:55] if sentences else "")) if keyword else (sentences[0][:65] if sentences else "")
    if len(title) > 70:
        title = title[:67] + "..."

    slug_src = (keyword or title).lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug_src)
    slug = re.sub(r"\s+", "-", slug).strip("-")[:60]

    stop = set("the a an and or but of to in on at for with by is are was were be been being have has had do does did this that these those i you we they it he she as not no so if then from into over under".split())
    words = re.findall(r"[a-zA-Z]{3,}", plain.lower())
    freq: Dict[str, int] = {}
    for w in words:
        if w not in stop:
            freq[w] = freq.get(w, 0) + 1
    keywords = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:8]]

    all_words = re.findall(r"\S+", plain)
    avg_len = len(all_words) / len(sentences) if sentences else 0
    readability = "Easy" if avg_len <= 16 else ("Medium" if avg_len <= 22 else "Hard")
    readability_score = max(0, min(100, int(100 - (avg_len - 10) * 3)))

    density = 0.0
    if keyword and all_words:
        matches = plain.lower().count(keyword.lower())
        density = round(matches / len(all_words) * 100, 2)

    return {
        "suggestedTitle": title,
        "metaDescription": meta,
        "slug": slug,
        "keywords": keywords,
        "readability": readability,
        "readabilityScore": readability_score,
        "keywordDensity": density,
    }


@app.post("/api/detect")
async def detect(req: DetectReq, request: Request):
    ip = request.client.host if request.client else "unknown"
    if not check_rate(ip):
        return JSONResponse({"error": "Too many requests. Wait 60s."}, status_code=429)

    text = (req.text or "").strip()
    wc = word_count(text)
    if wc < MIN_DETECT_WORDS:
        return JSONResponse({"error": f"Minimum {MIN_DETECT_WORDS} words required. You have {wc}."}, status_code=400)

    text = text[:MAX_CHARS]
    local = local_metrics(text)

    try:
        scores = earlybird_predict(text)
        ai_key = next((k for k in scores if re.match(r"^(ai|label_1|machine|generated)$", k, re.IGNORECASE)), None)
        eb_pct = int(round(scores[ai_key] * 100)) if ai_key else None
    except Exception as e:
        print("[DETECT] Earlybird failed:", e)
        eb_pct = None

    if eb_pct is not None:
        if local["total_signals"] >= 4:
            final = round(local["base_score"] * 0.85 + eb_pct * 0.15)
        elif local["total_signals"] >= 2:
            final = round(local["base_score"] * 0.7 + eb_pct * 0.3)
        else:
            final = round(local["base_score"] * 0.4 + eb_pct * 0.6)
    else:
        final = local["base_score"]

    is_humanized = local["humanized_triggers"] > 0

    if is_humanized and final > 25:
        verdict = "Humanized AI Content"
        humanized_pct = min(88, max(52, final + 15))
        ai_pct = max(5, 100 - humanized_pct - 10)
        human_pct = max(0, 100 - ai_pct - humanized_pct)
    elif final >= 50:
        verdict = "AI-Generated"
        ai_pct, human_pct, humanized_pct = final, 100 - final, 0
    else:
        verdict = "Pure Human-Written"
        ai_pct, human_pct, humanized_pct = final, 100 - final, 0

    return {
        "verdict": verdict,
        "aiPercent": max(0, min(100, ai_pct)),
        "scores": {
            "aiPercent": max(0, min(100, ai_pct)),
            "humanizedPercent": max(0, min(100, humanized_pct)),
            "humanPercent": max(0, min(100, human_pct)),
        },
        "metrics": {
            "burstiness": local["burstiness"],
            "perplexity": local["perplexity"],
            "sentenceVariance": local["variance"],
            "aiPhrasesFound": local["detected"],
            "phrasePatternsFound": len(local["phrases_detected"]),
            "totalSignals": local["total_signals"],
            "earlybirdScore": eb_pct,
        },
        "source": "Earlybird-fast + Expanded Statistical (Render)",
        "wordCount": wc,
    }


@app.post("/api/humanize")
async def humanize(req: HumanizeReq, request: Request):
    ip = request.client.host if request.client else "unknown"
    if not check_rate(ip):
        return JSONResponse({"error": "Too many requests. Wait 60s."}, status_code=429)

    text = (req.text or "").strip()
    wc = word_count(text)
    if wc < MIN_HUMANIZE_WORDS:
        return JSONResponse({"error": f"Minimum {MIN_HUMANIZE_WORDS} words required. You have {wc}."}, status_code=400)

    keyword = re.sub(r'["\n\r]', " ", (req.keyword or "")[:100]).strip()
    tone = req.tone if req.tone in ("natural", "casual", "formal", "blog") else "natural"
    text = text[:MAX_CHARS]

    engine = "groq"
    try:
        raw = await groq_rewrite(text, keyword, tone, wc)
        final = local_cleanup(strip_preamble(raw), tone)
        final = inject_burstiness(final)
    except Exception as e:
        print("[HUMANIZE] Groq failed -> fallback:", e)
        engine = "local-fallback"
        final = local_humanize_fallback(text, tone)

    if not final or len(final) < 15:
        raise HTTPException(500, "Rewritten text too short.")

    return {
        "humanizedText": final,
        "originalWords": wc,
        "newWords": word_count(final),
        "keyword": keyword,
        "tone": tone,
        "engine": engine,
        "seo": seo_extras(final, keyword),
    }


@app.get("/")
def health():
    return {
        "status": "ok",
        "detector": MODEL_ID,
        "groq_enabled": bool(GROQ_API_KEY),
        "endpoints": ["/api/detect", "/api/humanize"],
    }


@app.post("/predict")
def predict_legacy(item: DetectReq, x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="bad api key")
    text = item.text.strip()
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="text too short")
    out = clf(text[:6000])
    rows = out[0] if isinstance(out[0], list) else out
    return {"scores": {r["label"]: float(r["score"]) for r in rows}}

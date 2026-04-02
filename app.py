from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict
from dotenv import load_dotenv
import os
import time
import httpx

load_dotenv()

# -----------------------------
# Config
# -----------------------------
APP_API_KEY = os.getenv("SS_API_KEY", "")  # your own API key for participants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Simple in-memory rate limit (per process, global)
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
_bucket = {"ts_min": 0, "count": 0}

# Minimal participant registry for small volunteer study
# Replace these demo tokens before real deployment.
PARTICIPANTS = {
    "P001": {"token": "replace_with_real_token_001", "enabled": True},
    "P002": {"token": "replace_with_real_token_002", "enabled": True},
    "P003": {"token": "replace_with_real_token_003", "enabled": True},
    "P004": {"token": "replace_with_real_token_004", "enabled": True},
    "P005": {"token": "replace_with_real_token_005", "enabled": True},
    "P006": {"token": "replace_with_real_token_006", "enabled": True},
    "P007": {"token": "replace_with_real_token_007", "enabled": True},
    "P008": {"token": "replace_with_real_token_008", "enabled": True},
}


def rate_limit_ok() -> bool:
    now_min = int(time.time() // 60)
    if _bucket["ts_min"] != now_min:
        _bucket["ts_min"] = now_min
        _bucket["count"] = 0
    _bucket["count"] += 1
    return _bucket["count"] <= RATE_LIMIT_PER_MIN


# -----------------------------
# Schema
# -----------------------------
Strategy = Literal["UNDERSTANDING", "COMFORTING", "EVOKING", "SCAFFOLDING_HABITS"]
Tone = Literal["gentle", "neutral", "humorous", "strict"]
MentalState = Optional[Literal["stress", "boredom", "inertia", "other"]]


class GenerateRequest(BaseModel):
    # Minimal participant auth for volunteer study
    participant_id: str = Field(..., min_length=1, max_length=50)
    participant_token: str = Field(..., min_length=1, max_length=200)

    # Word slots / context
    time_local: Optional[str] = None        # e.g., "22:36"
    place: Optional[str] = None             # e.g., "library"
    total_today_min: Optional[int] = None   # computed
    since_last_open_min: Optional[int] = None  # computed

    mental_state: MentalState = None
    user_goals: List[str] = Field(default_factory=list)
    habit: Optional[str] = None

    # Control
    strategy: Strategy
    tone_style: Tone = "gentle"

    # Safety + output constraints
    max_words: int = 30
    num_sentences: int = 2
    gender_neutral: bool = True
    no_blame: bool = True
    no_exaggeration: bool = True
    no_medical: bool = True

    # Dev
    return_prompt: bool = False


class GenerateResponse(BaseModel):
    ok: bool
    message: str
    strategy: Strategy
    prompt_used: Optional[str] = None


# -----------------------------
# Security helpers
# -----------------------------
def require_key(x_api_key: str):
    if APP_API_KEY and x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def require_participant(participant_id: str, participant_token: str):
    participant = PARTICIPANTS.get(participant_id)

    if not participant:
        raise HTTPException(status_code=403, detail="Unknown participant")

    if not participant.get("enabled", False):
        raise HTTPException(status_code=403, detail="Participant disabled")

    if participant.get("token") != participant_token:
        raise HTTPException(status_code=403, detail="Invalid participant token")


# -----------------------------
# Prompt template
# -----------------------------
def build_prompt(req: GenerateRequest) -> Dict[str, str]:
    # 1) Task setup / background
    background = (
        "You are a digital wellbeing coach. Generate a short, persuasive, non-coercive message "
        "to help the user pause and choose intentionally, not to shame them."
    )

    # 2) Description of current context
    ctx_lines = []
    if req.time_local:
        ctx_lines.append(f"- Time: {req.time_local}")
    if req.place:
        ctx_lines.append(f"- Place: {req.place}")
    if req.total_today_min is not None:
        ctx_lines.append(f"- Total TikTok today: {req.total_today_min} minutes")
    if req.since_last_open_min is not None:
        ctx_lines.append(f"- Time since last open: {req.since_last_open_min} minutes")
    if req.mental_state:
        ctx_lines.append(f"- Mental state: {req.mental_state}")
    if req.user_goals:
        ctx_lines.append(f"- User goals: {', '.join(req.user_goals)}")
    if req.habit:
        ctx_lines.append(f"- Habit/context: {req.habit}")

    context = "Current user data:\n" + ("\n".join(ctx_lines) if ctx_lines else "- (no extra context)")

    # 3) Constraints
    rules = [
        f"- Output must be <= {req.max_words} words.",
        f"- Output must be exactly {req.num_sentences} sentences (no bullet points).",
        "- Provide one small alternative action the user can do now."
    ]

    if req.gender_neutral:
        rules.append("- Use gender-neutral language.")
    if req.no_exaggeration:
        rules.append("- Avoid exaggerated praise or dramatic wording.")
    if req.no_blame:
        rules.append("- Do not blame or judge the user.")
    if req.no_medical:
        rules.append("- Do not provide medical advice or diagnose conditions.")

    optimization = "Constraints:\n" + "\n".join(rules)

    # 4) Strategy instruction
    strategy_map = {
        "UNDERSTANDING": (
            "Use a caring tone. Acknowledge the situation and invite reflection with one gentle question."
        ),
        "COMFORTING": (
            "Use empathy to soothe stress, validate feelings, and suggest a small calming action."
        ),
        "EVOKING": (
            "Remind them of their goals and autonomy; invite them to do a tiny goal-aligned step first."
        ),
        "SCAFFOLDING_HABITS": (
            "Give a concrete micro-habit or environmental tweak to interrupt the loop."
        ),
    }

    strategy_text = f"Persuasion strategy: {req.strategy}\n{strategy_map[req.strategy]}"

    system_msg = (
        background + "\n\n"
        "You must follow the constraints strictly. Return only the final message text."
    )
    user_msg = context + "\n\n" + optimization + "\n\n" + strategy_text

    return {"system": system_msg, "user": user_msg}


# -----------------------------
# OpenAI-compatible call
# -----------------------------
async def call_llm(system: str, user: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")

    url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.6,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=25) as client:
        response = await client.post(url, json=payload, headers=headers)

        print("OPENAI STATUS:", response.status_code)
        if response.status_code >= 400:
            print("OPENAI ERROR BODY:", response.text)
            raise RuntimeError(f"OpenAI error {response.status_code}: {response.text}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected OpenAI response format: {e}; body={data}")


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="ScrollSanity LLM API", version="1.1.0")

# CORS is not critical for Android-only usage, but harmless to keep for now.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, x_api_key: str = Header(default="")):
    require_key(x_api_key)
    require_participant(req.participant_id, req.participant_token)

    if not rate_limit_ok():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    prompt = build_prompt(req)

    try:
        msg = await call_llm(prompt["system"], prompt["user"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

    return GenerateResponse(
        ok=True,
        message=msg,
        strategy=req.strategy,
        prompt_used=(prompt["system"] + "\n\n" + prompt["user"]) if req.return_prompt else None,
    )
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, TypedDict
from dotenv import load_dotenv
import os
import time
import httpx

load_dotenv()

# -----------------------------
# Config
# -----------------------------
APP_API_KEY = os.getenv("SS_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
DEBUG_RETURN_PROMPT = os.getenv("DEBUG_RETURN_PROMPT", "false").lower() == "true"

_bucket = {"ts_min": 0, "count": 0}


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
Strategy = Literal[
    "UNDERSTANDING",
    "COMFORTING",
    "EVOKING",
    "SCAFFOLDING_HABITS",
    "EMERGENCY",
    "REFLECTION",
]

Tone = Literal["gentle", "neutral", "humorous", "strict"]
MentalState = Optional[Literal["stress", "bored", "inertia", "other"]]


class GenerateRequest(BaseModel):
    time_local: Optional[str] = None
    total_today_min: Optional[int] = None
    since_last_open_min: Optional[int] = None

    mental_state: MentalState = None
    user_goals: List[str] = Field(default_factory=list)
    habit: Optional[str] = None

    strategy: Strategy
    tone_style: Tone = "gentle"


class GenerateResponse(BaseModel):
    ok: bool
    message: str
    strategy: Strategy
    prompt_used: Optional[str] = None


# -----------------------------
# Server-side generation policy
# -----------------------------
class StrategyPolicy(TypedDict):
    max_words: int
    num_sentences: int
    strategy_instruction: str


DEFAULT_GENDER_NEUTRAL = True
DEFAULT_NO_BLAME = True
DEFAULT_NO_EXAGGERATION = True
DEFAULT_NO_MEDICAL = True
DEFAULT_NO_DIAGNOSIS = True

STRATEGY_POLICIES: Dict[str, StrategyPolicy] = {
    "UNDERSTANDING": {
        "max_words": 30,
        "num_sentences": 2,
        "strategy_instruction": (
            "Use a caring tone. Acknowledge the situation and invite reflection with one gentle question."
        ),
    },
    "COMFORTING": {
        "max_words": 30,
        "num_sentences": 2,
        "strategy_instruction": (
            "Use empathy to soothe stress, validate feelings, and suggest a small calming action."
        ),
    },
    "EVOKING": {
        "max_words": 30,
        "num_sentences": 2,
        "strategy_instruction": (
            "Remind the user of their goals and autonomy; invite one tiny goal-aligned step first."
        ),
    },
    "SCAFFOLDING_HABITS": {
        "max_words": 30,
        "num_sentences": 2,
        "strategy_instruction": (
            "Give one concrete micro-habit or environmental tweak to interrupt the loop."
        ),
    },
    "EMERGENCY": {
        "max_words": 10,
        "num_sentences": 1,
        "strategy_instruction": (
            "Be extremely brief, calm, and direct. Focus only on interrupting the loop with one immediate action."
        ),
    },
    "REFLECTION": {
        "max_words": 40,
        "num_sentences": 2,
        "strategy_instruction": (
            "Invite brief self-reflection and intentional choice. Keep it warm, concise, and autonomy-supportive."
        ),
    },
}


def get_strategy_policy(strategy: Strategy) -> StrategyPolicy:
    return STRATEGY_POLICIES[strategy]


# -----------------------------
# Language inference
# -----------------------------
def infer_language(req: GenerateRequest) -> str:
    text_parts: List[str] = []

    if req.habit:
        text_parts.append(req.habit)

    if req.user_goals:
        text_parts.extend(req.user_goals)

    combined = " ".join(text_parts)

    if any('\u4e00' <= ch <= '\u9fff' for ch in combined):
        return "zh"

    chinese_punctuation = "，。！？；：、（）《》“”‘’"
    if any(ch in chinese_punctuation for ch in combined):
        return "zh"

    return "en"


# -----------------------------
# Security helpers
# -----------------------------
def require_key(x_api_key: str):
    if APP_API_KEY and x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# -----------------------------
# Prompt template
# -----------------------------
def build_prompt(req: GenerateRequest) -> Dict[str, str]:
    policy = get_strategy_policy(req.strategy)
    inferred_lang = infer_language(req)

    background = (
        "You are a digital wellbeing coach. Generate a short, persuasive, non-coercive message "
        "to help the user pause and choose intentionally. Your role is to support reflection and agency, "
        "not to shame, diagnose, or pressure the user."
    )

    ctx_lines = []

    if req.time_local:
        ctx_lines.append(f"- Local time: {req.time_local}")
    if req.total_today_min is not None:
        ctx_lines.append(f"- Total short-video/social media use today: {req.total_today_min} minutes")
    if req.since_last_open_min is not None:
        ctx_lines.append(f"- Time since last open: {req.since_last_open_min} minutes")
    if req.mental_state:
        ctx_lines.append(f"- Reported mental state: {req.mental_state}")
    if req.user_goals:
        ctx_lines.append(f"- User goals: {', '.join(req.user_goals)}")
    if req.habit:
        ctx_lines.append(f"- Habit context: {req.habit}")
    if req.tone_style:
        ctx_lines.append(f"- Preferred tone style: {req.tone_style}")

    context = "Current user context:\n" + ("\n".join(ctx_lines) if ctx_lines else "- (no extra context)")

    rules = [
        f"- Output must be no more than {policy['max_words']} words.",
        f"- Output must be exactly {policy['num_sentences']} sentence(s).",
        "- Do not use bullet points, numbering, emojis, quotation marks, or hashtags.",
        "- Include one small, realistic action the user can do right now.",
        "- Keep the message natural and conversational.",
    ]

    if DEFAULT_GENDER_NEUTRAL:
        rules.append("- Use gender-neutral language.")
    if DEFAULT_NO_EXAGGERATION:
        rules.append("- Avoid exaggerated praise, dramatic wording, or overstatement.")
    if DEFAULT_NO_BLAME:
        rules.append("- Do not blame, shame, scold, or judge the user.")
    if DEFAULT_NO_MEDICAL:
        rules.append("- Do not provide medical advice or mention treatment.")
    if DEFAULT_NO_DIAGNOSIS:
        rules.extend([
            "- Do not diagnose, assess, or label the user with any medical, psychiatric, or clinical condition.",
            "- Do not describe the user as addicted, dependent, compulsive, disordered, or symptomatic.",
            "- Do not imply certainty that the user has a condition or problem identity.",
        ])

    if inferred_lang == "zh":
        rules.extend([
            "- If the user's context contains Chinese, reply primarily in natural Simplified Chinese.",
            "- Avoid translation-like wording.",
            "- A small amount of English is acceptable only if necessary, but prefer Chinese overall.",
        ])
    else:
        rules.append("- Reply in natural English.")

    optimization = "Constraints:\n" + "\n".join(rules)

    strategy_text = (
        f"Persuasion strategy: {req.strategy}\n"
        f"{policy['strategy_instruction']}"
    )

    tone_map = {
        "gentle": "Use warm, soft, non-pushy wording.",
        "neutral": "Use calm, clear, balanced wording.",
        "humorous": "Use light, kind humor only if it stays respectful and subtle.",
        "strict": "Use firm but non-shaming wording. Stay respectful and brief.",
    }
    tone_text = f"Tone guidance: {tone_map[req.tone_style]}"

    language_text = (
        "Language preference: Simplified Chinese."
        if inferred_lang == "zh"
        else "Language preference: English."
    )

    system_msg = (
        background + "\n\n"
        "You must follow all constraints strictly. Return only the final message text, with no explanation."
    )

    user_msg = (
        context + "\n\n" +
        optimization + "\n\n" +
        strategy_text + "\n\n" +
        tone_text + "\n\n" +
        language_text
    )

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
app = FastAPI(title="ScrollSanity LLM API", version="1.4.0")

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
        prompt_used=(prompt["system"] + "\n\n" + prompt["user"]) if DEBUG_RETURN_PROMPT else None,
    )
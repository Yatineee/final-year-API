import os
import asyncio
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

async def main():
    url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one short sentence."}
        ],
        "temperature": 0.2,
    }

    print("KEY EXISTS:", bool(OPENAI_API_KEY))
    print("KEY PREFIX:", OPENAI_API_KEY[:12] if OPENAI_API_KEY else None)
    print("URL:", url)
    print("MODEL:", OPENAI_MODEL)

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, json=payload)
        print("STATUS:", r.status_code)
        print("BODY:", r.text)

asyncio.run(main())
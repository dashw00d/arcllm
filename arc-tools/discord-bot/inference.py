import logging

from openai import AsyncOpenAI

from config import LLAMA_SERVER_URL, MODEL, MAX_TOKENS

log = logging.getLogger("bot.inference")

client = AsyncOpenAI(
    base_url=f"{LLAMA_SERVER_URL}/v1",
    api_key="none",
    timeout=120.0,
    default_headers={"X-Priority": "high"},
)


async def chat(messages: list[dict], thinking: bool | None = None) -> str:
    """Send a chat completion request. Returns the assistant reply text."""
    extra_body = {}
    if thinking is True:
        extra_body["reasoning_budget"] = 4096
    else:
        extra_body["reasoning_budget"] = 0

    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.8,
            top_p=0.95,
            extra_body=extra_body,
        )
        content = resp.choices[0].message.content
        return content or "(empty response)"
    except Exception:
        log.exception("Inference request failed")
        return "Sorry, something went wrong with the model. Try again in a sec."

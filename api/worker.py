#!/usr/bin/env python3

import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-0.6B")
DTYPE = getattr(torch, os.getenv("TORCH_DTYPE", "float16"))
DEVICE = os.getenv("MODEL_DEVICE", "xpu")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "256"))
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() == "true"

app = FastAPI(title="Arc Worker", version="0.1.0")
tokenizer = None
model = None
generation_lock = threading.Lock()


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0)
    stream: bool = False


def build_prompt(messages: list[Message]) -> str:
    assert tokenizer is not None
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [message.model_dump() for message in messages],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
    return "\n".join(f"{msg.role}: {msg.content}" for msg in messages) + "\nassistant:"


def generate_text(prompt: str, max_tokens: int, temperature: float) -> tuple[str, dict[str, int]]:
    assert tokenizer is not None
    assert model is not None

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature

    with generation_lock:
        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)

    prompt_tokens = int(inputs["input_ids"].shape[1])
    generated_ids = outputs[0][prompt_tokens:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].lstrip()
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(generated_ids.shape[0]),
        "total_tokens": prompt_tokens + int(generated_ids.shape[0]),
    }
    return text, usage


def sse_chunk(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def chat_stream(prompt: str, request: ChatCompletionRequest) -> Iterator[bytes]:
    assert tokenizer is not None
    assert model is not None

    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generate_kwargs: dict[str, Any] = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": request.max_tokens,
        "do_sample": request.temperature > 0,
    }
    if request.temperature > 0:
        generate_kwargs["temperature"] = request.temperature

    def run_generation() -> None:
        with generation_lock:
            with torch.no_grad():
                model.generate(**generate_kwargs)

    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()

    yield sse_chunk(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model or MODEL_ID,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )
    for text in streamer:
        if not text:
            continue
        yield sse_chunk(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model or MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
        )
    thread.join()
    yield sse_chunk(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model or MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield b"data: [DONE]\n\n"


def completion_stream(prompt: str, request: CompletionRequest) -> Iterator[bytes]:
    assert tokenizer is not None
    assert model is not None

    created = int(time.time())
    completion_id = f"cmpl-{uuid.uuid4().hex}"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generate_kwargs: dict[str, Any] = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": request.max_tokens,
        "do_sample": request.temperature > 0,
    }
    if request.temperature > 0:
        generate_kwargs["temperature"] = request.temperature

    def run_generation() -> None:
        with generation_lock:
            with torch.no_grad():
                model.generate(**generate_kwargs)

    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()

    for text in streamer:
        if not text:
            continue
        yield sse_chunk(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": request.model or MODEL_ID,
                "choices": [{"index": 0, "text": text, "finish_reason": None}],
            }
        )
    thread.join()
    yield sse_chunk(
        {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": request.model or MODEL_ID,
            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
        }
    )
    yield b"data: [DONE]\n\n"


@app.on_event("startup")
def startup() -> None:
    global tokenizer
    global model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE)
    model.to(DEVICE)
    model.eval()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


@app.get("/v1/models")
def list_models() -> dict[str, list[dict[str, str]]]:
    return {
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "local",
            }
        ]
    }


@app.post("/v1/chat/completions", response_model=None)
def chat_completions(request: ChatCompletionRequest) -> Any:
    prompt = build_prompt(request.messages)
    if request.stream:
        return StreamingResponse(chat_stream(prompt, request), media_type="text/event-stream")
    text, usage = generate_text(prompt, request.max_tokens, request.temperature)
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": request.model or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


@app.post("/v1/completions", response_model=None)
def completions(request: CompletionRequest) -> Any:
    if request.stream:
        return StreamingResponse(
            completion_stream(request.prompt, request),
            media_type="text/event-stream",
        )
    text, usage = generate_text(request.prompt, request.max_tokens, request.temperature)
    created = int(time.time())
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": created,
        "model": request.model or MODEL_ID,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": usage,
    }

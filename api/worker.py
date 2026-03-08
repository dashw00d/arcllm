#!/usr/bin/env python3

import json
import os
import re
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-0.6B")
DTYPE = getattr(torch, os.getenv("TORCH_DTYPE", "float16"))
DEVICE = os.getenv("MODEL_DEVICE", "xpu")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "256"))
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() == "true"
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "1"))
MAX_QUEUE_DEPTH = int(os.getenv("MAX_QUEUE_DEPTH", "2"))
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

app = FastAPI(title="Arc Worker", version="0.2.0")
tokenizer = None
model = None


class WorkerOverloadedError(RuntimeError):
    pass


class RequestLimiter:
    def __init__(self, max_concurrent: int, max_queue_depth: int) -> None:
        self.max_concurrent = max(1, max_concurrent)
        self.max_queue_depth = max(0, max_queue_depth)
        self.condition = threading.Condition()
        self.active = 0
        self.waiting = 0

    def acquire(self) -> None:
        with self.condition:
            if self.active >= self.max_concurrent and self.waiting >= self.max_queue_depth:
                raise WorkerOverloadedError("worker queue is full")
            self.waiting += 1
            try:
                while self.active >= self.max_concurrent:
                    self.condition.wait()
                self.active += 1
            finally:
                self.waiting -= 1

    def release(self) -> None:
        with self.condition:
            self.active -= 1
            self.condition.notify()

    def snapshot(self) -> dict[str, int]:
        with self.condition:
            return {
                "active": self.active,
                "waiting": self.waiting,
                "max_concurrent": self.max_concurrent,
                "max_queue_depth": self.max_queue_depth,
                "available_slots": max(0, self.max_concurrent - self.active),
            }


request_limiter = RequestLimiter(MAX_CONCURRENT_REQUESTS, MAX_QUEUE_DEPTH)


@contextmanager
def generation_slot() -> Iterator[None]:
    try:
        request_limiter.acquire()
    except WorkerOverloadedError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "worker_queue_full",
                "message": str(exc),
                "queue": request_limiter.snapshot(),
            },
        ) from exc
    try:
        yield
    finally:
        request_limiter.release()


def acquire_generation_slot() -> None:
    try:
        request_limiter.acquire()
    except WorkerOverloadedError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "worker_queue_full",
                "message": str(exc),
                "queue": request_limiter.snapshot(),
            },
        ) from exc


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    reasoning_content: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[Message]
    max_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    seed: int | None = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    prompt: str
    max_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    seed: int | None = None


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text"} and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def normalize_tool_arguments(arguments: Any) -> Any:
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        return arguments
    return arguments


def normalize_tool_calls(tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool_call in tool_calls or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function", tool_call)
        name = function.get("name")
        if not name:
            continue
        normalized.append(
            {
                "id": tool_call.get("id"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": normalize_tool_arguments(function.get("arguments")),
                },
            }
        )
    return normalized


def normalize_messages(messages: list[Message]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        payload: dict[str, Any] = {
            "role": message.role,
            "content": content_to_text(message.content),
        }
        if message.reasoning_content:
            payload["reasoning_content"] = message.reasoning_content
        tool_calls = normalize_tool_calls(message.tool_calls)
        if tool_calls:
            payload["tool_calls"] = tool_calls
        normalized.append(payload)
    return normalized


def normalize_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            continue
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {"type": "object", "properties": {}}),
                },
            }
        )
    return normalized


def resolve_tools_for_request(request: ChatCompletionRequest) -> tuple[list[dict[str, Any]], list[str]]:
    tools = normalize_tools(request.tools)
    if not tools:
        return [], []

    instructions: list[str] = []
    tool_choice = request.tool_choice
    if tool_choice in (None, "auto"):
        pass
    elif tool_choice == "none":
        return [], []
    elif tool_choice == "required":
        instructions.append("You must return at least one tool call for the next assistant message.")
    elif isinstance(tool_choice, dict):
        function = tool_choice.get("function", {})
        function_name = function.get("name")
        if not function_name:
            raise HTTPException(status_code=400, detail="tool_choice.function.name is required")
        filtered = [tool for tool in tools if tool["function"]["name"] == function_name]
        if not filtered:
            raise HTTPException(status_code=400, detail=f"unknown tool_choice function: {function_name}")
        tools = filtered
        instructions.append(
            f"You must call the function {function_name} for the next assistant message."
        )
    else:
        raise HTTPException(status_code=400, detail=f"unsupported tool_choice: {tool_choice}")

    if request.parallel_tool_calls is False:
        instructions.append("Call at most one function in the next assistant message.")
    return tools, instructions


def apply_tool_instructions(
    messages: list[dict[str, Any]],
    instructions: list[str],
) -> list[dict[str, Any]]:
    if not instructions:
        return messages
    instruction_text = "\n".join(instructions)
    updated = [dict(message) for message in messages]
    if updated and updated[0]["role"] == "system":
        prefix = content_to_text(updated[0].get("content"))
        updated[0]["content"] = f"{prefix}\n\n{instruction_text}".strip()
        return updated
    return [{"role": "system", "content": instruction_text}, *updated]


def build_chat_prompt(request: ChatCompletionRequest) -> tuple[str, bool]:
    assert tokenizer is not None
    messages = normalize_messages(request.messages)
    tools, instructions = resolve_tools_for_request(request)
    messages = apply_tool_instructions(messages, instructions)

    if getattr(tokenizer, "chat_template", None):
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": ENABLE_THINKING,
        }
        if tools:
            kwargs["tools"] = tools
        return tokenizer.apply_chat_template(messages, **kwargs), bool(tools)
    return "\n".join(
        f"{message['role']}: {content_to_text(message.get('content'))}" for message in messages
    ) + "\nassistant:", bool(tools)


def build_generation_kwargs(
    inputs: Any,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    streamer: TextIteratorStreamer | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
    }
    if streamer is not None:
        kwargs["streamer"] = streamer
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    if seed is not None:
        torch.manual_seed(seed)
    return kwargs


def decode_generated_text(outputs: Any, prompt_tokens: int) -> tuple[str, int]:
    assert tokenizer is not None
    generated_ids = outputs[0][prompt_tokens:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, int(generated_ids.shape[0])


def build_usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def generate_text(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
) -> tuple[str, dict[str, int]]:
    assert tokenizer is not None
    assert model is not None

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    generate_kwargs = build_generation_kwargs(inputs, max_tokens, temperature, top_p, seed)

    with generation_slot():
        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)

    prompt_tokens = int(inputs["input_ids"].shape[1])
    text, completion_tokens = decode_generated_text(outputs, prompt_tokens)
    return text, build_usage(prompt_tokens, completion_tokens)


def sse_chunk(payload: dict[str, Any] | str) -> bytes:
    if isinstance(payload, str):
        return f"data: {payload}\n\n".encode("utf-8")
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def start_generation_stream(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
) -> tuple[TextIteratorStreamer, threading.Thread]:
    assert tokenizer is not None
    assert model is not None

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    acquire_generation_slot()
    generate_kwargs = build_generation_kwargs(
        inputs,
        max_tokens,
        temperature,
        top_p,
        seed,
        streamer=streamer,
    )

    def run_generation() -> None:
        try:
            with torch.no_grad():
                model.generate(**generate_kwargs)
        finally:
            request_limiter.release()

    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()
    return streamer, thread


def clean_generated_text(text: str) -> str:
    return THINK_BLOCK_RE.sub("", text).strip()


def format_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments if arguments is not None else {}, separators=(",", ":"))


def parse_tool_call(match_text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(match_text)
    except json.JSONDecodeError:
        return None
    function = payload.get("function", payload)
    name = function.get("name")
    if not name:
        return None
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": format_tool_arguments(function.get("arguments")),
        },
    }


def parse_assistant_output(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    cleaned = clean_generated_text(text)
    tool_calls: list[dict[str, Any]] = []
    text_parts: list[str] = []
    cursor = 0

    for match in TOOL_CALL_RE.finditer(cleaned):
        text_parts.append(cleaned[cursor:match.start()])
        tool_call = parse_tool_call(match.group(1).strip())
        if tool_call is None:
            text_parts.append(match.group(0))
        else:
            tool_calls.append(tool_call)
        cursor = match.end()

    text_parts.append(cleaned[cursor:])
    content = "".join(text_parts).strip() or None
    return content, tool_calls


def build_chat_completion(
    request: ChatCompletionRequest,
    completion_id: str,
    created: int,
    text: str,
    usage: dict[str, int],
) -> dict[str, Any]:
    content, tool_calls = parse_assistant_output(text)
    message: dict[str, Any] = {"role": "assistant", "content": content}
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


def plain_chat_stream(prompt: str, request: ChatCompletionRequest) -> Iterator[bytes]:
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    streamer, thread = start_generation_stream(
        prompt,
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.seed,
    )

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
    yield sse_chunk("[DONE]")


def tool_aware_chat_stream(prompt: str, request: ChatCompletionRequest) -> Iterator[bytes]:
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    streamer, thread = start_generation_stream(
        prompt,
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.seed,
    )
    text = "".join(chunk for chunk in streamer if chunk)
    thread.join()

    content, tool_calls = parse_assistant_output(text)
    yield sse_chunk(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model or MODEL_ID,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )
    if tool_calls:
        for index, tool_call in enumerate(tool_calls):
            yield sse_chunk(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model or MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": tool_call["id"],
                                        "type": "function",
                                        "function": tool_call["function"],
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
        yield sse_chunk(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model or MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }
        )
        yield sse_chunk("[DONE]")
        return

    if content:
        yield sse_chunk(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model or MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
        )
    yield sse_chunk(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model or MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield sse_chunk("[DONE]")


def completion_stream(prompt: str, request: CompletionRequest) -> Iterator[bytes]:
    created = int(time.time())
    completion_id = f"cmpl-{uuid.uuid4().hex}"
    streamer, thread = start_generation_stream(
        prompt,
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.seed,
    )

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
    yield sse_chunk("[DONE]")


@app.on_event("startup")
def startup() -> None:
    global tokenizer
    global model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE)
    model.to(DEVICE)
    model.eval()


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "queue": request_limiter.snapshot(),
    }


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
    prompt, tool_mode = build_chat_prompt(request)
    if request.stream:
        stream = tool_aware_chat_stream if tool_mode else plain_chat_stream
        return StreamingResponse(stream(prompt, request), media_type="text/event-stream")

    text, usage = generate_text(
        prompt,
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.seed,
    )
    return build_chat_completion(
        request,
        f"chatcmpl-{uuid.uuid4().hex}",
        int(time.time()),
        text,
        usage,
    )


@app.post("/v1/completions", response_model=None)
def completions(request: CompletionRequest) -> Any:
    if request.stream:
        return StreamingResponse(
            completion_stream(request.prompt, request),
            media_type="text/event-stream",
        )
    text, usage = generate_text(
        request.prompt,
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.seed,
    )
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model or MODEL_ID,
        "choices": [{"index": 0, "text": clean_generated_text(text), "finish_reason": "stop"}],
        "usage": usage,
    }

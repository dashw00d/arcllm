#!/usr/bin/env python3

import asyncio
import copy
import json
import os
import time
import uuid
from pathlib import Path
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


WORKER_URLS = [
    url.strip()
    for url in os.getenv(
        "WORKER_URLS",
        "http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003",
    ).split(",")
    if url.strip()
]
ROOT = Path(os.getenv("LLM_STACK_ROOT", Path(__file__).resolve().parent.parent))
RUNTIME_DIR = Path(os.getenv("ARCLLM_RUNTIME_DIR", ROOT / "runtime"))
RESPONSE_STORE_PATH = Path(
    os.getenv("ARCLLM_RESPONSE_STORE_PATH", RUNTIME_DIR / "responses.jsonl")
)

app = FastAPI(title="Arc Router", version="0.2.0")
workers = deque(WORKER_URLS)
worker_lock = asyncio.Lock()
response_store: dict[str, dict[str, Any]] = {}
response_lock = asyncio.Lock()
client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))


def ensure_runtime_dirs() -> None:
    RESPONSE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_response_store_from_disk() -> None:
    ensure_runtime_dirs()
    response_store.clear()
    if not RESPONSE_STORE_PATH.exists():
        return

    with RESPONSE_STORE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            response_id = record.get("response_id")
            response = record.get("response")
            conversation = record.get("conversation")
            if not response_id or not isinstance(response, dict) or not isinstance(conversation, list):
                continue
            response_store[response_id] = {
                "response": response,
                "conversation": conversation,
            }


def append_response_record(
    response_id: str,
    response: dict[str, Any],
    conversation: list[dict[str, Any]],
) -> None:
    ensure_runtime_dirs()
    record = {
        "response_id": response_id,
        "response": response,
        "conversation": conversation,
    }
    with RESPONSE_STORE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


async def next_worker() -> str:
    async with worker_lock:
        if not workers:
            raise HTTPException(status_code=503, detail="no workers configured")
        workers.rotate(-1)
        return workers[-1]


async def proxy_json(path: str, request: Request, body: Any) -> JSONResponse:
    worker = await next_worker()
    try:
        response = await client.request(
            request.method,
            f"{worker}{path}",
            params=request.query_params,
            json=body,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(status_code=response.status_code, content=response.json())


async def proxy_stream(path: str, request: Request, body: Any) -> StreamingResponse:
    worker = await next_worker()
    headers = {}
    if request.headers.get("content-type"):
        headers["content-type"] = request.headers["content-type"]

    try:
        upstream = await client.send(
            client.build_request(
                request.method,
                f"{worker}{path}",
                params=request.query_params,
                content=json.dumps(body).encode("utf-8"),
                headers=headers,
            ),
            stream=True,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    async def iter_bytes() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        iter_bytes(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in {"input_text", "output_text", "text"} and isinstance(
                    item.get("text"), str
                ):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                elif isinstance(item.get("output"), str):
                    parts.append(item["output"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("output"), str):
            return content["output"]
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def parse_response_input_item(item: Any) -> list[dict[str, Any]]:
    if isinstance(item, str):
        return [{"role": "user", "content": item}]
    if not isinstance(item, dict):
        return [{"role": "user", "content": str(item)}]

    item_type = item.get("type")
    if item_type == "function_call_output":
        call_id = item.get("call_id") or item.get("tool_call_id") or item.get("id")
        return [
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": content_to_text(item.get("output", item.get("content"))),
            }
        ]
    if item_type == "function_call":
        call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:24]}"
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": item.get("name"),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }
                ],
            }
        ]
    if item_type == "message":
        return [
            {
                "role": item.get("role", "user"),
                "content": content_to_text(item.get("content")),
            }
        ]

    message: dict[str, Any] = {
        "role": item.get("role", "user"),
        "content": content_to_text(item.get("content")),
    }
    if item.get("tool_calls"):
        message["tool_calls"] = item["tool_calls"]
    if item.get("tool_call_id"):
        message["tool_call_id"] = item["tool_call_id"]
    return [message]


def normalize_response_input(input_value: Any) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in ensure_list(input_value):
        messages.extend(parse_response_input_item(item))
    return messages


async def store_response(
    response_id: str,
    response: dict[str, Any],
    conversation: list[dict[str, Any]],
) -> None:
    async with response_lock:
        response_copy = copy.deepcopy(response)
        conversation_copy = copy.deepcopy(conversation)
        response_store[response_id] = {
            "response": response_copy,
            "conversation": conversation_copy,
        }
        append_response_record(response_id, response_copy, conversation_copy)


async def load_stored_response(response_id: str) -> dict[str, Any]:
    async with response_lock:
        record = response_store.get(response_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown response id: {response_id}")
    return copy.deepcopy(record)


@app.on_event("startup")
async def startup() -> None:
    async with response_lock:
        load_response_store_from_disk()


async def build_responses_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    previous_response_id = body.get("previous_response_id")
    if previous_response_id:
        previous = await load_stored_response(previous_response_id)
        messages.extend(previous["conversation"])

    instructions = content_to_text(body.get("instructions")).strip()
    if instructions:
        if messages and messages[0].get("role") == "system":
            existing = content_to_text(messages[0].get("content")).strip()
            messages[0]["content"] = f"{existing}\n\n{instructions}".strip()
        else:
            messages.insert(0, {"role": "system", "content": instructions})

    messages.extend(normalize_response_input(body.get("input")))
    return messages


def build_chat_request(body: dict[str, Any], messages: list[dict[str, Any]], stream: bool) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
        "stream": stream,
        "max_tokens": body.get("max_output_tokens", body.get("max_tokens", 256)),
        "temperature": body.get("temperature", 0.0),
        "top_p": body.get("top_p", 1.0),
    }
    if "tools" in body:
        request["tools"] = body["tools"]
    if "tool_choice" in body:
        request["tool_choice"] = body["tool_choice"]
    if "parallel_tool_calls" in body:
        request["parallel_tool_calls"] = body["parallel_tool_calls"]
    if "seed" in body:
        request["seed"] = body["seed"]
    return request


def response_output_from_chat_message(message: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    output: list[dict[str, Any]] = []
    output_text = content_to_text(message.get("content")).strip()
    if output_text:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": output_text,
                        "annotations": [],
                    }
                ],
            }
        )
    for tool_call in message.get("tool_calls", []) or []:
        function = tool_call.get("function", {})
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "type": "function_call",
                "call_id": tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "name": function.get("name"),
                "arguments": function.get("arguments", "{}"),
                "status": "completed",
            }
        )
    return output, output_text


def build_responses_object(
    response_id: str,
    body: dict[str, Any],
    message: dict[str, Any],
    usage: dict[str, Any] | None,
    created_at: int,
) -> dict[str, Any]:
    output, output_text = response_output_from_chat_message(message)
    response: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": body.get("model"),
        "output": output,
        "output_text": output_text or None,
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "store": body.get("store", True),
        "tools": body.get("tools", []),
    }
    if "tool_choice" in body:
        response["tool_choice"] = body["tool_choice"]
    if body.get("previous_response_id"):
        response["previous_response_id"] = body["previous_response_id"]
    if usage:
        response["usage"] = {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
    return response


def build_conversation_with_assistant(
    messages: list[dict[str, Any]],
    message: dict[str, Any],
) -> list[dict[str, Any]]:
    conversation = copy.deepcopy(messages)
    conversation.append(
        {
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
        }
    )
    return conversation


async def create_response_json(body: dict[str, Any]) -> JSONResponse:
    messages = await build_responses_messages(body)
    chat_request = build_chat_request(body, messages, stream=False)
    worker = await next_worker()
    try:
        upstream = await client.post(f"{worker}/v1/chat/completions", json=chat_request)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    payload = upstream.json()
    if upstream.status_code >= 400:
        return JSONResponse(status_code=upstream.status_code, content=payload)

    assistant_message = payload["choices"][0]["message"]
    response_id = f"resp_{uuid.uuid4().hex}"
    response = build_responses_object(
        response_id,
        body,
        assistant_message,
        payload.get("usage"),
        int(time.time()),
    )
    conversation = build_conversation_with_assistant(messages, assistant_message)
    if body.get("store", True):
        await store_response(response_id, response, conversation)
    return JSONResponse(status_code=200, content=response)


def sse_bytes(payload: dict[str, Any] | str) -> bytes:
    if isinstance(payload, str):
        return f"data: {payload}\n\n".encode("utf-8")
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


async def iter_sse_events(upstream: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    async for line in upstream.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        yield json.loads(data)


async def create_response_stream(body: dict[str, Any]) -> StreamingResponse | JSONResponse:
    messages = await build_responses_messages(body)
    chat_request = build_chat_request(body, messages, stream=True)
    worker = await next_worker()
    try:
        upstream = await client.send(
            client.build_request("POST", f"{worker}/v1/chat/completions", json=chat_request),
            stream=True,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if upstream.status_code >= 400:
        try:
            payload = await upstream.aread()
            content = json.loads(payload.decode("utf-8"))
        finally:
            await upstream.aclose()
        return JSONResponse(status_code=upstream.status_code, content=content)

    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())

    async def event_stream() -> AsyncIterator[bytes]:
        accumulated_text = ""
        tool_calls: list[dict[str, Any]] = []
        finish_reason = "stop"

        yield sse_bytes(
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "model": body.get("model"),
                    "status": "in_progress",
                },
            }
        )

        try:
            async for event in iter_sse_events(upstream):
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        accumulated_text += delta["content"]
                        yield sse_bytes(
                            {
                                "type": "response.output_text.delta",
                                "response_id": response_id,
                                "delta": delta["content"],
                            }
                        )
                    for tool_call in delta.get("tool_calls", []) or []:
                        function = tool_call.get("function", {})
                        item = {
                            "id": f"fc_{uuid.uuid4().hex[:24]}",
                            "type": "function_call",
                            "call_id": tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                            "name": function.get("name"),
                            "arguments": function.get("arguments", "{}"),
                            "status": "completed",
                        }
                        tool_calls.append(
                            {
                                "id": item["call_id"],
                                "type": "function",
                                "function": {
                                    "name": item["name"],
                                    "arguments": item["arguments"],
                                },
                            }
                        )
                        yield sse_bytes(
                            {
                                "type": "response.output_item.added",
                                "response_id": response_id,
                                "output_index": len(tool_calls) - 1,
                                "item": item,
                            }
                        )
                        yield sse_bytes(
                            {
                                "type": "response.output_item.done",
                                "response_id": response_id,
                                "output_index": len(tool_calls) - 1,
                                "item": item,
                            }
                        )
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        finally:
            await upstream.aclose()

        assistant_message: dict[str, Any] = {"role": "assistant", "content": accumulated_text or None}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        response = build_responses_object(
            response_id,
            body,
            assistant_message,
            usage=None,
            created_at=created_at,
        )
        response["status"] = "completed"
        response["finish_reason"] = finish_reason
        conversation = build_conversation_with_assistant(messages, assistant_message)
        if body.get("store", True):
            await store_response(response_id, response, conversation)

        if accumulated_text:
            yield sse_bytes(
                {
                    "type": "response.output_text.done",
                    "response_id": response_id,
                    "text": accumulated_text,
                }
            )
        yield sse_bytes({"type": "response.completed", "response": response})
        yield sse_bytes("[DONE]")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    statuses = []
    for worker in WORKER_URLS:
        try:
            response = await client.get(f"{worker}/healthz")
            statuses.append({"worker": worker, "status": response.json()})
        except httpx.HTTPError as exc:
            statuses.append({"worker": worker, "error": str(exc)})
    return {"workers": statuses}


@app.get("/v1/models")
async def models() -> JSONResponse:
    if not WORKER_URLS:
        raise HTTPException(status_code=503, detail="no workers configured")
    response = await client.get(f"{WORKER_URLS[0]}/v1/models")
    return JSONResponse(status_code=response.status_code, content=response.json())


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> Any:
    body = json.loads(await request.body())
    if body.get("stream"):
        return await proxy_stream("/v1/chat/completions", request, body)
    return await proxy_json("/v1/chat/completions", request, body)


@app.post("/v1/completions", response_model=None)
async def completions(request: Request) -> Any:
    body = json.loads(await request.body())
    if body.get("stream"):
        return await proxy_stream("/v1/completions", request, body)
    return await proxy_json("/v1/completions", request, body)


@app.post("/v1/responses", response_model=None)
async def responses(request: Request) -> Any:
    body = json.loads(await request.body())
    if body.get("stream"):
        return await create_response_stream(body)
    return await create_response_json(body)


@app.get("/v1/responses/{response_id}")
async def get_response(response_id: str) -> JSONResponse:
    record = await load_stored_response(response_id)
    return JSONResponse(status_code=200, content=record["response"])

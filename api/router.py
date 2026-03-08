#!/usr/bin/env python3

import asyncio
import os
from collections import deque
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


WORKER_URLS = [
    url.strip()
    for url in os.getenv(
        "WORKER_URLS",
        "http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003",
    ).split(",")
    if url.strip()
]

app = FastAPI(title="Arc Router", version="0.1.0")
workers = deque(WORKER_URLS)
worker_lock = asyncio.Lock()
client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))


async def next_worker() -> str:
    async with worker_lock:
        if not workers:
            raise HTTPException(status_code=503, detail="no workers configured")
        workers.rotate(-1)
        return workers[-1]


async def proxy_json(path: str, request: Request) -> JSONResponse:
    worker = await next_worker()
    try:
        response = await client.request(
            request.method,
            f"{worker}{path}",
            params=request.query_params,
            json=await request.json(),
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(status_code=response.status_code, content=response.json())


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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    return await proxy_json("/v1/chat/completions", request)


@app.post("/v1/completions")
async def completions(request: Request) -> JSONResponse:
    return await proxy_json("/v1/completions", request)

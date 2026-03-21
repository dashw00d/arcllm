#!/usr/bin/env python3
"""arcllm-proxy — Lazy-loading reverse proxy for llama-server.

Behaves like Ollama: listens immediately, loads models on first request,
unloads after idle timeout. No VRAM used until a request arrives.

Model registry is defined in TOML config or env vars.
"""

import http.client
import http.server
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

# ── Configuration ──────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
LLAMA_SERVER = ROOT / "llama.cpp" / "build-sycl" / "bin" / "llama-server"
CACHE_MANAGER = ROOT / "scripts" / "cache-manager.sh"
LISTEN_PORT = int(os.environ.get("ARCLLM_PORT", 11435))
LISTEN_HOST = os.environ.get("ARCLLM_HOST", "0.0.0.0")
BACKEND_PORT = int(os.environ.get("ARCLLM_BACKEND_PORT", 18400))
IDLE_TIMEOUT = int(os.environ.get("ARCLLM_IDLE_TIMEOUT", 0))  # seconds, 0 = never unload
LOG_FILE = os.environ.get("ARCLLM_LOGFILE", "/tmp/arcllm-server.log")

# SYCL env (set before spawning llama-server)
SYCL_ENV = {
    "GGML_SYCL_DISABLE_GRAPH": "1",
    "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "0",
    "ZE_AFFINITY_MASK": "0,1,2",
    "GGML_SYCL_FUSED_MMQ": "1",
}

# ── Model Registry ────────────────────────────────────────────────────────

MODELS = {}


def _register(name, path, flags, aliases=None):
    # Parse -np from flags to know how many slots this model has
    n_parallel = 1
    parts = flags.split()
    for i, p in enumerate(parts):
        if p == "-np" and i + 1 < len(parts):
            n_parallel = int(parts[i + 1])
    entry = {"name": name, "path": str(path), "flags": flags, "aliases": aliases or [], "n_parallel": n_parallel}
    MODELS[name] = entry
    for a in entry["aliases"]:
        MODELS[a] = entry


SLOT_CACHE = str(ROOT / "cache" / "slots") + "/"

# ── Qwen3-32B Q4_K_M (dense, DEFAULT) ────────────────────────────────────
# 19GB model, 29GB free for KV. np=4 → slot 0 for Discord, slots 1-3 for churner.
# FUSED_MMQ gives +25% at high batch (global SYCL_ENV).
# Test: find max context that fits at np=4 and np=8.
_register(
    "qwen3-32b",
    ROOT / "models/Qwen/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf",
    f"--split-mode layer -ngl 999 --tensor-split 1,1,1"
    f" -c 32768 -fa on"
    f" -np 4 --slot-save-path {SLOT_CACHE}"
    f" --reasoning-budget 0",
    aliases=["32b", "qwen3-32b-q4", "default"],
)

# Single-slot reasoning variant — for interactive use where quality > throughput
_register(
    "qwen3-32b-fast",
    ROOT / "models/Qwen/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf",
    f"--split-mode layer -ngl 999 --tensor-split 1,1,1 -c 32768 -fa on -np 1 --slot-save-path {SLOT_CACHE} --reasoning-budget 200",
    aliases=["qwen3-32b-think", "qwen3-32b-reasoning"],
)

# ── Qwen3-30B-A3B MoE ───────────────────────────────────────────────────
# Abliterated model — fast MoE but outputs thinking as plain text (no <think> tags).
# Use for churning (short outputs where thinking leak doesn't matter).
# np=4 c=8192 → 2048 tokens/slot. -fa off: IGC crashes on MoE + flash attention.
_register(
    "qwen3-30b-moe",
    ROOT / "models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf",
    f"--split-mode layer -ngl 99 --tensor-split 1,1,1"
    f" -c 8192 -fa off"
    f" -np 4 --no-warmup --slot-save-path {SLOT_CACHE}"
    f" --reasoning-budget 0",
    aliases=["qwen3-30b", "30b-moe", "moe"],
)

# ── Backend Manager ────────────────────────────────────────────────────────

log = logging.getLogger("arcllm")


def _cache_mode() -> str:
    """Derive cache mode string from current SYCL_ENV (e.g. 'batched_nograph')."""
    if SYCL_ENV.get("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS", "1") == "0":
        mode = "batched"
    else:
        mode = "immediate"
    if SYCL_ENV.get("GGML_SYCL_DISABLE_GRAPH", "1") == "0":
        mode += "_graph"
    else:
        mode += "_nograph"
    return mode


def _cache_restore():
    """Restore L0 compiler cache before backend launch."""
    mode = _cache_mode()
    log.info("Restoring compiler cache (mode=%s)", mode)
    try:
        env = os.environ.copy()
        env.update(SYCL_ENV)
        r = subprocess.run(
            ["bash", str(CACHE_MANAGER), "restore", mode],
            env=env, capture_output=True, text=True, timeout=30,
        )
        if r.stdout.strip():
            for line in r.stdout.strip().splitlines():
                log.info("cache-mgr: %s", line)
        if r.returncode != 0:
            log.warning("cache-manager restore failed (rc=%d): %s", r.returncode, r.stderr.strip())
    except Exception as e:
        log.warning("cache-manager restore error: %s", e)


def _cache_save():
    """Save L0 compiler cache after healthy startup or on clean shutdown."""
    mode = _cache_mode()
    log.info("Saving compiler cache (mode=%s)", mode)
    try:
        env = os.environ.copy()
        env.update(SYCL_ENV)
        r = subprocess.run(
            ["bash", str(CACHE_MANAGER), "save", mode],
            env=env, capture_output=True, text=True, timeout=30,
        )
        if r.stdout.strip():
            for line in r.stdout.strip().splitlines():
                log.info("cache-mgr: %s", line)
        if r.returncode != 0:
            log.warning("cache-manager save failed (rc=%d): %s", r.returncode, r.stderr.strip())
    except Exception as e:
        log.warning("cache-manager save error: %s", e)


class BackendManager:
    """Manages a single llama-server subprocess, loading/unloading on demand."""

    def __init__(self):
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.current_model: str | None = None  # canonical name
        self.last_request_time = 0.0
        self._idle_thread: threading.Thread | None = None
        self._stopping = False
        self._compiler_cache_healthy = False

    def ensure_model(self, model_id: str, timeout: float = 300) -> bool:
        """Ensure the given model is loaded. Blocks until ready or timeout."""
        entry = MODELS.get(model_id)
        if not entry:
            return False

        canonical = entry["name"]

        with self.lock:
            self.last_request_time = time.monotonic()

            # Already loaded
            if self.current_model == canonical and self.process and self.process.poll() is None:
                return True

            # Wrong model or dead process — stop and start
            self._stop_backend_locked()
            ok = self._start_backend_locked(entry, timeout)
            if ok:
                gate.set_max_active(entry.get("n_parallel", 1))
            return ok

    def _start_backend_locked(self, entry: dict, timeout: float) -> bool:
        model_path = entry["path"]
        if not Path(model_path).exists():
            log.error("Model file not found: %s", model_path)
            return False

        canonical = entry["name"]
        flags = entry["flags"]

        cmd = [
            str(LLAMA_SERVER),
            "--model", model_path,
            "--host", "127.0.0.1",
            "--port", str(BACKEND_PORT),
            "--alias", canonical,
        ] + flags.split()

        env = os.environ.copy()
        env.update(SYCL_ENV)

        self._compiler_cache_healthy = False

        # Restore mode-specific L0 compiler cache before spawning backend
        _cache_restore()

        log_fh = open(LOG_FILE, "a")
        log.info("Starting llama-server for %s (pid will follow)", canonical)
        log_fh.write(f"\n{'='*60}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading model: {canonical}\n{'='*60}\n")
        log_fh.flush()

        self.process = subprocess.Popen(
            cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
        )
        log_fh.close()  # child inherited the fd, parent can close its copy
        log.info("llama-server started (PID %d) for %s", self.process.pid, canonical)
        self.current_model = canonical

        # Wait for health
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                log.error("llama-server died during startup (exit %d)", self.process.returncode)
                self._stop_backend_locked()
                return False
            try:
                conn = http.client.HTTPConnection("127.0.0.1", BACKEND_PORT, timeout=2)
                conn.request("GET", "/health")
                resp = conn.getresponse()
                body = json.loads(resp.read())
                conn.close()
                if body.get("status") == "ok":
                    log.info("Model %s ready", canonical)
                    # Save compiler cache only after a confirmed healthy startup.
                    _cache_save()
                    self._compiler_cache_healthy = True
                    self._slot_restore(canonical, entry.get("n_parallel", 1))
                    self._start_idle_watcher()
                    return True
            except Exception:
                pass
            time.sleep(2)

        log.error("Timeout waiting for %s to load", canonical)
        self._stop_backend_locked()
        return False

    @staticmethod
    def _slot_filename(model_name: str, slot_id: int) -> str:
        return f"{model_name}.slot{slot_id}.bin"

    def _slot_save_one(self, model_name: str, slot_id: int) -> bool:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", BACKEND_PORT, timeout=60)
            body = json.dumps({"filename": self._slot_filename(model_name, slot_id)}).encode()
            conn.request(
                "POST",
                f"/slots/{slot_id}?action=save",
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            resp = conn.getresponse()
            data = json.loads(resp.read())
            conn.close()
            if resp.status == 200:
                size_mb = data.get("size_mb", "?")
                tokens = data.get("tokens_saved", data.get("n_tokens", "?"))
                log.info("Slot %d saved: %s (%s MB, %s tokens)", slot_id, self._slot_filename(model_name, slot_id), size_mb, tokens)
                return True
            log.warning("Slot %d save failed: %s", slot_id, data)
        except Exception as e:
            log.warning("Slot %d save error: %s", slot_id, e)
        return False

    def _slot_save(self, model_name: str):
        """Save KV caches for every configured slot."""
        n_parallel = MODELS.get(model_name, {}).get("n_parallel", 1)
        saved = 0
        for slot_id in range(n_parallel):
            if self._slot_save_one(model_name, slot_id):
                saved += 1
        log.info("Saved %d/%d slots for %s", saved, n_parallel, model_name)
        return saved > 0

    def _slot_restore_one(self, model_name: str, slot_id: int, filename: str) -> bool:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", BACKEND_PORT, timeout=120)
            body = json.dumps({"filename": filename}).encode()
            conn.request(
                "POST",
                f"/slots/{slot_id}?action=restore",
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            resp = conn.getresponse()
            data = json.loads(resp.read())
            conn.close()
            if resp.status == 200:
                tokens = data.get("tokens_restored", data.get("n_tokens", "?"))
                log.info("Slot %d restored: %s (%s tokens)", slot_id, filename, tokens)
                return True
            log.warning("Slot %d restore failed from %s: %s", slot_id, filename, data)
        except Exception as e:
            log.warning("Slot %d restore error from %s: %s", slot_id, filename, e)
        return False

    def _slot_restore(self, model_name: str, n_parallel: int):
        """Restore KV caches for every configured slot, with slot-0 legacy fallback."""
        restored = 0
        for slot_id in range(n_parallel):
            candidates = [self._slot_filename(model_name, slot_id)]
            if slot_id == 0:
                candidates.append(f"{model_name}.bin")

            for filename in candidates:
                cache_file = Path(str(SLOT_CACHE) + filename)
                if not cache_file.exists():
                    continue
                if self._slot_restore_one(model_name, slot_id, filename):
                    restored += 1
                    break
            else:
                log.info("No cached slot file for %s slot %d", model_name, slot_id)

        log.info("Restored %d/%d slots for %s", restored, n_parallel, model_name)
        return restored > 0

    @staticmethod
    def _hw_reset_gpus():
        """Hardware-level GPU reset via sysfs after DEVICE_LOST."""
        for p in sorted(Path("/sys/class/drm").glob("card*/device/reset")):
            try:
                p.write_text("1")
            except Exception:
                pass
        # Clear Level Zero shader cache (may be corrupt after crash).
        for d in [Path("/tmp/neo_compiler_cache"), Path("/tmp/opencl_cache")]:
            if d.exists():
                subprocess.run(["rm", "-rf", str(d)], capture_output=True)

    def _stop_backend_locked(self):
        crashed = self.process and self.process.poll() is not None
        if self.process and self.process.poll() is None:
            # Save KV cache before stopping
            if self.current_model:
                self._slot_save(self.current_model)
            # Save compiler cache only after a known-healthy startup.
            if self._compiler_cache_healthy:
                _cache_save()
            else:
                log.info("Skipping compiler cache save for %s; startup never reached healthy state", self.current_model)
            log.info("Stopping llama-server (PID %d, model %s)", self.process.pid, self.current_model)
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        elif crashed:
            log.warning("llama-server crashed (exit %d) — performing GPU hardware reset",
                        self.process.returncode if self.process else -1)
            self._hw_reset_gpus()
            time.sleep(5)
        self.process = None
        self.current_model = None
        self._compiler_cache_healthy = False

    def stop(self):
        self._stopping = True
        with self.lock:
            self._stop_backend_locked()

    def touch(self):
        self.last_request_time = time.monotonic()

    def _start_idle_watcher(self):
        if IDLE_TIMEOUT <= 0 or (self._idle_thread and self._idle_thread.is_alive()):
            return
        self._idle_thread = threading.Thread(target=self._idle_loop, daemon=True)
        self._idle_thread.start()

    def _idle_loop(self):
        while not self._stopping:
            time.sleep(30)
            if self.current_model is None:
                return
            idle = time.monotonic() - self.last_request_time
            if idle >= IDLE_TIMEOUT:
                log.info("Idle timeout (%.0fs), unloading %s", idle, self.current_model)
                with self.lock:
                    self._stop_backend_locked()
                return

    def status(self) -> dict:
        with self.lock:
            if self.process and self.process.poll() is None:
                return {
                    "status": "loaded",
                    "model": self.current_model,
                    "pid": self.process.pid,
                    "idle_seconds": int(time.monotonic() - self.last_request_time),
                }
            return {"status": "idle", "model": None}


backend = BackendManager()

# ── Priority Gate ─────────────────────────────────────────────────────────
# High-priority requests (Discord bot) go first. Low-priority (churner)
# wait while any high-priority request is pending or in-flight.
# This prevents the churner from starving interactive users.


class PriorityGate:
    """Thread-safe gate that gives high-priority requests precedence.

    With max_active=1 (np=1): fully serializes, high jumps the queue.
    With max_active=2 (np=2): allows 2 concurrent requests (one per slot),
    but high-priority still preempts queued low-priority requests.
    """

    def __init__(self, max_active: int = 1):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._max_active = max_active
        self._high_pending = 0   # high-priority requests waiting or in-flight
        self._active = 0         # total requests currently forwarded to backend

    def set_max_active(self, n: int):
        with self._cond:
            self._max_active = n
            self._cond.notify_all()

    def acquire(self, priority: str, timeout: float = 300) -> bool:
        """Block until this request is allowed to proceed. Returns False on timeout."""
        with self._cond:
            if priority == "high":
                self._high_pending += 1
                # High-priority waits only for a free slot
                if not self._cond.wait_for(lambda: self._active < self._max_active, timeout=timeout):
                    self._high_pending -= 1
                    return False
                self._active += 1
                return True
            else:
                # Low-priority waits for: a free slot AND no high-priority pending
                if not self._cond.wait_for(
                    lambda: self._active < self._max_active and self._high_pending == 0,
                    timeout=timeout,
                ):
                    return False
                self._active += 1
                return True

    def release(self, priority: str):
        with self._cond:
            self._active -= 1
            if priority == "high":
                self._high_pending -= 1
            self._cond.notify_all()

    def status(self) -> dict:
        with self._lock:
            return {
                "high_pending": self._high_pending,
                "active": self._active,
                "max_active": self._max_active,
            }


gate = PriorityGate()

# ── HTTP Handler ───────────────────────────────────────────────────────────


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    """Reverse proxy that lazy-loads models on demand."""

    # Suppress default log spam
    def log_message(self, format, *args):
        log.debug(format, *args)

    def do_GET(self):
        if self.path == "/health":
            status = backend.status()
            status["queue"] = gate.status()
            self._respond_json(200, status)
            return
        if self.path in ("/v1/models", "/api/tags"):
            self._handle_models()
            return
        # Proxy other GETs
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_OPTIONS(self):
        self._proxy()

    def _handle_models(self):
        """Return available models without loading anything."""
        seen = set()
        models = []
        for key, entry in MODELS.items():
            if entry["name"] in seen:
                continue
            seen.add(entry["name"])
            exists = Path(entry["path"]).exists()
            models.append({
                "id": entry["name"],
                "object": "model",
                "aliases": entry["aliases"],
                "available": exists,
                "loaded": entry["name"] == backend.current_model,
            })
        self._respond_json(200, {"object": "list", "data": models})

    def _proxy(self):
        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # Priority: "high" for interactive (Discord), "low" for background (churner)
        priority = self.headers.get("X-Priority", "low").lower()
        if priority not in ("high", "low"):
            priority = "low"

        # Log request, inject slot assignment based on priority
        model_id = None
        if body:
            try:
                req = json.loads(body)
                model_id = req.get("model")
                msgs = req.get("messages", [])
                stream = req.get("stream", False)
                max_tok = req.get("max_tokens")

                # Slot assignment for multi-slot models (np>1):
                #   high → slot 0 (Discord gets consistent KV cache)
                #   low  → no pin (server assigns from any available slot)
                # Skip for single-slot models to avoid "slot not found" errors.
                n_par = MODELS.get(model_id, {}).get("n_parallel", 1) if model_id else 1
                if n_par > 1 and "id_slot" not in req and priority == "high":
                    req["id_slot"] = 0
                    body = json.dumps(req).encode()

                log.info("REQ %s model=%s msgs=%d stream=%s max_tokens=%s priority=%s slot=%s path=%s",
                         self.command, model_id, len(msgs), stream, max_tok, priority, req.get("id_slot"), self.path)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # For non-model requests, if backend is running just proxy through
        if not model_id:
            if backend.current_model and backend.process and backend.process.poll() is None:
                self._gated_forward(body, priority)
                return
            self._respond_json(503, {"error": {"message": "No model loaded. Send a request with a 'model' field to load one.", "type": "unavailable_error"}})
            return

        # Resolve model
        if model_id not in MODELS:
            # Try fuzzy match
            for key in MODELS:
                if model_id.lower() in key.lower() or key.lower() in model_id.lower():
                    model_id = key
                    break
            else:
                self._respond_json(404, {"error": {"message": f"Unknown model: {model_id}. GET /v1/models to see available.", "type": "not_found"}})
                return

        # Ensure model is loaded
        if not backend.ensure_model(model_id):
            self._respond_json(503, {"error": {"message": f"Failed to load model: {model_id}. Check {LOG_FILE}", "type": "unavailable_error"}})
            return

        self._gated_forward(body, priority)

    def _gated_forward(self, body: bytes, priority: str):
        """Acquire priority gate, forward to backend, release gate."""
        if not gate.acquire(priority, timeout=300):
            self._respond_json(503, {"error": {"message": "Request queue timeout", "type": "queue_error"}})
            return
        try:
            self._forward(body)
        finally:
            gate.release(priority)

    def _forward(self, body: bytes):
        """Forward request to backend llama-server, streaming the response."""
        backend.touch()
        try:
            fwd_headers = {k: v for k, v in self.headers.items()
                           if k.lower() not in ("host", "transfer-encoding")}
            if body:
                fwd_headers["Content-Length"] = str(len(body))
            conn = http.client.HTTPConnection("127.0.0.1", BACKEND_PORT, timeout=300)
            conn.request(self.command, self.path, body=body, headers=fwd_headers)
            resp = conn.getresponse()
        except Exception as e:
            self._respond_json(502, {"error": {"message": f"Backend error: {e}", "type": "proxy_error"}})
            return

        # Send response headers
        self.send_response(resp.status)
        for key, val in resp.getheaders():
            if key.lower() not in ("transfer-encoding",):
                self.send_header(key, val)
        self.end_headers()

        # Stream response body
        try:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, http.client.IncompleteRead, OSError):
            pass
        finally:
            conn.close()

    def _respond_json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ThreadedHTTPServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    def shutdown(signum, frame):
        log.info("Shutting down (signal %d)", signum)
        backend.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    server = ThreadedHTTPServer((LISTEN_HOST, LISTEN_PORT), ProxyHandler)
    log.info("arcllm-proxy listening on %s:%d (backend port %d)", LISTEN_HOST, LISTEN_PORT, BACKEND_PORT)
    log.info("Idle timeout: %s", f"{IDLE_TIMEOUT}s" if IDLE_TIMEOUT > 0 else "disabled")
    log.info("Models: %s", ", ".join(e["name"] for e in {id(v): v for v in MODELS.values()}.values()))
    log.info("No model loaded — waiting for requests...")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        backend.stop()
        server.server_close()


if __name__ == "__main__":
    main()

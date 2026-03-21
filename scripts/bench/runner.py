"""Server lifecycle + request firing + utilization capture."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import BenchConfig
from .env import BASE_ENV
from .monitor import Monitor, Utilization

LOG_DIR = Path("/tmp/bench_logs")
LOG_DIR.mkdir(exist_ok=True)

# GPU device reset paths (Arc A770 on i915 driver).
GPU_RESET_PATHS = sorted(Path("/sys/class/drm").glob("card*/device/reset"))


@dataclass
class BenchResult:
    name: str
    config: BenchConfig
    completed: int = 0
    failed: int = 0
    total_tokens: int = 0
    wall_time: float = 0.0
    total_tps: float = 0.0
    utilization: Utilization = field(default_factory=Utilization.empty)
    per_request: list[dict] = field(default_factory=list)
    error: str = ""

    def row(self) -> str:
        s = f"{self.completed}/{self.config.concurrent}"
        u = self.utilization.summary() if self.utilization.samples else ""
        e = f"  ERR: {self.error}" if self.error else ""
        return (f"{self.name:<28s} {s:>5s}  {self.total_tokens:>5d} tok  "
                f"{self.wall_time:>6.1f}s  {self.total_tps:>5.1f} t/s  {u}{e}")


class BenchRunner:
    """Manages GPU reset, server lifecycle, monitor, and request firing."""

    def __init__(self):
        self.monitor = Monitor()
        self._proc: Optional[subprocess.Popen] = None
        self._last_crashed: bool = False

    @staticmethod
    def _make_env(config: BenchConfig) -> dict[str, str]:
        env = BASE_ENV.copy()
        env.update(config.sycl_env())
        env["HOST"] = "127.0.0.1"
        return env

    # ── GPU management ────────────────────────────────────────────────

    def check_gpus(self) -> bool:
        try:
            r = subprocess.run(
                ["sycl-ls"], capture_output=True, text=True, timeout=10,
                env=self._make_env(BenchConfig()),
            )
            return r.stdout.count("level_zero:gpu") >= 3
        except Exception:
            return False

    @staticmethod
    def _kill_gpu_consumers():
        """Kill ALL processes that might hold GPU state."""
        for pat in ["llama-server", "llama-cli", "llama-bench", "arcllm-proxy"]:
            subprocess.run(["pkill", "-9", "-f", pat],
                           capture_output=True)

    @staticmethod
    def _hw_reset_gpus():
        """Hardware-level GPU reset via sysfs.

        Writing '1' to /sys/class/drm/cardN/device/reset triggers an i915
        GPU engine reset. This clears DEVICE_LOST state, flushes command
        queues, and resets the Level Zero runtime's view of the device.

        Without this, a DEVICE_LOST crash can leave the GPU in a degraded
        state where sycl-ls still sees it but kernels silently produce
        garbage or immediately fail.
        """
        for p in GPU_RESET_PATHS:
            try:
                p.write_text("1")
            except PermissionError:
                # Needs root or specific udev rules. Log but don't fail.
                pass
            except Exception:
                pass

    # Persistent JIT cache — survives DEVICE_LOST crash + GPU reset.
    _JIT_CACHE_BACKUP = Path("/home/ryan/llm-stack/cache/neo_compiler_cache")
    _JIT_CACHE_LIVE = [
        Path("/tmp/neo_compiler_cache"),
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "neo_compiler_cache",
    ]

    @classmethod
    def _save_jit_cache(cls):
        """Snapshot the L0 JIT cache to NVMe after a successful server startup.

        The L0 runtime JIT-compiles SPIR-V → GPU ISA on first launch (~90s).
        Saving the cache lets us restore it after a crash instead of recompiling.
        """
        for d in cls._JIT_CACHE_LIVE:
            if d.exists() and any(d.iterdir()):
                cls._JIT_CACHE_BACKUP.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["rm", "-rf", str(cls._JIT_CACHE_BACKUP)], capture_output=True)
                subprocess.run(["cp", "-a", str(d), str(cls._JIT_CACHE_BACKUP)], capture_output=True)
                print(f"  JIT cache saved ({sum(1 for _ in d.rglob('*'))} files)")
                return

    @classmethod
    def _restore_jit_cache(cls):
        """Restore the L0 JIT cache from NVMe backup after a crash.

        Replaces the (possibly corrupt) live cache with the known-good backup.
        Server startup drops from ~110s to ~10-20s.
        """
        if not cls._JIT_CACHE_BACKUP.exists():
            return False
        for d in cls._JIT_CACHE_LIVE:
            subprocess.run(["rm", "-rf", str(d)], capture_output=True)
            d.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-a", str(cls._JIT_CACHE_BACKUP), str(d)], capture_output=True)
            print(f"  JIT cache restored from backup")
            return True
        return False

    @classmethod
    def _flush_level_zero_cache(cls):
        """Clear Level Zero shader cache and compiled kernel cache.

        After a DEVICE_LOST, cached JIT-compiled kernels may be corrupt.
        The Level Zero runtime caches compiled SPIR-V in /tmp or
        XDG_CACHE_HOME. Clearing forces recompilation on next use.
        """
        cache_dirs = cls._JIT_CACHE_LIVE + [Path("/tmp/opencl_cache")]
        for d in cache_dirs:
            if d.exists():
                subprocess.run(["rm", "-rf", str(d)], capture_output=True)

    def reset_gpus(self, wait: int = 5, flush_cache: bool = False) -> bool:
        """Full GPU reset: kill processes → hw reset → (optionally flush caches) → verify.

        This ensures clean GPU state between tests. After a DEVICE_LOST
        crash, anything less than this can leave GPUs in degraded state
        where subsequent tests get false results.

        flush_cache: only set True after a confirmed DEVICE_LOST crash.
        Flushing the L0 JIT cache forces full recompilation (~90s penalty).
        Keeping the cache makes server startup ~10-20s instead of ~110s.
        """
        print("  resetting GPUs...")

        # 1. Kill everything holding GPU handles.
        self._kill_gpu_consumers()
        self._proc = None
        time.sleep(2)

        # 2. Hardware-level GPU reset (clears DEVICE_LOST state).
        self._hw_reset_gpus()

        # 3. After crash: flush corrupt cache, restore from NVMe backup.
        if flush_cache:
            self._flush_level_zero_cache()
            if self._restore_jit_cache():
                pass  # restored from backup — fast restart
            else:
                print("  no JIT cache backup — will recompile (~90s)")

        time.sleep(wait)

        # 4. Verify GPUs are responsive.
        for attempt in range(3):
            if self.check_gpus():
                print("  GPUs ready")
                return True
            print(f"  GPUs not ready, waiting ({attempt + 1}/3)...")
            time.sleep(10)

        print("  WARNING: GPUs not responding after reset")
        return False

    # ── Server lifecycle ──────────────────────────────────────────────

    def start_server(self, config: BenchConfig, timeout: int = 300) -> bool:
        if not config.server_bin.exists():
            print(f"  ERROR: binary not found: {config.server_bin}")
            return False
        if not config.model_path.exists():
            print(f"  ERROR: model not found: {config.model_path}")
            return False

        log = LOG_DIR / f"{config.name or 'test'}.log"
        args = config.server_args()

        flags = f"np={config.n_parallel} c={config.context}"
        if config.reasoning_budget:
            flags += f" think={config.reasoning_budget}"
        flags += f" graph={'OFF' if config.disable_graph else 'ON'}"
        print(f"  server: {flags}")

        fh = open(log, "w")
        self._proc = subprocess.Popen(
            args, env=self._make_env(config), stdout=fh, stderr=subprocess.STDOUT,
        )
        fh.close()
        print(f"  PID={self._proc.pid} log={log}")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                print(f"  server died (exit {self._proc.returncode})")
                for line in log.read_text().splitlines()[-8:]:
                    lo = line.lower()
                    if any(k in lo for k in ("error", "failed", "alloc", "device_lost")):
                        print(f"    {line.strip()}")
                self._proc = None
                return False
            try:
                resp = urllib.request.urlopen("http://127.0.0.1:8400/health", timeout=2)
                if resp.status == 200:
                    elapsed = timeout - (deadline - time.monotonic())
                    print(f"  ready ({elapsed:.0f}s)")
                    return True
            except Exception:
                pass
            time.sleep(3)

        print(f"  TIMEOUT ({timeout}s)")
        self.stop_server()
        return False

    def stop_server(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=3)
        self._proc = None

    # ── Request firing ────────────────────────────────────────────────

    @staticmethod
    def fire(config: BenchConfig) -> BenchResult:
        import aiohttp

        result = BenchResult(name=config.name or "test", config=config)

        async def send(session, idx):
            payload = {
                "model": config.model_path.stem,
                "messages": [{"role": "user", "content": config.prompt}],
                "max_tokens": config.max_tokens,
                "temperature": 0.7, "stream": False,
            }
            t0 = time.monotonic()
            try:
                async with session.post(
                    "http://127.0.0.1:8400/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.timeout),
                ) as resp:
                    body = await resp.json()
                    dt = time.monotonic() - t0
                    if resp.status != 200:
                        return {"idx": idx, "tokens": 0, "time_s": dt,
                                "tps": 0, "status": f"http:{resp.status}"}
                    tok = body.get("usage", {}).get("completion_tokens", 0)
                    return {"idx": idx, "tokens": tok, "time_s": round(dt, 2),
                            "tps": round(tok / dt, 1) if dt else 0, "status": "ok"}
            except asyncio.TimeoutError:
                return {"idx": idx, "tokens": 0,
                        "time_s": round(time.monotonic() - t0, 2),
                        "tps": 0, "status": "timeout"}
            except Exception as e:
                return {"idx": idx, "tokens": 0,
                        "time_s": round(time.monotonic() - t0, 2),
                        "tps": 0, "status": str(e)[:60]}

        async def go():
            async with aiohttp.ClientSession() as s:
                t0 = time.monotonic()
                res = await asyncio.gather(*(send(s, i) for i in range(config.concurrent)))
                return res, time.monotonic() - t0

        reqs, wall = asyncio.run(go())
        result.per_request = sorted(reqs, key=lambda r: r["idx"])
        result.completed = sum(1 for r in reqs if r["status"] == "ok")
        result.failed = config.concurrent - result.completed
        result.total_tokens = sum(r["tokens"] for r in reqs)
        result.wall_time = round(wall, 2)
        result.total_tps = round(result.total_tokens / wall, 1) if wall else 0
        return result

    # ── Full test lifecycle ───────────────────────────────────────────

    def run_test(self, config: BenchConfig) -> BenchResult:
        """reset GPUs → start monitor → start server → fire → collect util → stop."""
        name = config.name or config.summary()
        config = config.with_(name=name)

        print(f"\n{'═' * 64}")
        print(f"  {name}")
        print(f"  {config.summary()}")
        if config.patches:
            for p in config.patches:
                print(f"  PATCH: {p}")
        print(f"{'═' * 64}")

        self.reset_gpus(flush_cache=self._last_crashed)
        self.monitor.start()

        if not self.start_server(config):
            self.monitor.stop()
            return BenchResult(name=name, config=config, error="server failed")

        # Save JIT cache after first successful server startup.
        if not self._JIT_CACHE_BACKUP.exists():
            self._save_jit_cache()

        t_start = self.monitor.mark()
        result = self.fire(config)
        t_end = self.monitor.mark()

        result.name = name
        result.config = config
        result.utilization = self.monitor.snapshot(t_start, t_end)

        self.monitor.stop()

        print(f"  {result.completed}/{config.concurrent} ok  "
              f"{result.total_tokens} tok  {result.wall_time}s  "
              f"{result.total_tps} t/s")
        print(f"  {result.utilization.summary()}")
        if result.failed:
            for r in result.per_request:
                if r["status"] != "ok":
                    print(f"    req[{r['idx']}]: {r['status']} ({r['time_s']}s)")

        self.stop_server()
        # Track crash state so next test knows whether to flush+restore cache.
        self._last_crashed = result.failed > 0 or result.error
        time.sleep(2)
        return result

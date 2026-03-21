#!/usr/bin/env python3
"""GPU/CPU/RAM monitor — runs alongside benchmarks.

Spawns intel_gpu_top on each GPU, samples CPU from /proc/stat and RAM from
/proc/meminfo. Writes one-line summaries to stdout and a JSON-lines log file.

Usage:
    python3 bench_gpu_monitor.py --log /tmp/bench_gpu.jsonl [--interval 0.5]

    # From another process, get averages for a time window:
    python3 bench_gpu_monitor.py --summarize /tmp/bench_gpu.jsonl --after <epoch> --before <epoch>
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time


def read_cpu_usage():
    """Return overall CPU usage % since last call (or boot on first call)."""
    if not hasattr(read_cpu_usage, "_prev"):
        read_cpu_usage._prev = None
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        parts = line.split()
        # user nice system idle iowait irq softirq steal
        vals = [int(x) for x in parts[1:9]]
        idle = vals[3] + vals[4]  # idle + iowait
        total = sum(vals)
        if read_cpu_usage._prev is None:
            read_cpu_usage._prev = (idle, total)
            return 0.0
        prev_idle, prev_total = read_cpu_usage._prev
        read_cpu_usage._prev = (idle, total)
        d_total = total - prev_total
        d_idle = idle - prev_idle
        if d_total == 0:
            return 0.0
        return (1 - d_idle / d_total) * 100
    except Exception:
        return 0.0


def read_ram_used_gb():
    """Return RAM used in GB."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", 0)
        used_kb = total - available
        return used_kb / (1024 * 1024)
    except Exception:
        return 0.0


class GPUPoller:
    """Polls intel_gpu_top JSON for one GPU card."""

    def __init__(self, card_index: int, device_filter: str):
        self.card_index = card_index
        self.device_filter = device_filter
        self.utilization = 0.0
        self._proc = None
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        cmd = ["intel_gpu_top", "-J", "-s", "500", "-d", self.device_filter]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
        except FileNotFoundError:
            print(f"WARNING: intel_gpu_top not found, GPU{self.card_index} will show 0%",
                  file=sys.stderr)
            return
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        """Read intel_gpu_top JSON array stream.

        intel_gpu_top -J outputs a JSON array with periodic entries.
        Each entry is a JSON object on its own set of lines, separated by commas.
        We accumulate lines and try to parse each entry.
        """
        buf = ""
        while not self._stop.is_set():
            try:
                line = self._proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                # Skip array brackets
                if line in ("[", "]"):
                    continue
                # Strip trailing comma
                if line.endswith(","):
                    line = line[:-1]
                buf += line
                # Try to parse
                try:
                    obj = json.loads(buf)
                    buf = ""
                    self._update(obj)
                except json.JSONDecodeError:
                    # Accumulate more lines
                    continue
            except Exception:
                break

    def _update(self, obj):
        """Extract GPU busy % from intel_gpu_top JSON entry."""
        # Top-level has "engines" dict with utilization
        engines = obj.get("engines", {})
        # Sum all engine busy percentages, take the max (Render/3D is usually the one)
        max_busy = 0.0
        for engine_name, engine_data in engines.items():
            busy = engine_data.get("busy", 0.0)
            if busy > max_busy:
                max_busy = busy
        self.utilization = max_busy

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def find_gpu_devices():
    """Find Intel GPU device filters for intel_gpu_top.

    Uses `intel_gpu_top -L` to discover cards, returns pci:card=N filters.
    """
    devices = []
    try:
        result = subprocess.run(
            ["intel_gpu_top", "-L"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            # Lines like: card1  Intel Dg2 (Gen12)  pci:vendor=8086,device=56A0,card=0
            if line.startswith("card"):
                parts = line.split()
                # Find the pci: filter
                for part in parts:
                    if part.startswith("pci:"):
                        devices.append(part)
                        break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return devices


def run_monitor(log_path: str, interval: float):
    """Main monitoring loop."""
    devices = find_gpu_devices()
    if not devices:
        print("WARNING: No DRI render nodes found", file=sys.stderr)

    # Use up to 3 GPUs
    gpu_count = min(len(devices), 3)
    pollers = []
    for i in range(gpu_count):
        p = GPUPoller(i, devices[i])
        p.start()
        pollers.append(p)

    log_fh = open(log_path, "a")
    stop = threading.Event()

    def handle_signal(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Warm up CPU usage reader
    read_cpu_usage()
    time.sleep(0.2)

    try:
        while not stop.is_set():
            ts = time.time()
            cpu = read_cpu_usage()
            ram = read_ram_used_gb()
            gpu_utils = [p.utilization for p in pollers]

            # One-line summary
            gpu_str = " ".join(f"GPU{i}:{u:4.0f}%" for i, u in enumerate(gpu_utils))
            summary = f"{gpu_str} CPU:{cpu:4.0f}% RAM:{ram:.1f}G"
            print(f"\r{summary}", end="", flush=True)

            # JSON log line
            entry = {
                "ts": ts,
                "cpu": round(cpu, 1),
                "ram_gb": round(ram, 1),
            }
            for i, u in enumerate(gpu_utils):
                entry[f"gpu{i}"] = round(u, 1)
            log_fh.write(json.dumps(entry) + "\n")
            log_fh.flush()

            stop.wait(interval)
    finally:
        print()  # newline after \r output
        for p in pollers:
            p.stop()
        log_fh.close()


def summarize_log(log_path: str, after: float, before: float):
    """Print average GPU/CPU/RAM for a time window from the JSONL log."""
    entries = []
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    ts = e.get("ts", 0)
                    if after <= ts <= before:
                        entries.append(e)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print("{}", flush=True)
        return

    if not entries:
        print("{}", flush=True)
        return

    # Compute averages
    n = len(entries)
    result = {
        "samples": n,
        "cpu_avg": round(sum(e.get("cpu", 0) for e in entries) / n, 1),
        "ram_avg_gb": round(sum(e.get("ram_gb", 0) for e in entries) / n, 1),
    }
    # GPU averages — find which gpu keys exist
    for key in sorted(set(k for e in entries for k in e if k.startswith("gpu"))):
        vals = [e.get(key, 0) for e in entries]
        result[f"{key}_avg"] = round(sum(vals) / len(vals), 1)

    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description="GPU/CPU/RAM monitor")
    parser.add_argument("--log", default="/tmp/bench_gpu.jsonl", help="JSONL log path")
    parser.add_argument("--interval", type=float, default=0.5, help="Sample interval (seconds)")
    parser.add_argument("--summarize", action="store_true", help="Summarize mode: print averages")
    parser.add_argument("--after", type=float, default=0, help="Start epoch for summarize")
    parser.add_argument("--before", type=float, default=float("inf"), help="End epoch for summarize")
    args = parser.parse_args()

    if args.summarize:
        summarize_log(args.log, args.after, args.before)
    else:
        run_monitor(args.log, args.interval)


if __name__ == "__main__":
    main()

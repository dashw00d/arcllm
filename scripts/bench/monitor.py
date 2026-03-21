"""GPU/CPU/RAM monitor — runs in a background thread during tests.

Uses sysfs for GPU metrics (energy, frequency, temp) since intel_gpu_top
crashes with multi-GPU DG2 on v1.28. CPU from /proc/stat, RAM from /proc/meminfo.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Utilization:
    """Average utilization over a time window."""
    gpu_freq_mhz: list[float]   # per-GPU average frequency
    gpu_power_w: list[float]    # per-GPU average power draw (watts)
    gpu_temp_c: list[float]     # per-GPU average temp (celsius)
    cpu: float                  # overall CPU %
    ram_gb: float               # used RAM in GB
    samples: int

    def summary(self) -> str:
        parts = []
        for i, (f, w, t) in enumerate(zip(self.gpu_freq_mhz, self.gpu_power_w, self.gpu_temp_c)):
            parts.append(f"GPU{i}:{f:.0f}MHz/{w:.0f}W/{t:.0f}°C")
        parts.append(f"CPU:{self.cpu:.0f}%")
        parts.append(f"RAM:{self.ram_gb:.1f}G")
        return " ".join(parts)

    @staticmethod
    def empty() -> Utilization:
        return Utilization(gpu_freq_mhz=[], gpu_power_w=[], gpu_temp_c=[],
                           cpu=0, ram_gb=0, samples=0)


def _find_gpu_cards() -> list[dict]:
    """Find DRM card sysfs paths for Intel GPUs."""
    cards = []
    drm = Path("/sys/class/drm")
    for card_dir in sorted(drm.iterdir()):
        if not card_dir.name.startswith("card") or card_dir.name.count("-"):
            continue
        freq_path = card_dir / "gt_cur_freq_mhz"
        hwmon_dirs = sorted((card_dir / "device" / "hwmon").iterdir()) if (card_dir / "device" / "hwmon").exists() else []
        if freq_path.exists() and hwmon_dirs:
            cards.append({
                "name": card_dir.name,
                "freq_path": freq_path,
                "energy_path": hwmon_dirs[0] / "energy1_input",
                "temp_path": hwmon_dirs[0] / "temp1_input",
            })
    return cards


def _read_sysfs(path: Path) -> float:
    try:
        return float(path.read_text().strip())
    except Exception:
        return 0.0


def _read_cpu() -> tuple[int, int]:
    with open("/proc/stat") as f:
        parts = f.readline().split()
    vals = [int(x) for x in parts[1:9]]
    return vals[3] + vals[4], sum(vals)  # idle, total


def _read_ram_gb() -> float:
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                info[p[0].rstrip(":")] = int(p[1])
    return (info.get("MemTotal", 0) - info.get("MemAvailable", 0)) / (1024 * 1024)


class Monitor:
    """Background monitor. Start before test, stop after, query for averages."""

    def __init__(self):
        self._cards: list[dict] = []
        self._samples: list[dict] = []
        self._thread = None
        self._stop = threading.Event()
        self._prev_cpu = None
        self._prev_energy: list[float] = []
        self._prev_time: float = 0

    def start(self):
        self._cards = _find_gpu_cards()
        if not self._cards:
            print("  monitor: no GPU cards found in sysfs")

        # Warm up
        self._prev_cpu = _read_cpu()
        self._prev_energy = [_read_sysfs(c["energy_path"]) for c in self._cards]
        self._prev_time = time.monotonic()
        time.sleep(0.2)

        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            now = time.monotonic()
            dt = now - self._prev_time
            self._prev_time = now

            # CPU
            idle, total = _read_cpu()
            if self._prev_cpu:
                di = idle - self._prev_cpu[0]
                dtot = total - self._prev_cpu[1]
                cpu_pct = (1 - di / dtot) * 100 if dtot else 0
            else:
                cpu_pct = 0
            self._prev_cpu = (idle, total)

            sample = {
                "ts": time.time(),
                "cpu": round(cpu_pct, 1),
                "ram_gb": round(_read_ram_gb(), 1),
            }

            # Per-GPU
            for i, card in enumerate(self._cards):
                freq = _read_sysfs(card["freq_path"])
                temp = _read_sysfs(card["temp_path"]) / 1000  # mC → C
                energy = _read_sysfs(card["energy_path"])      # uJ
                # Power = delta_energy / delta_time
                if dt > 0 and i < len(self._prev_energy):
                    power_w = (energy - self._prev_energy[i]) / (dt * 1_000_000)
                else:
                    power_w = 0
                if i < len(self._prev_energy):
                    self._prev_energy[i] = energy
                sample[f"gpu{i}_freq"] = round(freq)
                sample[f"gpu{i}_power"] = round(power_w, 1)
                sample[f"gpu{i}_temp"] = round(temp, 1)

            self._samples.append(sample)
            self._stop.wait(0.5)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def snapshot(self, start: float = 0, end: float = float("inf")) -> Utilization:
        filtered = [s for s in self._samples if start <= s["ts"] <= end]
        if not filtered:
            return Utilization.empty()

        n = len(filtered)
        n_gpus = len(self._cards)

        freq_avgs = []
        power_avgs = []
        temp_avgs = []
        for i in range(n_gpus):
            freq_avgs.append(round(sum(s.get(f"gpu{i}_freq", 0) for s in filtered) / n))
            power_avgs.append(round(sum(s.get(f"gpu{i}_power", 0) for s in filtered) / n, 1))
            temp_avgs.append(round(sum(s.get(f"gpu{i}_temp", 0) for s in filtered) / n, 1))

        return Utilization(
            gpu_freq_mhz=freq_avgs,
            gpu_power_w=power_avgs,
            gpu_temp_c=temp_avgs,
            cpu=round(sum(s["cpu"] for s in filtered) / n, 1),
            ram_gb=round(sum(s["ram_gb"] for s in filtered) / n, 1),
            samples=n,
        )

    def mark(self) -> float:
        return time.time()

#!/usr/bin/env python3
import os
import signal
import subprocess
import time

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("AyatanaAppIndicator3", "0.1")

from gi.repository import AyatanaAppIndicator3 as AppIndicator3
from gi.repository import GLib, Gtk


HWMON_ROOT = "/sys/class/hwmon"
DRM_ROOT = "/sys/class/drm"
REFRESH_SECONDS = 2


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return None


def read_millic(path):
    text = read_text(path)
    if text is None:
        return None

    try:
        return int(text)
    except ValueError:
        return None


def format_temp(millic):
    if millic is None:
        return "--"
    return f"{round(millic / 1000)}C"


def hwmon_dirs():
    try:
        names = [name for name in os.listdir(HWMON_ROOT) if name.startswith("hwmon")]
    except OSError:
        return []
    return [os.path.join(HWMON_ROOT, name) for name in sorted(names)]


def get_cpu_temp():
    for directory in hwmon_dirs():
        if read_text(os.path.join(directory, "name")) == "coretemp":
            return read_millic(os.path.join(directory, "temp1_input"))
    return None


def get_gpu_temps():
    gpus = []

    for directory in hwmon_dirs():
        if read_text(os.path.join(directory, "name")) != "i915":
            continue

        device_path = os.path.realpath(os.path.join(directory, "device"))
        pci = os.path.basename(device_path)

        # Find the DRM card for this PCI device to read rc6
        card_name = None
        try:
            for name in sorted(os.listdir(DRM_ROOT)):
                if not name.startswith("card") or "-" in name:
                    continue
                card_device = os.path.realpath(os.path.join(DRM_ROOT, name, "device"))
                if os.path.basename(card_device) == pci:
                    card_name = name
                    break
        except OSError:
            pass

        gpus.append(
            {
                "pci": pci,
                "card": card_name,
                "temp": read_millic(os.path.join(directory, "temp1_input")),
                "fan": read_text(os.path.join(directory, "fan1_input")),
            }
        )

    gpus.sort(key=lambda gpu: gpu["pci"])
    return gpus


def get_cpu_usage(prev_stat):
    """Return (usage_percent, new_stat) from /proc/stat delta."""
    text = read_text("/proc/stat")
    if text is None:
        return None, prev_stat
    first_line = text.split("\n")[0]  # "cpu  user nice system idle ..."
    fields = [int(x) for x in first_line.split()[1:]]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)  # idle + iowait
    total = sum(fields)
    if prev_stat is None:
        return None, (idle, total)
    prev_idle, prev_total = prev_stat
    d_total = total - prev_total
    d_idle = idle - prev_idle
    if d_total == 0:
        return 0, (idle, total)
    return round(100 * (1 - d_idle / d_total)), (idle, total)


def get_gpu_rc6(card_name):
    """Read rc6_residency_ms for a card."""
    if card_name is None:
        return None
    path = os.path.join(DRM_ROOT, card_name, "gt/gt0/rc6_residency_ms")
    text = read_text(path)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


class TripleTempsIndicator:
    def __init__(self):
        self.indicator = AppIndicator3.Indicator.new(
            "triple-temps",
            "utilities-system-monitor-symbolic",
            AppIndicator3.IndicatorCategory.SYSTEM_SERVICES,
        )
        self.indicator.set_status(AppIndicator3.IndicatorStatus.ACTIVE)
        self.indicator.set_title("Triple Temps")
        self.indicator.set_label("Temps...", "Temps...")

        # State for utilization deltas
        self._cpu_stat = None
        self._prev_rc6 = {}  # card_name -> rc6_ms
        self._prev_time = None

        self.menu = Gtk.Menu()

        self.summary_item = Gtk.MenuItem(label="Temps...")
        self.summary_item.set_sensitive(False)
        self.menu.append(self.summary_item)

        self.menu.append(Gtk.SeparatorMenuItem())

        self.cpu_item = Gtk.MenuItem(label="CPU")
        self.cpu_item.set_sensitive(False)
        self.menu.append(self.cpu_item)

        self.gpu_items = []
        for index in range(3):
            item = Gtk.MenuItem(label=f"GPU {index + 1}")
            item.set_sensitive(False)
            self.gpu_items.append(item)
            self.menu.append(item)

        self.menu.append(Gtk.SeparatorMenuItem())

        open_psensor = Gtk.MenuItem(label="Open Psensor")
        open_psensor.connect("activate", self.on_open_psensor)
        self.menu.append(open_psensor)

        quit_item = Gtk.MenuItem(label="Quit")
        quit_item.connect("activate", self.on_quit)
        self.menu.append(quit_item)

        self.menu.show_all()
        self.indicator.set_menu(self.menu)

        self.refresh()
        GLib.timeout_add_seconds(REFRESH_SECONDS, self.refresh)

    def refresh(self):
        now = time.monotonic()
        cpu_temp = get_cpu_temp()
        gpu_temps = get_gpu_temps()

        # CPU utilization
        cpu_pct, self._cpu_stat = get_cpu_usage(self._cpu_stat)
        cpu_util_str = f"/{cpu_pct}%" if cpu_pct is not None else ""

        # GPU utilization from rc6 delta
        gpu_utils = {}
        for gpu in gpu_temps:
            card = gpu["card"]
            if card is None:
                continue
            rc6_now = get_gpu_rc6(card)
            if rc6_now is not None and card in self._prev_rc6 and self._prev_time is not None:
                dt_ms = (now - self._prev_time) * 1000
                if dt_ms > 0:
                    rc6_delta = rc6_now - self._prev_rc6[card]
                    util = max(0, min(100, round(100 * (1 - rc6_delta / dt_ms))))
                    gpu_utils[card] = util
            if rc6_now is not None:
                self._prev_rc6[card] = rc6_now

        self._prev_time = now

        # Build summary: CPU 51C/12%  G1 68C/0%  G2 49C/0%  G3 60C/5%
        parts = [f"CPU {format_temp(cpu_temp)}{cpu_util_str}"]
        for index, gpu in enumerate(gpu_temps):
            card = gpu["card"]
            util_str = f"/{gpu_utils[card]}%" if card in gpu_utils else ""
            parts.append(f"G{index + 1} {format_temp(gpu['temp'])}{util_str}")

        summary = "  ".join(parts)
        self.indicator.set_label(summary, summary)
        self.summary_item.set_label(summary)

        cpu_detail = f"CPU package: {format_temp(cpu_temp)}"
        if cpu_pct is not None:
            cpu_detail += f"  util {cpu_pct}%"
        self.cpu_item.set_label(cpu_detail)

        for index, item in enumerate(self.gpu_items):
            if index < len(gpu_temps):
                gpu = gpu_temps[index]
                fan_text = gpu["fan"] if gpu["fan"] is not None else "n/a"
                card = gpu["card"]
                util_text = f"  util {gpu_utils[card]}%" if card in gpu_utils else ""
                item.set_label(
                    f"GPU {index + 1} {gpu['pci']}: {format_temp(gpu['temp'])}{util_text}  fan {fan_text} RPM"
                )
                item.show()
            else:
                item.hide()

        return True

    def on_open_psensor(self, _item):
        subprocess.Popen(["psensor"])

    def on_quit(self, _item):
        Gtk.main_quit()


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    TripleTempsIndicator()
    Gtk.main()


if __name__ == "__main__":
    main()

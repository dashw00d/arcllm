"""Environment bootstrap: render group + SYCL env loading.

Handles everything so callers just `import bench` and go.
"""

from __future__ import annotations

import grp
import os
import pwd
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def ensure_render_group():
    """Add render group to this process so GPU DRI nodes are accessible."""
    try:
        render_gid = grp.getgrnam("render").gr_gid
    except KeyError:
        print("ERROR: 'render' group does not exist")
        return False

    if render_gid in os.getgroups():
        return True

    user = pwd.getpwuid(os.getuid()).pw_name
    members = grp.getgrnam("render").gr_mem
    user_groups = [g.gr_gid for g in grp.getgrall() if user in g.gr_mem]

    if render_gid not in user_groups and user not in members:
        print(f"ERROR: '{user}' not in render group.")
        print(f"  Fix: sudo usermod -a -G render {user}")
        return False

    try:
        os.setgroups(list(set(os.getgroups() + [render_gid])))
        return True
    except PermissionError:
        # Re-exec under sg render — source conda env so we get the right python
        env_script = ROOT / "env.sglang-xpu.sh"
        pkg_dir = str(Path(__file__).parent.parent)
        rest = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
        cmd = f"source {env_script} && cd {pkg_dir} && python3 -m bench {rest}"
        print(f"  re-launching with render group...")
        os.execvp("sg", ["sg", "render", "-c", f"bash -c '{cmd}'"])


def load_sycl_env() -> dict[str, str]:
    """Source env.sglang-xpu.sh in bash, return resulting env dict."""
    script = ROOT / "env.sglang-xpu.sh"
    if not script.exists():
        print(f"ERROR: {script} not found")
        sys.exit(1)

    r = subprocess.run(
        ["bash", "-c", f"source {script} && env -0"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        print(f"ERROR loading env: {r.stderr[:300]}")
        sys.exit(1)

    env = {}
    for entry in r.stdout.split('\0'):
        if '=' in entry:
            k, v = entry.split('=', 1)
            env[k] = v
    return env


# Run at import time
ensure_render_group()
BASE_ENV = load_sycl_env()

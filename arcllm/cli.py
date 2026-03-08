#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()
CONFIG_DIR = HOME / ".config" / "arcllm"
SYSTEMD_DIR = HOME / ".config" / "systemd" / "user"
CONFIG_FILE = CONFIG_DIR / "config.env"
SERVICE_FILE = SYSTEMD_DIR / "arcllm.service"
SERVICE_NAME = "arcllm.service"
STATE_DIR = HOME / ".local" / "state" / "arcllm"
LOG_DIR = ROOT / "logs"

DEFAULT_CONFIG = {
    "MODEL_ID": "Qwen/Qwen3-0.6B",
    "TORCH_DTYPE": "float16",
    "API_HOST": "127.0.0.1",
    "ROUTER_PORT": "8000",
    "GPU_IDS": "0,1,2",
    "WORKER_PORTS": "8001,8002,8003",
    "DEFAULT_MAX_NEW_TOKENS": "256",
    "ENABLE_THINKING": "false",
}


def ensure_dirs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def ensure_config() -> None:
    ensure_dirs()
    if CONFIG_FILE.exists():
        return
    example_config = ROOT / "config" / "config.env.example"
    if example_config.exists():
        CONFIG_FILE.write_text(example_config.read_text())
    else:
        lines = [f"{key}={value}\n" for key, value in DEFAULT_CONFIG.items()]
        CONFIG_FILE.write_text("".join(lines))


def load_config() -> dict[str, str]:
    ensure_config()
    config = DEFAULT_CONFIG.copy()
    for raw_line in CONFIG_FILE.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()
    return config


def write_service_file() -> None:
    ensure_dirs()
    content = f"""[Unit]
Description=ArcLLM local OpenAI-compatible API
After=network.target

[Service]
Type=simple
EnvironmentFile=%h/.config/arcllm/config.env
WorkingDirectory={ROOT}
ExecStart={ROOT / "bin" / "arcllm"} serve
Restart=on-failure
RestartSec=3
KillMode=control-group
TimeoutStopSec=30

[Install]
WantedBy=default.target
"""
    SERVICE_FILE.write_text(content)


def run(cmd: list[str], *, capture: bool = False, check: bool = True,
        env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
        env=env,
    )


def systemd_available() -> bool:
    try:
        run(["systemctl", "--user", "show-environment"], capture=True, check=True)
        return True
    except Exception:
        return False


def api_base(config: dict[str, str]) -> str:
    return f"http://{config['API_HOST']}:{config['ROUTER_PORT']}"


def api_request(config: dict[str, str], method: str, path: str,
                payload: dict | None = None) -> dict:
    url = f"{api_base(config)}{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"API error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"API unavailable: {exc.reason}") from exc


def config_env(config: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(config)
    env["LOG_DIR"] = str(LOG_DIR)
    return env


def cmd_serve(_: argparse.Namespace) -> int:
    config = load_config()
    env = config_env(config)
    os.execvpe(str(ROOT / "scripts" / "start_3gpu_cluster.sh"),
               [str(ROOT / "scripts" / "start_3gpu_cluster.sh")], env)
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    ensure_config()
    write_service_file()
    if not systemd_available():
        raise SystemExit("systemd --user is not available in this session")
    run(["systemctl", "--user", "daemon-reload"])
    if args.enable:
        run(["systemctl", "--user", "enable", SERVICE_NAME])
    run(["systemctl", "--user", "start", SERVICE_NAME])
    print(f"started {SERVICE_NAME}")
    return 0


def cmd_stop(_: argparse.Namespace) -> int:
    run(["systemctl", "--user", "stop", SERVICE_NAME], check=False)
    print(f"stopped {SERVICE_NAME}")
    return 0


def cmd_restart(_: argparse.Namespace) -> int:
    run(["systemctl", "--user", "restart", SERVICE_NAME])
    print(f"restarted {SERVICE_NAME}")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    config = load_config()
    result = run(["systemctl", "--user", "is-active", SERVICE_NAME],
                 capture=True, check=False)
    active = result.stdout.strip() if result.stdout else "inactive"
    print(f"service={active}")
    if active == "active":
        print(json.dumps(api_request(config, "GET", "/healthz"), indent=2))
    return 0 if active == "active" else 1


def cmd_logs(args: argparse.Namespace) -> int:
    cmd = ["journalctl", "--user", "-u", SERVICE_NAME, "-n", str(args.lines), "--no-pager"]
    if args.follow:
        cmd.remove("--no-pager")
        cmd.append("-f")
    os.execvp(cmd[0], cmd)
    return 0


def cmd_models(_: argparse.Namespace) -> int:
    config = load_config()
    print(json.dumps(api_request(config, "GET", "/v1/models"), indent=2))
    return 0


def cmd_health(_: argparse.Namespace) -> int:
    config = load_config()
    print(json.dumps(api_request(config, "GET", "/healthz"), indent=2))
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    config = load_config()
    prompt = args.prompt
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        raise SystemExit("prompt is required")
    payload = {
        "model": args.model or config["MODEL_ID"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    response = api_request(config, "POST", "/v1/chat/completions", payload)
    print(response["choices"][0]["message"]["content"])
    return 0


def cmd_env(_: argparse.Namespace) -> int:
    config = load_config()
    base_url = f"{api_base(config)}/v1"
    print(f"export OPENAI_BASE_URL={base_url}")
    print("export OPENAI_API_KEY=local")
    return 0


def cmd_config(_: argparse.Namespace) -> int:
    config = load_config()
    print(json.dumps(config, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arcllm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve")
    serve.set_defaults(func=cmd_serve)

    start = subparsers.add_parser("start")
    start.add_argument("--enable", action="store_true")
    start.set_defaults(func=cmd_start)

    stop = subparsers.add_parser("stop")
    stop.set_defaults(func=cmd_stop)

    restart = subparsers.add_parser("restart")
    restart.set_defaults(func=cmd_restart)

    status = subparsers.add_parser("status")
    status.set_defaults(func=cmd_status)

    logs = subparsers.add_parser("logs")
    logs.add_argument("-n", "--lines", type=int, default=100)
    logs.add_argument("-f", "--follow", action="store_true")
    logs.set_defaults(func=cmd_logs)

    models = subparsers.add_parser("models")
    models.set_defaults(func=cmd_models)

    health = subparsers.add_parser("health")
    health.set_defaults(func=cmd_health)

    chat = subparsers.add_parser("chat")
    chat.add_argument("prompt", nargs="?")
    chat.add_argument("--model")
    chat.add_argument("--max-tokens", type=int, default=256)
    chat.add_argument("--temperature", type=float, default=0.0)
    chat.set_defaults(func=cmd_chat)

    env_cmd = subparsers.add_parser("env")
    env_cmd.set_defaults(func=cmd_env)

    config_cmd = subparsers.add_parser("config")
    config_cmd.set_defaults(func=cmd_config)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

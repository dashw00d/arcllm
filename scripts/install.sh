#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${HOME}/.local/bin"
CONFIG_DIR="${HOME}/.config/arcllm"
SYSTEMD_DIR="${HOME}/.config/systemd/user"

mkdir -p "$BIN_DIR" "$CONFIG_DIR" "$SYSTEMD_DIR"

ln -sfn "$ROOT/bin/arcllm" "$BIN_DIR/arcllm"

if [ ! -f "$CONFIG_DIR/config.env" ]; then
  cp "$ROOT/config/config.env.example" "$CONFIG_DIR/config.env"
fi

cat > "$SYSTEMD_DIR/arcllm.service" <<EOF
[Unit]
Description=ArcLLM local OpenAI-compatible API
After=network.target

[Service]
Type=simple
EnvironmentFile=%h/.config/arcllm/config.env
WorkingDirectory=$ROOT
ExecStart=$ROOT/bin/arcllm serve
Restart=on-failure
RestartSec=3
KillMode=control-group
TimeoutStopSec=30

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload

echo "Installed arcllm to $BIN_DIR/arcllm"
echo "Config file: $CONFIG_DIR/config.env"
echo "Service file: $SYSTEMD_DIR/arcllm.service"
echo "Run: arcllm start"

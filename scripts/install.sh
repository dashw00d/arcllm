#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${HOME}/.local/bin"
CONFIG_DIR="${HOME}/.config/arcllm"
SYSTEMD_DIR="${HOME}/.config/systemd/user"
SHELL_LOCAL="${HOME}/.config/shell/local.sh"
ZSH_COMPLETIONS_DIR="${CONFIG_DIR}/completions"

mkdir -p "$BIN_DIR" "$CONFIG_DIR" "$SYSTEMD_DIR" "$ZSH_COMPLETIONS_DIR" "$(dirname "$SHELL_LOCAL")"

ln -sfn "$ROOT/bin/arcllm" "$BIN_DIR/arcllm"
ln -sfn "$ROOT/completions/_arcllm" "$ZSH_COMPLETIONS_DIR/_arcllm"
ln -sfn "$ROOT/completions/arcllm.bash" "$ZSH_COMPLETIONS_DIR/arcllm.bash"

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

touch "$SHELL_LOCAL"
if ! grep -Fq 'arcllm/completions/_arcllm' "$SHELL_LOCAL"; then
  cat >> "$SHELL_LOCAL" <<'EOF'

if [ -n "${ZSH_VERSION:-}" ] && [ -s "$HOME/.config/arcllm/completions/_arcllm" ]; then
  fpath=("$HOME/.config/arcllm/completions" $fpath)
  autoload -Uz compinit
  compinit -i
fi
EOF
fi

echo "Installed arcllm to $BIN_DIR/arcllm"
echo "Config file: $CONFIG_DIR/config.env"
echo "Service file: $SYSTEMD_DIR/arcllm.service"
echo "Zsh completion: $ZSH_COMPLETIONS_DIR/_arcllm"
echo "Run: arcllm start"

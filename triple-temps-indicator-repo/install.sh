#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin_dir="$HOME/.local/bin"
autostart_dir="$HOME/.config/autostart"
desktop_out="$autostart_dir/triple-temps-indicator.desktop"

mkdir -p "$bin_dir" "$autostart_dir"
install -m 755 "$repo_dir/triple-temps-indicator.py" "$bin_dir/triple-temps-indicator"

sed "s|@HOME@|$HOME|g" "$repo_dir/triple-temps-indicator.desktop.in" > "$desktop_out"
chmod 644 "$desktop_out"

cat <<EOF
Installed:
  $bin_dir/triple-temps-indicator
  $desktop_out

If dependencies are missing on Ubuntu:
  sudo apt-get install -y python3-gi gir1.2-gtk-3.0 gir1.2-ayatanaappindicator3-0.1

Optional:
  sudo apt-get install -y psensor
EOF

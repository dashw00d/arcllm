# Triple Temps Indicator

Small Ubuntu/GNOME top-bar indicator for:

- CPU package temperature
- Intel Arc GPU temperatures
- GPU fan RPM in the indicator menu

It was built for a multi-Arc Ubuntu workstation and reads temperatures from
`/sys/class/hwmon`.

## Requirements

- Ubuntu or another GNOME desktop with AppIndicator support
- `python3`
- `python3-gi`
- `gir1.2-gtk-3.0`
- `gir1.2-ayatanaappindicator3-0.1`

Optional:

- `psensor` for the "Open Psensor" menu item

## Install

```bash
./install.sh
```

The installer:

- copies the launcher to `~/.local/bin/triple-temps-indicator`
- installs an autostart entry to `~/.config/autostart/triple-temps-indicator.desktop`

## Run

```bash
triple-temps-indicator
```

For an immediate session launch after install:

```bash
systemd-run --user --unit=triple-temps-indicator --collect ~/.local/bin/triple-temps-indicator
```

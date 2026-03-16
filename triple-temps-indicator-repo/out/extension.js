import Clutter from 'gi://Clutter';
import GLib from 'gi://GLib';
import GObject from 'gi://GObject';
import St from 'gi://St';

import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import * as PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js';

const HWMON_ROOT = '/sys/class/hwmon';
const REFRESH_SECONDS = 2;

function readText(path) {
    try {
        const [ok, bytes] = GLib.file_get_contents(path);
        if (!ok)
            return null;
        return new TextDecoder().decode(bytes).trim();
    } catch (_) {
        return null;
    }
}

function readMilliC(path) {
    const text = readText(path);
    if (!text)
        return null;
    const value = Number.parseInt(text, 10);
    if (Number.isNaN(value))
        return null;
    return value;
}

function formatC(milliC) {
    if (milliC === null)
        return '--';
    return `${Math.round(milliC / 1000)}°`;
}

function getHwmonDirs() {
    const dirs = [];
    let dir = null;

    try {
        dir = GLib.Dir.open(HWMON_ROOT, 0);
        while (true) {
            const name = dir.read_name();
            if (name === null)
                break;
            if (name.startsWith('hwmon'))
                dirs.push(`${HWMON_ROOT}/${name}`);
        }
    } catch (_) {
        return [];
    } finally {
        dir?.close();
    }

    dirs.sort();
    return dirs;
}

function getCpuTemp() {
    for (const dir of getHwmonDirs()) {
        if (readText(`${dir}/name`) === 'coretemp')
            return readMilliC(`${dir}/temp1_input`);
    }

    return null;
}

function getGpuTemps() {
    const gpus = [];

    for (const dir of getHwmonDirs()) {
        if (readText(`${dir}/name`) !== 'i915')
            continue;

        const devicePath = GLib.file_read_link(`${dir}/device`);
        const pciAddress = devicePath.split('/').at(-1) ?? 'gpu';
        gpus.push({
            pciAddress,
            temp: readMilliC(`${dir}/temp1_input`),
        });
    }

    gpus.sort((a, b) => a.pciAddress.localeCompare(b.pciAddress));
    return gpus;
}

const TempIndicator = GObject.registerClass(
class TempIndicator extends PanelMenu.Button {
    _init() {
        super._init(0.0, 'Triple Temps', true);

        this._label = new St.Label({
            text: 'Temps...',
            y_align: Clutter.ActorAlign.CENTER,
        });

        this.add_child(this._label);
        this._refresh();
        this._timeoutId = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, REFRESH_SECONDS, () => {
            this._refresh();
            return GLib.SOURCE_CONTINUE;
        });
    }

    _refresh() {
        const cpuTemp = getCpuTemp();
        const gpuTemps = getGpuTemps();

        const parts = [`CPU ${formatC(cpuTemp)}`];
        gpuTemps.forEach((gpu, index) => {
            parts.push(`G${index + 1} ${formatC(gpu.temp)}`);
        });

        this._label.set_text(parts.join('  '));
    }

    destroy() {
        if (this._timeoutId) {
            GLib.Source.remove(this._timeoutId);
            this._timeoutId = null;
        }

        super.destroy();
    }
});

let indicator = null;

export default class TripleTempsExtension extends Extension {
    enable() {
        indicator = new TempIndicator();
        Main.panel.addToStatusArea('triple-temps', indicator, 0, 'right');
    }

    disable() {
        indicator?.destroy();
        indicator = null;
    }
}

"""Mide el consumo de recursos de un comando (pensado para envolver a runner.py).

Uso:
    python monitor.py --out output/x/_resources/nombre [--interval 1.0] [--disk-path output/x] -- \
        python runner.py ...
    python monitor.py summarize output/x/_resources        # tabla de todos los resúmenes

Muestrea cada `--interval` segundos, sin dependencias externas (Linux):
- RAM: RSS sumado de todo el árbol de procesos (/proc/<pid>/status).
- CPU: tiempo user+sys del árbol (/proc/<pid>/stat) → uso medio en núcleos.
- Disco: bytes leídos/escritos del árbol (/proc/<pid>/io), y opcionalmente el
  crecimiento de `--disk-path` (lo que la corrida dejó en disco).
- GPU (nvidia-smi): VRAM usada por los procesos del árbol (incluye el contexto CUDA,
  a diferencia de `gpu_peak_mb` del run.json, que es lo que reserva PyTorch), uso de
  la GPU (%) y potencia (W) del dispositivo → energía (Wh). Uso y potencia son del
  dispositivo completo: no correr otras cosas en la GPU mientras tanto.

Escribe `<out>.csv` (serie temporal) y `<out>.json` (resumen). Devuelve el código de
salida del comando.
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

CLK_TCK = os.sysconf("SC_CLK_TCK")


def _children_map():
    kids = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                stat = f.read()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        kids.setdefault(ppid, []).append(int(d))
    return kids


def _tree(root):
    kids = _children_map()
    out, stack = [], [root]
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(kids.get(p, []))
    return out


def _proc_sample(pid):
    """(rss_bytes, cpu_seconds, read_bytes, write_bytes) de un proceso, o None."""
    try:
        rss = 0
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                    break
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        cpu = (int(fields[11]) + int(fields[12])) / CLK_TCK  # utime + stime
        rb = wb = 0
        try:
            with open(f"/proc/{pid}/io") as f:
                for line in f:
                    k, v = line.split(":")
                    if k == "read_bytes":
                        rb = int(v)
                    elif k == "write_bytes":
                        wb = int(v)
        except OSError:
            pass
        return rss, cpu, rb, wb
    except (OSError, IndexError, ValueError):
        return None


def _nvsmi(args):
    try:
        r = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return [ln.split(", ") for ln in r.stdout.strip().splitlines() if ln.strip()]
    except (OSError, subprocess.TimeoutExpired):
        return []


def _gpu_sample(pids):
    """(vram_proc_mb, vram_device_mb, util_pct, power_w); None donde no hay dato."""
    pidset = set(pids)
    vram_proc = 0.0
    for row in _nvsmi(["--query-compute-apps=pid,used_memory"]):
        try:
            if int(row[0]) in pidset:
                vram_proc += float(row[1])
        except (ValueError, IndexError):
            pass
    dev = _nvsmi(["--query-gpu=memory.used,utilization.gpu,power.draw"])
    if not dev:
        return None, None, None, None

    def num(x):
        try:
            return float(x)
        except ValueError:
            return None

    mem, util, power = (num(v) for v in dev[0][:3])
    return vram_proc, mem, util, power


def _dir_size(path):
    total = 0
    for dp, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total


def run(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    disk0 = _dir_size(args.disk_path) if args.disk_path else None

    t0 = time.monotonic()
    proc = subprocess.Popen(args.cmd)
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda sig, _f: proc.send_signal(sig))

    # Acumuladores por pid: los procesos que terminan conservan su último valor.
    last = {}
    rows = []
    while True:
        alive = proc.poll() is None
        t = time.monotonic() - t0
        pids = _tree(proc.pid) if alive else []
        rss = 0
        for p in pids:
            s = _proc_sample(p)
            if s:
                last[p] = s
                rss += s[0]
        cpu = sum(v[1] for v in last.values())
        rb = sum(v[2] for v in last.values())
        wb = sum(v[3] for v in last.values())
        if alive:
            vram_proc, vram_dev, util, power = _gpu_sample(pids)
            rows.append([round(t, 3), rss, round(cpu, 3), rb, wb,
                         vram_proc, vram_dev, util, power])
        else:
            break
        # Espera el intervalo, pero sale apenas el proceso termina.
        try:
            proc.wait(timeout=args.interval)
        except subprocess.TimeoutExpired:
            pass
    rc = proc.returncode
    wall = time.monotonic() - t0

    cols = ["t_s", "rss_bytes", "cpu_s", "read_bytes", "write_bytes",
            "vram_proc_mb", "vram_device_mb", "gpu_util_pct", "power_w"]
    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)

    def col(i):
        return [r[i] for r in rows if r[i] is not None]

    def mean(v):
        return sum(v) / len(v) if v else None

    # Energía: integral de la potencia en el tiempo (regla del trapecio).
    energy_j = 0.0
    for a, b in zip(rows, rows[1:]):
        if a[8] is not None and b[8] is not None:
            energy_j += (a[8] + b[8]) / 2 * (b[0] - a[0])

    summary = {
        "cmd": args.cmd,
        "label": out.name,
        "returncode": rc,
        "interval_s": args.interval,
        "n_samples": len(rows),
        "wall_s": round(wall, 2),
        "ram_peak_mb": round(max(col(1), default=0) / 2**20, 1),
        "ram_mean_mb": round((mean(col(1)) or 0) / 2**20, 1),
        "cpu_s": round(cpu, 1),
        "cpu_mean_cores": round(cpu / wall, 2) if wall else None,
        "disk_read_mb": round(rb / 2**20, 2),
        "disk_write_mb": round(wb / 2**20, 2),
        "output_growth_mb": (round((_dir_size(args.disk_path) - disk0) / 2**20, 3)
                             if args.disk_path else None),
        "vram_proc_peak_mb": max(col(5), default=None),
        "vram_proc_mean_mb": round(mean(col(5)), 1) if col(5) else None,
        "vram_device_peak_mb": max(col(6), default=None),
        "gpu_util_mean_pct": round(mean(col(7)), 1) if col(7) else None,
        "gpu_util_max_pct": max(col(7), default=None),
        "power_mean_w": round(mean(col(8)), 1) if col(8) else None,
        "power_max_w": max(col(8), default=None),
        # None si la GPU no informa potencia (p. ej. GTX 1050 Ti → "[N/A]").
        "energy_wh": round(energy_j / 3600, 3) if col(8) else None,
    }
    with open(f"{out}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return rc


SUMMARY_COLS = [
    ("label", "corrida", "{}"), ("wall_s", "tiempo s", "{:.0f}"),
    ("ram_peak_mb", "RAM pico MB", "{:.0f}"), ("cpu_mean_cores", "CPU núcleos", "{:.2f}"),
    ("vram_proc_peak_mb", "VRAM pico MB", "{:.0f}"), ("gpu_util_mean_pct", "GPU %", "{:.0f}"),
    ("power_mean_w", "W medio", "{:.1f}"), ("energy_wh", "energía Wh", "{:.1f}"),
    ("disk_read_mb", "lee MB", "{:.1f}"), ("disk_write_mb", "escribe MB", "{:.1f}"),
    ("output_growth_mb", "salida MB", "{:.2f}"),
]


def summarize(path):
    files = sorted(Path(path).glob("*.json"))
    if not files:
        print(f"Sin resúmenes en {path}")
        return 1
    print("| " + " | ".join(h for _, h, _ in SUMMARY_COLS) + " |")
    print("|" + "---|" * len(SUMMARY_COLS))
    for fp in files:
        d = json.load(open(fp))
        cells = []
        for k, _, fmt in SUMMARY_COLS:
            v = d.get(k)
            cells.append("—" if v is None else fmt.format(v))
        print("| " + " | ".join(cells) + " |")
    return 0


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "summarize":
        sys.exit(summarize(sys.argv[2] if len(sys.argv) > 2 else "."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="prefijo de salida (sin extensión)")
    ap.add_argument("--interval", type=float, default=1.0, help="segundos entre muestras")
    ap.add_argument("--disk-path", default=None,
                    help="carpeta cuyo crecimiento se mide (lo que la corrida escribe)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- comando a medir")
    args = ap.parse_args()
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        ap.error("falta el comando después de --")
    sys.exit(run(args))


if __name__ == "__main__":
    main()

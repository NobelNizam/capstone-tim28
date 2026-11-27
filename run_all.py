"""Satu perintah reproduksi tim.

    python run_all.py [--seeds 1 2 3] [--final] [--hanya <folder> ...]

Menjalankan baseline starter dan ``models/<folder>/run.py`` setiap anggota untuk tiap seed, lalu
merakit ``reports/hasil_komparasi.md`` dari ``results/*/metrics.json`` dan ``results/*/tambahan.json``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

AKAR = Path(__file__).resolve().parent


def jalankan(perintah: list[str]) -> None:
    print("$", " ".join(perintah), flush=True)
    subprocess.run(perintah, check=True, cwd=AKAR)


def baca(folder: str) -> tuple[dict, dict]:
    d = AKAR / "results" / folder
    m = json.loads((d / "metrics.json").read_text("utf-8"))
    t = json.loads((d / "tambahan.json").read_text("utf-8")) if (d / "tambahan.json").exists() else {}
    return m, t


def ringkas(nilai: list[float]) -> str:
    if len(nilai) == 1:
        return f"{nilai[0]:.4f}"
    return f"{statistics.mean(nilai):.4f} ± {statistics.stdev(nilai):.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--hanya", nargs="*", default=None, help="folder model tertentu saja")
    ap.add_argument("--tanpa-baseline", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(AKAR))
    from src.protokol import data
    if not a.tanpa_baseline and getattr(data, "BASELINE_STARTER", True):
        jalankan([sys.executable, "-m", "src.common.baseline", "--seed", str(a.seeds[0])])
    anggota = sorted(p.name for p in (AKAR / "models").iterdir() if p.is_dir() and (p / "run.py").exists()
                     and p.name != "TEMPLATE" and (a.hanya is None or p.name in a.hanya))
    per_seed: dict[str, list[tuple[dict, dict]]] = {}
    for nama in anggota:
        for s in a.seeds:
            perintah = [sys.executable, str(AKAR / "models" / nama / "run.py"), "--seed", str(s)]
            if a.final and s == a.seeds[-1]:
                perintah.append("--final")
            jalankan(perintah)
            per_seed.setdefault(nama, []).append(baca(nama))

    for d in sorted((AKAR / "results").iterdir()):
        if d.is_dir() and (d / "metrics.json").exists() and (d / "metrics.json").stat().st_size > 0 and d.name not in per_seed:
            per_seed[d.name] = [baca(d.name)]

    kunci, label = getattr(data, "METRIK_TIM", ("val_mae", "val_MAE"))
    regresi = getattr(data, "TUGAS", "klasifikasi") == "regresi"
    k1, k2 = ("val_mae", "val_rmse") if regresi else ("val_acc", "val_macro_f1")
    baris = [f"| Folder | Model | seed | {k1} | {k2} | {label} | parameter | latensi (ms) | durasi (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for nama in ["baseline"] + [n for n in sorted(per_seed) if n != "baseline"]:
        if nama not in per_seed:
            continue
        daftar = per_seed[nama]
        m0, t0 = daftar[-1]
        acc = ringkas([m["metrik"][k1] for m, _ in daftar])
        f1 = ringkas([m["metrik"][k2] for m, _ in daftar])
        mae = ringkas([t[kunci] for _, t in daftar]) if all(kunci in t for _, t in daftar) else "—"
        seeds = ",".join(str(m["seed"]) for m, _ in daftar)
        baris.append(f"| {nama} | {m0.get('model', '')} | {seeds} | {acc} | {f1} | {mae} | "
                     f"{t0.get('parameter', '—')} | {t0.get('latensi_ms', '—')} | {m0.get('durasi_detik', 0):.0f} |")
    (AKAR / "reports").mkdir(exist_ok=True)
    out = AKAR / "reports" / "hasil_komparasi.md"
    out.write_text("# Hasil komparasi (dibangkitkan run_all.py)\n\n" + "\n".join(baris) + "\n", encoding="utf-8")
    print("tabel komparasi →", out.relative_to(AKAR))


if __name__ == "__main__":
    main()

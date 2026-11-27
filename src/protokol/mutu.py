"""Pemeriksaan mutu data tim 28 → reports/mutu-data.md + reports/mutu.json

    python -m src.protokol.mutu

Per sesi: kelengkapan, sambungan ESP32, aktuator yang benar-benar bergerak. Per blok: lama, titik prakiraan, rentang
daya kanal kipas dan suhu. Alasan target dipindah dari daya ke suhu (daya datar), galat persistensi tiap sinyal,
resolusi DHT11, dan berapa titik validasi yang suhunya benar-benar berubah.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime

import numpy as np

from src.common import manifest
from src.protokol import data


def _f(x, d: int = 2) -> str:
    return "—" if x is None else f"{x:.{d}f}".replace(".", ",")


def _persist(blok: list[dict], k: int, h: int) -> float:
    e = []
    for b in blok:
        v = b["nilai"][:, k]
        for t in range(data.RIWAYAT - 1, len(v) - max(data.HORIZON)):
            e.append(abs(v[t + h] - v[t]))
    return float(np.mean(e)) if e else float("nan")


def main() -> None:
    h: dict = {"sesi": {}}
    for s in data.SESI:
        f = data.FOLDER[s]
        log = (f / "log.txt").read_text(encoding="utf-8").strip().splitlines() if (f / "log.txt").exists() else []
        with (f / "aktuator.csv").open(encoding="utf-8") as g:
            akt = Counter(f"{r['nama']}={int(float(r['nilai']) > 0)}" for r in csv.DictReader(g))
        h["sesi"][s] = {"wifi_putus": sum("Wi-Fi ke ESP32 putus" in r for r in log), "aktuator": dict(akt),
                        "selesai": bool(log) and "Selesai" in log[-1]}
    blok = [b for b in data.blok() if len(b["menit"]) >= 30]
    c = data.contoh(); bg = data.bagian()
    h["blok"] = []
    for b in blok:
        v = b["nilai"]
        h["blok"].append({"nama": b["nama"], "menit": len(b["menit"]),
                          "selesai": f"{datetime.fromtimestamp(int(b['menit'][-1]) * 60, data.WIB):%H.%M}",
                          "titik": int((c["blok"] == b["nama"]).sum()),
                          "daya_W": [round(float(v[:, 0].min()), 2), round(float(v[:, 0].max()), 2)],
                          "suhu_C": [round(float(v[:, 2].min()), 1), round(float(v[:, 2].max()), 1)],
                          "peran": ("uji" if b["nama"] in data.UJI_BLOK else "validasi" if b["nama"] in data.VAL_BLOK
                                    else "latih" if (c["blok"] == b["nama"]).any() else "tidak dipakai (< 3 jam)")})
    pakai = [b for b in blok if (c["blok"] == b["nama"]).any()]
    h["persistensi"] = {nama: {f"h{x}": round(_persist(pakai, k, x), 3) for x in (15, 60)}
                        for k, nama in ((0, "daya_W"), (1, "tegangan_mV"), (2, "suhu_C"))}
    tim = [b for b in blok if b["nama"] in data.UJI_BLOK][0]["nilai"]
    h["daya_tim_maks_W"] = round(float(tim[:, 0].max()), 3)
    suhu_mentah = []
    with (data.FOLDER[data.SESI_TIM] / "lingkungan.csv").open(encoding="utf-8") as g:
        for r in csv.DictReader(g):
            try:
                x = float(r["suhu_C"])
            except ValueError:
                continue
            if x == x:
                suhu_mentah.append(x)
    bulat = Counter(int(x) for x in suhu_mentah).most_common(1)[0][0]
    desimal = Counter(int(round((x - bulat) * 10)) for x in suhu_mentah if int(x) == bulat)
    h["suhu_bulat_dominan"] = bulat
    h["suhu_desimal_min_maks"] = [min(desimal.values()), max(desimal.values())]
    for nama in ("latih", "val", "uji"):
        m = bg[nama]; y = c["y"][m]                       # y = perubahan suhu; galat persistensi = |y|
        ubah = np.abs(y).max(1) > data.AMBANG_PERISTIWA
        h[f"n_{nama}"] = int(m.sum()); h[f"berubah_{nama}"] = int(ubah.sum())
        h[f"persist_{nama}"] = round(float(np.abs(y).mean()), 3)
    (manifest.AKAR / "reports").mkdir(exist_ok=True)
    (manifest.AKAR / "reports" / "mutu.json").write_text(json.dumps(h, indent=2, ensure_ascii=False) + "\n", "utf-8")

    p = h["persistensi"]
    akt_tim = h["sesi"][data.SESI_TIM]["aktuator"]
    temuan = [
        f"**Daya datar, target dipindah ke suhu.** Daya kanal kipas (satu-satunya daya yang terukur) di sesi tim maksimum "
        f"{_f(h['daya_tim_maks_W'], 3)} W sepanjang 4 jam: lampu mati (di luar jam 7–18), kipas mati (suhu tidak melewati "
        f"28,5 °C), pompa menyala {akt_tim.get('pompa=1', 0)}× 30 s. Persistensi daya pada semua blok: MAE "
        f"{_f(p['daya_W']['h15'], 3)} W (15 menit) dan {_f(p['daya_W']['h60'], 3)} W (60 menit), hanya dari dua kejadian "
        "kipas mati/nyala dalam ±29 jam.",
        f"**Suhu berubah nyata.** Persistensi suhu MAE {_f(p['suhu_C']['h15'])} °C (15 menit) dan {_f(p['suhu_C']['h60'])} °C "
        "(60 menit). Hari kerja (18 dan 19 Nov): cahaya ruang naik sekitar 07.00–07.10, lalu suhu turun 3–5 °C dalam ±20 "
        "menit (dugaan: AC ruang dinyalakan). Minggu (15 Nov): suhu naik perlahan. Malam: datar.",
        f"**Desimal DHT11 tersebar rata.** Di sesi tim, nilai {bulat},0 sampai {bulat},9 °C masing-masing muncul "
        f"{h['suhu_desimal_min_maks'][0]}–{h['suhu_desimal_min_maks'][1]} kali (per detik). Dugaan: hanya bagian bulat yang "
        "bermakna (DHT11 ±2 °C); suhu per menit tampak meloncat ±1 °C, dan galat di bawah ±0,5 °C setara resolusi sensor.",
        f"**Titik yang suhunya berubah > {_f(data.AMBANG_PERISTIWA, 1)} °C dalam 60 menit:** latih {h['berubah_latih']} dari "
        f"{h['n_latih']}, validasi {h['berubah_val']} dari {h['n_val']}, uji {h['berubah_uji']} dari {h['n_uji']}. Uji (sesi tim, "
        "malam) hampir datar: angka uji nanti terutama mengukur kestabilan, bukan kemampuan meramal perubahan.",
        "**Beban pompa hanya di tegangan rel.** INA219 arus total (0x44) tidak terpasang; denyut pompa 30 s terlihat "
        f"sebagai turunnya bus_mV (persistensi tegangan {_f(p['tegangan_mV']['h15'], 0)} mV pada 15 menit).",
        "Program harian 14–19 Nov memakai pengaturan web *selalu menyala* untuk lampu, kipas, dan pompa (aturan jam dan "
        "termostat tidak berjalan); meski begitu daya kipas terukur 0 W sepanjang 15 dan 18 Nov (penyebabnya tidak "
        "tercatat di log).",
    ]
    baris = ["# Mutu data tim 28 (sesi tim + program harian bersama)", "", "Dibangkitkan `python -m src.protokol.mutu`.", "",
             "## Sesi", "", "| Sesi | Peran | Selesai normal | Wi-Fi putus | Perintah aktuator |", "|---|---|---|---|---|",
             *[f"| `{s}` | {'tim (uji)' if s == data.SESI_TIM else 'program harian'} | {'ya' if v['selesai'] else 'tidak'} | "
               f"{v['wifi_putus']} | {', '.join(f'{k} ×{n}' for k, n in sorted(v['aktuator'].items())) or '—'} |"
               for s, v in h["sesi"].items()],
             "", "## Blok bersambung", "", "| Blok (mulai) | Selesai | Menit | Titik prakiraan | Daya kipas (W) | Suhu (°C) | Peran |",
             "|---|---|---|---|---|---|---|",
             *[f"| {b['nama']} | {b['selesai']} | {b['menit']} | {b['titik']} | {_f(b['daya_W'][0])}–{_f(b['daya_W'][1])} | "
               f"{_f(b['suhu_C'][0], 1)}–{_f(b['suhu_C'][1], 1)} | {b['peran']} |" for b in h["blok"]],
             "", "## Galat persistensi per sinyal (semua blok terpakai)", "", "| Sinyal | 15 menit | 60 menit |", "|---|---|---|",
             f"| daya kanal kipas (W) | {_f(p['daya_W']['h15'], 3)} | {_f(p['daya_W']['h60'], 3)} |",
             f"| tegangan rel (mV) | {_f(p['tegangan_mV']['h15'], 1)} | {_f(p['tegangan_mV']['h60'], 1)} |",
             f"| suhu (°C) | {_f(p['suhu_C']['h15'])} | {_f(p['suhu_C']['h60'])} |",
             "", "## Temuan", "", *(f"- {t}" for t in temuan), ""]
    (manifest.AKAR / "reports" / "mutu-data.md").write_text("\n".join(baris), encoding="utf-8")
    print(json.dumps({k: v for k, v in h.items() if k != "sesi"}, ensure_ascii=False)[:700])


if __name__ == "__main__":
    main()

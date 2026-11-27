"""Analisis validasi → reports/kesalahan.md + reports/kesalahan.json

    python -m src.protokol.analisis

Per model: MAE per horizon, MAE rata-rata, skill terhadap persistensi, MAE pada titik yang suhunya berubah, dan
galat pada pagi validasi (jendela AC menyala) vs sisa malam.
"""
from __future__ import annotations

import json

import numpy as np

from src.common import manifest
from src.protokol import data

NAMA = {
    "baseline": "baseline starter (ridge)",
    "123450005": "TL+AE (123450005)",
    "123450033": "RNN (123450033)",
    "123450116": "Model 123450116",
    "123450117": "Model 123450117",
    "non-dl": "non-DL",
    "persistensi": "persistensi",
}


def _f(x, d: int = 2) -> str:
    return "—" if x is None else f"{x:.{d}f}".replace(".", ",")


def main() -> None:
    akar = manifest.AKAR
    b = data.bagian(); c = b["c"]
    y = c["y"][b["val"]].astype(np.float64)              # perubahan suhu; persistensi = perubahan 0
    jam = c["jam"][b["val"]]
    pagi = (jam >= 6) & (jam < 9)
    pred = {"persistensi": np.zeros_like(y)}
    for d in sorted((akar / "results").iterdir()):
        if (d / "pred_val.npy").exists():
            pred[d.name] = np.load(d / "pred_val.npy").reshape(len(y), -1).astype(np.float64)
    hasil = {}
    for nama, p in pred.items():
        e = np.abs(p - y)
        hasil[nama] = {**data.metrik_tim(y, p), "mae_pagi": round(float(e[pagi].mean()), 4),
                       "mae_malam": round(float(e[~pagi].mean()), 4)}
    kol = list(hasil)
    n_ubah = hasil["persistensi"]["val_n_berubah"]
    baris = ["# Analisis validasi", "",
             f"Validasi = blok {', '.join(data.VAL_BLOK)} ({len(y)} titik prakiraan; {n_ubah} titik suhunya berubah "
             f"> {_f(data.AMBANG_PERISTIWA, 1)} °C dalam 60 menit; {int(pagi.sum())} titik pukul 06–09). Galat dalam °C.", "",
             "| Ukuran | " + " | ".join(NAMA.get(k, k) for k in kol) + " |", "|---|" + "---|" * len(kol),
             "| **MAE rata-rata** | " + " | ".join(f"**{_f(hasil[k]['val_mae_rata'])}**" for k in kol) + " |"]
    for i, h in enumerate(data.KELAS):
        baris.append(f"| MAE {h} | " + " | ".join(_f(hasil[k]["val_mae_rinci"][h]) for k in kol) + " |")
    baris += ["| skill vs persistensi | " + " | ".join(_f(hasil[k]["val_skill"]) for k in kol) + " |",
              "| MAE titik berubah | " + " | ".join(_f(hasil[k]["val_mae_berubah"]) for k in kol) + " |",
              "| MAE pukul 06–09 | " + " | ".join(_f(hasil[k]["mae_pagi"]) for k in kol) + " |",
              "| MAE di luar 06–09 | " + " | ".join(_f(hasil[k]["mae_malam"]) for k in kol) + " |", "",
              "Skill = 1 − MAE model / MAE persistensi pada titik yang sama: positif = lebih baik daripada \"suhu nanti = "
              "suhu sekarang\".", ""]
    (akar / "reports").mkdir(exist_ok=True)
    (akar / "reports" / "kesalahan.md").write_text("\n".join(baris), encoding="utf-8")
    (akar / "reports" / "kesalahan.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(f"{len(kol)} kolom → reports/kesalahan.md")


if __name__ == "__main__":
    main()

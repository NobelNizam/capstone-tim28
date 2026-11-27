"""Pembanding non-deep learning tim 28 (rencana tim): persistensi dan rata-rata musiman per jam.

    python models/non-dl/run.py --seed 1 [--final]

Tiga calon tanpa pelatihan iteratif, dipilih dengan MAE validasi (semuanya tercatat di tambahan.json):
- ``persistensi``: suhu(t+h) = suhu(t).
- ``musiman``: suhu(t+h) = rata-rata suhu data latih pada jam WIB yang sama dengan t+h (jam yang tidak ada di data
  latih memakai rata-rata seluruh data latih).
- ``persistensi_musiman``: suhu(t) + [musiman(jam t+h) − musiman(jam t)]; jam yang tidak ada di data latih → selisih 0.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402
from src.tim import latih  # noqa: E402

FOLDER = Path(__file__).resolve().parent.name


def tabel_musiman(c: dict, m: np.ndarray) -> dict[int, float]:
    """Rata-rata suhu (absolut) per jam WIB dari titik latih (jam target = jam titik awal + h)."""
    kumpul: dict[int, list[float]] = {}
    for jam, q, y in zip(c["jam"][m], c["persist"][m], c["y"][m]):
        for h, v in zip(data.HORIZON, y):
            kumpul.setdefault(int((jam + h / 60) % 24), []).append(float(q + v))
    return {j: float(np.mean(v)) for j, v in kumpul.items()}


def ramal(nama: str, c: dict, m: np.ndarray, S: dict[int, float], umum: float) -> np.ndarray:
    q, jam = c["persist"][m].astype(np.float64), c["jam"][m]
    keluar = np.zeros((len(q), len(data.HORIZON)))
    for i, h in enumerate(data.HORIZON):
        jt = ((jam + h / 60) % 24).astype(int); j0 = (jam % 24).astype(int)
        if nama == "persistensi":
            keluar[:, i] = q
        elif nama == "musiman":
            keluar[:, i] = [S.get(int(j), umum) for j in jt]
        else:
            keluar[:, i] = q + [S[int(a)] - S[int(b)] if int(a) in S and int(b) in S else 0.0 for a, b in zip(jt, j0)]
    return (keluar - q[:, None]).astype(np.float32)        # target protokol = perubahan suhu dari menit t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()
    t0 = time.time()
    data.muat(standar=False)
    b = data.bagian(); c = b["c"]
    S = tabel_musiman(c, b["latih"]); umum = float(np.mean(list(S.values())))
    yva = c["y"][b["val"]]
    skor = {n: data.metrik_tim(yva, ramal(n, c, b["val"], S, umum))
            for n in ("persistensi", "musiman", "persistensi_musiman")}
    pilih = min(skor, key=lambda k: skor[k]["val_mae_rata"])
    pred_val = ramal(pilih, c, b["val"], S, umum)
    pred_test = ramal(pilih, c, b["uji"], S, umum) if a.final else None
    tambahan = {"jalur": "non-dl", "seed": a.seed, "dipilih": pilih, "calon": skor,
                "jam_musiman_latih": sorted(S), **data.metrik_tim(yva, pred_val)}
    latih.tulis_hasil(FOLDER, f"non-dl-{pilih}", a.seed, pred_val, pred_test, time.time() - t0, a.perangkat, tambahan)


if __name__ == "__main__":
    main()

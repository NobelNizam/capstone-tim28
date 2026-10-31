"""Penilai hasil model untuk kasus apa pun (milik starter, jangan diubah).

    python -m src.common.evaluate --nim 12345678 --model cnn --seed 1 \
        --pred-val results/12345678/pred_val.npy --durasi 540 --perangkat "cpu laptop"
    # sprint terakhir, SEKALI, oleh Penjaga Data:
    python -m src.common.evaluate ... --pred-test results/12345678/pred_test.npy

Label benar diambil dari ``src.protokol.data.muat()`` (protokol tim yang dibekukan) dan jenis
tugas dari ``src.protokol.data.TUGAS`` (``klasifikasi`` | ``regresi``). Menulis
``results/<nim>/metrics.json`` berskema baku Workbench: ``metrik`` (angka), ``seed`` (bulat),
``data_hash`` (heksadesimal), ``durasi_detik``, ``perangkat``, ditambah ``model`` dan ukuran data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import manifest, split

AKAR = manifest.AKAR


def akurasi(y: np.ndarray, p: np.ndarray) -> float:
    return float((y == p).mean())


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    f1 = []
    for c in np.unique(np.concatenate([y, p])):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def metrik_klasifikasi(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if p.ndim == 2:
        p = p.argmax(axis=1)
    y, p = y.astype(np.int64).ravel(), p.astype(np.int64).ravel()
    return {"acc": akurasi(y, p), "macro_f1": macro_f1(y, p)}


def metrik_regresi(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = y.astype(np.float64).ravel(), p.astype(np.float64).ravel()
    e = y - p
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()),
            "r2": float(1 - (e ** 2).sum() / ss_tot) if ss_tot > 0 else 0.0}


def _muat_pred(path: str, n: int) -> np.ndarray:
    p = np.load(path)
    if (p.shape[0] if p.ndim else 0) != n:
        raise SystemExit(f"{path}: panjang {p.shape[0] if p.ndim else 0}, seharusnya {n}")
    return p


def nilai(pred_val: str, pred_test: str | None = None) -> tuple[dict[str, float], dict[str, int]]:
    from src.protokol import data  # protokol tim (dibekukan Sprint 0)

    tugas = getattr(data, "TUGAS", "klasifikasi")
    hitung = metrik_regresi if tugas == "regresi" else metrik_klasifikasi
    _, _, _, y_val, _, y_test = data.muat()
    p_val = _muat_pred(pred_val, len(y_val))
    metrik = {f"val_{k}": v for k, v in hitung(np.asarray(y_val), p_val).items()}
    ukuran = {"n_val": int(len(y_val)), "n_test": 0}
    if pred_test:
        if y_test is None:
            raise SystemExit("protokol tim tidak menyediakan y_test")
        p_te = _muat_pred(pred_test, len(y_test))
        metrik.update({f"test_{k}": v for k, v in hitung(np.asarray(y_test), p_te).items()})
        ukuran["n_test"] = int(len(y_test))
    return metrik, ukuran


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nim", required=True)
    ap.add_argument("--model", required=True, help="jalur: cnn | tl | rnn | trf | ae | baseline")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pred-val", required=True)
    ap.add_argument("--pred-test", default=None, help="hanya pada sprint terakhir, sekali")
    ap.add_argument("--durasi", type=float, default=None, help="detik pelatihan")
    ap.add_argument("--perangkat", default="cpu laptop")
    ap.add_argument("--out", default=None, help="bawaan results/<nim>/metrics.json")
    a = ap.parse_args()

    metrik, ukuran = nilai(a.pred_val, a.pred_test)
    hasil = {"metrik": {k: round(float(v), 6) for k, v in metrik.items()}, "seed": int(a.seed),
             "data_hash": manifest.data_hash(f"split:{split.SEED_SPLIT}".encode()),
             "perangkat": a.perangkat, "model": a.model, **ukuran}
    if a.durasi is not None:
        hasil["durasi_detik"] = float(a.durasi)
    out = Path(a.out) if a.out else AKAR / "results" / a.nim / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(hasil, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(hasil["metrik"]), "→", out.relative_to(AKAR) if out.is_relative_to(AKAR) else out)


if __name__ == "__main__":
    main()

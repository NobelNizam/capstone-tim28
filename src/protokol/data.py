"""Pemuat data kasus tim — MILIK TIM, dibekukan setelah Sprint 0 dengan persetujuan pembimbing.

Isi oleh Penjaga Data & Evaluasi. Kontrak yang dipakai penilai dan baseline starter:

    TUGAS = "klasifikasi"  # atau "regresi"
    muat() -> (X_train, y_train, X_val, y_val, X_test, y_test)

- X berupa array NumPy (bentuk bebas: [N, fitur], [N, C, H, W], [N, langkah, fitur]);
  y klasifikasi = label bulat 0..K-1, y regresi = float.
- Split latih/validasi WAJIB memakai ``src.common.split`` (seed 2026) agar sama di semua laptop.
- y_test boleh ``None`` bila dataset tidak punya bagian uji; bila ada, data uji hanya dievaluasi
  sekali di sprint terakhir.
- Berkas data dibaca dari ``data/`` dan dicantumkan di ``data/MANIFEST.json`` (sidik data).

Contoh lengkap: ``contoh_fashion_mnist.py`` di folder ini.
"""
from __future__ import annotations

import numpy as np

from src.common import manifest, split

TUGAS = "klasifikasi"


def muat():
    """Ganti isi fungsi ini dengan pemuat dataset kasus tim Anda."""
    m = manifest.baca()
    raise NotImplementedError(
        f"Isi src/protokol/data.py untuk dataset {m.get('sumber', '?')} (lihat contoh_fashion_mnist.py)."
    )
    # --- pola yang lazim ---------------------------------------------------------------
    # import pandas as pd
    # df = pd.read_csv(manifest.AKAR / "data" / "latih.csv")
    # X = df.drop(columns=["target"]).to_numpy(np.float32); y = df["target"].to_numpy()
    # tr, va = split.indeks(len(X))               # atau split.indeks_urut(len(X)) untuk deret waktu
    # uji = pd.read_csv(manifest.AKAR / "data" / "uji.csv")   # bila ada
    # return X[tr], y[tr], X[va], y[va], X_uji, y_uji

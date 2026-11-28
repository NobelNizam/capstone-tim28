"""Pemisahan latih/validasi/uji per SATUAN (ulangan, partisipan, blok), bukan per jendela.

Jendela dari satu kondisi sangat mirip satu sama lain. Bila jendela diacak satu per satu, jendela
"kembar" masuk ke latih dan uji sekaligus dan akurasi tampak tinggi padahal model hanya menghafal
kondisi. Karena itu ``src.common.split.indeks`` (split beku starter, seed 2026) dipakai atas
DAFTAR SATUAN, lalu setiap jendela mengikuti satuannya.
"""
from __future__ import annotations

import numpy as np

from src.common import split


def bagi_satuan(satuan: np.ndarray, porsi_uji: float, porsi_val: float) -> tuple[list, list, list]:
    """Kembalikan (satuan_latih, satuan_val, satuan_uji), masing-masing terurut."""
    unik = np.array(sorted(set(satuan.tolist()), key=lambda v: (str(type(v)), v)))
    sisa_idx, uji_idx = split.indeks(len(unik), porsi_val=porsi_uji)
    sisa = unik[np.sort(sisa_idx)]
    lat_idx, val_idx = split.indeks(len(sisa), porsi_val=porsi_val)
    return (sorted(sisa[lat_idx].tolist()), sorted(sisa[val_idx].tolist()), sorted(unik[uji_idx].tolist()))


def topeng(satuan: np.ndarray, daftar: list) -> np.ndarray:
    return np.isin(satuan, np.asarray(daftar))


def periksa_tidak_bocor(*kelompok: np.ndarray) -> None:
    """Gagal bila satu satuan muncul di lebih dari satu bagian (latih/val/uji)."""
    himpunan = [set(np.asarray(k).tolist()) for k in kelompok]
    for i in range(len(himpunan)):
        for j in range(i + 1, len(himpunan)):
            irisan = himpunan[i] & himpunan[j]
            if irisan:
                raise AssertionError(f"satuan bocor antara bagian {i} dan {j}: {sorted(irisan)[:5]}")
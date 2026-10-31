"""Split beku untuk kasus apa pun (milik starter, jangan diubah).

``indeks(n, porsi_val=0.1, seed=2026)`` memberi ``(idx_latih, idx_val)`` yang sama di semua
laptop untuk ``n`` data latih yang sama. Untuk deret waktu, ``indeks_urut(n)`` memberi split
kronologis (validasi = bagian akhir) tanpa pengacakan.
"""
from __future__ import annotations

import numpy as np

SEED_SPLIT = 2026


def indeks(n: int, porsi_val: float = 0.1, seed: int = SEED_SPLIT) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < porsi_val < 1:
        raise ValueError("porsi_val harus di antara 0 dan 1")
    urut = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(round(n * porsi_val)))
    return urut[n_val:], urut[:n_val]


def indeks_urut(n: int, porsi_val: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Split kronologis untuk deret waktu: validasi adalah ``porsi_val`` terakhir."""
    n_val = max(1, int(round(n * porsi_val)))
    semua = np.arange(n)
    return semua[: n - n_val], semua[n - n_val:]

"""Contoh protokol tim yang lengkap: Fashion-MNIST dari berkas IDX resmi.

Untuk memakainya sebagai protokol tim: salin isi berkas ini ke ``data.py``, tulis
``data/MANIFEST.json`` yang mencantumkan keempat berkas IDX, dan sesuaikan ``PROTOKOL.md``.
Berkas IDX Fashion-MNIST sudah ada di laptop mahasiswa Deep Learning
(``~/workbench-workspace/deep-learning/data/raw/FashionMNIST/raw``); salin ke ``data/``.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from src.common import manifest, split

TUGAS = "klasifikasi"
FOLDER = manifest.AKAR / "data" / "FashionMNIST" / "raw"


def _baca(nama: str) -> bytes:
    p = FOLDER / nama
    return p.read_bytes() if p.exists() else gzip.decompress((p.with_name(nama + ".gz")).read_bytes())


def _idx(bagian: str) -> tuple[np.ndarray, np.ndarray]:
    awalan = "train" if bagian == "train" else "t10k"
    img = _baca(f"{awalan}-images-idx3-ubyte")
    lbl = _baca(f"{awalan}-labels-idx1-ubyte")
    n = int.from_bytes(img[4:8], "big")
    X = np.frombuffer(img, dtype=np.uint8, offset=16).reshape(n, 28, 28)
    y = np.frombuffer(lbl, dtype=np.uint8, offset=8).astype(np.int64)
    return X, y


def muat():
    X, y = _idx("train")
    tr, va = split.indeks(len(X), porsi_val=0.1)
    Xte, yte = _idx("test")
    return X[tr], y[tr], X[va], y[va], Xte, yte

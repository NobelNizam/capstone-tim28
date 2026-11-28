"""Label gabungan untuk tugas multi-label: satu kelompok saling eksklusif + beberapa label biner.

Penilai starter (``src.common.evaluate``) hanya mengenal satu label bulat per contoh. Agar penilai itu
tetap dipakai tanpa diubah, label dikodekan menjadi satu bilangan::

    kode = utama × 2^B + Σ biner_i × 2^(B-1-i)

Akurasi dari penilai starter = semua label benar sekaligus (*subset accuracy*). Model TIDAK memakai
softmax atas semua kode: kepala model mengeluarkan K logit untuk kelompok utama (softmax) dan B logit
biner (sigmoid), lalu ``dekode`` menggabungkannya kembali menjadi kode. F1 per kelas dihitung di sini.
"""
from __future__ import annotations

import numpy as np


class KodekMulti:
    def __init__(self, nama_utama: list[str], nama_biner: list[str]):
        self.nama_utama = list(nama_utama)
        self.nama_biner = list(nama_biner)
        self.K, self.B = len(nama_utama), len(nama_biner)
        self.n_keluaran = self.K + self.B

    # --- kode <-> label terpisah --------------------------------------------------------------
    def kode(self, utama: np.ndarray, biner: np.ndarray) -> np.ndarray:
        bobot = 2 ** np.arange(self.B - 1, -1, -1)
        return (np.asarray(utama) * 2 ** self.B + np.asarray(biner) @ bobot).astype(np.int64)

    def pisah(self, y: np.ndarray) -> np.ndarray:
        """kode [N] → [N, 1 + B] (kolom 0 = utama, sisanya biner 0/1)."""
        y = np.asarray(y, np.int64)
        biner = (y[:, None] >> np.arange(self.B - 1, -1, -1)[None, :]) & 1
        return np.concatenate([(y >> self.B)[:, None], biner], axis=1)

    def gabung(self, Y: np.ndarray) -> np.ndarray:
        return self.kode(Y[:, 0], Y[:, 1:])

    # --- untuk model ----------------------------------------------------------------------------
    def loss(self, logit, y, w=None):
        """Cross-entropy kelompok utama + BCE label biner (PyTorch)."""
        import torch
        import torch.nn.functional as F

        u = torch.div(y, 2 ** self.B, rounding_mode="floor")
        b = torch.stack([(y >> (self.B - 1 - i)) & 1 for i in range(self.B)], 1).float()
        return F.cross_entropy(logit[:, : self.K], u, weight=w) + F.binary_cross_entropy_with_logits(logit[:, self.K:], b)

    def dekode(self, logit: np.ndarray) -> np.ndarray:
        return self.kode(logit[:, : self.K].argmax(1), (logit[:, self.K:] > 0).astype(np.int64))

    # --- metrik tim -----------------------------------------------------------------------------
    def f1_per_kelas(self, y: np.ndarray, p: np.ndarray) -> dict[str, float]:
        Y, P = self.pisah(y), self.pisah(p)
        hasil = {}
        for i, nama in enumerate(self.nama_utama):
            hasil[nama] = _f1(Y[:, 0] == i, P[:, 0] == i)
        for j, nama in enumerate(self.nama_biner, 1):
            hasil[nama] = _f1(Y[:, j] == 1, P[:, j] == 1)
        return hasil

    def skor(self, y: np.ndarray, p: np.ndarray) -> float:
        """Rata-rata F1 per kelas (semua kelas utama + semua label biner): dasar pemilihan epoch."""
        return float(np.mean(list(self.f1_per_kelas(y, p).values())))


def _f1(benar: np.ndarray, tebak: np.ndarray) -> float:
    tp = int((benar & tebak).sum()); fp = int((~benar & tebak).sum()); fn = int((benar & ~tebak).sum())
    return 0.0 if tp == 0 else round(2 * tp / (2 * tp + fp + fn), 4)


class KodekRegresi:
    """Regresi satu nilai lewat jalur kodek ``src.tim.latih``: model mengeluarkan 1 nilai untuk target terstandar
    ``(y − mu) / sd`` (mu, sd dari data latih), ``dekode`` mengembalikan satuan asli, dan pemilihan epoch memakai
    MAE validasi (``skor`` = −MAE, makin besar makin baik)."""

    dtype_y = np.float32
    K = None

    def __init__(self, mu: float, sd: float):
        self.mu, self.sd = float(mu), float(sd) or 1.0
        self.n_keluaran = 1

    def loss(self, logit, y, w=None):
        import torch.nn.functional as F
        return F.mse_loss(logit[:, 0], (y - self.mu) / self.sd)

    def dekode(self, logit: np.ndarray) -> np.ndarray:
        return (np.asarray(logit)[:, 0] * self.sd + self.mu).astype(np.float64)

    def skor(self, y: np.ndarray, p: np.ndarray) -> float:
        return -float(np.abs(np.asarray(p, np.float64) - np.asarray(y, np.float64)).mean())


class KodekBiner:
    """Beberapa label biner tanpa kelompok utama (mis. kipas / pompa / lampu nyala-mati), dikodekan satu bilangan
    0 … 2^B − 1 untuk penilai starter. Kepala model = B logit sigmoid; ``skor`` = rata-rata F1 per label."""

    K = None

    def __init__(self, nama_biner: list[str]):
        self.nama_biner = list(nama_biner)
        self.B = len(nama_biner)
        self.n_keluaran = self.B

    def kode(self, biner: np.ndarray) -> np.ndarray:
        return (np.asarray(biner) @ (2 ** np.arange(self.B - 1, -1, -1))).astype(np.int64)

    def pisah(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, np.int64)
        return (y[:, None] >> np.arange(self.B - 1, -1, -1)[None, :]) & 1

    def gabung(self, Y: np.ndarray) -> np.ndarray:
        return self.kode(Y)

    def loss(self, logit, y, w=None):
        import torch
        import torch.nn.functional as F
        b = torch.stack([(y >> (self.B - 1 - i)) & 1 for i in range(self.B)], 1).float()
        return F.binary_cross_entropy_with_logits(logit, b)

    def dekode(self, logit: np.ndarray) -> np.ndarray:
        return self.kode((np.asarray(logit) > 0).astype(np.int64))

    def f1_per_kelas(self, y: np.ndarray, p: np.ndarray) -> dict[str, float]:
        Y, P = self.pisah(y), self.pisah(p)
        return {n: _f1(Y[:, j] == 1, P[:, j] == 1) for j, n in enumerate(self.nama_biner)}

    def skor(self, y: np.ndarray, p: np.ndarray) -> float:
        return float(np.mean(list(self.f1_per_kelas(y, p).values())))


class KodekSpektrum:
    """Target berupa spektrogram (pembersihan ucapan): model mengeluarkan taksiran spektrogram bersih terkompresi
    ``|S|^0,3`` berbentuk sama dengan target [N, F, T]. Loss = MSE, ``dekode`` = identitas, pemilihan epoch memakai
    −MSE validasi (makin besar makin baik). Penilai starter (regresi) meratakan larik, jadi bentuk [N, F, T] sah."""

    dtype_y = np.float32
    K = None
    n_keluaran = 1

    def loss(self, logit, y, w=None):
        import torch.nn.functional as F
        return F.mse_loss(logit, y)

    def dekode(self, logit: np.ndarray) -> np.ndarray:
        return np.asarray(logit, np.float32)

    def skor(self, y: np.ndarray, p: np.ndarray) -> float:
        return -float(np.mean((np.asarray(p, np.float64) - np.asarray(y, np.float64)) ** 2))


class KodekRegresiVektor:
    """Regresi beberapa nilai sekaligus (mis. jumlah objek per kelas): target [N, K] distandarkan per kolom dari data
    latih, loss MSE, ``dekode`` mengembalikan satuan asli [N, K], pemilihan epoch = −MAE rata-rata semua kolom."""

    dtype_y = np.float32
    K = None

    def __init__(self, mu: np.ndarray, sd: np.ndarray):
        self.mu = np.asarray(mu, np.float32); self.sd = np.where(np.asarray(sd) > 0, sd, 1.0).astype(np.float32)
        self.n_keluaran = len(self.mu)

    def loss(self, logit, y, w=None):
        import torch
        import torch.nn.functional as F
        return F.mse_loss(logit, (y - torch.from_numpy(self.mu)) / torch.from_numpy(self.sd))

    def dekode(self, logit: np.ndarray) -> np.ndarray:
        return (np.asarray(logit) * self.sd + self.mu).astype(np.float32)

    def skor(self, y: np.ndarray, p: np.ndarray) -> float:
        return -float(np.abs(np.asarray(p, np.float64) - np.asarray(y, np.float64)).mean())
"""Deret sensor ESP32 (imu.csv, daya.csv): baca dengan jam ESP32, potong jendela tanpa celah.

Sensor dikirim ESP32 lewat Wi-Fi. ``t_host`` adalah waktu TIBA di komputer meja: saat sambungan tersendat
data tiba bergerombol, dan saat sambungan putus sampel hilang. Jam yang benar untuk jarak antarsampel adalah
``t_esp_ms`` (jam ESP32, tidak pernah mundur). Jam itu dipetakan ke jam komputer dengan selisih
``t_host − t_esp`` persentil 1 (kiriman tercepat ≈ tanpa antrean), lalu:

- jendela yang memuat celah sampel > ``CELAH_MAKS`` s (sampel hilang saat Wi-Fi putus) DITOLAK;
- jendela yang sah dicuplik ulang ke laju tetap dengan interpolasi linear per kanal.

Baris CSV yang rusak (terpotong saat ditulis) dilewati dan dihitung.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CELAH_MAKS = 0.05


@dataclass
class Deret:
    t: np.ndarray          # [n] waktu host terpetakan dari jam ESP32 (detik epoch)
    X: np.ndarray          # [n, kanal]
    kanal: list[str]
    rusak: int             # baris CSV yang dilewati
    selisih: float         # t_host − t_esp yang dipakai

    def celah(self) -> np.ndarray:
        return np.diff(self.t)


def baca(folder: Path, berkas: str, kolom: list[str], saring: tuple[str, str] | None = None) -> Deret:
    """Baca ``berkas`` (mis. imu.csv), hanya baris ``saring`` (mis. ("imu", "k")), kolom nilai ``kolom``."""
    th, te, nilai, rusak = [], [], [], 0
    with (Path(folder) / berkas).open(encoding="utf-8", errors="replace") as f:
        pembaca = csv.reader(f)
        kepala = next(pembaca)
        idx = [kepala.index(k) for k in kolom]
        i_s = kepala.index(saring[0]) if saring else None
        for r in pembaca:
            if len(r) != len(kepala):
                rusak += 1
                continue
            if saring and r[i_s] != saring[1]:
                continue
            try:
                vals = [float(r[i]) for i in idx]
                a, b = float(r[0]), float(r[1])
            except ValueError:
                rusak += 1
                continue
            if any(np.isnan(v) for v in vals):
                continue
            th.append(a); te.append(b / 1000.0); nilai.append(vals)
    th, te = np.asarray(th), np.asarray(te)
    urut = np.argsort(te, kind="stable")
    th, te, X = th[urut], te[urut], np.asarray(nilai, np.float32)[urut]
    selisih = float(np.percentile(th - te, 1)) if len(th) else 0.0
    return Deret(t=te + selisih, X=X, kanal=list(kolom), rusak=rusak, selisih=selisih)


def potong(d: Deret, t0: float, t1: float, laju: float, celah_maks: float = CELAH_MAKS) -> np.ndarray | None:
    """[kanal, round((t1−t0)·laju)] pada laju tetap, atau None bila ada celah > ``celah_maks`` s."""
    i0, i1 = np.searchsorted(d.t, [t0, t1])
    tt = d.t[max(i0 - 1, 0):min(i1 + 1, len(d.t))]
    if len(tt) < 2 or tt[0] > t0 + celah_maks or tt[-1] < t1 - celah_maks:
        return None
    if np.diff(tt).max() > celah_maks:
        return None
    grid = t0 + np.arange(int(round((t1 - t0) * laju))) / laju
    XX = d.X[max(i0 - 1, 0):min(i1 + 1, len(d.t))]
    return np.stack([np.interp(grid, tt, XX[:, k]) for k in range(XX.shape[1])]).astype(np.float32)


class StandarKanal:
    """Rata-rata dan simpangan per kanal dari data latih; X [N, kanal, T]."""

    def __init__(self, X_latih: np.ndarray):
        self.mu = X_latih.mean(axis=(0, 2), keepdims=True)
        self.sd = X_latih.std(axis=(0, 2), keepdims=True) + 1e-6

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd).astype(np.float32)


def fitur_statistik(X: np.ndarray, laju: float, pita: int = 8) -> np.ndarray:
    """Fitur non-DL per kanal: rata-rata, simpangan, RMS, puncak, kemencengan, keruncingan, energi ``pita`` pita FFT
    (dalam log). X [N, kanal, T] (sebelum normalisasi) → [N, kanal·(6+pita)]."""
    mu = X.mean(-1, keepdims=True)
    Z = X - mu
    sd = Z.std(-1) + 1e-9
    rms = np.sqrt((X ** 2).mean(-1))
    puncak = np.abs(Z).max(-1)
    skew = (Z ** 3).mean(-1) / sd ** 3
    kurt = (Z ** 4).mean(-1) / sd ** 4
    spek = np.abs(np.fft.rfft(Z, axis=-1)) ** 2
    batas = np.linspace(0, spek.shape[-1], pita + 1).astype(int)
    energi = np.stack([np.log(spek[..., a:b].sum(-1) + 1e-9) for a, b in zip(batas, batas[1:])], -1)
    F = np.concatenate([mu[..., 0, None], sd[..., None], rms[..., None], puncak[..., None], skew[..., None],
                        kurt[..., None], energi], -1)
    return F.reshape(len(X), -1).astype(np.float32)
"""Baseline starter untuk kasus apa pun (milik starter, jangan diubah).

    python -m src.common.baseline --seed 2026

Klasifikasi: regresi logistik (softmax) NumPy; regresi: ridge. Dilatih pada split latih dari
protokol tim, dievaluasi pada validasi, ditulis ke ``results/baseline/metrics.json`` lewat
``evaluate`` (jalur yang sama dengan model mahasiswa). Fitur diratakan ke 2 dimensi dan
distandarkan; untuk data mentah yang besar ini hanya batas bawah, bukan pembanding serius.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np

from . import manifest

AKAR = manifest.AKAR


def _rata(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X.reshape(len(X), -1)


def _standar(Xtr: np.ndarray, Xva: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xva - mu) / sd


def logistik(Xtr, ytr, Xva, seed: int, epoch: int = 5, lr: float = 0.1, batch: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = int(ytr.max()) + 1
    W = np.zeros((Xtr.shape[1], k), np.float32); b = np.zeros(k, np.float32)
    n = len(Xtr)
    for _ in range(epoch):
        for i in rng.permutation(n).reshape(-1)[: n - n % batch].reshape(-1, batch):
            z = Xtr[i] @ W + b
            z -= z.max(1, keepdims=True)
            p = np.exp(z); p /= p.sum(1, keepdims=True)
            p[np.arange(batch), ytr[i]] -= 1
            p /= batch
            W -= lr * (Xtr[i].T @ p + 1e-4 * W); b -= lr * p.sum(0)
    return (Xva @ W + b).argmax(1)


def ridge(Xtr, ytr, Xva, lam: float = 1.0) -> np.ndarray:
    Xb = np.hstack([Xtr, np.ones((len(Xtr), 1), np.float32)])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, Xb.T @ ytr.astype(np.float32))
    return np.hstack([Xva, np.ones((len(Xva), 1), np.float32)]) @ w


def main() -> None:
    from src.protokol import data

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    t0 = time.time()
    Xtr, ytr, Xva, _, _, _ = data.muat()
    Xtr, Xva = _standar(_rata(Xtr), _rata(Xva))
    if getattr(data, "TUGAS", "klasifikasi") == "regresi":
        pred = ridge(Xtr, np.asarray(ytr), Xva)
    else:
        pred = logistik(Xtr, np.asarray(ytr, dtype=np.int64), Xva, a.seed).astype(np.int64)
    durasi = time.time() - t0
    out = AKAR / "results" / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", pred)
    subprocess.run([sys.executable, "-m", "src.common.evaluate", "--nim", "baseline", "--model", "baseline",
                    "--seed", str(a.seed), "--pred-val", str(out / "pred_val.npy"),
                    "--durasi", f"{durasi:.1f}", "--perangkat", "cpu laptop"], check=True, cwd=AKAR)


if __name__ == "__main__":
    main()

"""Alat bantu pelatihan milik tim (bukan milik starter): seed, loop latih, prediksi, tulis hasil.

Dipakai ``models/*/run.py`` supaya perbandingan antarjalur adil: optimizer, jadwal, pemilihan
epoch terbaik pada data validasi, dan cara menulis hasil sama. Arsitektur dan hiperparameter
tetap milik masing-masing jalur.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]


def atur_seed(seed: int) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(os.environ.get("TIM_THREADS", "4")))


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    kelas = np.unique(np.concatenate([y, p]))
    f = []
    for c in kelas:
        tp = int(((p == c) & (y == c)).sum()); fp = int(((p == c) & (y != c)).sum()); fn = int(((p != c) & (y == c)).sum())
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def prediksi(model, X: np.ndarray, batch: int = 256) -> np.ndarray:
    """Logit [N, K] dalam mode evaluasi."""
    import torch

    model.eval()
    keluar = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            keluar.append(model(torch.from_numpy(np.ascontiguousarray(X[i:i + batch]))).numpy())
    return np.concatenate(keluar)


def latih(model, Xtr, ytr, Xva, yva, *, seed: int, epoch: int = 15, lr: float = 1e-3, batch: int = 64,
          wd: float = 1e-4, bobot_kelas: bool = True, cetak: str = "", kodek=None) -> dict:
    """Latih model dengan AdamW + cosine; simpan bobot epoch dengan skor validasi terbaik."""
    import copy

    import torch

    g = torch.Generator().manual_seed(seed)
    X = torch.from_numpy(np.ascontiguousarray(Xtr))
    y = torch.from_numpy(np.asarray(ytr, getattr(kodek, "dtype_y", np.int64)))
    k = int(max(ytr.max(), yva.max())) + 1
    w = None
    if bobot_kelas and kodek is not None and getattr(kodek, "K", None) is None:
        pass
    elif bobot_kelas and kodek is not None:
        n = np.bincount(kodek.pisah(ytr)[:, 0], minlength=kodek.K).astype(np.float32)
        w = torch.tensor(n.sum() / (kodek.K * np.maximum(n, 1)), dtype=torch.float32)
    elif bobot_kelas:
        n = np.bincount(ytr, minlength=k).astype(np.float32)
        w = torch.tensor(n.sum() / (k * np.maximum(n, 1)), dtype=torch.float32)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    jadwal = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epoch)
    riwayat, terbaik, bobot = [], (float("-inf"), 0), None
    for e in range(1, epoch + 1):
        model.train()
        urut = torch.randperm(len(X), generator=g)
        total = 0.0
        for i in range(0, len(X), batch):
            j = urut[i:i + batch]
            opt.zero_grad()
            if kodek is not None:
                loss = kodek.loss(model(X[j]), y[j], w)
            else:
                loss = torch.nn.functional.cross_entropy(model(X[j]), y[j], weight=w)
            loss.backward()
            opt.step()
            total += loss.item() * len(j)
        jadwal.step()
        if kodek is not None:
            p = kodek.dekode(prediksi(model, Xva))
            f1, acc = kodek.skor(yva, p), float((p == yva).mean())
        else:
            p = prediksi(model, Xva).argmax(1)
            f1, acc = macro_f1(yva, p), float((p == yva).mean())
        riwayat.append({"epoch": e, "loss": round(total / len(X), 4), "val_f1": round(f1, 4), "val_acc": round(acc, 4)})
        if cetak:
            print(f"[{cetak}] epoch {e:2d} loss {total / len(X):.4f} val_f1 {f1:.4f} val_acc {acc:.4f}", flush=True)
        if f1 > terbaik[0]:
            terbaik, bobot = (f1, e), copy.deepcopy(model.state_dict())
    model.load_state_dict(bobot)
    return {"riwayat": riwayat, "epoch_terbaik": terbaik[1]}


def jumlah_parameter(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def latensi_ms(model, contoh: np.ndarray, ulang: int = 30) -> float:
    """Median waktu inferensi satu jendela di CPU (ms), untuk perbandingan kasar antarjalur."""
    import torch

    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(contoh[:1]))
    t = []
    with torch.no_grad():
        for _ in range(ulang + 5):
            t0 = time.perf_counter(); model(x); t.append(time.perf_counter() - t0)
    return round(float(np.median(t[5:])) * 1000, 2)


def tulis_hasil(folder: str, jalur: str, seed: int, pred_val: np.ndarray, pred_test: np.ndarray | None,
                durasi: float, perangkat: str, tambahan: dict | None = None) -> None:
    """Simpan prediksi, panggil penilai starter (metrics.json), dan tulis tambahan.json milik tim."""
    out = AKAR / "results" / folder
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))
    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", folder, "--model", jalur, "--seed", str(seed),
                "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}", "--perangkat", perangkat]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)
    if tambahan is not None:
        (out / "tambahan.json").write_text(json.dumps(tambahan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

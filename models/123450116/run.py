"""Jalur CNN 1-D (123450116): perubahan suhu +15/+30/+45/+60 menit dari jendela [6 kanal, 120 menit].

    python models/123450116/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name
JALUR = "cnn"
# kernel 5 = jendela 5 menit; model kecil karena latih hanya ~566 titik; dropout dan weight decay
# lebih besar untuk menahan overfitting; berhenti bila MAE validasi tidak membaik 5 epoch.
HIPER = {"epoch": 25, "lr": 1e-3, "batch": 64, "wd": 1e-3, "dropout": 0.3,
         "kernel": 5, "kanal": (16, 32, 32), "sabar": 5}


def buat(n_kanal: int, n_keluaran: int):
    import torch.nn as nn

    k = HIPER["kernel"]
    a, b, c = HIPER["kanal"]

    def blok(i, o):
        return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm1d(o), nn.ReLU(),
                             nn.MaxPool1d(2))

    return nn.Sequential(blok(n_kanal, a), blok(a, b), blok(b, c), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                         nn.Dropout(HIPER["dropout"]), nn.Linear(c, n_keluaran))


def latih_dan_prediksi(seed: int, final: bool):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    Xtr, ytr, Xva, yva, Xte, _ = data.muat()
    Xtr, Xva = np.asarray(Xtr, np.float32), np.asarray(Xva, np.float32)
    ytr, yva = np.asarray(ytr, np.float32), np.asarray(yva, np.float32)
    mu, sd = ytr.mean(0), ytr.std(0) + 1e-6          # target distandarkan dari data latih saja

    model = buat(Xtr.shape[1], ytr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=HIPER["lr"], weight_decay=HIPER["wd"])
    jadwal = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HIPER["epoch"])
    X, Y = torch.from_numpy(Xtr), torch.from_numpy((ytr - mu) / sd)
    g = torch.Generator().manual_seed(seed)

    def prediksi(A):
        model.eval()
        with torch.no_grad():
            z = model(torch.from_numpy(np.ascontiguousarray(A, dtype=np.float32))).numpy()
        return z * sd + mu                              # kembali ke satuan °C

    terbaik = (float("inf"), 0, None)
    for e in range(1, HIPER["epoch"] + 1):
        model.train()
        urut = torch.randperm(len(X), generator=g)
        total = 0.0
        for i in range(0, len(X), HIPER["batch"]):
            j = urut[i:i + HIPER["batch"]]
            opt.zero_grad()
            loss = F.mse_loss(model(X[j]), Y[j])
            loss.backward()
            opt.step()
            total += loss.item() * len(j)
        jadwal.step()
        mae = float(np.abs(prediksi(Xva) - yva).mean())
        print(f"[{NIM}] epoch {e:2d} loss {total / len(X):.4f} val_mae {mae:.4f}", flush=True)
        if mae < terbaik[0]:
            terbaik = (mae, e, copy.deepcopy(model.state_dict()))
        elif e - terbaik[1] >= HIPER["sabar"]:
            break
    model.load_state_dict(terbaik[2])
    print(f"[{NIM}] epoch terbaik {terbaik[1]}, val_mae {terbaik[0]:.4f}")
    pred_test = prediksi(Xte) if final and Xte is not None else None
    return prediksi(Xva), pred_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()
    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0
    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", pred_val)
    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
                "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat]
    if pred_test is not None:
        np.save(out / "pred_test.npy", pred_test)
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()

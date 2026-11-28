"""Kerangka run.py anggota. Salin folder ini menjadi models/<nim>/ lalu isi bagian MODEL.

    python models/<nim>/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM
JALUR = "cnn"                                   # cnn | tl | rnn | trf | ae


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Kembalikan (pred_val, pred_test|None). Ganti isi fungsi ini dengan model Anda."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr, Xva, _, Xte, _ = data.muat()
    # ---- MODEL (contoh minimal; ganti dengan arsitektur jalur Anda) ----
    def ke_tensor(X):
        return torch.tensor(np.asarray(X, dtype=np.float32)).reshape(len(X), -1)
    X, y = ke_tensor(Xtr), torch.tensor(np.asarray(ytr))
    model = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 10))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)       # tulis alasan pilihan optimizer di catatan.md
    for _ in range(2):
        for i in range(0, len(X), 128):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(X[i:i + 128]), y[i:i + 128])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred_val = model(ke_tensor(Xva)).argmax(1).numpy()
        pred_test = model(ke_tensor(Xte)).argmax(1).numpy() if final and Xte is not None else None
    return pred_val, pred_test


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
    np.save(out / "pred_val.npy", np.asarray(pred_val))
    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
                "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()

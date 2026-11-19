"""Model RNN (LSTM) untuk meramal daya meja 15 menit ke depan.

    python models/123450033/run.py --seed 1 --simulasi   # uji kerangka dengan data pura-pura
    python models/123450033/run.py --seed 1              # latih dengan data meja asli (butuh data.py)
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
JALUR = "rnn"                                   # cnn | tl | rnn | trf | ae

# ---- Pengaturan model (nanti diubah-ubah saat tuning di Sprint 2) ----
UKURAN_TERSEMBUNYI = 32   # banyaknya "ingatan" LSTM
DROPOUT = 0.2             # regularisasi: mematikan 20% neuron secara acak saat latihan
LAJU_BELAJAR = 1e-3       # learning rate untuk Adam
EPOCH = 10
UKURAN_BATCH = 64


# ---------------------------------------------------------------------------
# 1. DATA SIMULASI (sementara, sampai data.py dari Penjaga Data siap)
# ---------------------------------------------------------------------------
def data_simulasi(seed: int, panjang: int = 120, ke_depan: int = 15):
    """Data pura-pura yang BENTUKNYA sama dengan data meja.

    X : (N, 120, 3) -> 120 menit terakhir, 3 fitur: daya, suhu, jam
    y : (N,)        -> daya 15 menit setelah jendela berakhir
    Polanya meniru program harian: lampu jam 8-16, kipas jika suhu > 29 C, pompa tiap 120 menit.
    """
    rng = np.random.default_rng(seed)

    def buat(n):
        mulai = rng.uniform(0, 24 * 60, size=n)                         # menit mulai acak dalam sehari
        menit = mulai[:, None] + np.arange(panjang + ke_depan)[None, :]
        jam = (menit / 60) % 24
        suhu = 27 + 2.5 * np.sin(2 * np.pi * (jam - 9) / 24) + rng.normal(0, 0.2, menit.shape)
        lampu = ((jam >= 8) & (jam < 16)) * 1.0
        kipas = (suhu > 29) * 0.8
        pompa = ((menit % 120) < 1) * 0.5
        daya = lampu + kipas + pompa + rng.normal(0, 0.05, menit.shape)
        X = np.stack([daya[:, :panjang], suhu[:, :panjang], jam[:, :panjang]], axis=-1)
        y = daya[:, panjang + ke_depan - 1]
        return X.astype(np.float32), y.astype(np.float32)

    Xtr, ytr = buat(2000)
    Xva, yva = buat(400)
    Xte, yte = buat(400)
    return Xtr, ytr, Xva, yva, Xte, yte


# ---------------------------------------------------------------------------
# 2. MODEL RNN
# ---------------------------------------------------------------------------
def buat_model(n_fitur: int):
    import torch.nn as nn

    class ModelRNN(nn.Module):
        def __init__(self):
            super().__init__()
            # LSTM membaca 120 menit satu per satu sambil "mengingat" polanya
            self.lstm = nn.LSTM(n_fitur, UKURAN_TERSEMBUNYI, batch_first=True)
            self.dropout = nn.Dropout(DROPOUT)
            # Ingatan terakhir LSTM diubah menjadi 1 angka: tebakan daya
            self.keluar = nn.Linear(UKURAN_TERSEMBUNYI, 1)

        def forward(self, x):                 # x: (batch, 120, n_fitur)
            _, (h, _) = self.lstm(x)          # h[-1]: ingatan setelah membaca menit ke-120
            return self.keluar(self.dropout(h[-1])).squeeze(-1)

    return ModelRNN()


# ---------------------------------------------------------------------------
# 3. LATIH DAN PREDIKSI
# ---------------------------------------------------------------------------
def latih_dan_prediksi(seed: int, final: bool, simulasi: bool):
    """Kembalikan (pred_val, pred_test|None, y_val, X_val)."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if simulasi:
        Xtr, ytr, Xva, yva, Xte, _ = data_simulasi(seed)
    else:
        Xtr, ytr, Xva, yva, Xte, _ = data.muat()
    Xtr, Xva = np.asarray(Xtr, np.float32), np.asarray(Xva, np.float32)
    ytr = np.asarray(ytr, np.float32)

    # Normalisasi: rata-rata dan simpangan baku dihitung dari data LATIH saja
    # (aturan kasus tim: "normalisasi dari data latih saja")
    rx, sx = Xtr.mean(axis=(0, 1)), Xtr.std(axis=(0, 1)) + 1e-8
    ry, sy = ytr.mean(), ytr.std() + 1e-8

    def ke_tensor(X):
        return torch.tensor((np.asarray(X, np.float32) - rx) / sx)

    X, y = ke_tensor(Xtr), torch.tensor((ytr - ry) / sy)
    model = buat_model(X.shape[2])
    # Adam: menyesuaikan langkah belajar tiap bobot sendiri, biasanya stabil untuk LSTM
    opt = torch.optim.Adam(model.parameters(), lr=LAJU_BELAJAR)
    fungsi_rugi = torch.nn.L1Loss()   # L1 = MAE, sama dengan metrik utama kasus

    for epoch in range(EPOCH):
        model.train()
        urutan = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), UKURAN_BATCH):
            idx = urutan[i:i + UKURAN_BATCH]
            opt.zero_grad()
            rugi = fungsi_rugi(model(X[idx]), y[idx])
            rugi.backward()
            opt.step()
            total += rugi.item() * len(idx)
        print(f"epoch {epoch + 1}/{EPOCH}  rugi latih {total / len(X):.4f}")

    model.eval()
    with torch.no_grad():
        pred_val = model(ke_tensor(Xva)).numpy() * sy + ry           # kembalikan ke satuan asli
        pred_test = (model(ke_tensor(Xte)).numpy() * sy + ry) if final and Xte is not None else None
    return pred_val, pred_test, np.asarray(yva, np.float32), Xva


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--simulasi", action="store_true", help="uji kerangka dengan data pura-pura")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()
    t0 = time.time()
    pred_val, pred_test, yva, Xva = latih_dan_prediksi(a.seed, a.final, a.simulasi)
    durasi = time.time() - t0

    if a.simulasi:
        # Data pura-pura: hanya dicetak, TIDAK ditulis ke metrics.json
        mae_rnn = np.mean(np.abs(pred_val - yva))
        mae_persistensi = np.mean(np.abs(Xva[:, -1, 0] - yva))   # tebakan "nilai terakhir"
        print(f"[SIMULASI] MAE RNN {mae_rnn:.4f} | MAE persistensi {mae_persistensi:.4f} | {durasi:.1f} detik")
        return

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

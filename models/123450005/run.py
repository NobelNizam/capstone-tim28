"""Transfer Learning ResNet-18 + Autoencoder untuk peramalan perubahan suhu Tim 28.

Input:
    120 menit terakhir dengan 6 kanal protokol tim:
    suhu_rel, cahaya_log_rel, daya_rel, tegangan_rel, jam_sin, jam_cos -> (N, 6, 120).

Output:
    Prediksi perubahan suhu pada menit +15, +30, +45, dan +60 -> (N, 4) (°C).

Representasi:
    Enam kanal deret waktu dibagi menjadi dua kelompok 3-kanal, masing-masing diubah
    menjadi citra Gramian Angular Summation Field (GASF) 224x224 untuk diproses oleh
    ResNet-18 pralatih ImageNet (fitur 512 + 512 = 1024 dimensi), disandi oleh
    Autoencoder (1024 -> 128), lalu diteruskan ke kepala regresi 4 horizon.

Jalankan dari root repositori:
    python models/123450005/run.py --seed 1 --simulasi   # uji kerangka pada data simulasi
    python models/123450005/run.py --seed 1              # latih pada data primer meja

Evaluasi data uji (khusus sprint terakhir):
    python models/123450005/run.py --seed 1 --final
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# KONFIGURASI
# ============================================================

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

# Gunakan cache lokal di workspace jika tersedia agar berjalan tanpa akses luar.
_CACHE_LOKAL = AKAR / "data" / "torch_cache"
if _CACHE_LOKAL.exists():
    os.environ.setdefault("TORCH_HOME", str(_CACHE_LOKAL))

from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name
JALUR = "tl"

PANJANG_INPUT = 120
JUMLAH_FITUR = 6
JUMLAH_TARGET = 4

BATCH_SIZE_AE = 64
BATCH_SIZE_REGRESI = 32
BATCH_SIZE_PREDIKSI = 64

EPOCH_AE = 20
EPOCH_REGRESI = 25

LEARNING_RATE_AE = 1e-3
LEARNING_RATE_REGRESI = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATA SIMULASI (UJI KERANGKA SEBELUM / TANPA DATA PRIMER)
# ============================================================

def data_simulasi(seed: int, panjang: int = PANJANG_INPUT):
    """Data simulasi berformat sama dengan protokol tim: X (N, 6, 120), y (N, 4)."""
    rng = np.random.default_rng(seed)
    horizon = (15, 30, 45, 60)
    h_maks = max(horizon)

    def buat(n: int):
        mulai = rng.uniform(0, 24 * 60, size=n)
        menit = mulai[:, None] + np.arange(panjang + h_maks)[None, :]
        jam = (menit / 60.0) % 24.0
        suhu = 26.5 - 2.0 * ((jam >= 7) & (jam < 10)) + 0.8 * np.sin(2 * np.pi * jam / 24.0)
        suhu += rng.normal(0.0, 0.25, size=menit.shape)
        cahaya = np.log1p(np.where((jam >= 7) & (jam < 18), 350.0, 2.0) + rng.uniform(0, 10, menit.shape))
        kipas = (suhu > 27.5).astype(np.float32) * 0.7
        tegangan = 5000.0 - 200.0 * kipas + rng.normal(0.0, 5.0, menit.shape)

        t_idx = panjang - 1
        j_rad = jam[:, :panjang] / 24.0 * 2.0 * np.pi
        fisik = np.stack(
            [
                suhu[:, :panjang] - suhu[:, t_idx : t_idx + 1],
                cahaya[:, :panjang] - cahaya[:, t_idx : t_idx + 1],
                kipas[:, :panjang] - kipas[:, t_idx : t_idx + 1],
                tegangan[:, :panjang] - tegangan[:, t_idx : t_idx + 1],
                np.sin(j_rad),
                np.cos(j_rad),
            ],
            axis=1,
        ).astype(np.float32)
        y = np.stack([suhu[:, t_idx + h] - suhu[:, t_idx] for h in horizon], axis=1).astype(np.float32)
        return fisik, y

    Xtr, ytr = buat(400)
    Xva, yva = buat(120)
    Xte, yte = buat(80)
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    return (
        ((Xtr - mu) / sd).astype(np.float32),
        ytr,
        ((Xva - mu) / sd).astype(np.float32),
        yva,
        ((Xte - mu) / sd).astype(np.float32),
        yte,
    )


# ============================================================
# VALIDASI DAN NORMALISASI DATA
# ============================================================

def siapkan_input(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Memvalidasi bentuk (N, 6, 120) dan menormalisasi per kanal dari data latih."""
    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 2 and X.shape[1] == PANJANG_INPUT * JUMLAH_FITUR:
        X = X.reshape(-1, JUMLAH_FITUR, PANJANG_INPUT)
    elif X.ndim == 3 and X.shape[1:] == (PANJANG_INPUT, JUMLAH_FITUR):
        X = X.transpose(0, 2, 1)

    if X.ndim != 3 or X.shape[1:] != (JUMLAH_FITUR, PANJANG_INPUT):
        raise ValueError(
            f"Input harus berbentuk (N, {JUMLAH_FITUR}, {PANJANG_INPUT}), diperoleh {X.shape}."
        )
    if len(X) == 0:
        raise ValueError("Data input kosong.")
    if not np.isfinite(X).all():
        raise ValueError("Input mengandung NaN atau nilai tak hingga.")

    if fit:
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)

    if mean is None or std is None:
        raise ValueError("Statistik normalisasi data latih belum tersedia.")

    return ((X - mean) / std).astype(np.float32), mean, std


def siapkan_target(y: np.ndarray, nama: str) -> np.ndarray:
    """Memvalidasi target regresi 4 horizon berbentuk (N, 4)."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != JUMLAH_TARGET:
        raise ValueError(f"{nama} harus berbentuk (N, {JUMLAH_TARGET}), diperoleh {y.shape}.")
    if not np.isfinite(y).all():
        raise ValueError(f"{nama} mengandung NaN atau nilai tak hingga.")
    return y.astype(np.float32)


# ============================================================
# REPRESENTASI DERET WAKTU MENJADI CITRA GASF
# ============================================================

def _gasf_3kanal(x: torch.Tensor) -> torch.Tensor:
    """Ubah (batch, 3, 120) menjadi citra GASF ternormalisasi ImageNet (batch, 3, 224, 224)."""
    nilai_min = x.amin(dim=2, keepdim=True)
    nilai_max = x.amax(dim=2, keepdim=True)
    rentang = torch.where((nilai_max - nilai_min) < 1e-8, 1.0, nilai_max - nilai_min)

    z = torch.clamp(2.0 * (x - nilai_min) / rentang - 1.0, -1.0, 1.0)
    sin_theta = torch.sqrt(torch.clamp(1.0 - z.pow(2), min=0.0))

    gasf = z.unsqueeze(-1) * z.unsqueeze(-2) - sin_theta.unsqueeze(-1) * sin_theta.unsqueeze(-2)
    gasf = F.interpolate(gasf, size=(224, 224), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=gasf.device, dtype=gasf.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=gasf.device, dtype=gasf.dtype).view(1, 3, 1, 1)
    return (gasf - mean) / std


def ubah_ke_gasf_dua_grup(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Memecah (batch, 6, 120) menjadi dua citra RGB GASF (batch, 3, 224, 224)."""
    if x.ndim != 3 or x.shape[1:] != (JUMLAH_FITUR, PANJANG_INPUT):
        raise ValueError(f"Input GASF harus (N, {JUMLAH_FITUR}, {PANJANG_INPUT}), diperoleh {tuple(x.shape)}.")
    return _gasf_3kanal(x[:, :3, :]), _gasf_3kanal(x[:, 3:, :])


# ============================================================
# AUTOENCODER & MODEL TRANSFER LEARNING
# ============================================================

class PenyandiAE(nn.Module):
    """Autoencoder untuk menyandi fitur ResNet-18 menjadi representasi laten."""

    def __init__(self, dim_input: int = 1024, dim_laten: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_input, 256),
            nn.ReLU(),
            nn.Linear(256, dim_laten),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim_laten, 256),
            nn.ReLU(),
            nn.Linear(256, dim_input),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        laten = self.encoder(x)
        return self.decoder(laten), laten


class ModelTL(nn.Module):
    """ResNet-18 pralatih (beku) + Autoencoder penyandi fitur + kepala regresi 4 horizon."""

    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.ae = PenyandiAE(dim_input=1024, dim_laten=128)
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, JUMLAH_TARGET),
        )
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def ekstrak_fitur(self, x: torch.Tensor) -> torch.Tensor:
        g1, g2 = ubah_ke_gasf_dua_grup(x)
        with torch.no_grad():
            f1 = self.backbone(g1)
            f2 = self.backbone(g2)
        return torch.cat([f1, f2], dim=1)

    def forward_fitur(self, fitur: torch.Tensor) -> torch.Tensor:
        _, laten = self.ae(fitur)
        return self.regressor(laten)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_fitur(self.ekstrak_fitur(x))


# ============================================================
# EKSTRAKSI FITUR & PELATIHAN DUA TAHAP
# ============================================================

def ekstrak_semua_fitur(model: ModelTL, X: torch.Tensor, batch_size: int = BATCH_SIZE_AE) -> torch.Tensor:
    model.eval()
    hasil = []
    with torch.no_grad():
        for awal in range(0, len(X), batch_size):
            xb = X[awal : awal + batch_size].to(DEVICE)
            hasil.append(model.ekstrak_fitur(xb).cpu())
    return torch.cat(hasil, dim=0)


def latih_ae(model: ModelTL, fitur_latih: torch.Tensor, seed: int) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(fitur_latih), batch_size=BATCH_SIZE_AE, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.ae.parameters(), lr=LEARNING_RATE_AE, weight_decay=WEIGHT_DECAY)
    riwayat = []
    for epoch in range(1, EPOCH_AE + 1):
        model.ae.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            rekonstruksi, _ = model.ae(batch)
            loss = F.mse_loss(rekonstruksi, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        rata = total_loss / len(fitur_latih)
        riwayat.append({"epoch": epoch, "loss_ae": round(rata, 6)})
        print(f"AE Epoch {epoch:02d}/{EPOCH_AE} - mse: {rata:.6f}", flush=True)
    return riwayat


def latih_regresi(
    model: ModelTL,
    fitur_latih: torch.Tensor,
    y_latih: torch.Tensor,
    fitur_val: torch.Tensor,
    y_val_asli: np.ndarray,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    seed: int,
) -> dict:
    """Melatih encoder AE + kepala regresi dengan L1Loss pada target terstandar; pilih epoch MAE validasi terbaik."""
    generator = torch.Generator().manual_seed(seed)
    y_std_tr = (y_latih - y_mean) / y_std
    loader = DataLoader(
        TensorDataset(fitur_latih, y_std_tr),
        batch_size=BATCH_SIZE_REGRESI,
        shuffle=True,
        generator=generator,
    )
    params = list(model.ae.encoder.parameters()) + list(model.regressor.parameters())
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE_REGRESI, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH_REGRESI)

    terbaik = (float("inf"), 0)
    bobot_terbaik = None
    riwayat = []

    for epoch in range(1, EPOCH_REGRESI + 1):
        model.train()
        total_loss = 0.0
        for fitur, target in loader:
            fitur, target = fitur.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            pred = model.forward_fitur(fitur)
            loss = F.l1_loss(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(fitur)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred_val_norm = model.forward_fitur(fitur_val.to(DEVICE)).cpu()
            pred_val = (pred_val_norm * y_std + y_mean).numpy()
        val_mae = float(np.abs(pred_val - y_val_asli).mean())
        loss_tr = total_loss / len(fitur_latih)
        riwayat.append({"epoch": epoch, "loss_latih": round(loss_tr, 6), "val_mae": round(val_mae, 4)})
        print(f"Regresi Epoch {epoch:02d}/{EPOCH_REGRESI} - loss: {loss_tr:.4f} - val_mae: {val_mae:.4f}", flush=True)

        if val_mae < terbaik[0]:
            terbaik = (val_mae, epoch)
            bobot_terbaik = copy.deepcopy(model.state_dict())

    if bobot_terbaik is not None:
        model.load_state_dict(bobot_terbaik)
    return {"epoch_terbaik": terbaik[1], "val_mae_terbaik": round(terbaik[0], 4), "riwayat": riwayat}


def prediksi_dari_fitur(model: ModelTL, fitur: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> np.ndarray:
    model.eval()
    hasil = []
    with torch.no_grad():
        for awal in range(0, len(fitur), BATCH_SIZE_PREDIKSI):
            fb = fitur[awal : awal + BATCH_SIZE_PREDIKSI].to(DEVICE)
            out = model.forward_fitur(fb).cpu() * y_std + y_mean
            hasil.append(out.numpy())
    return np.concatenate(hasil, axis=0).astype(np.float32)


def latensi_ms(model: ModelTL, contoh_x: torch.Tensor, ulang: int = 10) -> float:
    model.eval()
    x1 = contoh_x[:1].to(DEVICE)
    waktu = []
    with torch.no_grad():
        for _ in range(ulang + 2):
            t0 = time.perf_counter()
            model(x1)
            waktu.append(time.perf_counter() - t0)
    return round(float(np.median(waktu[2:])) * 1000.0, 2)


# ============================================================
# PROSES UTAMA MODEL
# ============================================================

def latih_dan_prediksi(seed: int, final: bool, simulasi: bool):
    if not simulasi and getattr(data, "TUGAS", None) != "regresi":
        raise ValueError('src.protokol.data.TUGAS harus bernilai "regresi".')

    Xtr, ytr, Xva, yva, Xte, yte = data_simulasi(seed) if simulasi else data.muat()
    Xtr, mean, std = siapkan_input(Xtr, fit=True)
    Xva, _, _ = siapkan_input(Xva, mean=mean, std=std)
    ytr = siapkan_target(ytr, "Target latih")
    yva = siapkan_target(yva, "Target validasi")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_train = torch.tensor(Xtr, dtype=torch.float32)
    y_train = torch.tensor(ytr, dtype=torch.float32)
    X_valid = torch.tensor(Xva, dtype=torch.float32)

    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = torch.where(y_train.std(dim=0, keepdim=True) < 1e-6, 1.0, y_train.std(dim=0, keepdim=True))

    print("=" * 60)
    print(f"TRANSFER LEARNING RESNET-18 + AE ({'SIMULASI' if simulasi else 'DATA PRIMER'})")
    print("=" * 60)
    print("Perangkat       :", DEVICE)
    print("Data latih      :", Xtr.shape, "| Target:", ytr.shape)
    print("Data validasi   :", Xva.shape, "| Target:", yva.shape)

    model = ModelTL().to(DEVICE)

    print("\n[1/3] Ekstraksi fitur ResNet-18 (GASF 2 grup x 3 kanal)...")
    fitur_latih = ekstrak_semua_fitur(model, X_train)
    fitur_valid = ekstrak_semua_fitur(model, X_valid)
    print("Dimensi fitur:", tuple(fitur_latih.shape))

    print("\n[2/3] Melatih Autoencoder penyandi fitur...")
    riwayat_ae = latih_ae(model, fitur_latih, seed=seed)

    print("\n[3/3] Melatih encoder AE + kepala regresi...")
    info_reg = latih_regresi(model, fitur_latih, y_train, fitur_valid, yva, y_mean, y_std, seed=seed)

    pred_val = prediksi_dari_fitur(model, fitur_valid, y_mean, y_std)
    pred_test = None
    if final:
        if Xte is None:
            raise ValueError("Mode --final membutuhkan X_test dari data.muat().")
        Xte_norm, _, _ = siapkan_input(Xte, mean=mean, std=std)
        fitur_uji = ekstrak_semua_fitur(model, torch.tensor(Xte_norm, dtype=torch.float32))
        pred_test = prediksi_dari_fitur(model, fitur_uji, y_mean, y_std)

    n_param_latih = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    n_param_total = int(sum(p.numel() for p in model.parameters()))
    info = {
        "jalur": JALUR,
        "seed": seed,
        "hiper": {
            "backbone": "resnet18_imagenet_beku",
            "dim_fitur_resnet": 1024,
            "dim_laten_ae": 128,
            "epoch_ae": EPOCH_AE,
            "epoch_regresi": EPOCH_REGRESI,
            "lr": LEARNING_RATE_REGRESI,
            "wd": WEIGHT_DECAY,
            "dropout": 0.2,
            "loss_regresi": "L1Loss",
        },
        "parameter": n_param_latih,
        "parameter_total": n_param_total,
        "latensi_ms": latensi_ms(model, X_valid),
        "epoch_terbaik": info_reg["epoch_terbaik"],
        "riwayat_ae": riwayat_ae,
        "riwayat": info_reg["riwayat"],
        **data.metrik_tim(yva, pred_val),
    }
    return pred_val, pred_test, yva, info


# ============================================================
# MAIN DAN EVALUASI RESMI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer Learning ResNet-18 + Autoencoder Tim 28.")
    parser.add_argument("--seed", type=int, required=True, help="Seed eksperimen.")
    parser.add_argument("--final", action="store_true", help="Evaluasi data uji pada sprint terakhir.")
    parser.add_argument("--simulasi", action="store_true", help="Uji kerangka pada data simulasi.")
    parser.add_argument("--perangkat", default="cpu laptop", help="Keterangan perangkat yang digunakan.")
    args = parser.parse_args()

    if args.seed < 0:
        raise ValueError("Seed harus bernilai nol atau lebih.")

    mulai = time.time()
    pred_val, pred_test, yva, info = latih_dan_prediksi(args.seed, args.final, args.simulasi)
    durasi = time.time() - mulai

    if args.simulasi:
        print(
            f"[SIMULASI] val_mae_rata={info['val_mae_rata']:.4f} | "
            f"persistensi={info['val_mae_persistensi']:.4f} | "
            f"skill={info['val_skill']} | {durasi:.1f}s"
        )
        return

    output = AKAR / "results" / NIM
    output.mkdir(parents=True, exist_ok=True)
    file_pred_val = output / "pred_val.npy"
    np.save(file_pred_val, pred_val)

    perintah = [
        sys.executable, "-m", "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(args.seed),
        "--pred-val", str(file_pred_val),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", args.perangkat,
    ]
    if pred_test is not None:
        file_pred_test = output / "pred_test.npy"
        np.save(file_pred_test, pred_test)
        perintah.extend(["--pred-test", str(file_pred_test)])

    subprocess.run(perintah, check=True, cwd=AKAR)
    (output / "tambahan.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Selesai dalam {durasi:.1f}s -> {output / 'metrics.json'}")


if __name__ == "__main__":
    main()

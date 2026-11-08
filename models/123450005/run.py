"""Transfer Learning ResNet-18 + Autoencoder untuk peramalan daya Tim 28.

Input:
    120 menit terakhir dengan 3 fitur:
    daya, suhu, dan jam.

Output:
    Prediksi daya pada menit ke-15, 30, dan 60.

Representasi:
    Deret waktu diubah menjadi Gramian Angular Summation Field (GASF)
    untuk diproses oleh ResNet-18 pralatih ImageNet.

Jalankan dari root repositori:
    python models/123450005/run.py --seed 1

Evaluasi data uji (khusus sprint terakhir):
    python models/123450005/run.py --seed 1 --final
"""

from __future__ import annotations

import argparse
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

from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name
JALUR = "tl"

PANJANG_INPUT = 120
JUMLAH_FITUR = 3
JUMLAH_TARGET = 3

BATCH_SIZE_AE = 64
BATCH_SIZE_REGRESI = 32
BATCH_SIZE_PREDIKSI = 64

EPOCH_AE = 20
EPOCH_REGRESI = 30

LEARNING_RATE_AE = 1e-3
LEARNING_RATE_REGRESI = 1e-3
WEIGHT_DECAY = 1e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# VALIDASI DAN NORMALISASI DATA
# ============================================================

def siapkan_input(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Memvalidasi dan menormalisasi input berdasarkan data latih."""

    X = np.asarray(X, dtype=np.float32)

    # Mendukung bentuk data (N, 360).
    if X.ndim == 2 and X.shape[1] == PANJANG_INPUT * JUMLAH_FITUR:
        X = X.reshape(-1, PANJANG_INPUT, JUMLAH_FITUR)

    if X.ndim != 3:
        raise ValueError(
            f"Input harus memiliki 3 dimensi, diperoleh {X.shape}."
        )

    if X.shape[1:] != (PANJANG_INPUT, JUMLAH_FITUR):
        raise ValueError(
            "Bentuk input harus (N, 120, 3), "
            f"diperoleh {X.shape}."
        )

    if len(X) == 0:
        raise ValueError("Data input kosong.")

    if not np.isfinite(X).all():
        raise ValueError("Input mengandung NaN atau nilai tak hingga.")

    if fit:
        mean = X.mean(axis=(0, 1), keepdims=True)
        std = X.std(axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)

    if mean is None or std is None:
        raise ValueError("Statistik normalisasi data latih belum tersedia.")

    X = (X - mean) / std

    return X.astype(np.float32), mean, std


def siapkan_target(y: np.ndarray, nama: str) -> np.ndarray:
    """Memvalidasi target regresi dengan tiga horizon prediksi."""

    y = np.asarray(y, dtype=np.float32)

    if y.ndim == 1:
        if y.size % JUMLAH_TARGET != 0:
            raise ValueError(
                f"{nama} memiliki {y.size} nilai. "
                "Jumlah nilai harus dapat dibagi menjadi 3 target."
            )
        y = y.reshape(-1, JUMLAH_TARGET)

    if y.ndim != 2 or y.shape[1] != JUMLAH_TARGET:
        raise ValueError(
            f"{nama} harus berbentuk (N, 3), diperoleh {y.shape}."
        )

    if not np.isfinite(y).all():
        raise ValueError(f"{nama} mengandung NaN atau nilai tak hingga.")

    return y.astype(np.float32)


# ============================================================
# REPRESENTASI DERET WAKTU MENJADI CITRA GASF
# ============================================================

def ubah_ke_gasf(x: torch.Tensor) -> torch.Tensor:
    """Mengubah deret waktu menjadi tiga kanal citra GASF.

    Input:
        (batch, 120, 3)

    Output:
        (batch, 3, 224, 224)
    """

    if x.ndim != 3 or x.shape[1:] != (120, 3):
        raise ValueError(
            f"Input GASF harus (N, 120, 3), diperoleh {tuple(x.shape)}."
        )

    # Susun menjadi (batch, fitur, waktu).
    x = x.transpose(1, 2)

    # Normalisasi setiap fitur dalam setiap jendela ke [-1, 1].
    nilai_min = x.amin(dim=2, keepdim=True)
    nilai_max = x.amax(dim=2, keepdim=True)

    rentang = nilai_max - nilai_min
    rentang = torch.where(rentang < 1e-8, 1.0, rentang)

    x = 2.0 * (x - nilai_min) / rentang - 1.0
    x = torch.clamp(x, -1.0, 1.0)

    # Sudut polar.
    sin_theta = torch.sqrt(
        torch.clamp(1.0 - x.pow(2), min=0.0)
    )

    # GASF: cos(theta_i + theta_j).
    gasf = (
        x.unsqueeze(-1) * x.unsqueeze(-2)
        - sin_theta.unsqueeze(-1) * sin_theta.unsqueeze(-2)
    )

    # Ukuran citra disesuaikan dengan masukan ResNet-18.
    gasf = F.interpolate(
        gasf,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

    # Normalisasi sesuai statistik ImageNet.
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=gasf.device,
        dtype=gasf.dtype,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=gasf.device,
        dtype=gasf.dtype,
    ).view(1, 3, 1, 1)

    return (gasf - mean) / std


# ============================================================
# AUTOENCODER
# ============================================================

class PenyandiAE(nn.Module):
    """Autoencoder untuk mempelajari representasi laten fitur ResNet."""

    def __init__(
        self,
        dim_input: int = 512,
        dim_laten: int = 128,
    ):
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

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        laten = self.encoder(x)
        rekonstruksi = self.decoder(laten)

        return rekonstruksi, laten


# ============================================================
# MODEL TRANSFER LEARNING
# ============================================================

class ModelTL(nn.Module):
    """ResNet-18 pralatih + Autoencoder + kepala regresi."""

    def __init__(self):
        super().__init__()

        try:
            from torchvision.models import (
                resnet18,
                ResNet18_Weights,
            )
        except ImportError as exc:
            raise ImportError(
                "TorchVision tidak dapat diimpor. "
                "Periksa kesesuaian versi PyTorch dan TorchVision."
            ) from exc

        # Menggunakan bobot pralatih ImageNet.
        try:
            self.backbone = resnet18(
                weights=ResNet18_Weights.DEFAULT
            )
        except Exception as exc:
            raise RuntimeError(
                "Bobot ResNet-18 pralatih gagal dimuat. "
                "Periksa koneksi internet atau ketersediaan bobot "
                "ImageNet pada cache TorchVision."
            ) from exc

        # Menghapus lapisan klasifikasi ImageNet.
        self.backbone.fc = nn.Identity()

        # Membekukan seluruh parameter ResNet.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # Autoencoder sebagai penyandi fitur.
        self.ae = PenyandiAE(
            dim_input=512,
            dim_laten=128,
        )

        # Kepala regresi menghasilkan tiga horizon.
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, JUMLAH_TARGET),
        )

        self.backbone.eval()

    def train(self, mode: bool = True):
        """Menjaga backbone tetap dalam mode evaluasi."""

        super().train(mode)
        self.backbone.eval()

        return self

    def ekstrak_fitur(self, x: torch.Tensor) -> torch.Tensor:
        """Menghasilkan fitur 512 dimensi dari ResNet-18."""

        citra = ubah_ke_gasf(x)

        # Backbone dibekukan sehingga tidak menghitung gradien.
        with torch.no_grad():
            fitur = self.backbone(citra)

        return fitur

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        fitur = self.ekstrak_fitur(x)
        _, laten = self.ae(fitur)
        prediksi = self.regressor(laten)

        return prediksi


# ============================================================
# EKSTRAKSI FITUR DATA
# ============================================================

def ekstrak_semua_fitur(
    model: ModelTL,
    X: torch.Tensor,
    batch_size: int = BATCH_SIZE_AE,
) -> torch.Tensor:
    """Mengekstrak fitur ResNet secara bertahap."""

    model.eval()
    hasil = []

    with torch.no_grad():
        for awal in range(0, len(X), batch_size):
            xb = X[awal:awal + batch_size].to(DEVICE)
            fitur = model.ekstrak_fitur(xb)
            hasil.append(fitur.cpu())

    if not hasil:
        raise ValueError("Tidak ada data untuk ekstraksi fitur.")

    return torch.cat(hasil, dim=0)


# ============================================================
# PELATIHAN AUTOENCODER
# ============================================================

def latih_ae(
    model: ModelTL,
    fitur_latih: torch.Tensor,
    seed: int,
) -> None:
    """Melatih Autoencoder untuk merekonstruksi fitur ResNet."""

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataset = TensorDataset(fitur_latih)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_AE,
        shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.Adam(
        model.ae.parameters(),
        lr=LEARNING_RATE_AE,
        weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(EPOCH_AE):
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

        rata_rata = total_loss / len(dataset)

        print(
            f"AE Epoch {epoch + 1:02d}/{EPOCH_AE} "
            f"- loss: {rata_rata:.6f}"
        )


# ============================================================
# PELATIHAN KEPALA REGRESI
# ============================================================

def latih_regresi(
    model: ModelTL,
    fitur_latih: torch.Tensor,
    y_latih: torch.Tensor,
    seed: int,
) -> None:
    """Melatih encoder AE dan kepala regresi."""

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataset = TensorDataset(
        fitur_latih,
        y_latih,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_REGRESI,
        shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.Adam(
        list(model.ae.encoder.parameters())
        + list(model.regressor.parameters()),
        lr=LEARNING_RATE_REGRESI,
        weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(EPOCH_REGRESI):
        model.train()
        total_loss = 0.0

        for fitur, target in loader:
            fitur = fitur.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()

            laten = model.ae.encoder(fitur)
            prediksi = model.regressor(laten)

            loss = F.mse_loss(prediksi, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(fitur)

        rata_rata = total_loss / len(dataset)

        print(
            f"Regresi Epoch {epoch + 1:02d}/{EPOCH_REGRESI} "
            f"- loss: {rata_rata:.6f}"
        )


# ============================================================
# PREDIKSI
# ============================================================

def prediksi(
    model: ModelTL,
    X: torch.Tensor,
) -> np.ndarray:
    """Menghasilkan prediksi tiga horizon secara bertahap."""

    model.eval()
    hasil = []

    with torch.no_grad():
        for awal in range(0, len(X), BATCH_SIZE_PREDIKSI):
            xb = X[awal:awal + BATCH_SIZE_PREDIKSI].to(DEVICE)

            output = model(xb)
            hasil.append(output.cpu().numpy())

    if not hasil:
        raise ValueError("Tidak ada data untuk diprediksi.")

    return np.concatenate(hasil, axis=0).astype(np.float32)


# ============================================================
# PROSES UTAMA MODEL
# ============================================================

def latih_dan_prediksi(
    seed: int,
    final: bool,
) -> tuple[np.ndarray, np.ndarray | None]:

    # Pastikan jenis tugas sesuai dengan peramalan.
    if getattr(data, "TUGAS", None) != "regresi":
        raise ValueError(
            'src.protokol.data.TUGAS harus bernilai "regresi". '
            "Koordinasikan perubahan protokol dengan Penjaga Data "
            "dan Evaluasi."
        )

    # Memuat data melalui protokol tim.
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()

    # Validasi data latih dan validasi.
    Xtr, mean, std = siapkan_input(Xtr, fit=True)
    Xva, _, _ = siapkan_input(Xva, mean=mean, std=std)

    ytr = siapkan_target(ytr, "Target latih")
    yva = siapkan_target(yva, "Target validasi")

    if len(Xtr) != len(ytr):
        raise ValueError(
            "Jumlah data input dan target latih tidak sama."
        )

    if len(Xva) != len(yva):
        raise ValueError(
            "Jumlah data input dan target validasi tidak sama."
        )

    if len(Xtr) == 0 or len(Xva) == 0:
        raise ValueError("Data latih atau validasi kosong.")

    # Konversi ke tensor.
    X_train = torch.tensor(Xtr, dtype=torch.float32)
    y_train = torch.tensor(ytr, dtype=torch.float32)
    X_valid = torch.tensor(Xva, dtype=torch.float32)

    # Menetapkan seed.
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=" * 60)
    print("PELATIHAN TRANSFER LEARNING + AUTOENCODER")
    print("=" * 60)
    print("Perangkat       :", DEVICE)
    print("Data latih      :", Xtr.shape)
    print("Target latih    :", ytr.shape)
    print("Data validasi   :", Xva.shape)
    print("Target validasi :", yva.shape)

    # Inisialisasi model.
    model = ModelTL().to(DEVICE)

    # Ekstraksi fitur menggunakan ResNet-18.
    print("\n[1/3] Mengekstrak fitur ResNet-18...")

    fitur_latih = ekstrak_semua_fitur(model, X_train)

    print("Dimensi fitur:", tuple(fitur_latih.shape))

    # Melatih Autoencoder.
    print("\n[2/3] Melatih Autoencoder...")

    latih_ae(
        model=model,
        fitur_latih=fitur_latih,
        seed=seed,
    )

    # Melatih encoder AE dan kepala regresi.
    print("\n[3/3] Melatih kepala regresi...")

    latih_regresi(
        model=model,
        fitur_latih=fitur_latih,
        y_latih=y_train,
        seed=seed,
    )

    # Prediksi validasi.
    pred_val = prediksi(model, X_valid)

    if pred_val.shape != yva.shape:
        raise ValueError(
            f"Dimensi prediksi validasi {pred_val.shape} "
            f"tidak sama dengan target {yva.shape}."
        )

    # Prediksi data uji hanya ketika --final digunakan.
    pred_test = None

    if final:
        if Xte is None or yte is None:
            raise ValueError(
                "Mode --final membutuhkan X_test dan y_test "
                "dari data.muat()."
            )

        Xte, _, _ = siapkan_input(
            Xte,
            mean=mean,
            std=std,
        )

        yte = siapkan_target(yte, "Target uji")

        if len(Xte) != len(yte):
            raise ValueError(
                "Jumlah data input dan target uji tidak sama."
            )

        X_test = torch.tensor(Xte, dtype=torch.float32)
        pred_test = prediksi(model, X_test)

        if pred_test.shape != yte.shape:
            raise ValueError(
                f"Dimensi prediksi uji {pred_test.shape} "
                f"tidak sama dengan target {yte.shape}."
            )

    print("\nUkuran input       :", tuple(Xtr.shape))
    print("Ukuran target      :", tuple(ytr.shape))
    print("Ukuran prediksi val:", tuple(pred_val.shape))

    if pred_test is not None:
        print("Ukuran prediksi uji:", tuple(pred_test.shape))

    return pred_val, pred_test


# ============================================================
# MAIN DAN EVALUASI RESMI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Transfer Learning + Autoencoder Tim 28."
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed eksperimen.",
    )

    parser.add_argument(
        "--final",
        action="store_true",
        help="Menghasilkan prediksi data uji pada sprint terakhir.",
    )

    parser.add_argument(
        "--perangkat",
        default=str(DEVICE),
        help="Keterangan perangkat yang digunakan.",
    )

    args = parser.parse_args()

    if args.seed < 0:
        raise ValueError("Seed harus bernilai nol atau lebih.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    mulai = time.time()

    pred_val, pred_test = latih_dan_prediksi(
        seed=args.seed,
        final=args.final,
    )

    durasi = time.time() - mulai

    # Membuat folder hasil.
    output = AKAR / "results" / NIM
    output.mkdir(parents=True, exist_ok=True)

    # Menyimpan prediksi validasi.
    file_pred_val = output / "pred_val.npy"
    np.save(file_pred_val, pred_val)

    # Menyiapkan perintah evaluator resmi.
    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(args.seed),
        "--pred-val", str(file_pred_val),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", args.perangkat,
    ]

    # Menyimpan prediksi uji hanya pada mode final.
    if pred_test is not None:
        file_pred_test = output / "pred_test.npy"
        np.save(file_pred_test, pred_test)

        perintah.extend([
            "--pred-test",
            str(file_pred_test),
        ])

    print("\nMenjalankan evaluator resmi...")

    subprocess.run(
        perintah,
        check=True,
        cwd=AKAR,
    )

    print("\n" + "=" * 60)
    print("PROSES SELESAI")
    print("=" * 60)
    print("Seed       :", args.seed)
    print("Perangkat  :", args.perangkat)
    print("Durasi     :", f"{durasi:.1f} detik")
    print("Prediksi   :", file_pred_val)
    print("Metrik     :", output / "metrics.json")


if __name__ == "__main__":
    main()
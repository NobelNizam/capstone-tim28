# Catatan Eksperimen Transfer Learning + Autoencoder

**Nama:** Efi Defiyati  
**NIM:** 123450005  
**Tim:** 28  
**Judul:** Prakiraan Suhu Ruang Meja 15–60 Menit ke Depan (adaptasi dari prakiraan energi)  
**Jalur Model:** Transfer Learning (`tl`) + Autoencoder penyandi fitur

## 1. Tujuan & Penyesuaian Protokol

Model meramalkan perubahan suhu ruang meja pada 4 horizon (`+15`, `+30`, `+45`, dan `+60` menit) dari riwayat 120 menit terakhir (`6` kanal protokol tim: `suhu_rel`, `cahaya_log_rel`, `daya_rel`, `tegangan_rel`, `jam_sin`, `jam_cos`).

Pendekatan menggabungkan **Transfer Learning** (ResNet-18 pralatih ImageNet) dengan **Autoencoder** sebagai penyandi fitur untuk menghasilkan representasi laten sebelum kepala regresi.

## 2. Arsitektur Model

1. **Representasi GASF 2×3 kanal:** Enam kanal masukan `(N, 6, 120)` dipecah menjadi dua kelompok 3-kanal (`[:3]` dan `[3:]`), masing-masing diubah menjadi citra Gramian Angular Summation Field (GASF) berukuran `224 × 224` dan dinormalisasi sesuai statistik ImageNet.
2. **ResNet-18 pralatih (beku):** Mengekstrak fitur `512` dimensi dari masing-masing kelompok citra → gabungan `1024` fitur. Seluruh parameter backbone dibekukan (`requires_grad=False`, mode `eval`).
3. **Autoencoder (`PenyandiAE`):** `1024 → 256 → 128` (encoder) dan `128 → 256 → 1024` (decoder), dilatih terlebih dahulu untuk merekonstruksi fitur ResNet-18 dengan MSE.
4. **Kepala Regresi:** `128 → 64 → 4` dengan `Dropout(0.2)`, dilatih bersama encoder AE untuk memprediksi 4 horizon sekaligus.

## 3. Alasan Optimizer & Regularisasi

- **Optimizer (`AdamW` + `CosineAnnealingLR`):** Menyesuaikan laju belajar adaptif per parameter pada lapisan linear serta menurunkan learning rate secara halus sepanjang epoch agar konvergen stabil pada dataset kecil (`566` titik latih).
- **Loss Regresi (`L1Loss` pada target terstandar):** Selaras langsung dengan metrik utama tim (MAE).
- **Regularisasi:** Backbone ResNet-18 dibekukan, `Dropout(0.2)` pada kepala regresi, `weight_decay = 1e-4`, dan pemilihan bobot pada epoch dengan `val_mae` terbaik (epoch ke-3 pada seed 1).

## 4. Ringkasan Hyperparameter

| Parameter | Nilai |
|---|---:|
| Bentuk input | `(N, 6, 120)` |
| Bentuk target | `(N, 4)` (`+15, +30, +45, +60` menit) |
| Backbone | ResNet-18 (`ResNet18_Weights.DEFAULT`, beku) |
| Dimensi fitur ResNet (2 grup) | 1024 |
| Dimensi laten AE | 128 |
| Batch size AE / regresi | 64 / 32 |
| Epoch AE / regresi | 20 / 25 |
| Learning rate | `1e-3` (`CosineAnnealingLR`) |
| Weight decay | `1e-4` |
| Dropout | `0.2` |
| Loss AE / regresi | MSE / L1Loss (target terstandar) |

## 5. Hasil Sprint 1 (Seed 1)

### Uji Kerangka Data Simulasi (`--simulasi --seed 1`)
- `val_mae_rata`: `0,3572 °C` vs persistensi `0,3707 °C` (`skill = +0,0364`), durasi `46,9 s`.

### Uji Data Primer Meja (`--seed 1`)
- **`data_hash`**: `83cb911ea1c23c1b80c3bcca11493efb` (cocok dengan `MANIFEST.json`).
- **Epoch terbaik**: `3` dari `25` (validasi mulai naik setelah epoch 3 akibat keterbatasan jumlah kejadian pagi hari kerja pada data latih).
- **Durasi**: `66,1 detik` (CPU laptop), parameter terlatih `600.004` (total `11.776.516`), latensi inferensi `95,93 ms`.

| Model | MAE rata-rata (°C) | RMSE (°C) | +15 m | +30 m | +45 m | +60 m | Skill vs persistensi | MAE titik berubah |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Persistensi / non-DL terpilih** | **0,4765** | 1,2371 | 0,2350 | 0,3981 | 0,5535 | 0,7194 | 0,000 | 2,0062 |
| **TL + AE (`123450005`)** | **0,4989** | **1,2255** | 0,2411 | 0,3974 | 0,5754 | 0,7817 | -0,047 | **1,9894** |
| Baseline starter (ridge) | 2,9942 | 7,6699 | 1,39 | 2,39 | 3,85 | 4,34 | -5,28 | 5,60 |

## 6. Rencana Sprint 2

- Pencarian hyperparameter dan minimal 2 *ablation study* (mis. perbandingan tanpa AE vs dengan AE, variasi ukuran laten/dropout, serta fine-tuning parsial vs beku penuh).
- Menjalankan 3 seed penuh (`--seeds 1 2 3`) pada konfigurasi akhir setelah penambahan data pagi hari kerja.

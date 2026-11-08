# Catatan Eksperimen Transfer Learning + Autoencoder

**Nama:** Efi Defiyati  
**NIM:** 123450005  
**Tim:** 28  
**Judul:** Prakiraan Konsumsi Energi Meja 15–60 Menit ke Depan  
**Jalur Model:** Transfer Learning (TL)

## 1. Tujuan Model

Model dikembangkan untuk memperkirakan konsumsi daya meja pada 15, 30, dan 60 menit ke depan berdasarkan data historis selama 120 menit. Fitur masukan terdiri atas daya, suhu, dan jam.

Pendekatan yang digunakan menggabungkan Transfer Learning dengan Autoencoder sebagai penyandi fitur untuk menghasilkan representasi laten yang digunakan dalam proses regresi.

## 2. Arsitektur Model

Arsitektur terdiri atas tiga komponen utama:

1. **ResNet-18 pralatih:** Menggunakan bobot pralatih ImageNet sebagai ekstraktor fitur. Parameter backbone dibekukan agar proses pelatihan lebih ringan.
2. **Autoencoder:** Mempelajari representasi laten dari fitur yang dihasilkan ResNet-18. Encoder memetakan 512 fitur menjadi 128 fitur laten, sedangkan decoder merekonstruksi kembali 512 fitur.
3. **Regressor:** Menggunakan representasi laten dari encoder untuk menghasilkan tiga nilai prediksi konsumsi daya pada horizon 15, 30, dan 60 menit.

## 3. Prapemrosesan Data

Tahapan prapemrosesan yang diterapkan meliputi:

1. Memastikan masukan memiliki bentuk `(N, 120, 3)`.
2. Melakukan normalisasi fitur berdasarkan rata-rata dan standar deviasi data latih.
3. Mengubah deret waktu menjadi representasi citra menggunakan Gramian Angular Summation Field (GASF).
4. Menyesuaikan ukuran representasi menjadi 224 × 224 piksel.
5. Melakukan normalisasi menggunakan statistik ImageNet sebelum masukan diberikan ke ResNet-18.

Statistik normalisasi fitur dihitung dari data latih dan digunakan kembali pada data validasi maupun data uji.

## 4. Tahapan Pelatihan

Pelatihan dilakukan melalui dua tahap.

**Tahap pertama: Pelatihan Autoencoder**

Autoencoder dilatih untuk merekonstruksi fitur hasil ekstraksi ResNet-18. Fungsi kerugian yang digunakan adalah Mean Squared Error (MSE).

**Tahap kedua: Pelatihan Regresi**

Encoder Autoencoder dan kepala regresi dilatih menggunakan target konsumsi daya. ResNet-18 tetap dibekukan selama tahap pelatihan.

## 5. Hyperparameter

| Parameter | Nilai |
|---|---:|
| Panjang input | 120 langkah |
| Jumlah fitur input | 3 |
| Jumlah target | 3 |
| Backbone | ResNet-18 |
| Bobot pralatih | ImageNet |
| Dimensi fitur ResNet | 512 |
| Dimensi laten AE | 128 |
| Hidden layer AE | 256 |
| Hidden layer regresi | 64 |
| Batch size AE | 64 |
| Batch size regresi | 32 |
| Epoch AE | 20 |
| Epoch regresi | 30 |
| Optimizer | Adam |
| Learning rate AE | 0,001 |
| Learning rate regresi | 0,001 |
| Weight decay | 0,00001 |
| Dropout | 0,2 |
| Fungsi kerugian | MSE |
| Seed pengujian awal | 1 |

## 6. Evaluasi Model

Evaluasi dilakukan menggunakan evaluator resmi repositori melalui `src.common.evaluate`.

Metrik untuk tugas regresi meliputi:

- RMSE (Root Mean Squared Error).
- MAE (Mean Absolute Error).
- R² (Koefisien Determinasi).

Pengujian normal menggunakan data latih dan validasi. Data uji hanya digunakan pada tahap akhir sesuai ketentuan evaluasi tim.

## 7. Status Eksperimen

| Aktivitas | Status |
|---|---|
| Penyusunan arsitektur model | Selesai |
| Implementasi ResNet-18 | Selesai |
| Implementasi Autoencoder | Selesai |
| Implementasi kepala regresi | Selesai |
| Implementasi normalisasi | Selesai |
| Integrasi pemanggilan evaluator | Selesai |
| Pemeriksaan sintaks | Menunggu pengujian |
| Pengujian menggunakan data simulasi | Belum dilakukan |
| Pengujian menggunakan data meja | Menunggu ketersediaan data |
| Pencatatan hasil evaluasi | Menunggu pengujian |
| Ablation study | Belum dilakukan |

## 8. Catatan Pengembangan

Implementasi model masih memerlukan pengujian menggunakan data simulasi untuk memastikan kesesuaian dimensi masukan dan keluaran serta keberhasilan proses pelatihan.

Pengujian menggunakan data meja dilakukan setelah dataset tersedia dan pemuat data pada protokol tim selesai diimplementasikan.

Pencarian hyperparameter dan ablation study akan dicatat setelah eksperimen dilakukan. Hasil evaluasi tidak dicantumkan sebelum proses pengujian selesai.
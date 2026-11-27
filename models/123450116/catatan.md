# Catatan jalur CNN 1-D (NIM 123450116)

## Ringkasan
Model CNN 1-D ini meramalkan perubahan suhu ruang 15, 30, 45, dan 60 menit ke depan.
Masukannya data 2 jam terakhir (6 kanal x 120 menit). Dilatih dengan seed 1 pada data primer meja.

## Hasil (validasi, seed 1)
- MAE rata-rata 4 horizon: 0,569 °C
- RMSE: 1,188
- R2: -0,030
- Epoch terbaik: 1 (latihan berhenti di epoch 6 karena galat validasi naik)
- data_hash: 83cb911ea1c23c1b80c3bcca11493efb (sama dengan MANIFEST tim)
- Pembanding persistensi: MAE 0,4765, skill CNN -0,195

## Pilihan model dan alasannya
- 3 blok Conv1d dengan kernel 5, karena data per menit sehingga kernel 5 melihat pola 5 menit.
- Jumlah kanal kecil (16, 32, 32), karena data latih hanya 566 titik dan model besar mudah menghafal.
- BatchNorm dan ReLU di tiap blok, lalu MaxPool 2 untuk memperpendek deret waktu.
- Rata-rata global atas waktu, dropout 0,3, lalu satu lapisan Linear dengan 4 keluaran.
- Optimizer AdamW (lr 0,001, weight decay 0,001), untuk menahan overfitting.
- Loss MSE pada target yang distandarkan (rata-rata dan simpangan hanya dari data latih).
- Early stopping dengan kesabaran 5 epoch: berhenti bila MAE validasi tidak membaik.

## Temuan
- Galat validasi paling kecil di epoch 1 (0,569), lalu naik sampai 0,969 di epoch 6. Ini tanda overfitting.
- CNN belum mengalahkan persistensi (skill negatif).
- Dugaan penyebab: data latih hanya punya sedikit kejadian suhu turun (satu pagi hari kerja), jadi model belum bisa belajar meramalkannya.

## Yang belum dilakukan
Belum ada pencarian hyperparameter atau ablation. Di Sprint 1 hanya satu konfigurasi dan satu seed.

## Rencana Sprint 2
- Latih dengan seed 2 dan 3, lalu laporkan rata-rata dan simpangannya.
- Coba kernel 3 atau 7, dropout 0,2 sampai 0,5, dan jumlah kanal berbeda.
- Tambahkan fitur hari kerja dan akhir pekan.

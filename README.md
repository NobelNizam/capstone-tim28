# Capstone Deep Learning — starter tim

Template repositori tim untuk **Capstone Deep Learning: Satu Kasus, Lima Keluarga Model**
(Program Studi Sains Data ITERA, Data Science Workbench). Satu tim = satu repositori yang
dibuat dari template ini; setiap anggota memiliki satu keluarga model dan satu folder.

**Kasus dan dataset tiap tim berbeda.** Keduanya ditetapkan tim pengajar (dosen pengampu bersama
pembimbing) dan tercatat di kartu *Kasus tim* pada Papan Tim Workbench beserta tautan
datasetnya. Starter ini tidak memuat dataset apa pun; yang dibakukan adalah **cara
mengevaluasi dan melaporkan**, sehingga lima model tim dibandingkan secara adil dan hasilnya
dapat direproduksi.

## Struktur

```text
src/common/        milik starter — jangan diubah: evaluate (penilai), split (split beku), manifest (data_hash), baseline
src/protokol/      milik tim, DIBEKUKAN setelah Sprint 0 dengan persetujuan pembimbing: data.py (pemuat kasus tim), PROTOKOL.md
data/              dataset kasus tim (tidak di-commit; lihat data/MANIFEST.json)
models/<nim>/      milik satu anggota — kode model, run.py
results/<nim>/     berkas hasil anggota: pred_val.npy, pred_test.npy, metrics.json
reports/individu/<nim>.md   laporan perorangan (≤ 1200 kata)
reports/tim.md     laporan tim (6 bagian)
AI_LOG.md          catatan pemakaian AI tim
run_all.py         satu perintah: jalankan semua model dan rakit tabel komparasi
```

## Sprint 0: membekukan protokol tim

Penjaga Data \& Evaluasi mengerjakan ini bersama tim, lalu meminta persetujuan pembimbing
lewat *pull request* ke `main`:

1. Unduh dataset dari tautan di kartu *Kasus tim* ke `data/` (folder ini tidak di-commit).
   Isi `data/MANIFEST.json` dengan tautan sumber dan daftar berkas; `src.common.manifest`
   menghitung `data_hash` dari isi berkas itu, sehingga semua anggota memakai data yang sama.
2. Isi `src/protokol/data.py`: fungsi `muat()` mengembalikan `X_train, y_train, X_val, y_val,
   X_test, y_test` memakai split beku dari `src.common.split` (seed 2026). Contoh lengkap
   untuk Fashion-MNIST ada di `src/protokol/contoh_fashion_mnist.py`.
3. Tulis `src/protokol/PROTOKOL.md`: jenis tugas (`klasifikasi` atau `regresi`), metrik utama,
   ukuran split, anggaran komputasi, dan aturan evaluasi data uji (sekali, di sprint terakhir).
4. Jalankan baseline untuk memastikan jalur evaluasi hidup:

   ```bash
   pip install -r requirements.txt
   python -m src.common.baseline --seed 2026
   ```

   Hasilnya di `results/baseline/metrics.json`: batas bawah yang harus dilampaui setiap model.

Setelah *pull request* protokol digabung, `src/protokol/` tidak diubah lagi sampai akhir;
perubahan hanya lewat *pull request* yang disetujui pembimbing.

## Sprint 1 dan seterusnya: model tiap anggota

1. Salin `models/TEMPLATE/` menjadi `models/<nim>/`, isi `run.py` sesuai `models/README.md`.
2. Jalankan tiga seed pada konfigurasi akhir:

   ```bash
   python models/<nim>/run.py --seed 1 && python models/<nim>/run.py --seed 2 && python models/<nim>/run.py --seed 3
   ```

   Setiap run menulis `results/<nim>/metrics.json` lewat `src.common.evaluate`; Local Runner
   memeriksanya (skema `metrik/seed/data_hash`, isi sama dengan HEAD) saat *Periksa artefak*
   ditekan di aplikasi DSWorkbench.
3. Perintah reproduksi tim (dinilai pada sprint terakhir): `python run_all.py`.

## Yang dibakukan untuk semua tim

- Split latih/validasi beku dari data latih dengan seed 2026 (`src/common/split.py`); data uji
  hanya dievaluasi **sekali** oleh Penjaga Data pada sprint terakhir (`--final`).
- Metrik: klasifikasi → akurasi dan macro-F1; regresi → RMSE, MAE, R². Metrik utama ditulis di
  `PROTOKOL.md`. Hasil konfigurasi akhir dilaporkan sebagai rerata ± simpangan baku dari 3 seed.
- Anggaran komputasi ≤ 20 menit pelatihan per run di CPU laptop (atau sesuai yang disetujui
  pembimbing untuk kasus tim); catat `--durasi` dan `--perangkat`.
- `data_hash` dihitung dari berkas di `data/MANIFEST.json` dan split; jangan diisi manual.

## Jalur model

| Jalur | Keluarga | Modul |
|---|---|---|
| `cnn` | CNN (1D/2D sesuai data) | 05 |
| `tl` | Transfer learning / fine-tuning model pralatih | 06 |
| `rnn` | RNN → LSTM/GRU | 07–08 |
| `trf` | Transformer encoder | 09 |
| `ae` | Autoencoder → representasi untuk klasifikasi/regresi/anomali | — |

Pembimbing boleh menyesuaikan pilihan jalur dengan sifat data kasus tim. Syarat semua model:
optimizer dengan alasan tertulis, ≥ 1 regularisasi, tuning hyperparameter dengan catatan
pencarian, ≥ 2 ablation, 3 seed pada konfigurasi akhir.

Lisensi: MIT. Repo ini masuk Galeri Capstone sebagai portofolio setelah semester berakhir.

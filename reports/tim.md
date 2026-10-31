# Laporan tim — <Nama tim>

Satu dokumen bersama, enam bagian. Angka di bagian Hasil ditarik dari `results/*/metrics.json`
(tabel otomatis: `reports/hasil_komparasi.md` dari `python run_all.py`).

## Ringkasan
Apa yang dibandingkan, pada kasus apa, dan satu kalimat temuan utama.

## Protokol
Apa yang membuat perbandingan antar-model adil: split beku (seed 2026), metrik, anggaran
komputasi, jumlah seed, cara data uji dievaluasi (sekali, oleh siapa, kapan).

## Hasil
Tabel komparasi lima model (+ baseline): val_acc dan val_macro_f1 rerata ± sb dari 3 seed,
jumlah parameter, durasi, test_acc (sekali). Satu analisis kesalahan per model (kelas yang
sering tertukar).

## Analisis
Mengapa hasilnya berbeda: penjelasan mekanistik yang diuji (bukan hanya "A lebih baik dari
B"); bedakan fakta eksperimen, interpretasi, dan dugaan.

## Rekomendasi
Model mana yang layak untuk anggaran komputasi ini, dengan alasan dari bukti dan batas
berlakunya.

## Pembagian kerja
| Anggota | NIM | Jalur | Tugas sampingan | Bagian laporan yang ditulis |
|---|---|---|---|---|

Komponen yang tidak tersedia (bila ada): siapa, mengapa, dan dampaknya pada komparasi.

# Laporan perorangan — <Nama> (<NIM>), jalur <cnn|tl|rnn|trf|ae>

Salin berkas ini menjadi `reports/individu/<nim>.md`. Paling banyak 1200 kata di luar tabel
dan kode. Angka harus ada di `results/<nim>/metrics.json` yang sudah di-*commit*.

## 1. Komponen saya
Dua–tiga kalimat: model yang dipilih dan perannya dalam komparasi tim.

## 2. Keputusan desain
Satu keputusan utama (arsitektur, optimizer, regularisasi, atau praproses), alternatif yang
ditolak, dan alasan menolaknya.

## 3. Kegagalan dan pelajaran
Satu kegagalan nyata (eksperimen gagal, bug, pendekatan yang dibuang) dan perubahan yang
diambil karenanya.

## 4. Hasil
Rerata ± simpangan baku 3 seed pada konfigurasi akhir (val_acc, val_macro_f1), hasil ablation
(≥ 2), durasi per run, dan batas berlakunya. Sebut seed dan run yang dirujuk.

| Konfigurasi | seed | val_acc | val_macro_f1 | durasi (s) |
|---|---|---|---|---|

## 5. Yang akan saya ubah jika ada satu sprint lagi
Tiga–lima kalimat.

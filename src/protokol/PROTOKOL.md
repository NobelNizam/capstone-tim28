# Protokol evaluasi tim — Tim 28: Prakiraan suhu ruang meja 15–60 menit

| Butir | Isi |
|---|---|
| Kasus | Meramalkan perubahan suhu ruang meja 15, 30, 45, dan 60 menit ke depan dari 2 jam terakhir |
| Sumber dataset | Sesi tim `sesi-20261119-202155` (19 Nov 2026 20.21–00.21, aturan program harian tim 28) + rekaman program harian bersama 14, 15, 18, dan 19 Nov (sesi tim00); berkas + sha256 di `data/MANIFEST.json` |
| Jenis tugas (`TUGAS`) | regresi 4 keluaran: **perubahan** suhu dari menit t ke t+15, +30, +45, +60 (°C); suhu ramalan = suhu(t) + keluaran, galatnya sama |
| **Penyimpangan dari rencana** | Rencana awal (`rencana/t28-prakiraan-energi.yaml`): prakiraan **daya** meja 15, 30, 60 menit. Daya hanya terukur di INA219 kipas (0x40; arus total 0x44 tidak terpasang), dan di sesi tim daya itu datar (0,000 W selama 4 jam karena lampu dan kipas mati menurut aturan, pompa 3× 30 s). Target dipindah ke **suhu ruang** (masukan termostat kipas di aturan tim) yang berubah nyata, dengan horizon lengkap 15, 30, 45, dan 60 menit |
| Masukan | menit t−119..t, 6 kanal: suhu, log1p cahaya, daya kanal kipas, tegangan rel (keempatnya **relatif terhadap nilainya di menit t**), sin/cos jam WIB → `[6, 120]` |
| Satuan pemisahan | **Blok rekaman bersambung** menurut waktu: latih = 3 blok harian terlama (`13-11 23.59`, `15-11 05.00`, `18-11 00.00`); validasi = blok harian terbaru (`19-11 00.00`); uji = sesi tim sendiri (`19-11 20.21`) |
| Ukuran latih / validasi / uji | 566 / 376 / 62 titik awal prakiraan (bergeser 1 menit, saling tumpang tindih) |
| Metrik utama | MAE rata-rata 4 horizon (°C); MAE per horizon; **skill vs persistensi** (1 − MAE/MAE persistensi; persistensi = perubahan 0); MAE pada titik yang suhunya berubah > 0,5 °C |
| Praproses | rata-rata per menit; celah ≤ 5 menit diisi interpolasi linear; normalisasi per kanal dari data latih saja |
| Anggaran komputasi | ≤ 20 menit CPU laptop per jalur per seed |
| Evaluasi data uji | sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | `83cb911ea1c23c1b80c3bcca11493efb` (terverifikasi dari `results/baseline/metrics.json`) |

## Alasan desain protokol

1. **Mengapa target dipindah dari daya ke suhu ruang:** Pada sesi tim malam dan rekaman program harian, daya kanal kipas nyaris datar (persistensi daya 0,022 W pada 15 menit dan 0,057 W pada 60 menit) sehingga tidak ada pola beban yang memadai untuk dipelajari. Sebaliknya, suhu ruang (DHT11) berubah nyata (3–5 °C pada pagi hari kerja).
2. **Mengapa perubahan suhu & masukan relatif, bukan absolut:** Tingkat absolut suhu dan status kipas berbeda antarblok rekaman. Menyatakan target sebagai selisih `suhu(t+h) - suhu(t)` dan 4 kanal fisik sebagai selisih terhadap menit `t` mencegah model menghafal tingkat absolut blok dan menjadikan persistensi sebagai titik acuan nol.
3. **Mengapa pemisahan per blok & urutan waktu:** Titik awal yang bertetangga per menit berbagi 119 menit jendela masukan. Memisahkan secara acak per titik akan menimbulkan *data leakage*, sehingga pemisahan wajib per blok bersambung dan kronologis.

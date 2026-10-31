# Protokol evaluasi tim — <Nama tim>

Diisi Penjaga Data & Evaluasi pada Sprint 0, digabung ke `main` lewat *pull request* yang
disetujui pembimbing, lalu **dibekukan**. Perubahan setelah itu hanya lewat *pull request*
baru yang disetujui pembimbing.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | … |
| Sumber dataset | … (tautan dari kartu Kasus tim) |
| Jenis tugas (`TUGAS`) | klasifikasi / regresi |
| Target / label | … |
| Ukuran data latih / validasi / uji | … / … / … |
| Cara split | `split.indeks` seed 2026, porsi validasi 10 % (atau `indeks_urut` untuk deret waktu) |
| Metrik utama untuk komparasi | macro-F1 (klasifikasi) / RMSE (regresi) |
| Praproses yang sama untuk semua model | … |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop (atau yang disetujui pembimbing) |
| Evaluasi data uji | sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | … (dari `results/baseline/metrics.json`) |

Disetujui pembimbing pada: … (tautan *pull request*)

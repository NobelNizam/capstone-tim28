# AI_LOG — Tim 28

Satu baris per pemakaian AI yang bermakna. Kolom terakhir mencatat apa yang diperiksa dan diubah dari keluarannya.

| Tanggal | Siapa (NIM) | Alat | Untuk apa | Apa yang diubah dari keluarannya |
|---|---|---|---|---|
| 2026-11-03 | 12345678 | Copilot | kerangka DataLoader | ganti normalisasi, tambah seed worker |
| 2026-11-25 | 123450033 | Claude | menyesuaikan run.py RNN untuk data meja (keluaran 4 horizon, urutan data) | memperbaiki sendiri indentasi yang salah saat menempel kode, lalu menguji dengan --simulasi sampai hasilnya sama dengan sebelum diubah |
| 2026-11-27 | 123450033 | Claude | menyiapkan data.py, deret.py, pisah.py, kodek.py dari Repo Kunci agar run.py bisa jalan pada data meja asli | membersihkan karakter tak terlihat (U+200B) yang ikut tertempel, memeriksa bentuk data (566/376/62 contoh) dan sha256 28 berkas semuanya OK |
| 2026-11-27 | 123450033 | Claude | memahami penyebab error MANIFEST.json dan manifest.py | memulihkan manifest.py yang tertimpa isi JSON dengan git checkout, lalu menjalankan ulang run.py seed 1 sampai metrics.json terbentuk |
| 2026-11-25 | 123450005 | Antigravity (Gemini) | Menyiapkan infrastruktur protokol data (`src/protokol/*`), `data/MANIFEST.json`, `run_all.py`, dan tes protokol pra-training | Disesuaikan agar memverifikasi fungsi tanpa mengarang metrik hasil sebelum data primer diekstrak; header `PROTOKOL.md` dan `non-dl/catatan.md` dibersihkan dari angka contoh |
| 2026-11-27 | 123450005 | Antigravity (Gemini) | Menyelaraskan `models/123450005/run.py` (GASF + ResNet-18 + AE) ke masukan `[6, 120]` dan target `[4]` serta menambah mode `--simulasi` | Menambahkan pemecahan 6 kanal menjadi 2 citra RGB GASF (`1024` fitur ResNet-18), seleksi epoch terbaik berdasar `val_mae` dengan `L1Loss`, cache lokal `data/torch_cache`, serta verifikasi langsung pada `--simulasi` dan data primer seed 1 |

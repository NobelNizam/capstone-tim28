# AI_LOG — Tim 28

Satu baris per pemakaian AI yang bermakna. Kolom terakhir mencatat apa yang diperiksa dan diubah dari keluarannya.

| Tanggal | Siapa (NIM) | Alat | Untuk apa | Apa yang diubah dari keluarannya |
|---|---|---|---|---|
| 2026-11-25 | 123450005 | Antigravity (Gemini) | Menyiapkan infrastruktur protokol data (`src/protokol/*`), `data/MANIFEST.json`, `run_all.py`, dan tes protokol pra-training | Disesuaikan agar memverifikasi fungsi tanpa mengarang metrik hasil sebelum data primer diekstrak; header `PROTOKOL.md` dan `non-dl/catatan.md` dibersihkan dari angka contoh |
| 2026-11-27 | 123450005 | Antigravity (Gemini) | Menyelaraskan `models/123450005/run.py` (GASF + ResNet-18 + AE) ke masukan `[6, 120]` dan target `[4]` serta menambah mode `--simulasi` | Menambahkan pemecahan 6 kanal menjadi 2 citra RGB GASF (`1024` fitur ResNet-18), seleksi epoch terbaik berdasar `val_mae` dengan `L1Loss`, cache lokal `data/torch_cache`, serta verifikasi langsung pada `--simulasi` dan data primer seed 1 |

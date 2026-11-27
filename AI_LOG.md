# AI_LOG — <Nama tim>

Satu baris per pemakaian AI yang bermakna (ChatGPT, Claude, Copilot, Bravais, dan sejenisnya).
Ringkas dan jujur; transkrip penuh tidak diminta. Yang dinilai bukan "pakai AI atau tidak",
melainkan "mengerti atau tidak" — isi kolom terakhir dengan apa yang Anda ubah dari keluarannya.

| Tanggal | Siapa (NIM) | Alat | Untuk apa | Apa yang diubah dari keluarannya |
|---|---|---|---|---|
| 2026-11-03 | 12345678 | Copilot | kerangka DataLoader | ganti normalisasi, tambah seed worker |
| 2026-11-25 | 123450033 | Claude | menyesuaikan run.py RNN untuk data meja (keluaran 4 horizon, urutan data) | memperbaiki sendiri indentasi yang salah saat menempel kode, lalu menguji dengan --simulasi sampai hasilnya sama dengan sebelum diubah |
| 2026-11-27 | 123450033 | Claude | menyiapkan data.py, deret.py, pisah.py, kodek.py dari Repo Kunci agar run.py bisa jalan pada data meja asli | membersihkan karakter tak terlihat (U+200B) yang ikut tertempel, memeriksa bentuk data (566/376/62 contoh) dan sha256 28 berkas semuanya OK |
| 2026-11-27 | 123450033 | Claude | memahami penyebab error MANIFEST.json dan manifest.py | memulihkan manifest.py yang tertimpa isi JSON dengan git checkout, lalu menjalankan ulang run.py seed 1 sampai metrics.json terbentuk |
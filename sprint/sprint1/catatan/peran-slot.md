# Peran slot meja — Tim 28

Rencana tim 28 menggunakan program harian dengan aturan (lampu 07–18, kipas di atas 28,5 °C, pompa tiap
90 menit selama 30 s) yang dijalankan 4 jam (16 tahap × 15 menit).

| Peran | Tugas | Selesai bila |
|---|---|---|
| Penanggung jawab jadwal | Memilih jam sesi yang melewati pergantian aturan (mis. 05.00–09.00 atau 16.00–20.00) | Log memuat perubahan lampu/kipas, bukan hanya pompa |
| Pengamat | Mengecek kartu *Eksperimen berjalan* di awal dan akhir (`n / 16` tahap) | Tahap 16/16 selesai |
| Penjaga data | Mengunduh arsip sesi dan program harian ke `data/`, mengisi `data/MANIFEST.json`, menjalankan `python -m src.protokol.mutu` | `reports/mutu-data.md` dihasilkan & di-commit |

## Catatan kesesuaian jam sesi & aturan

- Bila sesi tim dijalankan pada malam hari di luar jam lampu (07–18) dan di bawah ambang suhu kipas (28,5 °C), hanya pompa (3× 30 s) yang aktif dan tidak terukur langsung pada kanal kipas INA219 (`0x40`), sehingga daya kipas terbaca 0 W.
- Pastikan aturan/jadwal slot atau pemilihan target telah diverifikasi sebelum memulai sesi rekaman berikutnya.

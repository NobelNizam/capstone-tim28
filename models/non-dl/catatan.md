# Catatan pembanding non-deep learning (persistensi dan rata-rata musiman per jam)

## Calon pembanding (sesuai rencana tim)
- **persistensi**: suhu(t+h) = suhu(t) (perubahan = 0).
- **musiman**: rata-rata suhu data latih pada jam WIB yang sama dengan `t+h`.
- **persistensi_musiman**: `suhu(t) + [musiman(jam t+h) − musiman(jam t)]`.

## Status
Skrip `models/non-dl/run.py` sudah siap dijalankan melalui `python models/non-dl/run.py --seed 1` atau `python run_all.py --seeds 1` setelah berkas data sesi diunduh ke `data/sesi-*/`.

# Kontrak `models/<nim>/run.py`

Setiap anggota memiliki satu folder `models/<nim>/` (hanya pemiliknya yang boleh mengubah).
Isinya bebas (PyTorch dianjurkan, sesuai Modul 03–09), dengan satu syarat: skrip `run.py`
yang dapat dipanggil `run_all.py` dan menulis hasil lewat penilai dosen.

```bash
python models/<nim>/run.py --seed 1            # latih + prediksi validasi
python models/<nim>/run.py --seed 1 --final    # sprint terakhir: + prediksi data uji (sekali)
```

Yang wajib dilakukan `run.py`:

1. Memuat data lewat `src.common.split.muat_split()` (jangan membuat split sendiri).
2. Menetapkan seed dari argumen `--seed` ke NumPy dan PyTorch (`torch.manual_seed`).
3. Menulis prediksi validasi ke `results/<nim>/pred_val.npy` (array int label 0–9, panjang 6.000).
   Dengan `--final`, juga `results/<nim>/pred_test.npy` (panjang 10.000) dari `src.common.data.muat("test")`.
4. Memanggil penilai dosen, bukan menghitung metrik sendiri:

   ```python
   subprocess.run([sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", "cnn",
                   "--seed", str(seed), "--pred-val", "results/NIM/pred_val.npy",
                   "--durasi", f"{durasi:.1f}", "--perangkat", "cpu laptop"], check=True, cwd=AKAR)
   ```

5. Menyimpan catatan pencarian hyperparameter dan hasil ablation di `models/<nim>/catatan.md`
   (angka ringkasan 3 seed juga ditulis di laporan perorangan).

`metrics.json` yang ditulis penilai adalah **berkas hasil** yang diperiksa Local Runner
(*Periksa artefak* di aplikasi DSWorkbench). Jangan menyuntingnya dengan tangan; jalankan
ulang `run.py` bila ingin mengganti angkanya, lalu *commit*.

Folder `models/TEMPLATE/` berisi kerangka `run.py` yang sudah memenuhi kontrak ini.

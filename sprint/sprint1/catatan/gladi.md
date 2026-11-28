# Gladi aturan dengan mode simulasi

```bash
cd meja-uji-5v/perekam
MEJA_TAHAP_MENIT=0.5 python meja.py --simulasi harian --tim 28 --jam 0.1 --aturan ../rencana/t28-prakiraan-energi.yaml
```

Simulasi menulis `daya.csv`, `lingkungan.csv`, `aktuator.csv`, dan log tahap. Periksa di `aktuator.csv` aktuator mana yang benar-benar berganti menurut aturan pada jam gladi; bila tidak ada yang bergerak, jadwal sesi atau ambang aturan perlu disesuaikan sebelum slot.

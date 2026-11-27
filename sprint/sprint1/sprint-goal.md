# Sprint Goal Sprint 1 — Tim 28

> Data sesi tim dan program harian masuk ke repo sebagai deret per menit dengan pemisahan per blok menurut waktu,
> target prakiraan yang benar-benar berubah (suhu ruang 15–60 menit) ditetapkan dari pemeriksaan mutu, pembanding
> baseline dan non-DL (persistensi) berjalan, serta jalur model tim dilatih pada data primer meja (seed 1).

## Checklist Definition of Done Sprint 1

- [x] Rencana eksperimen disetujui (`rencana/t28-prakiraan-energi.yaml`) & kesepakatan peran tim (`docs/portofolio.md`).
- [x] Peran operasional slot meja & panduan gladi simulasi (`sprint/sprint1/catatan/peran-slot.md`, `sprint/sprint1/catatan/gladi.md`).
- [x] Manifes data (`data/MANIFEST.json`), protokol pemuat deret & anti-kebocoran blok (`src/protokol/data.py`, `PROTOKOL.md`), serta verifikasi protokol (`tests/test_t28.py`).
- [x] Ekstrak 7 arsip sesi ke `data/sesi-*/` dan periksa mutu data (`python -m src.protokol.mutu` → `reports/mutu-data.md`).
- [x] Baseline starter (`python -m src.common.baseline --seed 2026`) & pembanding non-DL (`models/non-dl/run.py --seed 1`) valid dengan `data_hash = 83cb911ea1c23c1b80c3bcca11493efb`.
- [x] Jalur TL+AE (`123450005`): uji simulasi (`--simulasi`) dan pelatihan data primer seed 1 (`results/123450005/metrics.json` valid, `val_mae = 0,4989 °C`, `run_all.py` & `src.protokol.analisis` berjalan).
- [x] Naskah Demo Sprint 1 (`sprint/sprint1/DEMO.md`) dan Log Sprint 1 Koordinator/TL (`sprint/sprint1/123450005.md`) siap.
- [ ] Merge jalur `123450033` (RNN) & penambahan/pelatihan jalur `123450116` dan `123450117` pada data primer (seed 1).
- [ ] Log Sprint 1 seluruh anggota terisi di Papan Tim Workbench & PR ke `main` digabung.

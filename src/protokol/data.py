"""Pemuat data tim 28: prakiraan suhu ruang meja 15, 30, 45, dan 60 menit ke depan dari 2 jam terakhir.

Kontrak starter: ``TUGAS`` dan ``muat() -> X_train, y_train, X_val, y_val, X_test, y_test``.

Keputusan protokol (rinciannya di PROTOKOL.md):
- **Penyimpangan dari rencana: target = suhu ruang, bukan daya.** Daya meja hanya terukur di INA219 kipas (0x40;
  INA219 arus total 0x44 tidak terpasang). Di semua sesi kipas menyala atau mati berjam-jam, dan di sesi tim sendiri
  (19-11 20.21–00.21) kipas mati sepanjang 4 jam: daya = 0 W, persistensi sempurna, tidak ada yang bisa diramalkan.
  Suhu ruang (DHT11) berubah nyata: AC ruang menurunkan suhu 3–5 °C di pagi hari kerja, dan suhu naik perlahan
  tanpa AC. Suhu juga masukan termostat kipas di aturan tim. Masukan tetap rencana tim (daya + suhu + jam, 2 jam
  terakhir), ditambah cahaya ruang dan tegangan rel.
- Satu langkah = satu menit: rata-rata daya kanal kipas (W, negatif kecil = offset sensor → 0), tegangan rel
  (bus_mV), suhu, dan cahaya (log1p lux, dipangkas 2000 lux). Celah ≤ ``CELAH_ISI_MENIT`` menit diisi interpolasi
  linear; celah lebih panjang memutus blok.
- Satu contoh = titik awal t: masukan menit t−119..t ([6, 120]), target = **perubahan** suhu dari menit t ke t+15,
  t+30, t+45, t+60 → y [4] (°C; prakiraan suhu = suhu(t) + y, galatnya sama). Empat kanal fisik (suhu, log cahaya,
  daya, tegangan) dinyatakan **relatif terhadap nilainya di menit t**, ditambah sin/cos jam WIB. Alasannya:
  tingkat absolut berbeda per blok (kipas menyala sepanjang blok validasi, hampir tidak pernah di latih); target
  dan masukan absolut membuat jalur DL menghafal tingkat blok (PROTOKOL.md).
  Persistensi = meramal perubahan 0. ``KODEK = KodekRegresiVektor``.
- Satuan pemisahan = **blok rekaman** (rentang bersambung ≥ 3 jam), menurut waktu: latih = tiga blok harian terlama,
  validasi = blok harian terbaru (19-11 pagi), uji = sesi tim sendiri (19-11 malam), sesuai kasus tim ("slot sendiri
  sebagai uji"). Blok 16-11 terputus-putus (< 30 menit per potong) dan tidak dipakai.
- Normalisasi per kanal dari data latih saja.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone

import numpy as np

from src.common import manifest
from src.protokol import deret, pisah
from src.protokol.kodek import KodekRegresiVektor

TUGAS = "regresi"
HORIZON = (15, 30, 45, 60)
KELAS = [f"+{h} menit" for h in HORIZON]
KODEK: KodekRegresiVektor | None = None
METRIK_TIM = ("val_mae_rata", "MAE rata-rata 4 horizon (°C)")

SESI = ["sesi-20261114-000000", "sesi-20261115-050000", "sesi-20261118-000000", "sesi-20261118-050001",
        "sesi-20261119-000000", "sesi-20261119-050000", "sesi-20261119-202155"]
SESI_TIM = "sesi-20261119-202155"
RIWAYAT = 120
CELAH_ISI_MENIT = 5
LUX_MAKS = 2000.0
KANAL = ["suhu_rel", "cahaya_log_rel", "daya_rel", "tegangan_rel", "jam_sin", "jam_cos"]  # *_rel = x − x(t)
VAL_BLOK = ["19-11 00.00"]             # program harian 19-11 00.00–09.14 (blok harian terbaru)
UJI_BLOK = ["19-11 20.21"]             # sesi tim sendiri
AMBANG_PERISTIWA = 0.5                  # |suhu(t+h) − suhu(t)| > 0,5 °C di salah satu horizon = titik "berubah"
WIB = timezone(timedelta(hours=7))
VERSI_FITUR = "t28-suhu-selisih-v2"

FOLDER = {s: manifest.AKAR / "data" / s for s in SESI}
CACHE = manifest.AKAR / "data" / "cache"


def _angka(baris: list[str], idx: list[int]) -> list[float] | None:
    try:
        return [float(baris[i]) for i in idx]
    except (ValueError, IndexError):
        return None                     # baris kepala yang terulang di tengah berkas, baris terpotong


def _per_menit(folder) -> dict[int, list[float]]:
    """{menit epoch: [daya_W, tegangan_V, suhu_C, lux]} satu sesi (rata-rata per menit; nan bila tidak ada)."""
    jum: dict[int, list[float]] = {}
    with (folder / "daya.csv").open(encoding="utf-8", errors="replace") as f:
        r = csv.reader(f); k = next(r); idx = [k.index(c) for c in ("t_host", "bus_mV", "arus_mA")]
        for b in r:
            v = _angka(b, idx)
            if v is None or v[1] != v[1] or v[2] != v[2]:
                continue
            s = jum.setdefault(int(v[0] // 60), [0.0, 0.0, 0])
            s[0] += max(v[1] * v[2] / 1e6, 0.0); s[1] += v[1]; s[2] += 1
    ling: dict[int, list[float]] = {}
    with (folder / "lingkungan.csv").open(encoding="utf-8", errors="replace") as f:
        r = csv.reader(f); k = next(r); idx = [k.index(c) for c in ("t_host", "suhu_C", "lux")]
        for b in r:
            v = _angka(b, idx)
            if v is None:
                continue
            s = ling.setdefault(int(v[0] // 60), [0.0, 0, 0.0, 0])
            if v[1] == v[1]:
                s[0] += v[1]; s[1] += 1
            if v[2] == v[2]:
                s[2] += min(v[2], LUX_MAKS); s[3] += 1
    hasil = {}
    for m, (p, v, n) in jum.items():
        l = ling.get(m, [0.0, 0, 0.0, 0])
        hasil[m] = [p / n, v / n, l[0] / l[1] if l[1] else np.nan, l[2] / l[3] if l[3] else np.nan]
    return hasil


def nama_blok(menit: np.ndarray) -> str:
    """Nama blok = waktu mulainya (WIB), mis. '19-11 20.21'."""
    return f"{datetime.fromtimestamp(int(menit[0]) * 60, WIB):%d-%m %H.%M}"


def jam_wib(menit: np.ndarray) -> np.ndarray:
    return ((np.asarray(menit) * 60 + 7 * 3600) % 86400) / 3600.0


def blok() -> list[dict]:
    """Blok bersambung: {nama, menit [M], nilai [M, 4] (daya, tegangan, suhu, lux) setelah celah pendek diisi}."""
    semua: dict[int, list[float]] = {}
    for s in SESI:
        semua.update(_per_menit(FOLDER[s]))
    menit = np.array(sorted(semua))
    putus = np.where(np.diff(menit) > CELAH_ISI_MENIT + 1)[0] + 1
    hasil = []
    for bag in np.split(menit, putus):
        penuh = np.arange(bag[0], bag[-1] + 1)
        nilai = np.full((len(penuh), 4), np.nan)
        for i, m in enumerate(penuh):
            if m in semua:
                nilai[i] = semua[m]
        for k in range(4):                      # isi celah ≤ CELAH_ISI_MENIT (dan ujung blok) secara linear
            ada = ~np.isnan(nilai[:, k])
            if ada.sum() >= 2:
                nilai[:, k] = np.interp(np.arange(len(penuh)), np.flatnonzero(ada), nilai[ada, k])
        hasil.append({"nama": nama_blok(penuh), "menit": penuh, "nilai": nilai})
    return hasil


def _sidik() -> str:
    h = hashlib.sha256(VERSI_FITUR.encode())
    for s in SESI:
        for nama in ("daya.csv", "lingkungan.csv"):
            st = (FOLDER[s] / nama).stat()
            h.update(f"{s}/{nama}:{st.st_size}".encode())
    h.update(json.dumps([HORIZON, RIWAYAT, CELAH_ISI_MENIT, LUX_MAKS, KANAL]).encode())
    return h.hexdigest()[:16]


def contoh() -> dict:
    """Semua titik awal (belum dinormalisasi), di-cache. Kunci: X [N, 6, 120], y [N, 4] (perubahan suhu), blok, menit
    (menit epoch titik awal), persist (suhu menit t, untuk kembali ke suhu absolut), jam (jam WIB titik awal),
    blok_info (json)."""
    hilang = [s for s in SESI if not FOLDER[s].exists()]
    if hilang:
        raise FileNotFoundError(f"data/{hilang[0]}/ belum ada: unduh arsip sesi tim dan rekaman program harian dari "
                                "kartu Data meja lalu ekstrak ke data/ (lihat data/MANIFEST.json).")
    berkas = CACHE / f"{_sidik()}.npz"
    if berkas.exists():
        with np.load(berkas) as z:
            return {k: z[k] for k in z.files}
    kol = {k: [] for k in ("X", "y", "blok", "menit", "persist", "jam")}
    info = []
    H = max(HORIZON)
    for b in blok():
        m, v = b["menit"], b["nilai"]
        info.append({"nama": b["nama"], "mulai": int(m[0]), "menit": len(m),
                     "titik": max(0, len(m) - RIWAYAT - H + 1)})
        if len(m) < RIWAYAT + H:
            continue
        j = jam_wib(m) / 24 * 2 * np.pi
        seri = np.stack([v[:, 2], np.log1p(v[:, 3]), v[:, 0], v[:, 1], np.sin(j), np.cos(j)]).astype(np.float32)
        for t in range(RIWAYAT - 1, len(m) - H):
            w = seri[:, t - RIWAYAT + 1:t + 1].copy()
            w[:4] -= w[:4, -1:]
            kol["X"].append(w)
            kol["y"].append([v[t + h, 2] - v[t, 2] for h in HORIZON])
            kol["blok"].append(b["nama"]); kol["menit"].append(int(m[t]))
            kol["persist"].append(v[t, 2]); kol["jam"].append(float(jam_wib(m[t])))
    hasil = {k: np.asarray(x, np.float32 if k in ("X", "y", "persist") else None) for k, x in kol.items()}
    hasil["blok_info"] = np.array(json.dumps(info, ensure_ascii=False))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(berkas, **hasil)
    return hasil


def bagian() -> dict:
    c = contoh()
    uji = np.isin(c["blok"], UJI_BLOK); val = np.isin(c["blok"], VAL_BLOK)
    lat = ~uji & ~val
    return {"c": c, "blok": {"latih": sorted(set(c["blok"][lat].tolist())), "val": VAL_BLOK, "uji": UJI_BLOK},
            "latih": lat, "val": val, "uji": uji}


def metrik_tim(y: np.ndarray, p: np.ndarray, logit: np.ndarray | None = None) -> dict:
    """y, p = perubahan suhu [N, 4]. MAE per horizon dan rata-ratanya (°C, sama dengan galat suhu absolut); MAE
    persistensi (= |y|) pada titik yang sama, skill (1 − MAE/MAE persistensi), dan MAE pada titik yang suhunya
    berubah > AMBANG_PERISTIWA."""
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    e = np.abs(p - y); ep = np.abs(y)
    ubah = ep.max(1) > AMBANG_PERISTIWA
    return {"val_mae_rata": round(float(e.mean()), 4),
            "val_mae_rinci": {k: round(float(e[:, i].mean()), 4) for i, k in enumerate(KELAS)},
            "val_mae_persistensi": round(float(ep.mean()), 4),
            "val_skill": round(float(1 - e.mean() / ep.mean()), 4) if ep.mean() > 0 else None,
            "val_mae_berubah": round(float(e[ubah].mean()), 4) if ubah.any() else None,
            "val_mae_persistensi_berubah": round(float(ep[ubah].mean()), 4) if ubah.any() else None,
            "val_n_berubah": int(ubah.sum())}


def muat(standar: bool = True):
    """X [N, 6, 120] float32 (dinormalisasi per kanal dari data latih bila ``standar``), y [N, 4] perubahan suhu (°C)."""
    global KODEK
    b = bagian(); c = b["c"]
    pisah.periksa_tidak_bocor(c["blok"][b["latih"]], c["blok"][b["val"]], c["blok"][b["uji"]])
    X, y = c["X"], c["y"].astype(np.float32)
    if standar:
        X = deret.StandarKanal(X[b["latih"]])(X)
    KODEK = KodekRegresiVektor(y[b["latih"]].mean(0), y[b["latih"]].std(0))
    return (X[b["latih"]], y[b["latih"]], X[b["val"]], y[b["val"]], X[b["uji"]], y[b["uji"]])

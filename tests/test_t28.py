"""Tes protokol tim 28: jam WIB, nama blok, metrik, blok terpisah, uji target acak (data meja bila ada)."""
import numpy as np
import pytest

from src.protokol import data


def test_jam_wib_dan_nama_blok():
    m = np.array([1795021200 // 60])                     # 2026-11-19 00:00 WIB
    assert data.jam_wib(m)[0] == pytest.approx(0.0) and data.nama_blok(m) == "19-11 00.00"


def test_metrik_persistensi_adalah_perubahan_nol():
    y = np.array([[1.0, 1, 1, 1], [-1, -1, -1, -1]])
    assert data.metrik_tim(y, np.zeros_like(y))["val_skill"] == 0.0
    m = data.metrik_tim(y, y + 0.5)
    assert m["val_mae_rata"] == 0.5 and m["val_skill"] == 0.5 and m["val_n_berubah"] == 2


nyata = pytest.mark.skipif(not all(f.exists() for f in data.FOLDER.values()), reason="data belum ada")


@nyata
def test_blok_terpisah_dan_bentuk():
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()
    b = data.bagian(); c = b["c"]
    assert not set(c["blok"][b["latih"]]) & set(c["blok"][b["val"]].tolist() + c["blok"][b["uji"]].tolist())
    assert Xtr.shape[1:] == (len(data.KANAL), data.RIWAYAT) and ytr.shape[1] == len(data.HORIZON)
    assert len(Xva) > 0 and len(Xte) > 0
    # target +15 = suhu titik awal 15 menit kemudian (blok yang sama) − suhu titik ini; masukan relatif = 0 di menit t
    i = np.flatnonzero(b["latih"])[0]
    assert c["persist"][i + 15] - c["persist"][i] == pytest.approx(c["y"][i, 0], abs=1e-4)
    assert np.allclose(c["X"][:, :4, -1], 0)


@nyata
def test_target_acak_tidak_lebih_baik_dari_rata_rata():
    from sklearn.ensemble import RandomForestRegressor

    b = data.bagian(); c = b["c"]
    F = c["X"][:, :, ::10].reshape(len(c["X"]), -1)
    ytr = c["y"][b["latih"]][np.random.default_rng(0).permutation(int(b["latih"].sum()))]
    m = RandomForestRegressor(100, random_state=0, n_jobs=1).fit(F[b["latih"]], ytr)
    yva = c["y"][b["val"]]
    acak = np.abs(m.predict(F[b["val"]]) - yva).mean()
    rata = np.abs(c["y"][b["latih"]].mean(0) - yva).mean()      # penebak rata-rata perubahan
    assert acak > 0.8 * rata, f"MAE target acak {acak:.2f} terlalu rendah dibanding rata-rata {rata:.2f}"

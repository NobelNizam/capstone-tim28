"""Sidik data (``data_hash``) dari berkas yang tercantum di ``data/MANIFEST.json`` (milik starter).

Bentuk MANIFEST.json (diisi Penjaga Data pada Sprint 0)::

    {"sumber": "https://…",            # tautan dataset dari kartu Kasus tim
     "berkas": ["data/latih.csv", "data/uji.csv"],
     "catatan": "versi 2026-10, 12.340 baris"}

``data_hash`` = sha256 atas isi semua berkas (urut nama) + parameter split, dipotong 32 hex.
Semua anggota tim menghasilkan nilai yang sama bila datanya sama.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

AKAR = Path(__file__).resolve().parents[2]
MANIFEST = AKAR / "data" / "MANIFEST.json"


def baca() -> dict:
    if not MANIFEST.exists():
        raise FileNotFoundError("data/MANIFEST.json belum ada; isi pada Sprint 0 (lihat README).")
    m = json.loads(MANIFEST.read_text("utf-8"))
    if not isinstance(m.get("berkas"), list) or not m["berkas"]:
        raise ValueError("MANIFEST.json: 'berkas' wajib daftar jalur berkas data.")
    return m


def data_hash(tambahan: bytes = b"") -> str:
    m = baca()
    h = hashlib.sha256()
    for nama in sorted(m["berkas"]):
        p = AKAR / nama
        if not p.exists():
            raise FileNotFoundError(f"berkas data {nama} tidak ada; unduh dari {m.get('sumber', '?')}")
        h.update(nama.encode("utf-8"))
        with p.open("rb") as f:
            for potongan in iter(lambda: f.read(1 << 20), b""):
                h.update(potongan)
    h.update(tambahan)
    return h.hexdigest()[:32]

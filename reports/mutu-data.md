# Mutu data tim 28 (sesi tim + program harian bersama)

Dibangkitkan `python -m src.protokol.mutu`.

## Sesi

| Sesi | Peran | Selesai normal | Wi-Fi putus | Perintah aktuator |
|---|---|---|---|---|
| `sesi-20261114-000000` | program harian | ya | 0 | kipas=1 ×1, lampu=1 ×1, pompa=1 ×1 |
| `sesi-20261115-050000` | program harian | ya | 0 | lampu=1 ×1 |
| `sesi-20261118-000000` | program harian | ya | 0 | — |
| `sesi-20261118-050001` | program harian | ya | 18 | lampu=0 ×2, lampu=1 ×18 |
| `sesi-20261119-000000` | program harian | ya | 0 | kipas=1 ×1, lampu=1 ×1, pompa=1 ×1 |
| `sesi-20261119-050000` | program harian | ya | 58 | kipas=0 ×6, kipas=1 ×131, lampu=0 ×3, lampu=1 ×144, pompa=0 ×4, pompa=1 ×142 |
| `sesi-20261119-202155` | tim (uji) | ya | 0 | buzzer=0 ×1, kipas=0 ×1, lampu=0 ×1, pompa=0 ×4, pompa=1 ×3 |

## Blok bersambung

| Blok (mulai) | Selesai | Menit | Titik prakiraan | Daya kipas (W) | Suhu (°C) | Peran |
|---|---|---|---|---|---|---|
| 13-11 23.59 | 05.00 | 302 | 123 | 0,00–0,86 | 24,9–25,4 | latih |
| 15-11 05.00 | 10.00 | 301 | 122 | 0,00–0,00 | 26,5–27,9 | latih |
| 18-11 00.00 | 08.19 | 500 | 321 | 0,00–0,00 | 22,4–28,6 | latih |
| 19-11 00.00 | 09.14 | 555 | 376 | 0,44–0,92 | 22,3–27,6 | validasi |
| 19-11 20.21 | 00.21 | 241 | 62 | 0,00–0,00 | 25,4–27,6 | uji |

## Galat persistensi per sinyal (semua blok terpakai)

| Sinyal | 15 menit | 60 menit |
|---|---|---|
| daya kanal kipas (W) | 0,022 | 0,057 |
| tegangan rel (mV) | 10,8 | 26,5 |
| suhu (°C) | 0,19 | 0,64 |

## Temuan

- **Daya datar, target dipindah ke suhu.** Daya kanal kipas (satu-satunya daya yang terukur) di sesi tim maksimum 0,000 W sepanjang 4 jam: lampu mati (di luar jam 7–18), kipas mati (suhu tidak melewati 28,5 °C), pompa menyala 3× 30 s. Persistensi daya pada semua blok: MAE 0,022 W (15 menit) dan 0,057 W (60 menit), hanya dari dua kejadian kipas mati/nyala dalam ±29 jam.
- **Suhu berubah nyata.** Persistensi suhu MAE 0,19 °C (15 menit) dan 0,64 °C (60 menit). Hari kerja (18 dan 19 Nov): cahaya ruang naik sekitar 07.00–07.10, lalu suhu turun 3–5 °C dalam ±20 menit (dugaan: AC ruang dinyalakan). Minggu (15 Nov): suhu naik perlahan. Malam: datar.
- **Desimal DHT11 tersebar rata.** Di sesi tim, nilai 27,0 sampai 27,9 °C masing-masing muncul 1164–1331 kali (per detik). Dugaan: hanya bagian bulat yang bermakna (DHT11 ±2 °C); suhu per menit tampak meloncat ±1 °C, dan galat di bawah ±0,5 °C setara resolusi sensor.
- **Titik yang suhunya berubah > 0,5 °C dalam 60 menit:** latih 162 dari 566, validasi 80 dari 376, uji 0 dari 62. Uji (sesi tim, malam) hampir datar: angka uji nanti terutama mengukur kestabilan, bukan kemampuan meramal perubahan.
- **Beban pompa hanya di tegangan rel.** INA219 arus total (0x44) tidak terpasang; denyut pompa 30 s terlihat sebagai turunnya bus_mV (persistensi tegangan 11 mV pada 15 menit).
- Program harian 14–19 Nov memakai pengaturan web *selalu menyala* untuk lampu, kipas, dan pompa (aturan jam dan termostat tidak berjalan); meski begitu daya kipas terukur 0 W sepanjang 15 dan 18 Nov (penyebabnya tidak tercatat di log).

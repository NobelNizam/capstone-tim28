# Analisis validasi

Validasi = blok 19-11 00.00 (376 titik prakiraan; 80 titik suhunya berubah > 0,5 °C dalam 60 menit; 135 titik pukul 06–09). Galat dalam °C.

| Ukuran | persistensi | TL+AE (123450005) | baseline starter (ridge) | non-DL |
|---|---|---|---|---|
| **MAE rata-rata** | **0,48** | **0,50** | **2,99** | **0,48** |
| MAE +15 menit | 0,23 | 0,24 | 1,39 | 0,23 |
| MAE +30 menit | 0,40 | 0,40 | 2,39 | 0,40 |
| MAE +45 menit | 0,55 | 0,58 | 3,85 | 0,55 |
| MAE +60 menit | 0,72 | 0,78 | 4,34 | 0,72 |
| skill vs persistensi | 0,00 | -0,05 | -5,28 | 0,00 |
| MAE titik berubah | 2,01 | 1,99 | 5,60 | 2,01 |
| MAE pukul 06–09 | 1,22 | 1,25 | 7,69 | 1,22 |
| MAE di luar 06–09 | 0,06 | 0,08 | 0,36 | 0,06 |

Skill = 1 − MAE model / MAE persistensi pada titik yang sama: positif = lebih baik daripada "suhu nanti = suhu sekarang".

# Hasil komparasi (dibangkitkan run_all.py)

| Folder | Model | seed | val_mae | val_rmse | MAE rata-rata 4 horizon (°C) | parameter | latensi (ms) | durasi (s) |
|---|---|---|---|---|---|---|---|---|
| baseline | baseline | 1 | 2.9942 | 7.6698 | — | — | — | 0 |
| 123450005 | tl | 1 | 0.4979 | 1.2322 | 0.4979 | 600004 | 102.24 | 114 |
| 123450033 | rnn | 1 | 0.5122 | 1.2818 | — | — | — | 4 |
| 123450116 | cnn | 1 | 0.5693 | 1.1878 | — | — | — | 4 |
| non-dl | non-dl-persistensi | 1 | 0.4765 | 1.2371 | 0.4765 | — | — | 0 |

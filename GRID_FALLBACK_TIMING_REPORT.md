# Grid Fallback Timing Report

Test tarihi: 2026-05-08

Test akisi:

1. Sabit ROI crop: `(1819, 361, 2906, 1124)`
2. Normal hizli decode: local range + ZXing DataMatrix
3. Normal decode basarisizsa: calibrated 20x20 grid sampling fallback

## Sonuclar

| Gorsel | Sonuc | Decode Text | Normal Sure | Fallback Sure | Toplam Sure |
|---|---:|---|---:|---:|---:|
| `Kamera_2_20260507_164715.jpg` | fallback OK | `#8B3886177B ###=0892601561701S` | 1176 ms | 83 ms | 1259 ms |
| `Kamera_2_20260507_165041.jpg` | fallback OK | `#8B3886177A ###=0892603261701S` | 694 ms | 99 ms | 793 ms |

## Entegre CLI Testi

Komut:

```powershell
python dark_surface_datamatrix_reader.py --image <image> --out <out> --roi 1514 321 3406 1316 --full-image-variants 24 --skip-od
```

| Gorsel | Sonuc | Method | Decode Text | Toplam CLI Sure |
|---|---:|---|---|---:|
| `Kamera_2_20260507_164715.jpg` | OK | `calibrated_20x20_grid_sampling` | `#8B3886177B ###=0892601561701S` | 4604 ms |
| `Kamera_2_20260507_165041.jpg` | OK | `calibrated_20x20_grid_sampling` | `#8B3886177A ###=0892603261701S` | 4226 ms |
| `Kamera_2_20260507_165342.jpg` | OK | `zxing_local_range` | `#8B3886177B ###=0892602481701S` | 2214 ms |

Not: CLI sureleri Python process baslangici ve dosya yazma maliyetini de icerir.

## Yorum

Grid fallback'in kendisi pahali degil. Bu testte fallback asamasi yaklasik `80-100 ms` ek sure getirdi.

Toplam sureyi belirleyen kisim normal ZXing denemesi. Bu nedenle uretim akisinda normal deneme sayisi sinirli tutulmali, basarisizlik durumunda hizlica 20x20 grid sampling fallback'e gecilmelidir.

Onerilen uretim akisi:

```text
ROI crop
local_range + ZXing, sinirli 3-4 varyant
basarisizsa calibrated 20x20 grid sampling
fail log
```

Bu yaklasimla fail olan iki gorsel de okunmustur ve toplam sure yaklasik `0.8 - 1.3 saniye` bandindadir.

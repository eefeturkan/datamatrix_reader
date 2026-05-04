# Data Matrix OD + Decode High-Level Design

## Amaç

Crop edilmemiş endüstriyel görüntüden Data Matrix bölgesini bulmak, bu bölgeyi işleyip `20x20` bit matrisine çevirmek ve barkod metnini decode etmektir.

Mevcut çalışan akış:

```text
full image
  -> YOLO OD detection
  -> bbox crop
  -> 20x20 grid frame fit
  -> local range preprocessing
  -> cell score extraction
  -> bit matrix render
  -> zxingcpp decode
  -> decoded text
```

## Dosya ve Kod Konumları

Ana çalışan modül:

- `experiments/step13_od_then_range_grid_decode.py`

Export/handoff modülü:

- `experiments/step14_handoff_export.py`

Streamlit UI:

- `app.py`

OD modeli:

- `best.pt`

## 1. Object Detection Aşaması

Kod konumu:

- `experiments/step13_od_then_range_grid_decode.py`
- Fonksiyon: `detect_datamatrix(...)`

Ne yapar:

1. `YOLO(best.pt)` modeli yüklenir.
2. Crop edilmemiş görüntü üzerinde inference yapılır.
3. Data Matrix aday bbox'ları alınır.
4. En yüksek confidence değerine sahip bbox öncelikli denenir.

Örnek çıktı:

```text
det0: conf=0.8399 xyxy=[1677.8, 481.2, 2192.7, 1033.3]
```

## 2. Crop Aşaması

Kod konumu:

- `experiments/step13_od_then_range_grid_decode.py`
- Fonksiyon: `run(...)`
- Fonksiyon: `decode_crop(...)`

Ne yapar:

1. OD bbox koordinatları tam görüntüden kesilir.
2. Crop görüntüsü kaydedilir.
3. Bundan sonraki tüm Data Matrix okuma işlemi bu crop üzerinde yapılır.

Çıktı örneği:

```text
artifacts/od_range_decode/saf_det0_crop.png
```

## 3. Grid Frame Fit Aşaması

Kod konumu:

- `experiments/step13_od_then_range_grid_decode.py`
- Fonksiyon: `fit_frame(...)`
- Yardımcı eski modül: `experiments/step07_combined_edge_fit.py`

Ne yapar:

1. Crop içinde dot-peen noktalarından ağırlıklı nokta adayları çıkarılır.
2. `20x20` Data Matrix grid'i için frame fit yapılır.
3. Grid şu parametrelerle temsil edilir:

```text
origin = sol üst grid başlangıcı
vx     = bir hücre sağa gidiş vektörü
vy     = bir hücre aşağı gidiş vektörü
```

Neden gerekli:

Data Matrix hafif eğimli, perspektifli veya crop içinde tam hizalı olmayabilir. Bu yüzden hücre merkezleri tek tek bu grid vektörleriyle hesaplanır.

## 4. Preprocess Aşaması

Kod konumu:

- `experiments/step13_od_then_range_grid_decode.py`
- Fonksiyon: `local_range_response(...)`

Uygulanan preprocessing:

```python
response = dilate(gray, 7x7) - erode(gray, 7x7)
```

Yani gri görüntü üzerinde `7x7 local range response` çıkarılır.

Neden bu kullanılıyor:

Dot-peen Data Matrix'te bilgi mutlak siyahlıkta değil, metal üzerindeki lokal kabartı/çukur kontrastındadır. Bu yüzden klasik threshold yerine lokal kontrast cevabı daha stabil sonuç verir.

Çıktı örneği:

```text
artifacts/od_range_decode/saf_det0_range_response.png
```

## 5. Cell Score Extraction

Kod konumu:

- `experiments/step13_od_then_range_grid_decode.py`
- Fonksiyon: `sample_scores(...)`

Ne yapar:

1. Grid üzerinde `20x20 = 400` hücre merkezi hesaplanır.
2. Her hücre merkezinden `13x13` patch alınır.
3. Patch içindeki `p90` değeri hücre skoru olarak kullanılır.

Neden `13x13 patch`:

Tek piksele bakmak yerine merkezin çevresindeki küçük alan değerlendirilir. Dot biraz kaymışsa yakalanır; komşu hücre çok fazla içeri alınmaz.

## 6. Bit Matrix Oluşturma

Kod konumu:

- `experiments/step13_od_then_range_grid_decode.py`
- Fonksiyon: `decode_crop(...)`

Ne yapar:

1. 400 hücre skoru quantile threshold ile binary yapılır.
2. Denenen quantile aralığı:

```text
0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52
```

3. Oluşan matris siyah/beyaz Data Matrix gibi render edilir.

Çıktı örneği:

```text
artifacts/od_range_decode/saf_det0_q0.47_bits.png
```

## 7. Decode Aşaması

Kod konumu:

- `experiments/step13_od_then_range_grid_decode.py`
- Fonksiyon: `decode_bits(...)`

Ne yapar:

1. Bit matrisi 4 rotasyonda denenir.
2. Gerekirse invert hali de denenir.
3. Render edilen bit matrisi `zxingcpp` decoder'a verilir.

Decode edilen text burada oluşur:

```python
decoded.update(result.text for result in zxingcpp.read_barcodes(rendered))
```

Başarılı örnek:

```text
#8B3886177B ###=0302604331701S
```

## 8. Barkod Numarasıyla Kaydetme

Kod konumu:

- `experiments/step14_handoff_export.py`

Ne yapar:

1. `step13` akışını çalıştırır.
2. Başarılı decode sonucu bulursa barkod text'ini dosya adına uygun hale getirir.
3. Crop ve işlenmiş pipeline görselini barkod numarasıyla export eder.

Çıktı klasörü:

```text
artifacts/barcode_exports/
```

Örnek dosyalar:

```text
artifacts/barcode_exports/8B3886177B_0302604331701S_crop.png
artifacts/barcode_exports/8B3886177B_0302604331701S_pipeline.png
artifacts/barcode_exports/8B3886177B_0302604331701S_range_response.png
artifacts/barcode_exports/8B3886177B_0302604331701S_bits.png
artifacts/barcode_exports/8B3886177B_0302604331701S_metadata.txt
```

## Özet

Bu sistemde OD sadece Data Matrix bölgesini bulmak için kullanılır. Asıl okuma başarısı crop sonrası uygulanan local range preprocessing ve `20x20` grid sampling yönteminden gelir.

Mühendislik entegrasyonu için ana kritik fonksiyonlar:

- `detect_datamatrix(...)`
- `fit_frame(...)`
- `local_range_response(...)`
- `sample_scores(...)`
- `decode_bits(...)`

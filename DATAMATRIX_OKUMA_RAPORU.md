# Data Matrix Okuma Raporu

## Kapsam

Bu çalışma iki odak görüntü üzerinde yapılmıştır:

- `cropped.png`
- `yenitest.png`
- `yenitest2.png`

Amaç, fabrika ortamında benzer açı ve ışık koşullarında çekilecek dot-peen Data Matrix işaretlerini okuyabilecek hızlı ve genellenebilir bir görüntü işleme zinciri belirlemektir.

## Başarılı yöntem

İki görüntü de aynı yöntemle başarıyla okunmuştur:

1. Görüntü gri seviyeye çevrildi.
2. Dot-peen izlerini öne çıkarmak için lokal range cevabı üretildi:
   - `response = dilate(gray, 7x7) - erode(gray, 7x7)`
3. Data Matrix yapısı `20x20` hücreli sabit grid olarak ele alındı.
4. Her hücrenin beklenen merkezinde `13x13` patch örneklendi.
5. Her patch için `p90` değeri hücre doluluk skoru olarak alındı.
6. 400 hücre skoru quantile eşikleriyle ikili bit matrisine çevrildi.
7. Bit matrisi quiet-zone ile yeniden render edildi.
8. Render edilen matris `zxingcpp` ile decode edildi.

Bu yaklaşım klasik threshold, Hough-only nokta doluluğu ve genel decoder denemelerine göre daha stabil sonuç verdi. Ana sebep, dot-peen işaretlerde bilginin mutlak koyulukta değil, lokal kabartı/çukur kontrastında taşınmasıdır.

## Sonuçlar

| Görüntü | Başarılı quantile aralığı | Decode sonucu |
| --- | --- | --- |
| `cropped.png` | `q=0.48`, `q=0.49`, `q=0.50` | `#8B3886177B ###=0302604331701S` |
| `yenitest.png` | `q=0.47`, `q=0.48`, `q=0.49`, `q=0.50` | `#8B3886177B ###=0302604331701S` |
| `yenitest2.png` | `q=0.48`, `q=0.49`, `q=0.50` | `#8B3886177B ###=0302604331701S` |

## Tekrar çalıştırma

```powershell
python experiments\step12_range_grid_decode.py
```

Beklenen çıktı dosyaları:

- `artifacts/range_grid_decode/summary.txt`
- `artifacts/range_grid_decode/results.csv`
- `artifacts/range_grid_decode/cropped_pipeline.png`
- `artifacts/range_grid_decode/yenitest_pipeline.png`
- `artifacts/range_grid_decode/yenitest2_pipeline.png`

## Neden farklı bit matrisleri aynı sonucu verebiliyor?

Data Matrix ECC200, Reed-Solomon hata düzeltme kullanır. Bu yüzden render edilen bit matrisinde bazı hücreler hatalı olsa bile decoder aynı payload'u geri çıkarabilir.

Önemli nokta: hata düzeltme kapasitesi doğrudan "kaç kare yanlış olabilir?" diye sabit bir sayı değildir. Hata düzeltme codeword seviyesinde çalışır; bir codeword birden fazla modülden oluşur ve modüller matrise dağıtılmış durumdadır. Bu nedenle:

- Birkaç yanlış modül hiç sorun yaratmayabilir.
- Aynı sayıda yanlış modül, kritik codeword'lere denk gelirse decode bozulabilir.
- Finder/timing kenarlarında hata varsa önce geometrik okuma bozulabilir.
- Biz bit matrisini yeniden render ettiğimiz için decoder geometriyle değil, daha çok ECC düzeltmesiyle uğraşır.

Bu testlerde `q=0.47..0.50` gibi birkaç farklı eşik aynı metni döndürdüğü için sonuç daha güvenilir kabul edilebilir. Üretim tarafında aynı payload'un birden fazla yakın eşikte okunması güven skoru olarak kullanılabilir.

GS1 DataMatrix guideline tablosuna göre `20x20` sembolde toplam `22` codeword, `18` data codeword ve `9/15` maksimum correctable error/erasure codeword kapasitesi vardır. Bu değer modül/kare sayısı değil, codeword seviyesindeki kapasitedir.
- `artifacts/range_grid_decode/cropped_grid_overlay.png`
- `artifacts/range_grid_decode/yenitest_grid_overlay.png`
- `artifacts/range_grid_decode/cropped_range_response.png`
- `artifacts/range_grid_decode/yenitest_range_response.png`
- `artifacts/range_grid_decode/*_bits.png`

## Kanıt görselleri

Pipeline preview dosyaları her görüntü için şu sırayı gösterir:

1. Orijinal görüntü
2. Kalibre edilmiş `20x20` grid
3. Lokal range response
4. Decode veren render edilmiş bit matrisi

Bu dosyalar rapor/sunum için doğrudan kullanılabilir:

- `artifacts/range_grid_decode/cropped_pipeline.png`
- `artifacts/range_grid_decode/yenitest_pipeline.png`

## Mevcut sınırlama

Bu aşamada grid frame değerleri iki odak görüntü için kalibre edilmiş olarak script içinde tutulmaktadır. Fabrikadaki sabit kamera/ışık kurulumunda bu kabul edilebilir bir ilk sürümdür; üretim versiyonunda grid frame şu yollardan biriyle otomatikleştirilmelidir:

- Tek seferlik kamera kalibrasyonu ve sabit ROI/grid parametresi.
- Hough/lattice tabanlı otomatik grid başlangıcı.
- Finder pattern skoruyla küçük offset/pitch düzeltmesi.

## Önerilen üretim akışı

1. Kamerayı ve ışığı sabitle.
2. Bir referans parça ile `20x20` grid frame kalibrasyonu yap.
3. Her yeni görüntüde aynı grid frame etrafında küçük offset araması uygula.
4. Hücre skorlarını lokal range response üzerinden üret.
5. `q=0.47..0.50` aralığında bit matrislerini dene.
6. Decode sonucu geldiyse kabul et; birden fazla q aynı sonucu veriyorsa güven skoru yüksek kabul et.

## Kısa sonuç cümlesi

Bu iki test görüntüsünde Data Matrix, lokal kontrast tabanlı ön işleme ve kalibre edilmiş `20x20` grid sampling yöntemiyle başarıyla çıkarılmıştır. Klasik görüntü threshold yöntemlerinden farklı olarak hücre doluluğu, dot-peen izlerinin lokal range cevabı üzerinden ölçülmüş ve bit matrisi yeniden render edilerek `zxingcpp` ile okunmuştur.

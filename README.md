# Data Matrix Reader v1

Bu proje, zor okunan dot-peen / noktalı Data Matrix goruntulerini Python ile okuyup sonucu bir web arayuzunde gostermek icin hazirlandi.

Ana hedef:
- Kullanici bir goruntu yukler.
- Sistem birden fazla on isleme ve reconstruction yaklasimi dener.
- Okunan metni ekranda basar.
- Sonucu ureten islenmis veya sentetik Data Matrix goruntusunu da gosterir.

## Su Anki Durum

Calisan ornekler:
- `test4.png`
- `test5.png`

Bu iki goruntu su payload'a cozuluyor:

```text
#8B3886177B ###=0302604331701S
```

Onemli not:
- `test4` ve `test5` ayni fiziksel sembolun farkli parlakliktaki goruntuleri gibi davraniyor.
- Sistem artik bu iki dosya icin ayni sentetik 20x20 Data Matrix goruntusunu uretmeli.
- Bu davranis kritik kabul edilmeli; ileride yapilacak tuning bunu bozmamali.

Henuz cozulmemis / sonraya birakilmis dosyalar:
- `test1.png`
- `cropped.png`

Bekleyen ana problem:
- `test1` tarafinda aci / perspektif toleransi daha da guclendirilmeli.
- `cropped` tarafinda acik zemin ve komponent ayrisma davranisi ayri ele alinmali.

## Repo Yapisi

```text
app.py
requirements.txt
src/
  datamatrix_reader/
    __init__.py
    pipeline.py
```

Dosyalar:
- `app.py`: Streamlit arayuzu
- `src/datamatrix_reader/pipeline.py`: tum decode, preprocess, reconstruction ve skor mantigi
- `src/datamatrix_reader/__init__.py`: public API export

## Calistirma

Gereksinimler:

```bash
py -3.12 -m pip install -r requirements.txt
```

Uygulamayi baslatma:

```bash
py -3.12 -m streamlit run app.py
```

## Public API

Paket disina acilan fonksiyon:

```python
from datamatrix_reader import decode_image
```

Kullanim:

```python
result = decode_image(image_bytes)
```

`DecodeResult` alanlari:
- `text`: okunan metin veya `None`
- `engine`: sonucu veren engine
- `stage_name`: sonucu ureten asama etiketi
- `score`: siralama skoru
- `processed_image`: ekranda gosterilen gri tonlu goruntu
- `alternatives`: alternatif aday listesi

## Yuksek Seviye Mimari

Pipeline iki ana kola ayrilir:

1. Reconstruction-first kolu
- Ham grayscale goruntu uzerinden calisir.
- Amac semboldeki nokta / iz yapisindan tekrar temiz bir 20x20 Data Matrix olusturmaktir.
- Basarili olursa dogrudan bu kolun sonucu doner.

2. Klasik preprocess fallback kolu
- Reconstruction sonuc vermezse devreye girer.
- CLAHE, sharpen, invert, threshold, morphology, upscale, rotation gibi varyantlarla `zxingcpp` ve `pylibdmtx` dener.

Pratikte su an en degerli kol reconstruction koludur. `test4` ve `test5` bununla cozulmektedir.

## Reconstruction Akisi

`pipeline.py` icindeki temel akis:

1. Goruntu yuklenir.
- `decode_image(...)`
- RGB -> BGR -> grayscale

2. Reconstruction icin ROI adaylari uretilir.
- `full` goruntu
- inset / square crop adaylari
- `row20-*` lokalize crop adaylari

Onemli urun karari:
- Kullanici genelde zaten Data Matrix'e kirpilmis goruntu veriyor.
- Bu nedenle "tum goruntu once" yaklasimi esas alinmali.
- Ek ROI / crop yaklasimlari ana yol degil yardimci yol olmalidir.

3. `row20` lokalizasyonu ile 20 modulluk ust satir aranir.
- `Data Matrix` 20x20 kabul ediliyor.
- Ust satirdaki modullerden bir progression fit edilmeye calisiliyor.

4. Response map'ler uretilir.
- `mix`
- `tophat`
- `blackhat`

5. Reconstruction denemesi yapilir.
- Dikey ofset
- pitch
- shear
- threshold quantile

Her kombinasyonda 20x20 bit matrisi tahmin edilir.

6. Tahmin edilen bit matrisi orientation kontrolunden gecer.
- solid border ve alternating border mantigi kullanilir

7. Sentetik Data Matrix render edilir.
- `_render_bits(...)`
- Buyutulmus siyah-beyaz raster uretilir

8. `zxingcpp` ile pure image decode denenir.
- `_decode_pure_render(...)`

## Hizlandirma ve Aday Secimi

Pipeline su an 3 reconstruction profilini kullaniyor:

- `FAST`
- `REFINE`
- `FULL`

Amaç:
- Once hizli arama ile aday bulmak
- Belirsizlik varsa ayni ROI uzerinde daha dar ama daha guclu bir refine gecisi yapmak
- Tum pahali exhaustive aramayi en sona itmek

Bu mantik ozellikle `test5` icin eklendi.

## Neden Ozel Bir Secim Mantigi Eklendi

Gecmiste su regression yasandi:
- `test5` dogru payload'i okuyordu
- ama gosterilen sentetik matris gorsel olarak yanlisti
- decoder yine de ayni text'i veriyordu cunku `ECC200` hata duzeltme hatalari tolere edebiliyordu

Bu nedenle su kural eklendi:
- Ayni text'i veren birden fazla reconstruct adayi varsa
- sadece "decode etti" diye secilmez
- once referans Data Matrix yerlesimi uretilir
- sonra adaylar bu referans yerlesime Hamming uzakligina gore siralanir
- en yakin aday secilir

Bu referans, `pylibdmtx.encode(...)` ile uretilir.

Kritik sonuc:
- `test5` icin gosterilen sentetik goruntu tekrar `test4` ile birebir ayni hale getirildi

## Kritik Fonksiyonlar

`decode_image`
- Tum orkestrasyonun giris noktasi

`_generate_roi_candidates`
- Reconstruction ve fallback decode icin goruntu adaylari uretir

`_generate_row_localized_rois`
- 20 modulluk ust satira benzeyen crop adaylari bulur

`_reconstruct_datamatrix_candidates`
- Response map + grid fitting + sentetik matris + pure decode zinciri

`_select_decoded_group_candidate`
- Ayni text'i veren adaylar arasindan en guvenilir olanini secer

`_reference_bits_for_text`
- `pylibdmtx.encode(...)` ile referans Data Matrix bit matrisi uretir

`_ambiguous_reconstruction_rois`
- Hizli aramada belirsizlik varsa refine gerekip gerekmedigine karar verir

## Test Verileri Hakkinda Bilinenler

`test4.png`
- Dogru calisiyor
- Referans kabul edilen sentetik goruntu burada elde edildi

`test5.png`
- `test4` ile ayni payload
- Parlaklik farki yuzunden once yanlis sentetik secilebiliyordu
- Su an secilen sentetik goruntu `test4` ile birebir ayni olmalidir

`test1.png`
- Mevcut sorun daha cok aci / perspektif
- 20'lik row localization daha toleransli hale getirilmeli

`cropped.png`
- Ayri davranis gosteren acik arka planli veri
- Sonraki iterasyonda ele alinmali

## Baska Bir Agent Icin Onemli Notlar

Bu projede bozulmamasi gereken ana kararlar:

- Gosterilen sentetik matris sadece decode edilebilir degil, fiziksel sembole de sadik olmali.
- `test4` ve `test5` ayni sentetik matrisle sonuclanmalı.
- Kullanici zaten kirpilmis goruntu veriyor; gereksiz crop mantigini ana yol yapma.
- Reconstruction kolu ana cozum; klasik threshold kolu yalnizca fallback.
- `test5`te hiz ugruna tekrar "dogru text ama yanlis sentetik matris" regresyonuna dusme.

Yeni agent bir degisiklik yapacaksa su dosyayi once okumali:
- [src/datamatrix_reader/pipeline.py](src/datamatrix_reader/pipeline.py)

Ozellikle su bolgeleri incelemeli:
- reconstruction profilleri
- `_build_reconstructed_candidates`
- `_reconstruct_datamatrix_candidates`
- `_select_decoded_group_candidate`
- `_reference_bits_for_text`

## Kisa Ozet

Bu proje klasik barkod okutmadan ziyade reconstruction odakli bir Data Matrix okuyucusudur.

Su an:
- web arayuzu var
- public API var
- `test4` ve `test5` cozuluyor
- sentetik matris gosteriliyor

Sonraki ana is:
- `test1`
- sonra `cropped`

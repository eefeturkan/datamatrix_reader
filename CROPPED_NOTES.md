# `cropped.png` Notes

Bu dosya, `cropped.png` icin daha once denenip sonuc vermeyen yontemleri tekrar etmemek icin tutuluyor.

## Hedef

- `cropped.png` icin dogru sentetik Data Matrix uretmek
- `test4.png` ve `test5.png` akisini bozmamak

## Goruntu Gozlemleri

- Acik renkli, dokulu metal yuzey
- Dot-peen izleri koyu arka planli `test4/test5` gorsellerinden farkli
- Kadraj daha buyuk, sembol daha genis bir alana yayiliyor
- Sol kenar ve ust satir gozle secilebiliyor, ama texture ve parlamalar fazla yalanci cevap uretiyor

## Olculen Farklar

- `test4.png`: koyu arka plan, reconstruction calisiyor
- `test5.png`: koyu arka plan, reconstruction calisiyor
- `cropped.png`: acik arka plan, mevcut reconstruction ve klasik preprocess fallback decode vermiyor

## Denenip Calismayan Yontemler

### 1. Mevcut reconstruction akisi

- `decode_image(cropped.png)` sonucu `text=None`
- Mevcut reconstruction branch anlamli decode cikarmadi

### 2. Mevcut klasik preprocess fallback

- Ham gri, invert, CLAHE, blackhat, tophat
- Otsu / adaptive threshold
- 1x-4x upscale
- `zxingcpp` ve `pylibdmtx`
- Sonuc: decode yok

### 3. Mevcut `row20` lokalizasyonu

- Fast path: `row20` ROI uretemedi
- Slow path: yalnizca `row20-base-blackhat-0.92-9` gibi tek bir ROI cikti
- Bu ROI ile fast / refine / full reconstruct decode vermedi

### 4. Manuel row crop denemeleri

- `blackhat q=0.92` uzerinden birden fazla row index manuel crop denendi
- Row 2, 4, 9, 13, 14, 16 gibi adaylar reconstruct'e sokuldu
- Sonuc: decode yok

### 5. Manuel kutu / bbox denemeleri

- Morphology ile buyuk konturlerden ROI kutulari cikarildi
- Ornek kutular:
  - `bh96`
  - `bh94`
  - `th96`
  - `th92`
- Bu ROI'lerde reconstruction yine decode vermedi

### 6. Generic ROI + klasik decode denemeleri

- Siki ROI kutulari icinde `_build_image_candidates(...)` + `zxingcpp/pylibdmtx` denendi
- Sonuc: decode yok

### 7. Hough / blob merkezleri ile kaba izgara

- `SimpleBlobDetector`
- `HoughCircles`
- Problem: texture yuzunden fazla yalanci merkez
- Kmeans ile `20 x` ve `20 y` merkez cikarildi
- Bu merkezlerle sentetik matris render edilip decode denendi
- Sonuc: decode yok

### 8. Hough merkezlerinden row / column anchor

- Ust satir ve sol kolon adaylari Hough merkezlerinden cikarildi
- Eksik nokta toleransli progression fit denendi
- Izgara kurulup dot-presence ile bit matrisi dolduruldu
- Sonuc: decode yok

### 9. Global parametre brute-force

- `x0 / y0 / pitch / shear` taramasi
- Hough merkezleriyle dot-presence doldurma
- `blackhat` response ile global grid score doldurma
- Mevcut brute-force'lar anlamli decode vermedi

### 10. Belirsiz hucre flip aramasi

- `debug_quick.py` mantigindaki sabit grid (`x0=63`, `y0=18`, `xp=17.1`, `yp=18.3`) uzerinden score matrisi alindi
- Quantile threshold ile olusan sentetik matrislerde threshold'a en yakin hucreler belirsiz kabul edildi
- 1, 2 ve sinirli 3 hucre flip kombinasyonlari decode icin denendi
- Sonuc: decode yok

### 11. Top grid adaylari + beam search

- `debug_quick.py` seed araligi etrafinda genis ama hedefli bir grid aday taramasi yapildi
- Adaylar border / occupancy / orient skorlarina gore siralandi
- En iyi adaylar uzerinde belirsiz hucre beam search uygulandi
- Sonuc: decode yok

### 12. Debug artefakt decode taramasi

- `debug/` altindaki tum `cropped*.png` ara ciktlari `zxingcpp`, `zxingcpp pure`, `invert`, `scale`, `pylibdmtx` ile tarandi
- Sonuc: dogrudan decode veren ara temsil bulunmadi

### 13. Java ZXing pure decoder dogrulamasi

- `debug/cropped_bestmatrix_r*.png` gorselleri Java ZXing `datamatrix.decoder.Decoder` ile dogrudan bit matrisi olarak denendi
- Sonuc: tumu `FormatException`

### 14. Response-matched haritalar

- `debug_cropped6.py` tarafindaki `matched_r4..r7_pos` ve `smooth_*` response haritalari test edildi
- Sabit-grid sampling ve threshold ile dogrudan decode cikmadi
- Bu haritalar gorsel olarak daha temiz, ama mevcut karar mantigi ile yeterli degil

### 15. Bright-background fallback prototipi

- `pipeline.py` icine acik-zemin dot-peen icin ayri bir fallback iskeleti eklendi
- Icerik:
  - median-dark response
  - smooth blackhat / mix
  - matched-circle positive response
  - seedli grid search
  - belirsiz hucre beam search
- Durum:
  - yapisal olarak eklendi
  - henuz `cropped.png` icin decode vermiyor
  - performans tarafinda da ek tuning gerekiyor

## Simdiki En Iyi Hipotez

`cropped.png` icin en umutlu yol:

1. Nokta benzerligini temsil eden daha temiz bir `dot-likelihood` haritasi uretmek
- Ornek: dot template ortalamasi + `matchTemplate`
- Ya da ring/dot pattern icin daha iyi bir korelasyon cevabi

2. Bu harita uzerinden global 20x20 lattice fit yapmak
- Sadece row-localization'a bagli kalmamak
- `x0 / y0 / pitch / shear` icin response-maksimizasyon yapmak

3. Oluan grid hucrelerini row/column bazli degil dogrudan dot-likelihood ile doldurmak

## Kacinilacak Tekrarlar

- Sadece threshold varyasyonlari ekleyip yeniden brute-force yapmak
- Mevcut `row20` akisina biraz daha quantile eklemek
- Hough / blob sonucunu oldugu gibi kmeans'e verip decode beklemek
- Sabit-grid + az sayida hucre flip aramasi tek basina yeterli gorunmuyor
- Seed etrafinda top-grid + beam-search kombinasyonu da tek basina yeterli gorunmuyor
- Debug artefaktlardan hicbiri mevcut decoderlarla dogrudan okunmuyor
- Sorun artik daha cok `nokta skoru -> dogru bit matrisi` donusumunde

## Sonraki Adim

- Dot-template tabanli bir bright-background fallback branch prototiple
- `cropped.png` icin calisiyorsa pipeline'a son fallback olarak ekle
- `test4/test5` branchlerini degistirmeden koru

## Step-by-step Log

### Step 01 - Dot-likelihood haritalari

Kaydedilen ciktılar:
- [matched_r5_pos.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step01_dot_likelihood\matched_r5_pos.png)
- [matched_r5_pos_peaks.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step01_dot_likelihood\matched_r5_pos_peaks.png)
- [median_dark_31_peaks.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step01_dot_likelihood\median_dark_31_peaks.png)
- [marked_red_mask.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step01_dot_likelihood\marked_red_mask.png)

Uretilen response ailesi:
- `median_dark_21`
- `median_dark_31`
- `smooth_blackhat`
- `smooth_mix`
- `matched_r5_pos`
- `matched_r6_pos`
- `matched_r7_pos`

Olculen ozet:
- `median_dark_21`: 240 peak, en iyi satir yogunlugu `14`
- `median_dark_31`: 244 peak, en iyi satir yogunlugu `14`
- `smooth_blackhat`: 282 peak, en iyi satir yogunlugu `16`
- `smooth_mix`: 282 peak, en iyi satir yogunlugu `16`
- `matched_r5_pos`: 411 peak, en iyi satir yogunlugu `20`
- `matched_r6_pos`: 398 peak, en iyi satir yogunlugu `23`
- `matched_r7_pos`: 376 peak, en iyi satir yogunlugu `20`

Neden tek basina yetmedi:
- `matched_r5_pos` ve benzeri response'lar gercek noktalari daha iyi vurguluyor
- ama ayni anda arka plan dokusundan da fazla yalanci tepe uretiyor
- yani sorun artik “noktayi hic goremiyoruz” degil
- sorun “gercek nokta merkezi ile texture kaynakli sahte tepeyi ayiramiyoruz”

Bir sonraki mantikli adim:
- bu likelihood map'i dogrudan blob listesine cevirmek yerine
- 20x20 lattice'i topluca fit etmek
- yani “once nokta bul, sonra grid kur” degil
- “grid hipotezi kur, likelihood ustunde en iyi 20x20 yerlesimi bul” yaklasimi

### Step 01B - Insan isaretleri ile karsilastirma

Referans:
- [cropped_isaretli.png](d:\DATAGUESS\datamatrix_v2\cropped_isaretli.png)

Olculenler:
- Kirmizi isaretlerden cikan referans merkez sayisi: `201`
- Referans kapsama kutusu: `x=[69, 392]`, `y=[51, 372]`

Otomatik peak setleri ile karsilastirma (`tol=6 px`):
- `median_dark_31`
  - otomatik peak: `240`
  - eslesen referans nokta: `164`
  - precision: `0.683`
  - recall: `0.816`
- `matched_r5_pos`
  - otomatik peak: `405`
  - eslesen referans nokta: `200`
  - precision: `0.494`
  - recall: `0.995`

Yorum:
- `matched_r5_pos` neredeyse tum gercek noktalari goruyor
- ama texture kaynakli cok fazla yalanci tepe ekliyor
- `median_dark_31` daha temiz ama fazla kacirma yapiyor

Bu ne anlama geliyor:
- tek bir response ile dogrudan peak secmek dogru strateji degil
- daha iyi yol:
  - `matched_r5_pos` ile yuksek recall'i korumak
  - `median_dark_31` ile sahte tepe cezasi vermek
  - sonra bu iki harita ustunde birlikte lattice fit yapmak

### Step 01C - Isaretli gorselden grid geometrisi

Insan isaretlerinden cikan ek bulgular:
- referans nokta sayisi: `201`
- kapsama kutusu:
  - `x=[69, 392]`
  - `y=[51, 372]`
- yalnizca bbox tabanli pitch tahmini:
  - `x pitch ~= 17.007`
  - `y pitch ~= 16.901`

Bu neden onemli:
- onceki debug seed'lerinde kullandigimiz `yp ~= 18.3` buyuk olasilikla fazla yuksek
- `cropped.png` icin grid neredeyse kareye yakin
- yani seed tarafinda en buyuk hata `y` ekseninde

Neden row/column cluster sayilari daginik:
- isaretli gorsel tum 20x20 hucreleri degil, yalnizca siyah modulleri isaretliyor
- bu nedenle tek bir fiziksel satirda 6-12 adet nokta gorulmesi normal
- dolayisiyla `row20` mantigi burada yapisal olarak yanlis prior

Sonuc:
- yeni yaklasim satir bazli peak saymaya dayanmamali
- `20x20 affine lattice` dogrudan optimize edilmeli
- optimize ederken:
  - `matched_r5_pos` = pozitif kanit
  - `median_dark_31` = yalanci tepe cezasi
  - `x pitch ~ 17.0`
  - `y pitch ~ 16.9`
  - kucuk rotation / shear

### Step 02 - Isaretli noktalardan dogrudan lattice fit

Kaydedilen ciktilar:
- [marked_grid_overlay.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step02_marked_lattice\marked_grid_overlay.png)
- [raw_grid_overlay.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step02_marked_lattice\raw_grid_overlay.png)
- [matched_scores_heatmap.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step02_marked_lattice\matched_scores_heatmap.png)
- [median_scores_heatmap.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step02_marked_lattice\median_scores_heatmap.png)
- [combined_scores_heatmap.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step02_marked_lattice\combined_scores_heatmap.png)
- [candidate_q48.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step02_marked_lattice\candidate_q48.png)
- [candidate_q50.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step02_marked_lattice\candidate_q50.png)
- [candidate_q52.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step02_marked_lattice\candidate_q52.png)
- [candidate_q54.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step02_marked_lattice\candidate_q54.png)
- [candidate_q56.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step02_marked_lattice\candidate_q56.png)
- [summary.txt](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step02_marked_lattice\summary.txt)

Yapilan is:
- `cropped_isaretli.png` icindeki kirmizi merkezler cikarildi
- bu merkezlerin yakin komsularindan kaba grid bazis vektorleri olculdu
- dar bir aralikta:
  - origin
  - `vx`
  - `vy`
  - kucuk aci degisimleri
  taranarak 20x20 lattice fit edildi
- bulunan grid daha sonra isaretsiz `cropped.png` ustunde `matched_r5_pos` ve `median_dark_31` response'lari ile orneklendi
- `combined = matched - 0.55 * median` skoru uretildi
- bu skordan birden fazla quantile ile sentetik matris adaylari cizdirildi

Olculen en iyi geometri:
- origin = `(79.0, 52.0)`
- `vx = (15.605, 0.907)` -> uzunluk `15.631`
- `vy = (-1.511, 16.037)` -> uzunluk `16.108`
- acilar:
  - `vx ~= 3.33 deg`
  - `vy ~= 95.38 deg`

Neden decode olmadi:
- geometri tarafi artik tamamen kopuk degil; grid overlay dosyalari bunun makul oldugunu gosteriyor
- buna ragmen `combined_scores` uzerinden uretilen bit matrisleri decode olmadi
- kritik bulgu:
  - sol border neredeyse solid cikiyor
  - ust border ise beklenenden belirgin bicimde bozuk kaliyor
- yani kirilan yer artik:
  - `grid nerede?` sorusu degil
  - `bu grid uzerindeki hangi hucre gercekten siyah?` sorusu

Ara sonuc:
- insan isaretlerinden grid geometrisi cikarilabiliyor
- ama mevcut hucre skoru (`matched - 0.55 * median`) ust satirdaki siyah modulleri dogru ayiramiyor
- bu da `cropped.png` icin ana problemin geometri degil, hucre siniflandirma / occupancy karari oldugunu guclu bicimde teyit etti

Bir sonraki mantikli adim:
- isaretli noktalarla zayif supervision kullanmak
- yani her grid hucresi icin yalnizca tek bir `max(response)` yerine:
  - merkez koyulugu
  - halka kontrasti
  - yatay/dikey profil
  - local background farki
  ozelliklerini cikarip
- siyah modulu "nokta izi" olarak daha sadik siniflayan yeni bir hucre skoru tanimlamak

### Step 03 - Blob ve circle tabanli merkez tespiti

Kaydedilen ciktilar:
- [summary.txt](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step03_blob_analysis\summary.txt)
- [hough_gray_inv_overlay.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step03_blob_analysis\hough_gray_inv_overlay.png)
- [blob_matched_circ_overlay.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step03_blob_analysis\blob_matched_circ_overlay.png)
- [gray_inv.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step03_blob_analysis\gray_inv.png)
- [median_dark_31.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step03_blob_analysis\median_dark_31.png)
- [matched_r5_pos.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step03_blob_analysis\matched_r5_pos.png)

Deney seti:
- `SimpleBlobDetector`
  - `gray_inv`
  - `median_dark_31`
  - `matched_r5_pos`
  - `blackhat_17`
- `HoughCircles`
  - `gray_inv`
  - `median_dark_31`
  - `matched_r5_pos`

Olculer (`tol=6 px`, referans = `cropped_isaretli.png`):
- `hough_gray_inv`
  - count = `314`
  - matched = `177`
  - precision = `0.564`
  - recall = `0.881`
  - f1 = `0.687`
- `hough_median_dark`
  - count = `251`
  - matched = `146`
  - precision = `0.582`
  - recall = `0.726`
  - f1 = `0.646`
- `blob_matched_circ`
  - count = `367`
  - matched = `160`
  - precision = `0.436`
  - recall = `0.796`
  - f1 = `0.563`
- `blob_matched_loose`
  - count = `406`
  - matched = `159`
  - precision = `0.392`
  - recall = `0.791`
  - f1 = `0.524`

Ara sonuc:
- bu gorselde `HoughCircles`, su anki `SimpleBlobDetector` denemelerinden daha iyi merkez veriyor
- en iyi sonuc `gray_inv` uzerindeki Hough oldu
- bu, fiziksel izin "tam dolu daire" gibi degil ama halka/kenar gradyanina sahip olmasi ile uyumlu

Neden `SimpleBlobDetector` zayif kaldi:
- threshold + connected component tabanli oldugu icin texture kaynakli yalanci bloblari fazla iceri aliyor
- circularity ile temizleyince precision biraz artiyor ama recall sert dusuyor
- yani blob mantigi tek basina tum merkezleri guvenilir ayiramadi

Neden bu yine de degerli:
- artik elimizde sadece response-map peak'leri degil, ikinci bir merkez adayi ailesi var
- `HoughCircles` merkezleri, lattice fit icin daha saglam seed olarak kullanilabilir
- ozellikle dis cerceveyi ve sag kolon gibi kacan yerleri dogrulamak icin yardimci olabilir

Bir sonraki mantikli adim:
- `HoughCircles(gray_inv)` merkezlerini ana aday seti yapmak
- `matched_r5_pos` ile birlestirip ortak / yakin merkezleri guclendirmek
- sonra bu merkezlerden:
  - once dis cerceveyi
  - sonra 20x20 lattice'i
  cikan hibrid bir fit yapmak

### Step 04 - Hough + matched hibrid dis cerceve fit

Kaydedilen ciktilar:
- [summary.txt](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step04_hybrid_frame_fit\summary.txt)
- [hough_overlay.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step04_hybrid_frame_fit\hough_overlay.png)
- [matched_peaks_overlay.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step04_hybrid_frame_fit\matched_peaks_overlay.png)
- [hybrid_points_overlay.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step04_hybrid_frame_fit\hybrid_points_overlay.png)
- [hybrid_grid_overlay.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step04_hybrid_frame_fit\hybrid_grid_overlay.png)

Yapilan is:
- `HoughCircles(gray_inv)` merkezleri cikartildi
- `matched_r5_pos` lokal peak'leri cikartildi
- iki aday kume yakinlikla merge edildi
- merge edilmis merkezlerden bazis vektorleri tahmin edildi
- modulo-offset taramasi ile 20x20 cerceve fit edilmeye calisildi

Olculen sonuc:
- `hough_count = 314`
- `matched_peak_count = 220`
- `hybrid_count = 383`
- bulunan geometri:
  - `origin = (40.0, 35.0)`
  - `vx = (16.0, 0.5)` uzunluk `16.01`
  - `vy = (-2.0, 14.0)` uzunluk `14.14`
- fit kalitesi:
  - `good_count = 121`
  - `span_u = 27`
  - `span_v = 27`
  - `truth_good = 74`
  - `truth max_i = 23`
  - `truth max_j = 23`

Neden olmadi:
- merge edilen hibrit merkez kumesi cok kirli kaldi
- modulo-offset skoru yanlis periyodu kabul etti
- ozellikle `vy` uzunlugu dogrudan bozuldu (`14.14`), yani grid pitch daha fit asamasinda sapti
- `span_u/span_v = 27` cikmasi, bulunan kafesin 20x20 degil daha buyuk yalanci bir tekrar yapisini yakaladigini gosteriyor
- bu yuzden dis cerceve yine gercek sembol sinirina oturmadi

Sonuc:
- `Hough + matched + global offset fit` dogru yon degil
- problem merkez adayini biraz iyilestirmek degil
- problem, dis cerceveyi dogrudan semantik olarak bulmadan global tekrar yapisina kapilmak

Bir sonraki mantikli adim:
- artik merkezleri merge edip butun alanda grid aramak yerine
- once dis kenar seritlerini bulmak gerekiyor
- yani:
  - sagdaki gercek kolon
  - soldaki gercek kolon
  - ustteki gercek satir
  - alttaki gercek satir
  Hough / projection / line-RANSAC ile ayrica cikarilacak
- lattice fit bundan sonra sadece bu dort kenarin icine sikistirilacak

### Step 05 - LCN + Hessian determinant / blobness

Kaydedilen ciktilar:
- [summary.txt](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step05_hessian_blobness\summary.txt)
- [lcn_dark.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step05_hessian_blobness\lcn_dark.png)
- [det_multi.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step05_hessian_blobness\det_multi.png)
- [blob_multi.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step05_hessian_blobness\blob_multi.png)
- [blob_multi_top220_overlay.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step05_hessian_blobness\blob_multi_top220_overlay.png)

Yapilan is:
- `Local Contrast Normalization` uygulandi
- koyu cuukur cevabi icin `dark_lcn = max(-(I-mu)/sigma, 0)` uretildi
- bunun uzerinde Sobel tabanli ikinci turevlerle Hessian determinant ve eigenvalue-tabanli izotropik `blobness` haritalari cikartildi
- coklu olcek (`sigma = 1.4, 1.8, 2.2, 2.8`) max-pool ile birlestirildi

Ilk ham peak secimi sonucu:
- `det_multi`
  - count = `661`
  - matched = `167`
  - precision = `0.253`
  - recall = `0.831`
  - f1 = `0.387`
- `blob_multi`
  - count = `1135`
  - matched = `199`
  - precision = `0.175`
  - recall = `0.990`
  - f1 = `0.298`

Ara bulgu:
- Hessian/blobness haritasi neredeyse tum gercek noktalari goruyor
- fakat ham threshold ile asiri fazla yalanci tepe cikiyor
- yani bu harita tek basina son merkez listesi degil, yuksek recall'li bir "aday enerjisi" gibi davraniyor

Top-K kisma testi:
- ayni `blob_multi` haritasinda sadece en guclu `K` merkez tutulunca performans belirgin toparladi
- `K = 220` icin:
  - matched = `147`
  - precision = `0.668`
  - recall = `0.731`
  - f1 = `0.698`
- bu, simdiye kadar gorulen `hough_gray_inv` sonucuna (`f1 ~= 0.687`) cok yakin, hatta hafif daha iyi

Neden onemli:
- ilk kez kullanisli bir "enerji haritasi" cikti
- problem Hessian fikrinin yanlis olmasi degil
- problem, bu haritadan merkez secme asamasi
- ham quantile threshold kotu
- ama `top-K + NMS` mantigi umut veriyor

Sonuc:
- Hessian/blobness tamamen cop degil; aksine iyi bir recall kaynagi
- bu harita:
  - dogrudan decode icin yeterli degil
  - ama dis cerceve / grid fit icin kuvvetli aday alani uretebilir

Bir sonraki mantikli adim:
- `blob_multi_top220` merkezlerini kullanip
- sadece en dis satir/sutunlari RANSAC / projection ile cikarmak
- boylece Hessian'i tum alanda grid aramak yerine dis cerceve bulmak icin kullanmak

### Step 06 - 2D FFT band-pass filtering

Kaydedilen ciktilar:
- [summary.txt](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step06_fft_bandpass\summary.txt)
- [fft_p17p0_w0p014_amp.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step06_fft_bandpass\fft_p17p0_w0p014_amp.png)
- [fft_p17p0_w0p014_overlay.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step06_fft_bandpass\fft_p17p0_w0p014_overlay.png)
- [fft_p17p0_w0p014_mask.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step06_fft_bandpass\fft_p17p0_w0p014_mask.png)

Yapilan is:
- `LCN dark` goruntu uretildi
- goruntu 2D FFT uzayina alindi
- `~17 px` pitch'e karsilik gelen frekans (`f ~= 1/17`) etrafinda halka band-pass maskeleri denendi
- bazi varyantlarda `2*f` ikinci harmonik de eklendi
- ters FFT sonrasi olusan enerji haritasindan `top-K + NMS` ile merkezler secildi

Denenen varyantlar:
- `fft_p16p5_w0p010`
- `fft_p17p0_w0p010`
- `fft_p17p0_w0p014`
- `fft_p17p0_h2`
- `fft_p17p5_h2`

Genel sonuc:
- en iyi varyant `fft_p17p0_w0p014`
  - count = `220`
  - matched = `133`
  - precision = `0.605`
  - recall = `0.662`
  - f1 = `0.632`

Bu ne anlama geliyor:
- FFT band-pass tek basina en iyi genel merkez dedektoru olmadi
- genel skor olarak:
  - `Hessian blob_multi top220` (`f1 ~= 0.698`)
  - `Hough gray_inv` (`f1 ~= 0.687`)
  halen daha iyi

Ama kritik ek bulgu:
- sadece dis kenara yakin referans noktalar icin (`52` adet edge truth) FFT biraz daha iyi davrandi
- `fft_p17p0_w0p014`
  - edge matched = `37 / 52`
  - edge recall = `0.712`
- `Hessian blob_multi top220`
  - edge matched = `35 / 52`
  - edge recall = `0.673`

Yorum:
- FFT tum noktalar icin super bir cozum degil
- ama duzensiz metal dokusunu bastirip periyodik yapinin kenarlarini bir miktar belirginlestiriyor
- bu yuzden "tam merkez listesi" yerine "dis cerceve yardimci kanali" olarak anlamli olabilir

Sonuc:
- `FFT band-pass tek basina cozum degil`
- ama tamamen basarisiz da degil
- en guclu degeri:
  - dis kenar nokta ailelerini bulmada
  - Hessian / Hough ile birlikte yardimci kanal olarak kullanilmasi olabilir

Bir sonraki mantikli adim:
- `FFT best amp` + `Hessian blob_multi` + `Hough centers` birlikte kullanilarak
- yalnizca sol/sag/ust/alt kenar adaylari uzerinde bir edge-fit denemesi yapmak

### Step 07 - Birlesik edge-fit kirilmasi

Kaydedilen ciktilar:
- [combined_edge_overlay.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step07_combined_edge_fit\combined_edge_overlay.png)
- [summary.txt](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step07_combined_edge_fit\summary.txt)

Yapilan is:
- `Hough(gray_inv)` merkezleri
- `Hessian blob_multi top220`
- `FFT best amp top220`
  birlestirilip agirlikli aday nokta kumesi kuruldu
- yakin noktalar merge edildi
- kucuk aci/pitch araliginda birden fazla bazis denendi
- ama bu kez tum alana global fit yerine yalnizca 4 kenari guclu yapan cerceve skoru optimize edildi

Bulunan geometri:
- `origin = (91.955, 51.579)`
- `vx = (15.765, 1.047)` uzunluk `15.800`, aci `3.800 deg`
- `vy = (-1.394, 15.939)` uzunluk `16.000`, aci `95.000 deg`

Kritik sonuc:
- referans isaretlerin tumu bu kafese iyi oturuyor:
  - `truth_good = 201 / 201`
  - `truth_mean_err = 0.140`
  - `truth_min_i = 0`, `truth_max_i = 19`
  - `truth_min_j = 0`, `truth_max_j = 19`
- yani ilk kez geometri tarafi cozuldu
- dis cerceve artik 20x20 gercek frame ile uyumlu

Bu ne anlama geliyor:
- artik temel problem "grid nerede?" degil
- temel problem "bu sabit grid uzerindeki hangi hucre dolu?" sorusu
- bundan sonraki butun is occupancy / bit siniflandirma

Durum:
- bu adim basarili
- bundan sonra decode icin frame-fit degil, cell scoring ve bit reconstruction denenecek

### Step 08 - Frame uzerinden sentetik matris ve local bit duzeltme

Kaydedilen ciktilar:
- [frame_overlay.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step08_reconstruct_from_frame\frame_overlay.png)
- [vote_heatmap.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step08_reconstruct_from_frame\vote_heatmap.png)
- [combined_score_heatmap.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step08_reconstruct_from_frame\combined_score_heatmap.png)
- [q48_decoded_fix.png](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step08_reconstruct_from_frame\q48_decoded_fix.png)
- [summary.txt](d:\DATAGUESS\datamatrix_v2\artifacts\cropped_steps\step08_reconstruct_from_frame\summary.txt)

Yapilan is:
- Step 07'de bulunan dogru frame sabit tutuldu
- hucre dolulugu icin su kanallar birlestirildi:
  - point-vote
  - Hessian blobness
  - FFT band-pass
  - dark LCN
  - gray inverse
- bu kanallardan birlesik hucre skoru uretildi
- `q44..q56` arasi threshold adaylari denendi

Ilk sonuc:
- ham q-matrislerinin hicbiri dogrudan decode olmadi
- ama manuel isaretlerden kurulan "truth matrix" decode etti
- bu, geometri ve temel bit yerlestirmenin dogru oldugunu; yalnizca birkac hucrenin hatali kaldigini gosterdi

Kritik kirilma:
- `q48` adayinda en belirsiz hucreler arasinda kucuk bir local search denendi
- sadece `2` bit flip ile decode geldi:
  - flipler: `(16, 6)` ve `(7, 9)`
- sonuc payload:
  - `#8B3886177B ###=0302604331701S`

Sonuc:
- `cropped.png` icin calisan zincir bulundu:
  - bright frame fit
  - hucre skoru
  - local uncertain-cell search
  - zxing decode

Urunlestirme:
- ayni bright-background fallback mantigi `pipeline.py` icine eklendi
- smoke test sonucu:
  - `test4.png` -> dogru
  - `test5.png` -> dogru
  - `cropped.png` -> dogru

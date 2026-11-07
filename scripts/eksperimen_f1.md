# Eksperimen peningkatan F1 classifier EN

Tanggal: 27–28 Oktober 2026. Data: gold standard konsensus (919 kalimat). Skrip: `scripts/eksperimen_f1.py`. Hasil mentah: `results/eksperimen_f1/`.

## Ringkasan

- Beberapa cara yang tidak membutuhkan anotasi baru sudah dicoba: model linear, embedding beku, varian fine-tuning IndoBERT, ensemble, dan *domain-adaptive pretraining* (DAPT) pada korpus kurikulum.
- Model terbaik adalah IndoBERT yang di-DAPT. Pada validasi silang, macro-F1-nya **0,58**, naik 0,04 dari setelan notebook (0,54).
- Pada test set, kenaikan itu **tidak terbukti**. Macro-F1 test sekitar 0,38 untuk kedua model, dan interval kepercayaan 95%-nya selebar ±0,12.
- F1 belum mencapai tingkat yang memuaskan (≥ 0,70). Batasnya ada pada data: label yang awalnya disepakati rendah (κ putaran 1 = 0,24) dan contoh positif yang sedikit (14–107 per konstruk). Penyetelan model tidak bisa menembus batas ini.
- Model di notebook **tidak diganti**. Kenaikan yang belum terbukti di test tidak sebanding dengan tambahan langkah DAPT (±100 menit di GTX 1650 Ti).

## Protokol

Supaya skor test tidak bias, protokol berikut dipakai sejak awal:

1. **Pembagian data sama persis dengan notebook §7** (SEED 42).
   - Data dev = train + val = 781 kalimat.
   - Test = 138 kalimat. Jumlah positif per konstruk di test: 5, 17, 12, 9, 2, 5, 7, 2.
2. **Semua pilihan dibuat dengan validasi silang 5 lipatan pada dev.** Pilihan yang dimaksud: fitur, model, hiperparameter, jumlah epoch, dan threshold.
   - Threshold dinilai secara *cross-fitted*: lipatan k memakai threshold yang di-tuning pada prediksi lipatan lain.
3. **Kandidat akhir dipilih sebelum test dibuka.** Yang dipilih adalah CV tertinggi, yaitu IndoBERT + DAPT, epoch 8.
   - Test dibuka sekali untuk kandidat ini.
   - Test juga dibuka sekali untuk setelan notebook sebagai pembanding. Pembanding ini bukan bagian dari pemilihan.
4. **DAPT tidak memakai kalimat test.** DAPT dilatih dengan MLM pada 12.486 kalimat korpus tanpa label, dan kalimat test dikeluarkan.

## Hasil validasi silang (dev, 781 kalimat)

| Model | Macro-F1 | EN01 | EN02 | EN03 | EN04 | EN05 | EN06 | EN07 | EN08 |
|---|---|---|---|---|---|---|---|---|---|
| **IndoBERT + DAPT, epoch 8** | **0,581** | 0,68 | 0,57 | 0,61 | 0,67 | 0,50 | 0,38 | 0,75 | 0,48 |
| IndoBERT + DAPT + focal loss (γ = 2), epoch 7 | 0,561 | 0,50 | 0,58 | 0,64 | 0,69 | 0,40 | 0,41 | 0,66 | 0,62 |
| Ensemble semua model | 0,571 | 0,63 | 0,55 | 0,64 | 0,68 | 0,50 | 0,39 | 0,74 | 0,43 |
| IndoBERT, lr 3e-5, pos_weight ≤ 3 | 0,556 | 0,55 | 0,54 | 0,67 | 0,79 | 0,38 | 0,34 | 0,74 | 0,44 |
| IndoBERT, mean pooling, pos_weight ≤ 3 | 0,545 | 0,67 | 0,56 | 0,67 | 0,71 | 0,42 | 0,32 | 0,68 | 0,33 |
| IndoBERT, setelan notebook | 0,543 | 0,43 | 0,55 | 0,59 | 0,68 | 0,40 | 0,37 | 0,72 | 0,61 |
| IndoBERT, pos_weight ≤ 3 | 0,527 | 0,38 | 0,57 | 0,55 | 0,69 | 0,48 | 0,35 | 0,68 | 0,52 |
| TF-IDF kata + karakter, regresi logistik | 0,526 | 0,47 | 0,47 | 0,53 | 0,75 | 0,40 | 0,40 | 0,70 | 0,49 |
| Embedding IndoBERT beku, regresi logistik | 0,498 | 0,32 | 0,59 | 0,63 | 0,70 | 0,21 | 0,29 | 0,72 | 0,52 |
| Embedding multilingual-e5-base beku, regresi logistik | 0,481 | 0,33 | 0,46 | 0,54 | 0,69 | 0,42 | 0,24 | 0,63 | 0,54 |

Catatan:

- **Model sederhana tidak jauh tertinggal.** TF-IDF + regresi logistik (0,53) hampir setara dengan IndoBERT setelan notebook (0,54). Dengan data sebanyak ini, IndoBERT belum memberi keunggulan yang berarti.
- **Di antara semua perubahan, DAPT memberi kenaikan CV terbesar.** Kenaikan terbesarnya ada di EN01 (0,43 → 0,68). Kenaikan EN01 ini tidak terlihat di test (lihat di bawah).
- **Focal loss tidak membantu.** Macro-F1 CV turun dari 0,58 ke 0,56. Kenaikan di EN08 diimbangi penurunan di EN01, EN05, dan EN07. Karena kalah di CV, model ini tidak dievaluasi di test.
- **Class weight sudah dipakai sejak awal.** Bentuknya `pos_weight` pada `BCEWithLogitsLoss`, padanan multi-label dari `CrossEntropyLoss(weight=...)`, sebab satu kalimat bisa memuat beberapa konstruk. Batas maksimumnya diuji (10 vs 3), dan 10 sedikit lebih baik.
- **Konstruk tersulit adalah EN06 Motivation/Agency** (0,32–0,40 di semua model), sejalan dengan κ-nya yang paling rendah (0,03).

## Hasil test (138 kalimat, dibuka sekali)

| Model (dilatih pada 781 kalimat dev) | Macro-F1, 8 konstruk | 95% CI (bootstrap) | Macro-F1, 6 konstruk dengan ≥ 5 positif |
|---|---|---|---|
| IndoBERT + DAPT, epoch 8 (dipilih) | 0,378 | 0,24–0,48 | 0,438 |
| IndoBERT, setelan notebook (pembanding) | 0,380 | 0,21–0,47 | 0,396 |
| Notebook saat ini (dilatih pada 643 kalimat train) | 0,387 | – | 0,405 |

- Selisih DAPT − pembanding pada 6 konstruk: 95% CI −0,06 sampai +0,14. Peluang DAPT lebih baik menurut bootstrap 81%, sehingga belum bisa disebut lebih baik.
- EN05 dan EN08 hanya punya 2 positif di test. F1 keduanya di test (0 sampai 0,67) tidak bermakna.

**Kenapa test (±0,38) jauh di bawah CV (±0,55)?** Pola yang sama juga terlihat di notebook (val 0,55, test 0,39). Ada dua dugaan:

- 138 kalimat terlalu sedikit, sehingga skor test sangat dipengaruhi kebetulan.
- Kalimat test kebetulan lebih sulit.

CV memprediksi setiap kalimat dev tepat satu kali (781 kalimat), sehingga estimasinya jauh lebih stabil. Kedua angka sebaiknya dilaporkan.

## Kurva belajar

Pertanyaannya: apakah menambah anotasi akan menaikkan F1? Untuk menjawabnya, kedua model dilatih ulang dengan 25, 50, 75, dan 100% bagian latih tiap lipatan (subset bersarang), lalu dinilai dengan CV yang sama. Test tidak dipakai. Grafik: `results/eksperimen_f1/kurva_belajar.png`.

| Kalimat latih per lipatan | 156 | 312 | 469 | 625 |
|---|---|---|---|---|
| IndoBERT + DAPT, macro-F1 | 0,39 | 0,44 | 0,51 | 0,58 |
| TF-IDF + regresi logistik, macro-F1 (rata-rata 5 ulangan) | 0,31 | 0,37 | 0,45 | 0,53 |

- **Kurva belum mendatar.** Kenaikannya sekitar 0,10 setiap kali data dilipatgandakan. Menggandakan data (±800 kalimat beranotasi baru) diperkirakan membawa macro-F1 CV ke sekitar 0,65–0,68. Ini ekstrapolasi kasar, bukan jaminan.
- **EN01, EN05, EN07, EN08** masih naik tajam, sehingga paling diuntungkan oleh tambahan contoh positif. Titik EN05 dan EN08 sangat bergoyang karena hanya ±10 positif di data latih.
- **EN02, EN04** hampir mendatar. Menambah data saja tidak banyak membantu; konsistensi label lebih penting.
- **EN06** datar di sekitar 0,4 untuk IndoBERT. Ini sejalan dengan κ 0,03: masalahnya pada definisi, bukan jumlah contoh.

## Skenario: EN05 dan EN06 digabung

**Aturan.** Konstruk gabungan "Afektif-Motivasional" bernilai 1 jika kalimat berlabel EN05 atau EN06. Aturan ini ditetapkan sebelum hasil dilihat. Model IndoBERT + DAPT dilatih ulang untuk 7 label. Epoch dipilih dengan CV (epoch 10), lalu test dibuka sekali (`scripts/eksperimen_f1.py gabung`, hasil di `results/eksperimen_f1/final_gabung.json`).

| | κ putaran 1 | Positif gold | F1 CV | F1 test (positif test) |
|---|---|---|---|---|
| EN05 Emotion/Affect | 0,29 | 14 | 0,50 | 0,40 (2) |
| EN06 Motivation/Agency | 0,03 | 33 | 0,38 | 0,29 (5) |
| **EN05+EN06 digabung** | **0,22** | **44** | **0,54** | **0,31 (7)** |

| Macro-F1 | 8 konstruk | 7 konstruk (digabung) |
|---|---|---|
| CV dev | 0,581 | 0,596 |
| Test | 0,378 | 0,423 |

- **Konstruk gabungan lebih mudah dipelajari** daripada rata-rata EN05 dan EN06 (CV 0,54 vs 0,44). Meski begitu, κ-nya tetap rendah (0,22), sehingga labelnya masih kurang konsisten.
- **Kenaikan macro-F1 sebagian besar mekanis**, karena dua konstruk lemah menjadi satu. Kenaikan di test (0,38 → 0,42) masih jauh di dalam interval kepercayaan (±0,12).
- **Ada variasi antar-pelatihan.** Konstruk yang tidak diubah ikut bergeser ketika model dilatih ulang, misalnya EN01 CV 0,68 → 0,59. Selisih F1 per konstruk di bawah ±0,1 antar-skenario belum bisa dianggap nyata.
- **Penggabungan mengubah kerangka teori** (8 → 7 konstruk), sehingga harus diputuskan dan dijustifikasi oleh peneliti.

## Rekomendasi

1. **Laporkan F1 apa adanya**, lengkap dengan interval kepercayaan, dan sertakan CV 5 lipatan di samping test. Jangan memilih seed, threshold, atau kandidat berdasarkan test.
2. **Jalur yang paling mungkin menaikkan F1 tetap data** (lihat kurva belajar), dengan urutan berikut:
   - Perjelas codebook, terutama EN06, lalu uji κ di 100 kalimat baru.
   - Lakukan sampling terarah untuk EN01, EN05, EN07, dan EN08. Kandidat dipilih dengan kata kunci atau probabilitas model, dicampur dengan kalimat acak, dan dianotasi tanpa skor model. Kalimat ini hanya masuk data latih, sedangkan test tetap sampel acak.
   - Untuk EN02 dan EN04, cukup pastikan konsistensinya.
3. **Jika model DAPT ingin dipakai,** checkpoint-nya ada di `results/eksperimen_f1/dapt/` (±500 MB). Konfigurasinya `dapt` di skrip, epoch 8, dengan threshold di `results/eksperimen_f1/final_ft_dapt_e8.json`.

## Cara mengulang

```bash
.venv/bin/python scripts/eksperimen_f1.py linear          # TF-IDF dan embedding beku (CPU/GPU, ±15 menit)
.venv/bin/python scripts/eksperimen_f1.py finetune         # 4 konfigurasi × 5 lipatan (±23 menit per konfigurasi)
.venv/bin/python scripts/eksperimen_f1.py dapt             # ±100 menit
.venv/bin/python scripts/eksperimen_f1.py finetune dapt
.venv/bin/python scripts/eksperimen_f1.py ringkas
.venv/bin/python scripts/eksperimen_f1.py final ft_dapt@8  # sekali, membuka test set
```

Jalankan `import torch` sebelum sklearn. Urutan sebaliknya membuat proses *segfault* saat inferensi. Skrip sudah mengatur urutan ini.

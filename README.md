# Curriculum Audit: Analisis Konstruk Tingkat Kalimat pada Dokumen Kurikulum

Pipeline yang mengubah dokumen kurikulum berformat PDF menjadi label konstruk per kalimat, lalu merangkumnya per dokumen, fase, dan lapis kurikulum. Model klasifikasinya adalah IndoBERT multi-label. Setiap kalimat bisa memuat lebih dari satu konstruk.

Data pengembangannya adalah dokumen Biologi SMA Kurikulum Merdeka (Fase E dan F) dengan delapan konstruk Educational Neuroscience (EN01–EN08). Bagian yang bergantung pada mata pelajaran dan konstruk disimpan sebagai struktur konfigurasi di notebook, jadi pipeline yang sama bisa dipakai untuk mata pelajaran atau himpunan konstruk lain (lihat [Mengganti mata pelajaran atau konstruk](#mengganti-mata-pelajaran-atau-konstruk)).

## Isi repo

| Path | Isi |
|---|---|
| `Audit_Kurikulum_EN_CA_Pipeline_ID.ipynb` | Notebook utama (bahasa Indonesia) |
| `Audit_Kurikulum_EN_CA_Pipeline_EN.ipynb` | Notebook yang sama dalam bahasa Inggris |
| `scripts/eksperimen_f1.py` | Validasi silang, DAPT, baseline, kurva belajar, uji akhir |
| `scripts/eksperimen_f1.md` | Protokol dan hasil eksperimen F1 |
| `scripts/jalankan_notebook.py` | Menjalankan notebook dari awal sampai akhir tanpa editor |
| `data_pdf/` | Dokumen sumber (CP, modul ajar, buku siswa, panduan) |
| `corpus/` | Korpus kalimat dengan identitas tetap (`master_corpus.json`, `appended_docs.json`) |
| `annotation/` | Berkas anotasi annotator A dan B, serta `gold_standard.json` |
| `model/model_config.json` | Hiperparameter dan ambang per konstruk dari model terlatih |
| `results/` | Keluaran notebook dan eksperimen |

## Persiapan

Diperlukan Python 3.12 dan, untuk PDF tanpa lapisan teks, Tesseract OCR dengan data bahasa Indonesia.

```bash
sudo apt install tesseract-ocr tesseract-ocr-ind
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Model `indobenchmark/indobert-base-p2` diunduh otomatis dari Hugging Face saat pertama kali dipakai.

GPU CUDA dipakai bila tersedia. Pengaturan bawaan dirancang untuk GPU 4 GB (diuji pada GTX 1650 Ti): panjang maksimum 128 token, batch 16, FP32. Dengan 512 token, memori GPU 4 GB tidak cukup. Mixed precision (AMP) justru lebih lambat pada GPU tanpa tensor core.

## Menjalankan

Buka notebook dan jalankan semua cell secara berurutan, atau jalankan tanpa editor:

```bash
.venv/bin/python scripts/jalankan_notebook.py Audit_Kurikulum_EN_CA_Pipeline_ID.ipynb
```

Skrip ini berhenti di cell pertama yang error, dan output cell yang sudah jalan tetap disimpan ke notebook.

**Bobot model tidak ada di repo.** `model/best_model.pt` (±500 MB) melebihi batas ukuran file GitHub. Bobot ini dibuat oleh §7 notebook dalam sekitar 3 menit di GPU 4 GB. Karena itu, jalankan §7 sebelum §8–§13.

### Alur notebook

| § | Langkah | Keluaran |
|---|---|---|
| 1 | Daftar dokumen dan metadata | – |
| 2 | Penyusunan korpus (PyMuPDF, OCR untuk PDF tanpa teks, segmentasi Punkt) | `corpus/*.json` |
| 3 | Codebook EN01–EN08 | – |
| 4 | Penarikan sampel anotasi (diperiksa sama dengan berkas anotasi) | – |
| 5 | Reliabilitas antar-annotator (Cohen's κ) | – |
| 6 | Gold standard (aturan konservatif: 1 hanya jika kedua annotator 1) | `annotation/gold_standard.json` |
| 7 | Pelatihan IndoBERT multi-label dan penyetelan ambang per konstruk | `model/best_model.pt`, `model/model_config.json`, `results/training_report.json` |
| 7b | Membaca hasil validasi silang dan kurva belajar | – (dari `results/eksperimen_f1/`) |
| 8 | Pelabelan seluruh korpus | `results/corpus_en_predictions.json` |
| 9 | Ekstraksi konsep dan level Bloom | `results/concept_occurrences.json` |
| 10 | Ciri arsitektur kurikulum CA01–CA04 | `results/ca_analysis.json` |
| 11 | Integrasi konstruk × CA | `results/integration_8EN_4CA.json` |
| 12 | Keselarasan antarlapis kurikulum | `results/alignment_CP_module_textbook.json` |
| 13 | Pelabelan dokumen baru tanpa pelatihan ulang | `results/prediksi_EN_dokumen_disisihkan.xlsx` |

**Korpus tidak disusun ulang secara bawaan.** Anotasi merujuk ke identitas kalimat (`<document_id>_S<nomor>`), jadi `corpus/master_corpus.json` hanya dibaca. `REBUILD_CORPUS = True` di §2 mengekstraksi ulang korpus, tetapi identitas kalimat akan berubah dan anotasi tidak cocok lagi. PDF buku siswa kelas XII (124 MB) tidak ikut di repo karena alasan ukuran. Kalimatnya sudah ada di korpus, dan PDF ini hanya dibutuhkan untuk penyusunan ulang.

### Melabeli dokumen baru

Tambahkan dokumen ke daftar `DOCUMENTS` di §1 dengan `"split": "predict"`, letakkan PDF-nya di `data_pdf/`, lalu jalankan §13. Hasilnya berupa berkas Excel berisi satu baris per kalimat (label dan probabilitas) dan satu lembar ringkasan per dokumen.

## Eksperimen F1

Validasi silang, DAPT, dan kurva belajar dijalankan terpisah dari notebook (total ±6 jam GPU). Notebook §7b hanya membaca hasilnya.

```bash
.venv/bin/python scripts/eksperimen_f1.py linear          # TF-IDF dan embedding beku + regresi logistik (±15 menit)
.venv/bin/python scripts/eksperimen_f1.py finetune base pw3 mean_pw3 lr3_pw3  # fine-tuning × 5 lipatan (±20 menit per konfigurasi)
.venv/bin/python scripts/eksperimen_f1.py dapt             # domain-adaptive pretraining (±100 menit)
.venv/bin/python scripts/eksperimen_f1.py finetune dapt dapt_focal  # fine-tuning dari checkpoint DAPT
.venv/bin/python scripts/eksperimen_f1.py ringkas          # tabel semua kandidat (CV)
.venv/bin/python scripts/eksperimen_f1.py kurva            # kurva belajar (tanpa test)
.venv/bin/python scripts/eksperimen_f1.py gambar_kurva     # grafik kurva belajar
.venv/bin/python scripts/eksperimen_f1.py gabung           # skenario EN05 + EN06 digabung
.venv/bin/python scripts/eksperimen_f1.py final ft_dapt@8  # latih ulang pada dev, buka test sekali
```

Checkpoint DAPT (`results/eksperimen_f1/dapt/`, ±500 MB) juga tidak ada di repo. Checkpoint ini dibuat ulang oleh perintah `dapt`, dan harus ada sebelum konfigurasi `dapt` dan `dapt_focal` dijalankan. `finetune` tanpa argumen menjalankan semua konfigurasi, termasuk kedua konfigurasi DAPT itu. Protokol lengkap dan hasilnya ada di [`scripts/eksperimen_f1.md`](scripts/eksperimen_f1.md).

## Ringkasan hasil

| Ukuran | Nilai |
|---|---|
| Korpus | 13.146 kalimat dari 12 dokumen (+5 modul untuk uji dokumen baru) |
| Kalimat beranotasi ganda | 919 |
| Cohen's κ, rata-rata (berkas label akhir) | 0,863 (terendah 0,774) |
| Macro-F1 data uji (138 kalimat) | 0,387 |
| Macro-F1 validasi silang 5 lipatan | 0,542 (IndoBERT), 0,581 (IndoBERT + DAPT) |
| Kecepatan pelabelan | ±149 kalimat/detik (GTX 1650 Ti) |

κ dihitung dari berkas label akhir, yaitu setelah kedua annotator membahas perbedaan label di akhir setiap putaran. Data uji hanya 138 kalimat, sehingga skor ujinya sangat tidak pasti (selang kepercayaan 95% ±0,12). Kurva belajar belum mendatar, jadi menambah contoh positif berlabel adalah cara yang paling mungkin menaikkan F1. Persentase konstruk di §8–§13 adalah perkiraan model, bukan hitungan manusia.

## Mengganti mata pelajaran atau konstruk

Semua pengetahuan khusus mata pelajaran dan konstruk ada di struktur berikut:

| Struktur | Letak | Isi |
|---|---|---|
| `DOCUMENTS` | §1 | Daftar PDF beserta lapis, fase, kelas, dan status latih/prediksi |
| `NOISE` | §2 | Frasa berulang yang dibuang saat ekstraksi |
| `EN_COLS`, `EN_NAMES` | §0 | Daftar konstruk; jumlah keluaran model mengikuti panjangnya |
| `GUIDELINE` | §3 | Codebook: definisi, kriteria, aturan keputusan, contoh |
| `BIOLOGY_CONCEPTS`, `BLOOM_VERBS` | §9 | Leksikon konsep dan akar kata kerja Bloom |
| `PREREQ` | §10 | Pasangan konsep prasyarat |
| `MODEL_NAME`, `HP` | §7 | Encoder dan hiperparameter |

Mata pelajaran atau himpunan konstruk baru juga membutuhkan anotasi baru dan pelatihan ulang. Model yang ada hanya dilatih pada kalimat Biologi, dan kinerjanya pada mata pelajaran lain belum diukur.

Dua bagian masih terikat pada data pengembangan:

- Perbandingan antarfase di §10–§11 ditulis untuk pasangan fase E dan F.
- Analisis integrasi di §11 menyoroti EN03 dan EN04 secara khusus.

## Catatan

- `scripts/eksperimen_f1.py` mengimpor `torch` sebelum scikit-learn. Urutan sebaliknya membuat proses segfault saat inferensi.
- Berkas JSON ditulis lewat berkas sementara lalu diganti namanya, sehingga proses yang terputus tidak meninggalkan berkas terpotong.

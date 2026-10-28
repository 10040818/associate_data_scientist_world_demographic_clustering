# 🌐 World Demographic Clustering Intelligence
### Segmentasi Demografi 234 Negara & Wilayah Berbasis Unsupervised Machine Learning (K-Means)
> **Skema Sertifikasi Okupasi:** Associate Data Scientist (KKNI Level 5)  
> **Standar Acuan:** SKKNI No. 299 Tahun 2020 (Lengkap 9 Unit Kompetensi)  
> **Nomor Skema Asesmen:** `SSK22/BPPMPVKPTK/KKRPL/2022` — LSP BPPMPV KPTK / Kementerian Komunikasi dan Digital RI (KOMDIGI)  
> **Peneliti Utama & Pengembang Model:** **M. Iqbal Adenan**  
> **Live Web Dashboard:** [https://10040818.github.io/ukk/](https://10040818.github.io/ukk/)

---

[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-success?style=for-the-badge&logo=github)](https://10040818.github.io/ukk/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-v4.0-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Standard](https://img.shields.io/badge/Standard-SKKNI%20299--2020-indigo?style=for-the-badge)](https://komdigi.go.id/)

---

## 📌 Ringkasan Eksekutif & Latar Belakang

Penelitian ini mengimplementasikan algoritma **K-Means Clustering** untuk mengelompokkan 234 negara dan wilayah di dunia ke dalam **4 kuadran homogen** berdasarkan 4 pilar demografi objektif:
1. **Ukuran Populasi (Population 2022)**
2. **Luas Wilayah Teritorial (Area km²)**
3. **Kepadatan Penduduk (Density per km²)**
4. **Dinamika Pertumbuhan (Growth Rate)**

Seluruh tahapan proyek dirancang terstruktur dan terstandarisasi untuk memenuhi **9 Unit Kompetensi SKKNI No. 299 Tahun 2020** untuk Skema Okupasi **Associate Data Scientist**.

### 🎯 Key Performance Indicators (KPI) Penelitian:
* **Total Entitas Sensus:** 234 Negara & Teritori (100% data riil warga bumi, 0 missing value).
* **Representasi Populasi:** 7,97 Miliar Jiwa terakomodasi utuh tanpa sampling sintetis.
* **Reduksi Skewness:** Berhasil memangkas kemencengan data ekstrem sebesar **93%** (dari $+9{,}15$ ke $-0{,}60$) melalui transformasi logaritma natural `np.log1p` dengan prinsip **Zero Data Loss Policy** (tidak membuang China, India, atau Rusia).
* **Klaster Optimal ($K=4$):** Berdasarkan penurunan inersia WCSS Metode Elbow ($>60\%$) dan validasi **Silhouette Coefficient** tertinggi ($0{,}3138$).
* **Separasi Klaster PCA 2D:** Mampu mempertahankan **74,32%** varians kumulatif data mentah.

---

## 🏛️ Pemenuhan 9 Unit Kompetensi SKKNI No. 299 Tahun 2020

| No | Kode Unit | Judul Unit Kompetensi | Implementasi & Deliverable di Proyek | Cell Notebook |
| :-: | :--- | :--- | :--- | :-: |
| **1** | `J.62DMI00.004.1` | **Mengumpulkan Data** | Akuisisi 234 data negara dari sensus PBB/Kaggle, validasi skema 17 atribut CSV, inspeksi tipe data tabular. | Cell 1–3 |
| **2** | `J.62DMI00.005.1` | **Menelaah Data** | Eksplorasi dimensi dataset ($234 \times 17$), 8 dimensi deret waktu historis (1970–2022), ringkasan statistik deskriptif 5-angka (*Mean, Std, Min, Median, Max*), deteksi outlier via Boxplot. | Cell 4–10 |
| **3** | `J.62DMI00.006.1` | **Memvalidasi Data** | Audit kualitas data: $0$ missing values (100% lengkap), $0$ baris duplikasi, validasi konsistensi batas batas logis non-negatif. | Cell 11–16 |
| **4** | `J.62DMI00.007.1` | **Menentukan Objek Data** | Seleksi 4 fitur pilar demografi independen; eliminasi multikolinearitas data historis ($r > 0{,}99$) dan atribut identitas administratif. | Cell 17–20 |
| **5** | `J.62DMI00.008.1` | **Membersihkan Data** | Penerapan *Zero Data Loss Policy*: Outlier ekstrem tidak dihapus; ditransformasi via `np.log1p`, sukses memangkas kemencengan (skewness) 93%. | Cell 21–25 |
| **6** | `J.62DMI00.009.1` | **Mengkonstruksi Data** | Standarisasi multivariat menggunakan `StandardScaler` (Z-score mean=0, std=1) guna menyetarakan timbangan jarak Euclidean antar satuan dimensi. | Cell 26–27 |
| **7** | `J.62DMI00.010.1` | **Menentukan Label Data** | Pelabelan klaster 0–3 secara objektif; validasi batas pemisah klaster melalui Reduksi Dimensi PCA 2D (varians kumulatif 74,32%). | Cell 28–37 |
| **8** | `J.62DMI00.013.1` | **Membangun Model** | Pemodelan K-Means (`k-means++`, $n\_\text{init}=20$, $K=4$) dengan optimasi Metode Elbow (drop inersia $>60\%$) serta ekspor model `model_kmeans.pkl` & `scaler.pkl`. | Cell 38–45 |
| **9** | `J.62DMI00.014.1` | **Mengevaluasi Hasil Pemodelan** | Evaluasi skor Silhouette global ($0{,}3138$) dan profil per klaster; perumusan taksonomi 4 profil sosio-ekonomi global dan rekomendasi kebijakan PBB. | Cell 46–64 |

---

## 🎯 Mengapa Memilih K = 4 (Bukan K = 3, K = 5, atau K = 6)?

Penetapan nilai **$K = 4$** didasarkan pada metodologi validasi ilmiah ganda (*Dual-Metric Validation*) yang memadukan **Metode Siku (*Elbow Method*)**, **Evaluasi Kohesi & Separasi (*Silhouette Score*)**, serta **Kesehatan Distribusi Klaster**:

### 📊 Tabel Evaluasi Komparatif Kandidat $K$:

| Kandidat $K$ | Inersia WCSS | % Drop Inersia | Silhouette Score | Distribusi Anggota Klaster | Status & Alasan Keputusan |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$K = 2$** | $552{,}58$ | $-40{,}96\%$ | $0{,}4110$ | $166 	ext{ vs } 68$ negara | ❌ **Ditolak (*Underfitting*):** Terlalu dangkal/biner (hanya besar vs kecil), menghilangkan fenomena negara-kota mikro padat vs negara luas renggang. |
| **$K = 3$** | $440{,}76$ | $-20{,}24\%$ | $0{,}3010$ | $86, 84, 64$ negara | ⚠️ **Ditolak (*Sub-optimal*):** Skor siluet ($0{,}3010$) lebih rendah dibanding $K=4$. Negara berkembang tumbuh pesat masih bercampur baur dengan negara berpopulasi mapan. |
| **$K = 4$** | **$369{,}28$** | **$-16{,}22\%$** | **$0{,}3138$** | **$71, 64, 54, 45$ negara** |  **PILIHAN TERBAIK (OPTIMAL):** Titik siku tegas (penurunan varians $>60\%$), skor siluet tertinggi di antara kandidat siku, distribusi sangat seimbang tanpa singleton, merefleksikan 4 profil sosio-demografi nyata. |
| **$K = 5$** | $323{,}30$ | $-12{,}45\%$ | $0{,}3185$ | $75, 64, 49, 45, \mathbf{1}$ negara | ❌ **Ditolak (*Over-segmentation*):** Terbentuk klaster cacat kerdil (*singleton*) yang hanya berisi **1 negara saja** (overfitting data). |
| **$K = 6$** | $290{,}27$ | $-10{,}22\%$ | **$0{,}2743$** | Terlalu terpecah | ❌ **Ditolak (*Degradasi Struktur*):** Skor siluet anjlok di bawah batas layak $0{,}30$. Laju inersia jenuh (*diminishing returns*). Terjadi tumpang tindih (*overlapping*). |

### 💡 4 Argumen Inti Pengambilan Keputusan:
1. **Titik Belok Siku (*Elbow Point*):** Varians inersia terpangkas masif $>60{,}5\%$ dari $936{,}00$ ke $369{,}28$. Penurunan dari $K=3$ ke $K=4$ ($-16{,}22\%$) adalah penurunan signifikan terakhir sebelum kurva melandai di bawah $12\%$.
2. **Kekuatan Separasi Siluet:** Skor Silhouette $K=4$ ($0{,}3138$) mengungguli $K=3$ ($0{,}3010$) dan jauh melampaui $K=6$ ($0{,}2743$).
3. **Mencegah Cacat Klaster Singleton:** Pada $K=5$, terbentuk klaster kerdil beranggotakan 1 negara saja. Pada $K=4$, seluruh 234 negara terdistribusi sehat (71, 64, 54, 45 negara).
4. **Interpretasi Bisnis & Geografi Riil:** $K=4$ memetakan 4 kuadran arketipe demografi dunia: (0) Tumbuh Pesat, (1) Kota Mikro Padat, (2) Megapopulasi Mapan, dan (3) Teritorial Luas Renggang.

---

## 📊 Taksonomi 4 Profil Demografi Global

Berikut adalah sintesis karakteristik 4 kuadran profil demografi dunia hasil pemodelan K-Means:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      TAKSONOMI 4 PROFIL DEMOGRAFI GLOBAL                         │
├───────────┬──────────────┬──────────────┬──────────────────┬─────────────────────┤
│  Klaster  │ Total Negara │ % Penduduk   │ Karakteristik    │ Contoh Negara       │
│           │  & Proporsi  │  Bumi 2022   │ Utama            │ Representatif       │
├───────────┼──────────────┼──────────────┼──────────────────┼─────────────────────┤
│ Klaster 0 │  54 Negara   │    24,07%    │ Pertumbuhan      │ Pakistan, Nigeria,  │
│           │   (23,08%)   │ (1,92 Miliar)│ Pesat (+2,50%)   │ Mesir, Ethiopia     │
├───────────┼──────────────┼──────────────┼──────────────────┼─────────────────────┤
│ Klaster 1 │  64 Negara   │    0,35%     │ Kepadatan Tinggi │ Singapura, Monako,  │
│           │   (27,35%)   │ (27,9 Juta)  │ (1.338 jiwa/km²) │ Hong Kong, Bahrain  │
├───────────┼──────────────┼──────────────┼──────────────────┼─────────────────────┤
│ Klaster 2 │  71 Negara   │    68,70%    │ Megapopulasi     │ China, India, AS,   │
│           │   (30,34%)   │ (5,47 Miliar)│ Mapan (Rata 77M) │ Indonesia, Brasil   │
├───────────┼──────────────┼──────────────┼──────────────────┼─────────────────────┤
│ Klaster 3 │  45 Negara   │    6,89%     │ Teritorial Masif │ Rusia, Kanada,      │
│           │   (19,23%)   │  (549 Juta)  │ (Rata 1,39M km²) │ Australia, Aljazair │
└───────────┴──────────────┴──────────────┴──────────────────┴─────────────────────┘
```

### Rekomendasi Kebijakan Sosio-Ekonomi Berbasis Bukti (PBB / SDG):
* **Klaster 0:** Prioritas fasilitas kesehatan ibu-anak, pemenuhan gizi generasi muda (*anti-stunting*), ketahanan pangan, dan penciptaan lapangan kerja usia produktif.
* **Klaster 1:** Rekayasa tata ruang vertikal (*smart compact city*), penguatan transit massal nol-emisi, dan mitigasi kenaikan muka air laut global.
* **Klaster 2:** Reformasi jaminan hari tua (*aging population management*), transisi energi bersih terbarukan, dan peningkatan otomasi industri manufaktur.
* **Klaster 3:** Penggelaran infrastruktur satelit telekomunikasi daerah 3T, jalur logistik kereta jarak jauh, dan program konservasi lahan serta cadangan air.

---

## ⚡ Fitur Aplikasi Web Dashboard (`index.html`)

Aplikasi web dirancang mandiri (*Zero Server Dependency Single Page Application*) dan dapat dijalankan langsung di peramban atau di-hosting via GitHub Pages:
1. **Navigasi Responsif & Tailwind CSS v4:** Interface gelap modern dengan ambient radial gradient, tipografi Outfit & Plus Jakarta Sans.
2. **Matriks Alur 9 Unit SKKNI:** Visualisasi sekuensial tahapan kerja data science berstandar nasional.
3. **Taksonomi 4 Klaster Bebas Wrap Text:** Kartu profil demografi dengan indikator status berdenyut dan layout kartu dinamis.
4. **Visualisasi Interaktif (Chart.js):** 4 grafik visualisasi (Donut Klaster, Komposisi Benua, Radar Pilar Z-Score, dan Scatter Plot Populasi vs Kepadatan).
5. **Simulator Klasterisasi Real-Time:** 
   * Uji inferensi mandiri data negara baru berdasarkan log1p, scaler Z-score, dan perhitungan jarak Euclidean terdekat ke Centroid.
   * Dilengkapi fitur **Autocomplete Live Search 234 negara** dan tombol preset instan.
6. **Eksplorasi Dataset 234 Negara:** Tabel interaktif dengan pencarian instan, filter benua, filter klaster, sorting kolom (*Ascending/Descending*), dan navigasi pagination.
7. **Galeri 15 Visualisasi HD 300 DPI:** Lightbox modal preview untuk seluruh artefak grafik visualisasi hasil riset.
8. **Drawer Panduan Asesmen (32 Q&A):** Drawer panduan tanya jawab asesmen sidang SKKNI yang dapat dibuka dengan mengklik tombol **"9 Unit Kompeten"** di navbar.

---

## 📂 Struktur Direktori Repositori

```text
ukk/
├── index.html                   # Dashboard Web Interaktif (Tailwind CSS v4 + Chart.js)
├── analisis_kmeans.ipynb        # Jupyter Notebook Pemodelan ML Lengkap 9 Unit SKKNI
├── task.md                      # Checklist & Matriks Pemenuhan 9 Unit SKKNI
├── dokumentasi_tahapan.md       # Laporan Teknis Dokumentasi Lengkap
├── rangkuman.md                 # Bank Soal & Pembahasan Sidang Asesmen (32 Q&A)
├── .gitignore                   # Konfigurasi Git Ignore
│
├── dataset/                     # Direktori Sumber Data
│   ├── world_population.csv     # Dataset mentah 234 negara kependudukan dunia
│   └── social_media_impact...   # Dataset pendukung
│
├── output/                      # Direktori Artefak Pemodelan & Visualisasi
│   ├── model_kmeans.pkl         # Serialisasi Model K-Means (Scikit-Learn Joblib)
│   ├── scaler.pkl               # Serialisasi StandardScaler Parameter Pelatihan
│   ├── world_population_clustered.csv # Dataset final dengan label Klaster 0-3
│   ├── eda_distribusi_sebelum_transformasi.png
│   ├── eda_pairplot_fitur_terpilih.png
│   ├── eda_scatter_korelasi.png
│   ├── eda_standardized_features.png
│   ├── eda_top10_fitur_terpilih.png
│   ├── model_centroid_heatmap.png
│   ├── model_centroid_scatterplot.png
│   ├── model_distribusi_klaster.png
│   ├── model_elbow_method.png
│   ├── model_evaluasi_k_optimal.png
│   ├── model_komposisi_benua_klaster.png
│   ├── model_perbandingan_perubahan_klastering.png
│   ├── model_sebaran_klaster_pairplot.png
│   ├── model_silhouette_scores.png
│   └── perbandingan_sebelum_sesudah_preprocessing.png
│
└── laporan/                     # Berkas Portofolio & Dokumen Asesmen UKK
    ├── Laporan_Sertifikasi_Associate_Data_Scientist_9Unit.docx   # Laporan Analisis Data 9 Unit SKKNI (Word)
    ├── Laporan_Sertifikasi_Associate_Data_Scientist_9Unit.pdf    # Laporan Analisis Data 9 Unit SKKNI (PDF 24 Hal)
    ├── Dokumen_Bukti_Portofolio_Repositori_GitHub_EDA.docx       # Dokumen Bukti Portofolio Repositori & EDA (Word)
    ├── Dokumen_Bukti_Portofolio_Repositori_GitHub_EDA.pdf        # Dokumen Bukti Portofolio Repositori & EDA (PDF 14 Hal)
    ├── Slide_Presentasi_Sidang_UKK_Associate_Data_Scientist.pptx # Slide Sidang Presentasi UKK 16:9 (PPTX)
    └── Slide_Presentasi_Sidang_UKK_Associate_Data_Scientist.pdf  # Slide Sidang Presentasi UKK 16:9 (PDF 12 Slide)
```

---

## 📦 Berkas Resmi Uji Kompetensi Keahlian (UKK) / Asesmen Sertifikasi BNSP

Tiga berkas utama pemenuhan syarat unjuk kerja portofolio mandiri:

| No. Persyaratan Asesmen | Nama Berkas Dokumen | Format | Halaman / Slide | Deskripsi Cakupan & Pemenuhan Standar |
| :--- | :--- | :---: | :---: | :--- |
| **Butir 2: Laporan Data Analysis** | [`Laporan_Sertifikasi_Associate_Data_Scientist_9Unit.pdf`](laporan/Laporan_Sertifikasi_Associate_Data_Scientist_9Unit.pdf) | PDF & DOCX | 24 Halaman | Laporan komprehensif pemenuhan 9 Unit Kompetensi SKKNI No. 299 Tahun 2020 lengkap dengan koding Python terstruktur, visualisasi HD, tabel statistik, justifikasi ilmiah K=4, dan rekomendasi kebijakan. |
| **Butir 3: Bukti Portofolio Repositori & EDA** | [`Dokumen_Bukti_Portofolio_Repositori_GitHub_EDA.pdf`](laporan/Dokumen_Bukti_Portofolio_Repositori_GitHub_EDA.pdf) | PDF & DOCX | 14 Halaman | Dokumen bukti portofolio mandiri memuat rekam jejak Git commit riil (@10040818), arsitektur file tree repositori, ringkasan kode 9 Unit notebook, 14 grafik hasil EDA 300 DPI, dan tangkapan layar web dashboard live. |
| **Butir 4: Slide Presentasi Sidang UKK** | [`Slide_Presentasi_Sidang_UKK_Associate_Data_Scientist.pdf`](laporan/Slide_Presentasi_Sidang_UKK_Associate_Data_Scientist.pdf) | PDF & PPTX | 12 Slide (16:9) | Slide presentasi sidang asesmen berdesain Dark Executive 16:9 mencakup alur metodologi CRISP-DM, audit data, rekayasa fitur Zero Data Loss, justifikasi K=4, taksonomi 4 klaster, simulator web, dan penutup. |

---

## 🚀 Panduan Menjalankan Proyek Secara Lokal

### 1. Kloning Repositori
```bash
git clone https://github.com/10040818/ukk.git
cd ukk
```

### 2. Persiapan Lingkungan Python
Disarankan menggunakan virtual environment (Python 3.10+):
```bash
python -m venv env
# Aktivasi di Windows:
.\env\Scripts\activate
# Aktivasi di macOS/Linux:
source env/bin/activate
```

### 3. Instalasi Dependensi
```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

### 4. Menjalankan Jupyter Notebook
```bash
jupyter notebook analisis_kmeans.ipynb
```

### 5. Membuka Dashboard Web
Cukup buka file `index.html` langsung di browser favorit Anda (Google Chrome, Edge, Safari, Firefox), atau gunakan Live Server:
```bash
# Menggunakan python built-in HTTP server:
python -m http.server 8000
```
Buka peramban di `http://localhost:8000`.

---

## 👨‍💻 Peneliti Utama & Pengembang Model

* **Nama:** **M. Iqbal Adenan**
* **Peran:** Peneliti Utama & Pengembang Model (*Lead Data Scientist & Developer*)
* **Skema Okupasi:** Associate Data Scientist (KKNI Level 5)
* **Standar Kompetensi:** SKKNI No. 299 Tahun 2020
* **Repositori GitHub:** [github.com/10040818/ukk](https://github.com/10040818/ukk)

---

## 📄 Lisensi
Proyek ini dilisensikan di bawah [MIT License](LICENSE). Seluruh dataset bersumber dari data publik sensus kependudukan dunia PBB.


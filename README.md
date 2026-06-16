# 🏎️ BoxOne 

**BoxOne** adalah project data science sederhana berbasis **Formula 1** yang dibuat untuk menganalisis performa driver, constructor, dan memprediksi peluang finish **Top 10** menggunakan machine learning.

---

## 🚀 Live Demo

Link aplikasi Streamlit:

```text
https://boxone-beta.streamlit.app/
```

> Ganti link di atas dengan link Streamlit kamu setelah deploy berhasil.

---

## 📌 Fitur Project

Project ini memiliki beberapa fitur utama:

### 1. Overview Dashboard

Menampilkan ringkasan umum dataset Formula 1, seperti:

* Total data race
* Total driver
* Total constructor
* Rentang musim balapan
* Grafik jumlah race per tahun
* Top 10 driver berdasarkan total points

### 2. Driver Analysis

Fitur untuk menganalisis performa driver tertentu.

Data yang ditampilkan:

* Total race
* Total wins
* Total podiums
* Total points
* Grafik points per season
* Riwayat race driver

### 3. Constructor Analysis

Fitur untuk menganalisis performa constructor / tim Formula 1.

Data yang ditampilkan:

* Ranking constructor berdasarkan points
* Total race data
* Total wins
* Total podiums
* Total points
* Grafik performa constructor per season

### 4. Top 10 Predictor

Fitur machine learning sederhana untuk memprediksi apakah seorang pembalap berpeluang finish di posisi **Top 10** berdasarkan:

* Tahun balapan
* Grid start
* Constructor
* Circuit

Output prediksi:

```text
Finish Top 10 ✅
atau
Tidak Finish Top 10 ❌
```

---

## 🧠 Teknologi yang Digunakan

Project ini dibuat menggunakan:

* Python
* Pandas
* Plotly
* Scikit-learn
* Joblib
* Streamlit
* GitHub

---

## 📁 Struktur Folder

Struktur folder project:

```text
BoxOne/
│
├── data/
│   ├── races.csv
│   ├── results.csv
│   ├── drivers.csv
│   ├── constructors.csv
│   └── circuits.csv
│
├── notebook/
│   └── f1_analysis.ipynb
│
├── app.py
├── train_model.py
├── model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

Dataset yang digunakan adalah dataset historis Formula 1 yang berisi informasi tentang:

* Race
* Driver
* Constructor
* Circuit
* Result
* Grid start
* Points
* Finish position

File utama yang digunakan:

```text
races.csv
results.csv
drivers.csv
constructors.csv
circuits.csv
```

---

## ⚙️ Cara Menjalankan Project di Lokal

### 1. Clone repository

```bash
git clone https://github.com/Zaru96/BoxOne.git
cd BoxOne
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan training model

```bash
python train_model.py
```

Perintah ini akan membuat file:

```text
model.pkl
```

### 4. Jalankan aplikasi Streamlit

```bash
streamlit run app.py
```

Atau jika command `streamlit` tidak terbaca:

```bash
python -m streamlit run app.py
```

---

## 🤖 Machine Learning Model

Model yang digunakan adalah **Random Forest Classifier**.

Target prediksi:

```text
top_10 = 1 jika positionOrder <= 10
top_10 = 0 jika positionOrder > 10
```

Fitur yang digunakan:

```text
year
grid
constructor_name
circuit_name
```

Model disimpan menggunakan `joblib` dalam file:

```text
model.pkl
```

---

## 🧪 Alur Pengerjaan Project

Tahapan utama dalam project ini:

1. Persiapan project
2. Load dataset
3. Data understanding
4. Data merging
5. Data cleaning
6. Feature engineering
7. Exploratory data analysis
8. Persiapan data modeling
9. Training model
10. Evaluasi model
11. Simpan model
12. Membuat dashboard Streamlit
13. Deploy aplikasi

---

## 🌐 Deployment

Project ini dapat dideploy menggunakan **Streamlit Community Cloud**.

Langkah umum deployment:

1. Push project ke GitHub
2. Login ke Streamlit Community Cloud
3. Pilih repository `BoxOne`
4. Pilih branch `main`
5. Pilih file utama `app.py`
6. Klik deploy

---

## 👨‍💻 Author

Project ini dibuat oleh:

**Zaru96**

GitHub:

```text
https://github.com/Zaru96
```

---

## 🏁 Penutup
Project ini masih bisa dikembangkan lagi dengan fitur tambahan seperti:

* Driver comparison
* Constructor battle
* GOAT Index
* DNF rate analysis
* Podium predictor
* F1 season performance dashboard

---

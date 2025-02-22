# AI Rekomendasi Layanan Publik

Proyek ini adalah sistem rekomendasi layanan publik yang menggunakan pendekatan hybrid antara Naive Bayes dan Fuzzy Logic untuk memberikan rekomendasi layanan berdasarkan deskripsi masalah yang diberikan oleh pengguna.

## Fitur

- **Preprocessing Teks**: Membersihkan dan mempersiapkan teks untuk analisis.
- **Model Naive Bayes**: Melatih model Naive Bayes dengan data training untuk memprediksi layanan yang sesuai.
- **Fuzzy Logic**: Menghitung derajat keanggotaan fuzzy untuk setiap kategori layanan.
- **Kombinasi Prediksi**: Menggabungkan prediksi dari model Naive Bayes dan Fuzzy Logic untuk memberikan rekomendasi yang lebih akurat.

## Struktur Proyek

```plaintext
.
├── dataset.csv
├── main.py
├── poetry.lock
├── pyproject.toml
├── README.md
├── src
│   ├── HybridServiceRecommender.py
│   ├── __init__.py
└── tests
    └── __init__.py
```

## Instalasi

1. Clone repositori ini:

    ```sh
    git clone <URL_REPOSITORI>
    cd layanan-masyarakat
    ```

2. Install dependencies menggunakan Poetry:

    ```sh
    poetry install
    ```

## Penggunaan

1. Jalankan skrip [main.py](http://_vscodecontentref_/9):

    ```sh
    poetry run python main.py
    ```

2. Masukkan deskripsi masalah, usia, dan pendapatan bulanan Anda ketika diminta.

3. Sistem akan memberikan rekomendasi layanan berdasarkan input yang Anda berikan.

## Contoh

AI Rekomendasi Layanan Publik Masukkan deskripsi masalah Anda: Saya sedang demam dan butuh obat Masukkan usia Anda: 21 Masukkan pendapatan bulanan Anda (dalam Rupiah): 4600000

Deskripsi: Saya sedang demam dan butuh obat
Usia: 21
Pendapatan: Rp4,600,000.0

Rekomendasi Layanan:

- Telemedicine: 0.75
- Kunjungan Puskesmas: 0.65

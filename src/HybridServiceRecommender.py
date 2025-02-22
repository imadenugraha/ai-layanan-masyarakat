import pandas as pd
import re

from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity



class HybridServiceRecommender:
    def __init__(self):
        """
        Inisialisasi sistem rekomendasi hybrid yang menggabungkan Naive Bayes dan Fuzzy Logic.
        
        Parameters:
            training_data: DataFrame dengan kolom 'deskripsi_masalah' dan 'layanan_rekomendasi'
        """
        # Inisialisasi komponen Naive Bayes
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        self.nb_classifier = MultinomialNB()
        
        # Inisialisasi kategori layanan dan kata kunci untuk fuzzy logic
        self.service_keywords = {
            'kesehatan': [
                'sakit', 'demam', 'batuk', 'flu', 'pusing', 'vaksin', 
                'kesehatan', 'obat', 'pengobatan', 'medis'
            ],
            'pendidikan': [
                'sekolah', 'pendidikan', 'belajar', 'kursus', 'pelatihan',
                'skill', 'keterampilan', 'mengajar', 'akademik', 'beasiswa'
            ],
            'sosial': [
                'bantuan', 'ekonomi', 'miskin', 'susah', 'dukungan',
                'kesejahteraan', 'biaya', 'hidup', 'kebutuhan', 'PHK'
            ],
            'kependudukan': [
                'ktp', 'kk', 'kelahiran', 'kematian', 'penduduk', 'pindah',
                'datang', 'tinggal', 'lahir', 'mati'
            ]
        }
        
        # Mapping kategori ke layanan spesifik
        self.service_mapping = {
            'kesehatan': ['kunjungan_puskesmas', 'telemedicine', 'vaksinasi_keliling'],
            'pendidikan': ['bantuan_beasiswa', 'pendidikan_vokasi', 'kursus_online'],
            'sosial': ['program_lansia', 'bantuan_sosial', 'bantuan_ekonomi'],
            'kependudukan': ['pembuatan_ktp', 'pembuatan_kk', 'akta_kelahiran'],
        }
        
        # Mapping ke nama layanan yang lebih ramah pengguna
        self.service_display_names = {
            'kunjungan_puskesmas': 'Kunjungan Puskesmas',
            'telemedicine': 'Telemedicine',
            'vaksinasi_keliling': 'Vaksinasi Keliling',
            'bantuan_beasiswa': 'Bantuan Beasiswa',
            'pendidikan_vokasi': 'Pendidikan Vokasi',
            'kursus_online': 'Kursus Online',
            'program_lansia': 'Program Lansia',
            'bantuan_sosial': 'Bantuan Sosial',
            'bantuan_ekonomi': 'Bantuan Ekonomi',
            'pembuatan_ktp': 'Pembuatan KTP',
            'pembuatan_kk': 'Pembuatan KK',
            'akta_kelahiran': 'Akta Kelahiran'
        }
        
        # Load data training jika tersedia
        df = pd.read_csv('dataset.csv')
        training_data = df[['deskripsi_masalah', 'layanan_rekomendasi']]
        
        # Train model jika data training tersedia
        if training_data is not None:
            self.train(training_data)

    def preprocess_text(self, text: str) -> str:
        """Membersihkan dan mempersiapkan teks untuk analisis"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        return text

    def train(self, training_data: pd.DataFrame) -> None:
        """
        Melatih model Naive Bayes dengan data training
        
        Parameters:
            training_data: DataFrame dengan kolom 'deskripsi_masalah' dan 'layanan_rekomendasi'
        """
        # Preprocess deskripsi masalah
        processed_descriptions = [
            self.preprocess_text(desc) for desc in training_data['deskripsi_masalah']
        ]
        
        # Fit dan transform data training
        X = self.vectorizer.fit_transform(processed_descriptions)
        y = training_data['layanan_rekomendasi']
        
        # Train Naive Bayes classifier
        self.nb_classifier.fit(X, y)
        
        print("Model berhasil dilatih dengan data training")

    def get_naive_bayes_predictions(self, description: str) -> Dict[str, float]:
        """
        Mendapatkan probabilitas prediksi dari Naive Bayes untuk setiap layanan
        """
        # Preprocess dan vectorize input
        processed_desc = self.preprocess_text(description)
        X = self.vectorizer.transform([processed_desc])
        
        # Dapatkan probabilitas untuk setiap kelas
        probabilities = self.nb_classifier.predict_proba(X)[0]
        
        # Buat dictionary mapping layanan ke probabilitasnya
        service_probs = dict(zip(self.nb_classifier.classes_, probabilities))
        
        return service_probs

    def calculate_fuzzy_membership(self, description: str) -> Dict[str, float]:
        """
        Menghitung derajat keanggotaan fuzzy untuk setiap kategori
        """
        description = self.preprocess_text(description)
        memberships = {}
        
        for category, keywords in self.service_keywords.items():
            # Gabungkan keywords menjadi satu dokumen
            keyword_doc = ' '.join(keywords)
            
            # Hitung TF-IDF dan cosine similarity
            tfidf_matrix = TfidfVectorizer().fit_transform([description, keyword_doc])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Normalisasi nilai ke range [0,1]
            memberships[category] = max(0, min(1, similarity))
            
        return memberships

    def combine_predictions(self,
                          nb_predictions: Dict[str, float],
                          fuzzy_memberships: Dict[str, float],
                          age: int = None,
                          income: float = None,
                          nb_weight: float = 0.6) -> Dict[str, float]:
        """
        Menggabungkan prediksi Naive Bayes dengan fuzzy memberships
        """
        combined_scores = {}
    
        # Terapkan bobot untuk setiap metode
        fuzzy_weight = 1 - nb_weight
        
        # Gabungkan skor untuk setiap layanan
        for category, nb_prob in nb_predictions.items():
            # Dapatkan layanan spesifik untuk kategori ini
            services = self.service_mapping.get(category.lower(), [])
            
            for service in services:
                fuzzy_score = fuzzy_memberships.get(category.lower(), 0)
                combined_scores[service] = (
                    nb_prob * nb_weight + 
                    fuzzy_score * fuzzy_weight
                )
                
                # Modifikasi skor berdasarkan usia
                if age is not None:
                    if age >= 60 and service == 'program_lansia':
                        combined_scores[service] += 0.2
                    elif age <= 25 and service == 'bantuan_beasiswa':
                        combined_scores[service] += 0.2
                
                # Modifikasi skor berdasarkan pendapatan
                if income is not None and income < 3000000:
                    if service in ['bantuan_ekonomi', 'bantuan_sosial']:
                        combined_scores[service] += 0.2
        
        return combined_scores

    def get_recommendations(self, 
                          description: str,
                          age: int = None,
                          income: float = None,
                          top_n: int = 2) -> List[Tuple[str, float]]:
        """
        Mendapatkan rekomendasi layanan menggunakan pendekatan hybrid
        """
        # Dapatkan prediksi dari kedua model
        nb_predictions = self.get_naive_bayes_predictions(description)

        fuzzy_memberships = self.calculate_fuzzy_membership(description)
        
        # Gabungkan prediksi
        combined_scores = self.combine_predictions(
            nb_predictions,
            fuzzy_memberships,
            age,
            income
        )
        
        # Urutkan dan ambil top-n rekomendasi
        recommendations = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        recommendations = [(self.service_display_names[service], score) for service, score in recommendations]
        
        return recommendations
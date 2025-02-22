from src.HybridServiceRecommender import HybridServiceRecommender

def main():  
    # Inisialisasi dan train recommender
    recommender = HybridServiceRecommender()
    
    # Input user
    print("AI Rekomendasi Layanan Publik")
    description = input("Masukkan deskripsi masalah Anda: ")
    age = int(input("Masukkan usia Anda: "))
    income = float(input("Masukkan pendapatan bulanan Anda (dalam Rupiah): "))
    
    # Dapatkan rekomendasi
    recommendations = recommender.get_recommendations(
        description=description,
        age=age,
        income=income
    )
    
    # Tampilkan hasil
    print(f"\nDeskripsi: {description}")
    print(f"Usia: {age}")
    print(f"Pendapatan: Rp{income:,}")
    print("\nRekomendasi Layanan:")
    for service, score in recommendations:
        print(f"- {service}: {score:.2f}")  
        

if __name__ == "__main__":
    main()

# ========== IMPORT DAN SETUP ==========
# Mengimpor library yang diperlukan untuk API
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import pandas as pd

# Memuat model dan encoder yang telah dilatih
model = joblib.load('medicine_review_model.pkl')
le_manufacturer = joblib.load('manufacturer_encoder.pkl')
le_best_review = joblib.load('review_category_encoder.pkl')

print(f"Model expects {model.n_features_in_} features")

# Membuat aplikasi FastAPI
app = FastAPI()

from fastapi.responses import FileResponse

@app.get("/")
async def read_root():
    return FileResponse("index.html")

# Mengatur CORS untuk mengizinkan akses dari semua origin
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== MODEL INPUT DATA ==========
# Mendefinisikan struktur data input untuk prediksi obat
class MedicineData(BaseModel):
    medicine_name: Optional[str] = None  # Opsional
    composition: Optional[str] = None
    uses: Optional[str] = None
    side_effects: Optional[str] = None
    manufacturer: str

# ========== FUNGSI PEMBUAT FITUR ==========
# Fungsi bantu untuk membuat fitur dari data input pengguna
def create_features(data: MedicineData):
    # Menangani nilai kosong
    composition = data.composition or ''
    side_effects = data.side_effects or ''
    uses = data.uses or ''

    # Rekayasa fitur sesuai dengan yang dilakukan di train.py
    num_ingredients = composition.count('+') + 1  # Jumlah bahan
    num_side_effects = side_effects.count(',') + 1  # Jumlah efek samping
    num_uses = uses.count(',') + 1  # Jumlah penggunaan
    medicine_name = data.medicine_name or 'Unknown Medicine'  # Default jika kosong
    medicine_name_length = len(medicine_name)  # Panjang nama obat

    # Mengencode produsen
    try:
        manufacturer_encoded = le_manufacturer.transform([data.manufacturer])[0]
    except:
        manufacturer_encoded = 0  # Default untuk produsen yang tidak dikenal

    # Menghitung frekuensi produsen dari dataset
    df = pd.read_csv('Medicine_Details.csv')
    manufacturer_counts = df['Manufacturer'].value_counts()
    manufacturer_frequency = manufacturer_counts.get(data.manufacturer, 0)

    # Apakah merupakan tablet
    is_tablet = int('tablet' in medicine_name.lower())

    return np.array([[num_ingredients, num_side_effects, num_uses,
                      manufacturer_frequency, manufacturer_encoded,
                      medicine_name_length, is_tablet]])

# ========== ENDPOINT PREDIKSI ==========
# Mendefinisikan route untuk melakukan prediksi
@app.post("/predict")
def predict_medicine_review(data: MedicineData):
    # Membuat fitur dari data input
    input_features = create_features(data)

    # Melakukan prediksi
    prediction_encoded = model.predict(input_features)[0]
    prediction_proba = model.predict_proba(input_features)[0]

    # Mendekode prediksi ke kategori asli
    predicted_category = le_best_review.inverse_transform([prediction_encoded])[0]

    # Mapping ke bahasa Indonesia
    category_mapping = {
        "Excellent": "Bagus",
        "Average": "Rata-rata",
        "Poor": "Buruk"
    }
    predicted_category_id = category_mapping.get(predicted_category, predicted_category)

    # Membuat dictionary probabilitas untuk setiap kategori
    prob_dict = {category_mapping.get(category, category): float(prob) for category, prob in
                 zip(le_best_review.classes_, prediction_proba)}

    # Mengembalikan hasil prediksi sebagai JSON
    return {
        "predicted_review_category": predicted_category_id,
        "prediction_probabilities": prob_dict
    }

# ========== CARA MENJALANKAN SERVER ==========
# Jalankan server dengan perintah: uvicorn main:app --reload
# Ini akan memulai server FastAPI pada http://127.0.0.1:8000
# Dokumentasi API tersedia di http://127.0.0.1:8000/docs

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ========== 1. DATA UNDERSTANDING ==========
# Memuat dataset dan menampilkan informasi dasar untuk memahami struktur data
df = pd.read_csv('Medicine_Details.csv')

print("Info Dataset:")
print(f"Jumlah baris: {len(df)}")
print(f"\nKolom: {df.columns.tolist()}")
print(f"\nBeberapa baris pertama:")
print(df.head())

# Membuat variabel target: Kategori ulasan dengan persentase tertinggi
df['Best_Review'] = df[['Excellent Review %', 'Average Review %', 'Poor Review %']].idxmax(axis=1)
df['Best_Review'] = df['Best_Review'].str.replace(' Review %', '')

print(f"\nDistribusi Target:")
print(df['Best_Review'].value_counts())

# ========== 2. FEATURE CLEANING ==========
# Mengatasi nilai kosong pada kolom teks untuk mencegah error
df['Composition'] = df['Composition'].fillna('')
df['Side_effects'] = df['Side_effects'].fillna('')
df['Uses'] = df['Uses'].fillna('')

# ========== 3. FEATURE CREATION ==========
# Membuat fitur baru dari data mentah untuk meningkatkan informasi model
# Hitung bahan komposisi
df['Num_Ingredients'] = df['Composition'].str.count('\+') + 1

# Hitung efek samping
df['Num_Side_Effects'] = df['Side_effects'].str.count(',') + 1

# Hitung penggunaan
df['Num_Uses'] = df['Uses'].str.count(',') + 1

# Dapatkan frekuensi produsen
manufacturer_counts = df['Manufacturer'].value_counts()
df['Manufacturer_Frequency'] = df['Manufacturer'].map(manufacturer_counts)

# Tambahan fitur: Panjang nama obat
df['Medicine_Name_Length'] = df['Medicine Name'].str.len()

# Tambahan fitur: Apakah merupakan tablet (1 jika ya, 0 jika tidak)
df['Is_Tablet'] = df['Medicine Name'].str.contains('Tablet', case=False, na=False).astype(int)

# ========== 4. FEATURE TRANSFORMATION ==========
# Mengubah fitur kategorikal menjadi numerik untuk input model
le_manufacturer = LabelEncoder()
le_best_review = LabelEncoder()

df['Manufacturer_Encoded'] = le_manufacturer.fit_transform(df['Manufacturer'])
df['Best_Review_Encoded'] = le_best_review.fit_transform(df['Best_Review'])

# ========== 5. FEATURE SELECTION & EVALUASI ==========
# Memilih fitur yang relevan dan mengevaluasi model
feature_columns = ['Num_Ingredients', 'Num_Side_Effects', 'Num_Uses',
                   'Manufacturer_Frequency', 'Manufacturer_Encoded',
                   'Medicine_Name_Length', 'Is_Tablet']

X = df[feature_columns]
y = df['Best_Review_Encoded']

print(f"\nFitur yang dipilih: {feature_columns}")
print(f"\nStatistik fitur:")
print(X.describe())

# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nUkuran set pelatihan: {len(X_train)}")
print(f"Ukuran set pengujian: {len(X_test)}")

# Latih model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# Buat prediksi pada data testing
y_pred = model.predict(X_test)

# Evaluasi performa model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Model: {accuracy:.4f}")

print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=le_best_review.classes_))

print("\nMatriks Kebingungan:")
print(confusion_matrix(y_test, y_pred))

# Analisis kepentingan fitur
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nKepentingan Fitur:")
print(feature_importance)

# ========== 6. REFLEKSI TEKNIS ==========
# Simpan model dan encoder untuk penggunaan di masa depan
joblib.dump(model, 'medicine_review_model.pkl')
joblib.dump(le_manufacturer, 'manufacturer_encoder.pkl')
joblib.dump(le_best_review, 'review_category_encoder.pkl')

print("\n✓ Model, encoder produsen, dan encoder kategori ulasan telah disimpan!")

# Contoh prediksi untuk validasi model
sample_medicine = X_test.iloc[0:1]
prediction = model.predict(sample_medicine)
prediction_proba = model.predict_proba(sample_medicine)

predicted_category = le_best_review.inverse_transform(prediction)[0]
print(f"\n--- Contoh Prediksi ---")
print(f"Kategori Ulasan yang Diprediksi: {predicted_category}")
print(f"Probabilitas Prediksi: {dict(zip(le_best_review.classes_, prediction_proba[0]))}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv('real_drug_dataset.csv') # Pastikan nama file sesuai
print(df.head())

# 1. Membuat kolom target 'Efektivitas' berdasarkan Improvement_Score
def label_efektivitas(score):
    try:
        val = float(score)
        if val >= 8.0: return 'Tinggi'
        elif val >= 5.0: return 'Sedang'
        else: return 'Rendah'
    except:
        return 'Sedang'

# Gunakan kolom 'Improvement_Score' sesuai file
df['Efektivitas'] = df['Improvement_Score'].apply(label_efektivitas)

# 2. Lakukan Encoding pada data kategori
# mengubah Condition, Drug_Name, dan Side_Effects menjadi angka
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Condition_Enc'] = le.fit_transform(df['Condition'])
df['Drug_Enc'] = le.fit_transform(df['Drug_Name'])
df['Side_Effects_Enc'] = le.fit_transform(df['Side_Effects'])
df['Gender_Enc'] = le.fit_transform(df['Gender'])
print("Hasil Labeling dan Encoding:")
print(df[['Condition', 'Drug_Name', 'Improvement_Score', 'Efektivitas']].head())

# 3. Persiapan Data untuk Model
# fitur yang digunakan: Age, Gender, Condition, Drug, Dosage, Duration, dan Side Effects
features = ['Age', 'Gender_Enc', 'Condition_Enc', 'Drug_Enc', 
            'Dosage_mg', 'Treatment_Duration_days', 'Side_Effects_Enc']
X = df[features]
y = df['Efektivitas']

# Split data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Jumlah data training: {len(X_train)}")

# Latih Model Naive Bayes (Baseline)
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
print(f"Akurasi: {accuracy_score(y_test, y_pred)}")
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))

# Visualisasi Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Rendah', 'Sedang', 'Tinggi'], yticklabels=['Rendah', 'Sedang', 'Tinggi'])
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.show()
# Simpan model jika diperlukan
import joblib
joblib.dump(model, 'model_efektivitas_obat.pkl')
print("Model disimpan sebagai 'model_efektivitas_obat.pkl'")
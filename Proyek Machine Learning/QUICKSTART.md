# Quick Start Guide

## 🎯 Cara Cepat Menjalankan Aplikasi

### 1. Pastikan dependencies terinstall
```bash
pip install -r requirements.txt
```

### 2. Jalankan server
```bash
python run_server.py
# atau
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Buka browser
```
http://localhost:8000
```

## 🧪 Testing dengan cURL/Postman

### Predict Endpoint
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 45,
    "Gender": "Male",
    "Condition": "Hypertension",
    "Drug_Name": "Lisinopril",
    "Dosage_mg": 10,
    "Treatment_Duration_days": 30,
    "Side_Effects": "Dizziness"
  }'
```

Response:
```json
{
  "prediction": "Tinggi"
}
```

### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "model_ready": true
}
```

## 📱 Fitur Web Interface

- ✅ Responsive design (desktop & mobile)
- ✅ Real-time form validation
- ✅ Loading indicator
- ✅ Color-coded results (Tinggi/Sedang/Rendah)
- ✅ Error handling
- ✅ Reset form button

## 🚀 Siap untuk Deployment?

Lihat [DEPLOYMENT.md](DEPLOYMENT.md) untuk panduan lengkap deployment ke:
- ✅ Heroku
- ✅ PythonAnywhere  
- ✅ AWS/Google Cloud (dengan Docker)
- ✅ Server lokal/VPS

## 🔧 Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Port 8000 sudah digunakan | Ganti port: `uvicorn main:app --port 8001` |
| Model tidak terload | Pastikan file `models/model.joblib` ada |
| CORS error | Sudah ditangani di main.py |
| Halaman blank | Refresh browser atau clear cache |

## 📞 Support
- API: `http://localhost:8000/docs` (Swagger UI)
- Health: `http://localhost:8000/health`
- Main: `http://localhost:8000`

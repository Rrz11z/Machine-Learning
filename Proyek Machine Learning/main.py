from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import os
import joblib
import logging


app = FastAPI(title="Drug Effectiveness Predictor")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# paths for saved model/encoders
MODEL_PATH = os.path.join('models', 'model.joblib')
ENC_PATH = os.path.join('models', 'encoders.joblib')

# feature order used for training/prediction
FEATURES = ['Age', 'Gender_Enc', 'Condition_Enc', 'Drug_Enc', 'Dosage_mg', 'Treatment_Duration_days', 'Side_Effects_Enc']

logging.basicConfig(level=logging.INFO)


class PredictRequest(BaseModel):
    Age: float
    Gender: str
    Condition: str
    Drug_Name: str
    Dosage_mg: float
    Treatment_Duration_days: float
    Side_Effects: str


def label_efektivitas(score):
    try:
        val = float(score)
        if val >= 8.0:
            return 'Tinggi'
        elif val >= 5.0:
            return 'Sedang'
        else:
            return 'Rendah'
    except:
        return 'Sedang'


# Global containers for the trained model and encoders
MODEL = None
ENCODERS = {}


def fit_model():
    global MODEL, ENCODERS
    logging.info('Fitting model from dataset...')
    df = pd.read_csv('real_drug_dataset.csv')

    # create target
    if 'Efektivitas' not in df.columns:
        if 'Improvement_Score' in df.columns:
            df['Efektivitas'] = df['Improvement_Score'].apply(label_efektivitas)
        else:
            raise RuntimeError('Dataset missing Improvement_Score column to derive Efektivitas')

    # prepare encoders for categorical fields
    enc_gender = LabelEncoder()
    enc_condition = LabelEncoder()
    enc_drug = LabelEncoder()
    enc_side = LabelEncoder()

    df['Gender_Enc'] = enc_gender.fit_transform(df['Gender'].astype(str))
    df['Condition_Enc'] = enc_condition.fit_transform(df['Condition'].astype(str))
    df['Drug_Enc'] = enc_drug.fit_transform(df['Drug_Name'].astype(str))
    df['Side_Effects_Enc'] = enc_side.fit_transform(df['Side_Effects'].astype(str))

    ENCODERS = {
        'Gender': enc_gender,
        'Condition': enc_condition,
        'Drug_Name': enc_drug,
        'Side_Effects': enc_side,
    }

    X = df[FEATURES].fillna(0)
    y = df['Efektivitas']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    MODEL = MultinomialNB()
    MODEL.fit(X_train, y_train)

    # optional: log accuracy
    try:
        y_pred = MODEL.predict(X_test)
        acc = (y_pred == y_test).mean()
        logging.info(f"Model trained. Test accuracy ~ {acc:.4f}")
    except Exception:
        pass

    # save model and encoders
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(MODEL, MODEL_PATH)
    joblib.dump(ENCODERS, ENC_PATH)
    logging.info(f"Saved model to {MODEL_PATH} and encoders to {ENC_PATH}")


@app.on_event('startup')
def startup_event():
    global MODEL, ENCODERS
    try:
        # try load saved model/encoders
        if os.path.exists(MODEL_PATH) and os.path.exists(ENC_PATH):
            MODEL = joblib.load(MODEL_PATH)
            ENCODERS = joblib.load(ENC_PATH)
            logging.info(f"Loaded model from {MODEL_PATH}")
        else:
            fit_model()
    except Exception as e:
        logging.exception('Error during startup/model load')


def _encode_value(le: LabelEncoder, value: str):
    # ensure unseen values are handled by extending classes_
    if value not in le.classes_:
        le.classes_ = np.append(le.classes_, value)
    return int(le.transform([value])[0]) # type: ignore


@app.post('/predict')
def predict(req: PredictRequest):
    if MODEL is None or not ENCODERS:
        raise HTTPException(status_code=503, detail='Model not ready')

    enc = ENCODERS
    try:
        gender_enc = _encode_value(enc['Gender'], str(req.Gender))
        condition_enc = _encode_value(enc['Condition'], str(req.Condition))
        drug_enc = _encode_value(enc['Drug_Name'], str(req.Drug_Name))
        side_enc = _encode_value(enc['Side_Effects'], str(req.Side_Effects))

        # build DataFrame with same feature names used in training
        row = {
            'Age': req.Age,
            'Gender_Enc': gender_enc,
            'Condition_Enc': condition_enc,
            'Drug_Enc': drug_enc,
            'Dosage_mg': req.Dosage_mg,
            'Treatment_Duration_days': req.Treatment_Duration_days,
            'Side_Effects_Enc': side_enc,
        }
        x_df = pd.DataFrame([row], columns=FEATURES)

        pred = MODEL.predict(x_df)[0]
        return JSONResponse({'prediction': pred})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get('/', response_class=HTMLResponse)
def index():
    return FileResponse('index.html', media_type='text/html')


@app.get('/admin', response_class=HTMLResponse)
def admin():
    return FileResponse('admin.html', media_type='text/html')


@app.get('/health')
def health():
    return JSONResponse({'status': 'ok', 'model_ready': MODEL is not None})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)

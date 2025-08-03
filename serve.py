"""
serve.py - FastAPI serving for Fraud Detection model
"""
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.preprocessing import OneHotEncoder
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# --- Feature Engineering ---
def feature_engineering(df):
    if 'claim_amount' in df.columns and 'customer_avg_claim' in df.columns:
        df['claim_to_avg'] = df['claim_amount'] / (df['customer_avg_claim'] + 1)
    return df

class ClaimData(BaseModel):
    claim_amount: float
    customer_avg_claim: float
    customer_history: int
    submission_channel: str
    document_type: str

# --- API App ---
def create_app(model_path='fraud_model.pkl'):
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # or specify ["http://localhost:8000"] if you serve frontend from a server
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    model = joblib.load(model_path)
    encoder = joblib.load('encoder.pkl')
    @app.post('/predict')
    def predict(data: ClaimData):
        df = pd.DataFrame([data.dict()])
        cat_cols = ['submission_channel', 'document_type']
        num_cols = ['claim_amount', 'customer_avg_claim', 'customer_history']
        encoded = encoder.transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
        df = pd.concat([df[num_cols], encoded_df], axis=1)
        df = feature_engineering(df)
        pred = model.predict_proba(df)[:,1][0]
        return {'fraud_likelihood': float(pred)}
    return app

if __name__ == '__main__':
    app = create_app()
    uvicorn.run(app, host='0.0.0.0', port=8000)

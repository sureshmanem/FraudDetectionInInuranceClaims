# Insurance Fraud Detection Project

## Project Prompt

> I am developing a machine learning application for automated fraud detection in insurance claims. The project involves these steps:
> - Data Handling: Load, inspect, and preprocess historical insurance claim data (features: claim amount, customer history, submission details, supporting document metadata, etc.). Handle missing values, encode categorical variables, and engineer features that capture patterns or anomalies in claim behavior.
> - Modeling Approaches: If labeled fraud data is available: Use supervised learning (Random Forest, XGBoost, or similar classifiers). Address class imbalance using oversampling (SMOTE) or model class weighting. If labeled data is limited: Use unsupervised anomaly detection (LSTM autoencoders or Isolation Forest) to flag suspicious claims.
> - Analysis and Evaluation: Evaluate model(s) using metrics appropriate for imbalanced data (recall, precision, F1-score, ROC-AUC). Visualize and interpret model results with feature importance plots or SHAP/LIME for explainability.
> - Deployment and API: Serialize the trained model. Build a simple REST API (using Flask or FastAPI) that receives claim data and returns fraud likelihood or anomaly score.

## Model Explanation

This project uses a supervised machine learning approach (Random Forest Classifier) to predict the likelihood that an insurance claim is fraudulent. The model is trained on historical claim data with the following features:
- `claim_amount`: The amount claimed
- `customer_avg_claim`: The customer's average claim amount
- `customer_history`: Number of previous claims
- `submission_channel`: How the claim was submitted (online/agent)
- `document_type`: Type of supporting document (pdf/scan/photo)

**How the model works:**
- The data is preprocessed: missing values are imputed, categorical variables are one-hot encoded, and a feature `claim_to_avg` (claim amount divided by average claim) is engineered.
- Class imbalance is handled using SMOTE oversampling.
- The Random Forest model is trained to classify claims as fraudulent or not.
- When a new claim is submitted, the same preprocessing is applied, and the model outputs a probability (fraud likelihood) between 0 and 1.
- The frontend interprets this likelihood: if it is above 0.7, the claim is flagged as likely fraudulent; otherwise, it is considered likely genuine.

## Project Structure

```
FraudDetectionInInsuranceClaims/
├── claims_data.csv         # Sample insurance claims data
├── encoder.pkl             # Trained OneHotEncoder for categorical features
├── fraud_model.pkl         # Trained Random Forest model
├── train.py                # Script to train the model and encoder
├── serve.py                # FastAPI app for serving predictions
├── frontend.html           # Simple web UI for submitting claims
├── .gitignore              # Git ignore file
```

## How to Execute

### 1. Install Dependencies
Create a virtual environment and install required packages:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model
Run the training script to generate `fraud_model.pkl` and `encoder.pkl`:
```bash
python train.py
```

### 3. Start the Backend API
Run the FastAPI server:
```bash
python serve.py
```
The API will be available at [http://localhost:8000](http://localhost:8000)

### 4. Serve the Frontend
In a new terminal, start a simple HTTP server:
```bash
python -m http.server 8080
```
Open [http://localhost:8080/frontend.html](http://localhost:8080/frontend.html) in your browser.

### 5. Make Predictions
- Fill in the claim details in the web UI and click "Predict Fraud Likelihood".
- The result will show the fraud likelihood and an inference (fraudulent/genuine) based on the model's output.

---

**Note:**
- You can also test the API directly using curl or the Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs).
- Adjust the fraud threshold in `frontend.html` as needed for your use case.

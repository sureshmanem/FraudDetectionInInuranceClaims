"""
train.py - Model training and serialization for Fraud Detection
"""
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="`BaseEstimator._validate_data` is deprecated")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import shap

DATA_PATH = 'claims_data_old.csv'

# --- Preprocessing and Feature Engineering ---
def preprocess_data(df, label_col=None):
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if label_col and label_col in num_cols:
        num_cols.remove(label_col)
    imputer = SimpleImputer(strategy='mean')
    df[num_cols] = imputer.fit_transform(df[num_cols])
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if cat_cols:
        encoded = encoder.fit_transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
        df = pd.concat([df[num_cols], encoded_df], axis=1)
    else:
        df = df[num_cols]
    # After fitting encoder in preprocess_data, save it:
    joblib.dump(encoder, 'encoder.pkl')
    return df

def feature_engineering(df):
    if 'claim_amount' in df.columns and 'customer_avg_claim' in df.columns:
        df['claim_to_avg'] = df['claim_amount'] / (df['customer_avg_claim'] + 1)
    return df

# --- Modeling ---
def supervised_pipeline(df, label_col='is_fraud'):
    X = df.drop(label_col, axis=1)
    y = df[label_col]
    # Dynamically set k_neighbors for SMOTE based on smallest class size
    class_counts = y.value_counts()
    min_class_count = class_counts.min()
    k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1
    smote = SMOTE(k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred))
    print('ROC-AUC:', roc_auc_score(y_test, y_proba))
    return clf, X_test

def unsupervised_pipeline(df):
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly_score'] = iso.fit_predict(df)
    print(df['anomaly_score'].value_counts())
    return iso, df

def explain_model(clf, X_test):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    # For binary classification, shap_values is a list of 2 arrays
    if isinstance(shap_values, list) and len(shap_values) == 2:
        # Use the second array (class 1)
        values = shap_values[1]
    else:
        values = shap_values
    # Ensure shape matches
    if values.shape[1] != X_test.shape[1]:
        print(f"Warning: SHAP values shape {values.shape} does not match X_test shape {X_test.shape}. Skipping plot.")
        return
    shap.summary_plot(values, X_test)

def save_model(model, path='fraud_model.pkl'):
    joblib.dump(model, path)
    print(f'Model saved to {path}')

if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    label_col = 'is_fraud' if 'is_fraud' in df.columns else None
    if label_col:
        y = df[label_col]
        X = df.drop(label_col, axis=1)
        X = preprocess_data(X)
        X = feature_engineering(X)
        df = pd.concat([X, y], axis=1)
        model, X_test = supervised_pipeline(df, label_col)
        explain_model(model, X_test)
        save_model(model)
    else:
        df = preprocess_data(df)
        df = feature_engineering(df)
        model, df = unsupervised_pipeline(df)
        save_model(model, 'fraud_anomaly_model.pkl')

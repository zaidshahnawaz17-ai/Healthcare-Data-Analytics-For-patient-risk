"""
Model Training & Data Generation Pipeline for Healthcare Patient Risk Assessment.
Generates a realistic synthetic clinical dataset, trains a Machine Learning classifier
(Random Forest / Gradient Boosting), evaluates performance, and saves model artifacts.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import joblib

# Ensure target directories exist
OS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(OS_DIR, "models")
DATA_DIR = os.path.join(OS_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def generate_synthetic_clinical_data(n_samples=1500, seed=42):
    """
    Generate synthetic Electronic Health Record (EHR) data with realistic clinical risk relationships.
    """
    np.random.seed(seed)

    ages = np.random.randint(18, 92, size=n_samples)
    genders = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.48, 0.52])
    prior_inpatient_visits = np.random.negative_binomial(1, 0.4, size=n_samples)
    emergency_visits = np.random.negative_binomial(1, 0.5, size=n_samples)
    length_of_stay = np.random.randint(1, 18, size=n_samples)

    # Labs & Vitals
    glucose_level = np.random.normal(125, 40, size=n_samples).clip(70, 350)
    hba1c_level = np.random.normal(6.8, 1.8, size=n_samples).clip(4.5, 14.0)
    systolic_bp = np.random.normal(130, 20, size=n_samples).clip(90, 200)
    diastolic_bp = np.random.normal(82, 12, size=n_samples).clip(55, 120)
    heart_rate = np.random.normal(76, 14, size=n_samples).clip(50, 130)
    bmi = np.random.normal(28.5, 6.0, size=n_samples).clip(17.0, 50.0)

    # Comorbidities
    has_diabetes = ((hba1c_level > 6.5) | (np.random.rand(n_samples) < 0.25)).astype(int)
    has_hypertension = ((systolic_bp > 140) | (np.random.rand(n_samples) < 0.35)).astype(int)
    has_heart_failure = np.random.binomial(1, p=np.clip(0.18 + (ages > 65) * 0.15, 0, 1), size=n_samples)
    has_copd = np.random.binomial(1, p=np.clip(0.12 + (ages > 60) * 0.1, 0, 1), size=n_samples)
    has_ckd = np.random.binomial(1, p=np.clip(0.10 + (has_diabetes * 0.15), 0, 1), size=n_samples)
    has_asthma = np.random.binomial(1, p=0.14, size=n_samples)

    # Diagnoses & Discharge
    primary_diagnoses = np.random.choice(
        ['Cardiovascular', 'Endocrine', 'Respiratory', 'Renal', 'General Medicine'],
        size=n_samples, p=[0.30, 0.20, 0.22, 0.13, 0.15]
    )
    discharge_destinations = np.random.choice(
        ['Home', 'Home Health', 'SNF', 'Rehab'],
        size=n_samples, p=[0.55, 0.25, 0.13, 0.07]
    )

    # Calculate log-odds of 30-day readmission based on clinical domain knowledge
    log_odds = (
        -2.5
        + 0.02 * (ages - 50)
        + 0.45 * prior_inpatient_visits
        + 0.35 * emergency_visits
        + 0.06 * length_of_stay
        + 0.15 * (hba1c_level - 6.0)
        + 0.70 * has_heart_failure
        + 0.55 * has_ckd
        + 0.40 * has_copd
        + 0.30 * has_diabetes
        + 0.50 * (discharge_destinations == 'SNF')
        + 0.30 * (discharge_destinations == 'Home Health')
        + np.random.normal(0, 0.5, size=n_samples)
    )

    prob_readmission = 1 / (1 + np.exp(-log_odds))
    readmitted_30d = (prob_readmission > 0.42).astype(int)

    df = pd.DataFrame({
        'age': ages,
        'gender': genders,
        'prior_inpatient_visits': prior_inpatient_visits,
        'emergency_visits': emergency_visits,
        'length_of_stay': length_of_stay,
        'glucose_level': np.round(glucose_level, 1),
        'hba1c_level': np.round(hba1c_level, 1),
        'systolic_bp': np.round(systolic_bp).astype(int),
        'diastolic_bp': np.round(diastolic_bp).astype(int),
        'heart_rate': np.round(heart_rate).astype(int),
        'bmi': np.round(bmi, 1),
        'has_diabetes': has_diabetes,
        'has_hypertension': has_hypertension,
        'has_heart_failure': has_heart_failure,
        'has_copd': has_copd,
        'has_ckd': has_ckd,
        'has_asthma': has_asthma,
        'primary_diagnosis': primary_diagnoses,
        'discharge_destination': discharge_destinations,
        'readmitted_30d': readmitted_30d
    })

    return df


def build_and_train_pipeline(df):
    """
    Build automated preprocessing and compare Machine Learning classifiers.
    Selects best performing model based on ROC-AUC score.
    """
    X = df.drop(columns=['readmitted_30d'])
    y = df['readmitted_30d']

    numerical_cols = [
        'age', 'prior_inpatient_visits', 'emergency_visits', 'length_of_stay',
        'glucose_level', 'hba1c_level', 'systolic_bp', 'diastolic_bp',
        'heart_rate', 'bmi', 'has_diabetes', 'has_hypertension',
        'has_heart_failure', 'has_copd', 'has_ckd', 'has_asthma'
    ]
    categorical_cols = ['gender', 'primary_diagnosis', 'discharge_destination']

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ])

    # Candidate Models
    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=4,
            random_state=42,
            class_weight='balanced'
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=5,
            random_state=42
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_pipeline = None
    best_model_name = ""
    best_roc_auc = -1.0
    best_acc = -1.0
    best_cm = []
    best_feature_importances = []

    print("Training and evaluating candidate Healthcare Risk Classifiers...")

    for name, clf in candidates.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')

        print(f"\nModel: {name}")
        print(f"Accuracy: {acc * 100:.2f}% | ROC-AUC: {roc_auc:.4f} | 5-Fold CV AUC: {cv_scores.mean():.4f}")

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_acc = acc
            best_pipeline = pipeline
            best_model_name = name
            best_cm = confusion_matrix(y_test, y_pred).tolist()

    # Get feature names after one-hot encoding
    cat_encoder = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    encoded_cat_cols = list(cat_encoder.get_feature_names_out(categorical_cols))
    all_feature_names = numerical_cols + encoded_cat_cols

    importances = best_pipeline.named_steps['classifier'].feature_importances_
    sorted_importances = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(all_feature_names, importances)],
        key=lambda x: x["importance"],
        reverse=True
    )

    print(f"\n---> Selected Winner: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")

    # Save artifacts
    model_path = os.path.join(MODELS_DIR, "readmission_model.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"Saved trained model pipeline to: {model_path}")

    metadata = {
        "model_name": best_model_name,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "all_feature_names": all_feature_names,
        "accuracy": float(best_acc),
        "roc_auc": float(best_roc_auc),
        "confusion_matrix": best_cm,
        "feature_importances": sorted_importances,
        "total_train_samples": len(X_train),
        "total_test_samples": len(X_test)
    }

    metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    return best_pipeline, metadata


if __name__ == "__main__":
    dataset = generate_synthetic_clinical_data(n_samples=1500)
    csv_path = os.path.join(DATA_DIR, "historical_patients.csv")
    dataset.to_csv(csv_path, index=False)
    print(f"Exported synthetic patient dataset to: {csv_path}")

    build_and_train_pipeline(dataset)

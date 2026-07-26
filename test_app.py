"""
Verification Unit & Integration Tests for Healthcare Risk Analytics Application.
"""

import os
import pytest
from fastapi.testclient import TestClient

from train_model import generate_synthetic_clinical_data, build_and_train_pipeline
from model_pipeline import predict_patient_readmission_risk, load_model_artifacts
from database import init_db, fetch_all_patient_assessments, get_patient_history, delete_patient_record, save_patient_record, save_risk_assessment
from app import app

client = TestClient(app)


def test_synthetic_data_generation():
    """Verify synthetic clinical dataset creation."""
    df = generate_synthetic_clinical_data(n_samples=100)
    assert len(df) == 100
    assert 'readmitted_30d' in df.columns
    assert 'age' in df.columns
    assert 'hba1c_level' in df.columns
    assert 'primary_diagnosis' in df.columns


def test_model_training_and_artifacts():
    """Verify ML pipeline training, metrics, and artifact dumping."""
    df = generate_synthetic_clinical_data(n_samples=200)
    pipeline, metadata = build_and_train_pipeline(df)

    assert metadata['accuracy'] > 0.5
    assert metadata['roc_auc'] > 0.5
    assert len(metadata['feature_importances']) > 0
    assert os.path.exists("models/readmission_model.joblib")
    assert os.path.exists("models/metadata.json")


def test_inference_and_xai_pipeline():
    """Test real-time ML inference and XAI risk breakdown."""
    patient_high_risk = {
        "name": "Test High Risk",
        "age": 78,
        "gender": "Male",
        "prior_inpatient_visits": 5,
        "emergency_visits": 4,
        "length_of_stay": 10,
        "glucose_level": 260.0,
        "hba1c_level": 11.2,
        "systolic_bp": 165,
        "diastolic_bp": 95,
        "heart_rate": 90,
        "bmi": 35.0,
        "has_diabetes": 1,
        "has_hypertension": 1,
        "has_heart_failure": 1,
        "has_copd": 1,
        "has_ckd": 1,
        "has_asthma": 0,
        "primary_diagnosis": "Cardiovascular",
        "discharge_destination": "SNF"
    }

    result = predict_patient_readmission_risk(patient_high_risk)

    assert result['risk_score'] > 40.0
    assert result['risk_category'] in ['High Risk', 'Medium Risk']
    assert len(result['top_risk_drivers']) > 0
    assert len(result['clinical_recommendations']) > 0


def test_database_history_and_deletion():
    """Test patient trajectory history retrieval and deletion."""
    init_db()
    p_id = "PAT-TEST-999"
    patient_data = {
        'patient_id': p_id,
        'name': "History Test Patient",
        'age': 60,
        'gender': 'Male',
        'prior_inpatient_visits': 1,
        'emergency_visits': 1,
        'length_of_stay': 3,
        'glucose_level': 110.0,
        'hba1c_level': 6.0,
        'systolic_bp': 120,
        'diastolic_bp': 80,
        'heart_rate': 72,
        'bmi': 25.0,
        'has_diabetes': 0,
        'has_hypertension': 0,
        'has_heart_failure': 0,
        'has_copd': 0,
        'has_ckd': 0,
        'has_asthma': 0,
        'primary_diagnosis': 'General Medicine',
        'discharge_destination': 'Home'
    }

    save_patient_record(patient_data)
    save_risk_assessment(p_id, 45.0, 'Medium Risk', [{'feature': 'Age'}], [], ['Follow up'])
    save_risk_assessment(p_id, 25.0, 'Low Risk', [], [{'feature': 'Glucose'}], ['Routine'])

    history = get_patient_history(p_id)
    assert len(history) == 2
    assert history[0]['risk_score'] == 45.0
    assert history[1]['risk_score'] == 25.0

    delete_patient_record(p_id)
    history_after = get_patient_history(p_id)
    assert len(history_after) == 0


def test_api_endpoints():
    """Integration test for FastAPI endpoints."""
    # 1. Test Root
    res_root = client.get("/")
    assert res_root.status_code == 200

    # 2. Test Predict API
    sample_patient = {
        "name": "API Test Patient",
        "age": 55,
        "gender": "Female",
        "prior_inpatient_visits": 1,
        "emergency_visits": 1,
        "length_of_stay": 3,
        "glucose_level": 120.0,
        "hba1c_level": 6.2,
        "systolic_bp": 125,
        "diastolic_bp": 80,
        "heart_rate": 72,
        "bmi": 24.5,
        "has_diabetes": 0,
        "has_hypertension": 1,
        "has_heart_failure": 0,
        "has_copd": 0,
        "has_ckd": 0,
        "has_asthma": 0,
        "primary_diagnosis": "General Medicine",
        "discharge_destination": "Home"
    }

    res_pred = client.post("/api/predict", json=sample_patient)
    assert res_pred.status_code == 200
    data = res_pred.json()
    assert 'risk_score' in data
    assert 'risk_category' in data
    assert 'top_risk_drivers' in data

    created_id = data['patient_id']

    # 3. Test Patient History Endpoint
    res_hist = client.get(f"/api/patient/{created_id}/history")
    assert res_hist.status_code == 200
    assert isinstance(res_hist.json(), list)

    # 4. Test Analytics API
    res_analytics = client.get("/api/analytics")
    assert res_analytics.status_code == 200
    analytics_data = res_analytics.json()
    assert analytics_data['total_patients'] > 0

    # 5. Test Patients Registry API
    res_patients = client.get("/api/patients")
    assert res_patients.status_code == 200
    assert isinstance(res_patients.json(), list)

    # 6. Test Model Info Endpoint
    res_model = client.get("/api/model/info")
    assert res_model.status_code == 200
    assert 'roc_auc' in res_model.json()

    # 7. Test Export CSV Endpoint
    res_export = client.get("/api/patients/export")
    assert res_export.status_code == 200
    assert "text/csv" in res_export.headers.get("content-type", "")

    # 8. Test Preset Profiles
    res_preset = client.get("/api/preset-patient/high_risk_diabetic")
    assert res_preset.status_code == 200
    assert res_preset.json()['name'] == "Eleanor Vance"

    # 9. Test Batch Prediction
    res_batch = client.post("/api/predict/batch", json=[sample_patient])
    assert res_batch.status_code == 200
    assert res_batch.json()['processed_count'] == 1

    # 10. Test Delete Endpoint
    res_del = client.delete(f"/api/patients/{created_id}")
    assert res_del.status_code == 200


if __name__ == "__main__":
    pytest.main(["-v", __file__])

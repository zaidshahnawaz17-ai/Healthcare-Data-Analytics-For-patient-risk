"""
PulseRisk AI Healthcare Data Analytics & Patient Risk Assessment FastAPI Backend Application.
"""

import os
import io
import uuid
import pandas as pd
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse
from pydantic import BaseModel, Field

from database import (
    init_db, save_patient_record, save_risk_assessment,
    fetch_all_patient_assessments, get_patient_history, delete_patient_record,
    seed_baseline_data_if_empty, get_db_connection
)
from model_pipeline import predict_patient_readmission_risk, load_model_artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager replacing deprecated on_event handlers."""
    init_db()
    load_model_artifacts()
    seed_baseline_data_if_empty(predict_patient_readmission_risk)
    yield


# App initialization
app = FastAPI(
    title="PulseRisk AI Healthcare Risk Analytics API",
    description="30-Day Hospital Readmission Risk Assessment & Clinical Decision Support Engine",
    version="1.1.0",
    lifespan=lifespan
)

OS_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(OS_DIR, "static")

# Mount static files
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Pydantic schema for Patient Data Input
class PatientInputSchema(BaseModel):
    patient_id: Optional[str] = None
    name: Optional[str] = "Anonymous Patient"
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    gender: str = Field(..., description="Male, Female, or Other")
    prior_inpatient_visits: int = Field(0, ge=0, description="Number of inpatient stays in past 12 months")
    emergency_visits: int = Field(0, ge=0, description="Number of ED visits in past 12 months")
    length_of_stay: int = Field(1, ge=1, description="Index hospitalization length of stay (days)")
    glucose_level: float = Field(100.0, ge=40.0, le=500.0, description="Blood Glucose (mg/dL)")
    hba1c_level: float = Field(5.7, ge=3.5, le=18.0, description="Glycated Hemoglobin (%)")
    systolic_bp: int = Field(120, ge=70, le=240, description="Systolic Blood Pressure (mmHg)")
    diastolic_bp: int = Field(80, ge=40, le=140, description="Diastolic Blood Pressure (mmHg)")
    heart_rate: int = Field(75, ge=30, le=200, description="Heart Rate (bpm)")
    bmi: float = Field(25.0, ge=10.0, le=65.0, description="Body Mass Index")
    has_diabetes: int = Field(0, ge=0, le=1)
    has_hypertension: int = Field(0, ge=0, le=1)
    has_heart_failure: int = Field(0, ge=0, le=1)
    has_copd: int = Field(0, ge=0, le=1)
    has_ckd: int = Field(0, ge=0, le=1)
    has_asthma: int = Field(0, ge=0, le=1)
    primary_diagnosis: str = Field("General Medicine", description="Cardiovascular, Endocrine, Respiratory, Renal, General Medicine")
    discharge_destination: str = Field("Home", description="Home, Home Health, SNF, Rehab")


@app.get("/")
def read_root():
    """Serve main web application index.html."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Index HTML not found.")
    return FileResponse(index_path)


@app.post("/api/predict")
def predict_patient_risk(patient: PatientInputSchema):
    """
    Process patient data, compute ML Readmission Risk %,
    generate Explainable AI (XAI) feature breakdown, and save record into SQLite.
    """
    patient_dict = patient.model_dump()

    if not patient_dict.get('patient_id'):
        patient_dict['patient_id'] = f"PAT-{uuid.uuid4().hex[:6].upper()}"

    results = predict_patient_readmission_risk(patient_dict)

    save_patient_record(patient_dict)
    save_risk_assessment(
        patient_dict['patient_id'],
        results['risk_score'],
        results['risk_category'],
        results['top_risk_drivers'],
        results['top_protective_factors'],
        results['clinical_recommendations']
    )

    results['patient_id'] = patient_dict['patient_id']
    results['patient_name'] = patient_dict['name']
    results['patient_data'] = patient_dict

    return results


@app.post("/api/predict/batch")
def predict_batch_patients(patients: List[PatientInputSchema]):
    """Process batch prediction for multiple patient records."""
    results_list = []
    for p in patients:
        p_dict = p.model_dump()
        if not p_dict.get('patient_id'):
            p_dict['patient_id'] = f"PAT-{uuid.uuid4().hex[:6].upper()}"

        res = predict_patient_readmission_risk(p_dict)
        save_patient_record(p_dict)
        save_risk_assessment(
            p_dict['patient_id'],
            res['risk_score'],
            res['risk_category'],
            res['top_risk_drivers'],
            res['top_protective_factors'],
            res['clinical_recommendations']
        )
        res['patient_id'] = p_dict['patient_id']
        res['patient_name'] = p_dict['name']
        results_list.append(res)

    return {"processed_count": len(results_list), "results": results_list}


@app.get("/api/analytics")
def get_population_analytics():
    """
    Return population-level analytics for interactive visual Plotly charts.
    """
    df = fetch_all_patient_assessments()

    if df.empty:
        return {
            "total_patients": 0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "avg_risk_score": 0.0,
            "risk_tier_distribution": {"High Risk": 0, "Medium Risk": 0, "Low Risk": 0},
            "age_group_risk": [],
            "prior_visits_correlation": [],
            "comorbidity_impact": [],
            "diagnosis_breakdown": []
        }

    total_patients = len(df)
    high_risk_count = int((df['risk_category'] == 'High Risk').sum())
    medium_risk_count = int((df['risk_category'] == 'Medium Risk').sum())
    low_risk_count = int((df['risk_category'] == 'Low Risk').sum())
    avg_risk_score = round(float(df['risk_score'].mean()), 1)

    tier_dist = {
        "High Risk": high_risk_count,
        "Medium Risk": medium_risk_count,
        "Low Risk": low_risk_count
    }

    # 1. Age Group Risk Breakdown
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 45, 60, 75, 100],
        labels=['<45', '45-59', '60-74', '75+']
    )
    age_risk = df.groupby('age_group', observed=False)['risk_score'].agg(['mean', 'count']).reset_index()
    age_risk_data = [
        {
            "age_group": str(row['age_group']),
            "avg_risk_score": round(float(row['mean']), 1) if not pd.isna(row['mean']) else 0.0,
            "patient_count": int(row['count'])
        }
        for _, row in age_risk.iterrows()
    ]

    # 2. Prior Visits correlation
    visit_corr = df.groupby('prior_inpatient_visits')['risk_score'].mean().reset_index()
    visit_corr_data = [
        {"prior_visits": int(row['prior_inpatient_visits']), "avg_risk_score": round(float(row['risk_score']), 1)}
        for _, row in visit_corr.iterrows()
    ]

    # 3. Comorbidity Impact Breakdown
    comorbidities = {
        "Heart Failure": "has_heart_failure",
        "Kidney Disease (CKD)": "has_ckd",
        "COPD": "has_copd",
        "Diabetes": "has_diabetes",
        "Hypertension": "has_hypertension",
        "Asthma": "has_asthma"
    }

    comorbidity_data = []
    for label, col in comorbidities.items():
        if col in df.columns:
            present_avg = df[df[col] == 1]['risk_score'].mean()
            absent_avg = df[df[col] == 0]['risk_score'].mean()
            comorbidity_data.append({
                "condition": label,
                "present_avg_risk": round(float(present_avg), 1) if not pd.isna(present_avg) else 0.0,
                "absent_avg_risk": round(float(absent_avg), 1) if not pd.isna(absent_avg) else 0.0,
                "count": int((df[col] == 1).sum())
            })

    comorbidity_data = sorted(comorbidity_data, key=lambda x: x['present_avg_risk'], reverse=True)

    # 4. Primary Diagnosis breakdown
    diag_df = df.groupby('primary_diagnosis')['risk_score'].agg(['mean', 'count']).reset_index()
    diag_data = [
        {
            "diagnosis": str(row['primary_diagnosis']),
            "avg_risk": round(float(row['mean']), 1) if not pd.isna(row['mean']) else 0.0,
            "count": int(row['count'])
        }
        for _, row in diag_df.iterrows()
    ]

    return {
        "total_patients": total_patients,
        "high_risk_count": high_risk_count,
        "medium_risk_count": medium_risk_count,
        "low_risk_count": low_risk_count,
        "avg_risk_score": avg_risk_score,
        "risk_tier_distribution": tier_dist,
        "age_group_risk": age_risk_data,
        "prior_visits_correlation": visit_corr_data,
        "comorbidity_impact": comorbidity_data,
        "diagnosis_breakdown": diag_data
    }


@app.get("/api/patients")
def get_patients_list(
    risk_category: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100
):
    """Retrieve list of assessed patients with optional risk tier filter or search term."""
    df = fetch_all_patient_assessments()
    if df.empty:
        return []

    if risk_category and risk_category != 'All':
        df = df[df['risk_category'] == risk_category]

    if search:
        search_lower = search.lower()
        df = df[
            df['name'].str.lower().str.contains(search_lower, na=False) |
            df['patient_id'].str.lower().str.contains(search_lower, na=False) |
            df['primary_diagnosis'].str.lower().str.contains(search_lower, na=False)
        ]

    df = df.head(limit)
    records = df.to_dict(orient='records')
    return records


@app.get("/api/patients/export")
def export_patients_csv():
    """Export all assessed patients as downloadable CSV file."""
    df = fetch_all_patient_assessments()
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=pulse_risk_patients_export.csv"}
    )


@app.get("/api/patient/{patient_id}/history")
def get_patient_risk_history(patient_id: str):
    """Retrieve all historical risk assessments over time for a patient."""
    history = get_patient_history(patient_id)
    if not history:
        raise HTTPException(status_code=404, detail="Patient history not found.")
    return history


@app.delete("/api/patients/{patient_id}")
def delete_patient(patient_id: str):
    """Delete a patient record and assessment history."""
    delete_patient_record(patient_id)
    return {"message": f"Patient {patient_id} deleted successfully."}


@app.get("/api/model/info")
def get_model_information():
    """Return ML model pipeline metadata, metrics, and feature importances."""
    _, metadata = load_model_artifacts()
    return metadata


@app.get("/api/preset-patient/{preset_type}")
def get_preset_patient_profile(preset_type: str):
    """Provide pre-filled clinical patient profiles for demonstration."""
    presets = {
        "high_risk_diabetic": {
            "name": "Eleanor Vance",
            "age": 74,
            "gender": "Female",
            "prior_inpatient_visits": 4,
            "emergency_visits": 3,
            "length_of_stay": 8,
            "glucose_level": 245.0,
            "hba1c_level": 10.4,
            "systolic_bp": 158,
            "diastolic_bp": 92,
            "heart_rate": 88,
            "bmi": 34.2,
            "has_diabetes": 1,
            "has_hypertension": 1,
            "has_heart_failure": 1,
            "has_copd": 0,
            "has_ckd": 1,
            "has_asthma": 0,
            "primary_diagnosis": "Endocrine",
            "discharge_destination": "SNF"
        },
        "moderate_risk_copd": {
            "name": "Arthur Pendelton",
            "age": 63,
            "gender": "Male",
            "prior_inpatient_visits": 1,
            "emergency_visits": 2,
            "length_of_stay": 4,
            "glucose_level": 130.0,
            "hba1c_level": 6.8,
            "systolic_bp": 136,
            "diastolic_bp": 84,
            "heart_rate": 82,
            "bmi": 27.5,
            "has_diabetes": 0,
            "has_hypertension": 1,
            "has_heart_failure": 0,
            "has_copd": 1,
            "has_ckd": 0,
            "has_asthma": 1,
            "primary_diagnosis": "Respiratory",
            "discharge_destination": "Home Health"
        },
        "low_risk_routine": {
            "name": "Clara Oswald",
            "age": 34,
            "gender": "Female",
            "prior_inpatient_visits": 0,
            "emergency_visits": 0,
            "length_of_stay": 2,
            "glucose_level": 95.0,
            "hba1c_level": 5.4,
            "systolic_bp": 118,
            "diastolic_bp": 76,
            "heart_rate": 70,
            "bmi": 22.8,
            "has_diabetes": 0,
            "has_hypertension": 0,
            "has_heart_failure": 0,
            "has_copd": 0,
            "has_ckd": 0,
            "has_asthma": 0,
            "primary_diagnosis": "General Medicine",
            "discharge_destination": "Home"
        },
        "elderly_heart_failure": {
            "name": "Harold Montgomery",
            "age": 82,
            "gender": "Male",
            "prior_inpatient_visits": 3,
            "emergency_visits": 2,
            "length_of_stay": 6,
            "glucose_level": 165.0,
            "hba1c_level": 7.2,
            "systolic_bp": 162,
            "diastolic_bp": 94,
            "heart_rate": 92,
            "bmi": 31.0,
            "has_diabetes": 1,
            "has_hypertension": 1,
            "has_heart_failure": 1,
            "has_copd": 1,
            "has_ckd": 0,
            "has_asthma": 0,
            "primary_diagnosis": "Cardiovascular",
            "discharge_destination": "SNF"
        },
        "complex_renal": {
            "name": "Beatrice Sterling",
            "age": 69,
            "gender": "Female",
            "prior_inpatient_visits": 2,
            "emergency_visits": 1,
            "length_of_stay": 7,
            "glucose_level": 190.0,
            "hba1c_level": 8.5,
            "systolic_bp": 148,
            "diastolic_bp": 88,
            "heart_rate": 80,
            "bmi": 29.2,
            "has_diabetes": 1,
            "has_hypertension": 1,
            "has_heart_failure": 0,
            "has_copd": 0,
            "has_ckd": 1,
            "has_asthma": 0,
            "primary_diagnosis": "Renal",
            "discharge_destination": "Home Health"
        }
    }

    if preset_type not in presets:
        raise HTTPException(status_code=404, detail="Preset profile not found.")

    return presets[preset_type]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

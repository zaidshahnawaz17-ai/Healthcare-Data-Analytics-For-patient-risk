"""
SQLite Database Layer for Healthcare Risk Analytics.
Manages patient demographic & clinical record persistence, assessment history,
patient risk trajectory tracking over time, and synthetic baseline population seeding.
"""

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime

OS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OS_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "healthcare_analytics.db")


def get_db_connection():
    """Create and return a SQLite database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables for patients and risk assessments."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        name TEXT,
        age INTEGER,
        gender TEXT,
        prior_inpatient_visits INTEGER,
        emergency_visits INTEGER,
        length_of_stay INTEGER,
        glucose_level REAL,
        hba1c_level REAL,
        systolic_bp INTEGER,
        diastolic_bp INTEGER,
        heart_rate INTEGER,
        bmi REAL,
        has_diabetes INTEGER,
        has_hypertension INTEGER,
        has_heart_failure INTEGER,
        has_copd INTEGER,
        has_ckd INTEGER,
        has_asthma INTEGER,
        primary_diagnosis TEXT,
        discharge_destination TEXT,
        created_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS risk_assessments (
        assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        risk_score REAL,
        risk_category TEXT,
        top_risk_drivers TEXT,
        top_protective_factors TEXT,
        clinical_recommendations TEXT,
        assessed_at TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id) ON DELETE CASCADE
    )
    """)

    conn.commit()
    conn.close()


def save_patient_record(patient_dict):
    """Save or update a patient record in SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()

    created_at = patient_dict.get('created_at', datetime.now().isoformat())

    cursor.execute("""
    INSERT OR REPLACE INTO patients (
        patient_id, name, age, gender, prior_inpatient_visits, emergency_visits,
        length_of_stay, glucose_level, hba1c_level, systolic_bp, diastolic_bp,
        heart_rate, bmi, has_diabetes, has_hypertension, has_heart_failure,
        has_copd, has_ckd, has_asthma, primary_diagnosis, discharge_destination, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_dict['patient_id'],
        patient_dict.get('name', f"Patient-{patient_dict['patient_id']}"),
        int(patient_dict['age']),
        patient_dict['gender'],
        int(patient_dict['prior_inpatient_visits']),
        int(patient_dict['emergency_visits']),
        int(patient_dict['length_of_stay']),
        float(patient_dict['glucose_level']),
        float(patient_dict['hba1c_level']),
        int(patient_dict['systolic_bp']),
        int(patient_dict['diastolic_bp']),
        int(patient_dict['heart_rate']),
        float(patient_dict['bmi']),
        int(patient_dict['has_diabetes']),
        int(patient_dict['has_hypertension']),
        int(patient_dict['has_heart_failure']),
        int(patient_dict['has_copd']),
        int(patient_dict['has_ckd']),
        int(patient_dict['has_asthma']),
        patient_dict['primary_diagnosis'],
        patient_dict['discharge_destination'],
        created_at
    ))

    conn.commit()
    conn.close()


def save_risk_assessment(patient_id, risk_score, risk_category, drivers, protective, recommendations, timestamp=None):
    """Save risk assessment results for a patient."""
    conn = get_db_connection()
    cursor = conn.cursor()
    assessed_at = timestamp if timestamp else datetime.now().isoformat()

    cursor.execute("""
    INSERT INTO risk_assessments (
        patient_id, risk_score, risk_category, top_risk_drivers,
        top_protective_factors, clinical_recommendations, assessed_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id,
        float(risk_score),
        risk_category,
        json.dumps(drivers),
        json.dumps(protective),
        json.dumps(recommendations),
        assessed_at
    ))

    conn.commit()
    conn.close()


def fetch_all_patient_assessments():
    """Retrieve full merged patient & latest assessment records for dashboard analytics."""
    conn = get_db_connection()
    query = """
    SELECT 
        p.*,
        r.risk_score,
        r.risk_category,
        r.top_risk_drivers,
        r.top_protective_factors,
        r.clinical_recommendations,
        r.assessed_at
    FROM patients p
    JOIN (
        SELECT patient_id, max(assessment_id) as max_id
        FROM risk_assessments
        GROUP BY patient_id
    ) latest ON p.patient_id = latest.patient_id
    JOIN risk_assessments r ON latest.max_id = r.assessment_id
    ORDER BY r.assessed_at DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_patient_history(patient_id: str):
    """Retrieve historical risk assessments over time for a patient."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT assessment_id, patient_id, risk_score, risk_category, 
               top_risk_drivers, top_protective_factors, clinical_recommendations, assessed_at
        FROM risk_assessments
        WHERE patient_id = ?
        ORDER BY assessed_at ASC
    """, (patient_id,))
    rows = cursor.fetchall()
    conn.close()

    history = []
    for r in rows:
        history.append({
            "assessment_id": r["assessment_id"],
            "patient_id": r["patient_id"],
            "risk_score": r["risk_score"],
            "risk_category": r["risk_category"],
            "top_risk_drivers": json.loads(r["top_risk_drivers"]) if r["top_risk_drivers"] else [],
            "top_protective_factors": json.loads(r["top_protective_factors"]) if r["top_protective_factors"] else [],
            "clinical_recommendations": json.loads(r["clinical_recommendations"]) if r["clinical_recommendations"] else [],
            "assessed_at": r["assessed_at"]
        })
    return history


def delete_patient_record(patient_id: str):
    """Delete patient and associated risk assessments."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM risk_assessments WHERE patient_id = ?", (patient_id,))
    cursor.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
    conn.commit()
    conn.close()


def seed_baseline_data_if_empty(predict_func):
    """Seed DB with synthetic baseline data if patient table is empty."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM patients")
    row_count = cursor.fetchone()['cnt']
    conn.close()

    if row_count > 0:
        return

    print("Database empty. Seeding historical patient data for population analytics...")
    csv_path = os.path.join(DATA_DIR, "historical_patients.csv")
    if not os.path.exists(csv_path):
        from train_model import generate_synthetic_clinical_data, build_and_train_pipeline
        df = generate_synthetic_clinical_data(1500)
        df.to_csv(csv_path, index=False)
        build_and_train_pipeline(df)

    raw_df = pd.read_csv(csv_path).head(150)
    name_list = [
        "John Doe", "Jane Smith", "Robert Johnson", "Emily Davis", "Michael Brown",
        "Sarah Miller", "David Wilson", "Taylor Anderson", "James Thomas", "Patricia Jackson",
        "Christopher White", "Amanda Harris", "Matthew Martin", "Olivia Thompson", "Daniel Garcia"
    ]

    for idx, row in raw_df.iterrows():
        p_id = f"PAT-{1000 + idx}"
        p_name = f"{name_list[idx % len(name_list)]}"

        patient_data = {
            'patient_id': p_id,
            'name': p_name,
            'age': int(row['age']),
            'gender': str(row['gender']),
            'prior_inpatient_visits': int(row['prior_inpatient_visits']),
            'emergency_visits': int(row['emergency_visits']),
            'length_of_stay': int(row['length_of_stay']),
            'glucose_level': float(row['glucose_level']),
            'hba1c_level': float(row['hba1c_level']),
            'systolic_bp': int(row['systolic_bp']),
            'diastolic_bp': int(row['diastolic_bp']),
            'heart_rate': int(row['heart_rate']),
            'bmi': float(row['bmi']),
            'has_diabetes': int(row['has_diabetes']),
            'has_hypertension': int(row['has_hypertension']),
            'has_heart_failure': int(row['has_heart_failure']),
            'has_copd': int(row['has_copd']),
            'has_ckd': int(row['has_ckd']),
            'has_asthma': int(row['has_asthma']),
            'primary_diagnosis': str(row['primary_diagnosis']),
            'discharge_destination': str(row['discharge_destination'])
        }

        # Run inference
        result = predict_func(patient_data)
        save_patient_record(patient_data)
        save_risk_assessment(
            p_id,
            result['risk_score'],
            result['risk_category'],
            result['top_risk_drivers'],
            result['top_protective_factors'],
            result['clinical_recommendations']
        )

    print("Baseline population database seeded successfully.")

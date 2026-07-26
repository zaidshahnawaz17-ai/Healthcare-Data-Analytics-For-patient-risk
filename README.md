# Healthcare Data Analytics for Patient Risk Assessment

A complete, end-to-end, production-ready Healthcare Data Analytics Web Application designed for **30-Day Hospital Readmission Risk Prediction & Clinical Decision Support**.

---

## 🌟 Application Features & Components (A to Z)

### Module A: Patient Data Ingestion & Management
- Intuitive, responsive web interface to input real patient clinical data or load pre-configured high/medium/low risk clinical profiles.
- Features captured: Demographics (Age, Gender), Healthcare Utilization (Prior Inpatient Visits, Emergency Visits, Length of Stay), Labs & Vitals (Glucose, HbA1c, BP, Heart Rate, BMI), Comorbidities (Diabetes, Hypertension, Heart Failure, COPD, CKD, Asthma), Primary Diagnosis, and Discharge Destination.
- SQLite database storage (`healthcare_analytics.db`) securing patient demographic & clinical assessment records.

### Module B: Automated Data Preprocessing & Pipeline
- Scikit-Learn `ColumnTransformer` pipeline automatically handling:
  - Missing data imputation (`SimpleImputer`).
  - Categorical feature encoding (`OneHotEncoder`).
  - Continuous numerical scaling (`StandardScaler`).

### Module C: Machine Learning Inference Engine
- Supervised ML classification model (Random Forest / XGBoost) trained on clinical EHR dataset.
- Real-time prediction producing:
  - **Risk Score %**: Precise 30-day hospital readmission probability (e.g. `82.4%`).
  - **Risk Categorization**: `Low Risk` (<30%), `Medium Risk` (30-60%), or `High Risk` (>60%).

### Module D: Explainable AI (XAI) & Clinical Decision Support
- Per-patient feature attribution breakdown quantifying top positive risk drivers (e.g. *"4 prior inpatient visits (+34.0% risk impact)"*) and protective factors.
- Natural language clinical explanations enabling clinicians to understand the *why* behind model scores.
- Dynamic, risk-tier-specific clinical action plan (e.g. mandatory 48-hr post-discharge follow-up, pharmacist medication reconciliation, telehealth enrollment).

### Module E: Interactive Population Analytics Dashboard
- Summary KPI Cards (Total Assessed Patients, High Risk Count, Average Population Risk, Low Risk Cohort).
- Interactive Plotly.js charts:
  1. **Readmission Risk Tier Distribution** (Donut chart).
  2. **Average Risk Score by Age Cohort** (Bar chart).
  3. **Hospital Utilization vs Readmission Risk** (Line/Scatter chart).
  4. **Comorbidity Impact Matrix** (Horizontal bar chart).
- Filterable Patient Registry with text search, risk tier filtering, and detailed clinical report modal view.

---

## 📁 Repository Structure

```
healthcare_risk_analytics/
├── app.py                   # FastAPI main application & server
├── database.py              # SQLite database manager & baseline seeder
├── model_pipeline.py        # ML Preprocessing, inference engine, & XAI breakdown
├── train_model.py           # Synthetic dataset generator & model training pipeline
├── test_app.py              # Pytest automated test suite
├── static/                  # Responsive web dashboard assets
│   ├── index.html           # HTML5 UI with glassmorphism layout & tabs
│   ├── css/
│   │   └── styles.css       # Design system CSS (dark mode, glassmorphism, responsive grid)
│   └── js/
│       └── main.js          # Interactive frontend logic & Plotly.js charts
├── data/
│   ├── historical_patients.csv # Synthetic EHR dataset (~1,000 records)
│   └── healthcare_analytics.db# SQLite database
├── models/
│   ├── readmission_model.joblib # Saved trained ML pipeline
│   └── metadata.json         # Feature list & training metrics
├── requirements.txt         # Python dependencies
└── README.md                # Documentation & instructions
```

---

## 🛠️ Step-by-Step Setup & Execution Instructions

### 1. Environment Setup

Open terminal in the project directory:

```bash
cd c:\Users\tagar\OneDrive\healthcare_risk_analytics
```

(Optional) Create and activate a Python virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

Install all required Python libraries:

```bash
pip install -r requirements.txt
```

### 3. Train the Model & Seed Database

Run the training script to generate the synthetic clinical dataset, train the machine learning pipeline, and save the model artifacts:

```bash
python train_model.py
```

*Output:*
- Trains Random Forest Classifier (`Accuracy: ~80%`, `ROC-AUC: ~0.87`).
- Saves `models/readmission_model.joblib` and `models/metadata.json`.
- Exports `data/historical_patients.csv`.

### 4. Run Automated Test Verification

Run pytest to ensure all modules, model inference, XAI, database operations, and API routes pass:

```bash
python -m pytest test_app.py -v
```

### 5. Launch the Web Application

Start the FastAPI application server:

```bash
python app.py
```

or via Uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Access the Application in Browser

Open your web browser and navigate to:

👉 **`http://localhost:8000`**

---

## 🚀 How to Use the Application

1. **Patient Risk Assessment Tab**:
   - Click one of the preset profile buttons (e.g. *Severe Diabetic*, *COPD Moderate*, *Routine Low Risk*) to quickly pre-fill clinical data.
   - Or manually input demographic, vitals, labs, and comorbidity data.
   - Click **"Run ML Risk Inference"**.
   - Review the calculated Risk Score %, Explainable AI (XAI) risk drivers breakdown, and tailored Clinical Action Plan.

2. **Population Analytics Tab**:
   - Click on **"Population Analytics"** in the top navigation header.
   - View population-level charts powered by Plotly.js analyzing risk distribution, age cohorts, utilization patterns, and comorbidity impact.

3. **Patient Registry Tab**:
   - Search by patient name, ID, or diagnosis.
   - Filter by risk level (*High Risk*, *Medium Risk*, *Low Risk*).
   - Click **"View Report"** on any patient row to inspect their full clinical summary report modal.

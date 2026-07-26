// PulseRisk AI Healthcare Analytics & Patient Risk Assessment Engine JavaScript

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    fetchKPIsAndAnalytics();
    fetchPatientRegistry();
});

// Global state variables
let registryData = [];
let selectedBatchCSV = null;
let currentAssessmentResult = null;

// Tab Navigation logic
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetTab = btn.getAttribute('data-tab');
            document.getElementById(targetTab).classList.add('active');

            if (targetTab === 'tab-analytics') {
                renderPopulationCharts();
            } else if (targetTab === 'tab-registry') {
                fetchPatientRegistry();
            }
        });
    });
}

// Load preset patient profile
async function loadPreset(presetType) {
    try {
        const response = await fetch(`/api/preset-patient/${presetType}`);
        if (!response.ok) throw new Error('Failed to fetch preset profile');
        const data = await response.json();

        document.getElementById('patient_name').value = data.name;
        document.getElementById('age').value = data.age;
        document.getElementById('gender').value = data.gender;
        document.getElementById('prior_inpatient_visits').value = data.prior_inpatient_visits;
        document.getElementById('emergency_visits').value = data.emergency_visits;
        document.getElementById('length_of_stay').value = data.length_of_stay;
        document.getElementById('glucose_level').value = data.glucose_level;
        document.getElementById('hba1c_level').value = data.hba1c_level;
        document.getElementById('systolic_bp').value = data.systolic_bp;
        document.getElementById('diastolic_bp').value = data.diastolic_bp;
        document.getElementById('heart_rate').value = data.heart_rate;
        document.getElementById('bmi').value = data.bmi;

        document.getElementById('has_diabetes').checked = data.has_diabetes === 1;
        document.getElementById('has_hypertension').checked = data.has_hypertension === 1;
        document.getElementById('has_heart_failure').checked = data.has_heart_failure === 1;
        document.getElementById('has_copd').checked = data.has_copd === 1;
        document.getElementById('has_ckd').checked = data.has_ckd === 1;
        document.getElementById('has_asthma').checked = data.has_asthma === 1;

        document.getElementById('primary_diagnosis').value = data.primary_diagnosis;
        document.getElementById('discharge_destination').value = data.discharge_destination;

    } catch (err) {
        console.error("Error loading preset:", err);
    }
}

// Submit Patient Data for ML Assessment
async function submitAssessment(event) {
    event.preventDefault();

    const btnEvaluate = document.getElementById('btn-evaluate');
    btnEvaluate.disabled = true;
    btnEvaluate.innerHTML = `<span class="spinner"></span> Computing Risk Inference...`;

    const payload = {
        name: document.getElementById('patient_name').value,
        age: parseInt(document.getElementById('age').value),
        gender: document.getElementById('gender').value,
        prior_inpatient_visits: parseInt(document.getElementById('prior_inpatient_visits').value),
        emergency_visits: parseInt(document.getElementById('emergency_visits').value),
        length_of_stay: parseInt(document.getElementById('length_of_stay').value),
        glucose_level: parseFloat(document.getElementById('glucose_level').value),
        hba1c_level: parseFloat(document.getElementById('hba1c_level').value),
        systolic_bp: parseInt(document.getElementById('systolic_bp').value),
        diastolic_bp: parseInt(document.getElementById('diastolic_bp').value),
        heart_rate: parseInt(document.getElementById('heart_rate').value),
        bmi: parseFloat(document.getElementById('bmi').value),
        has_diabetes: document.getElementById('has_diabetes').checked ? 1 : 0,
        has_hypertension: document.getElementById('has_hypertension').checked ? 1 : 0,
        has_heart_failure: document.getElementById('has_heart_failure').checked ? 1 : 0,
        has_copd: document.getElementById('has_copd').checked ? 1 : 0,
        has_ckd: document.getElementById('has_ckd').checked ? 1 : 0,
        has_asthma: document.getElementById('has_asthma').checked ? 1 : 0,
        primary_diagnosis: document.getElementById('primary_diagnosis').value,
        discharge_destination: document.getElementById('discharge_destination').value
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("Inference execution failed.");
        const result = await response.json();

        currentAssessmentResult = result;
        renderAssessmentResults(result);
        fetchKPIsAndAnalytics(); // Refresh KPIs

    } catch (err) {
        alert("Error executing model assessment: " + err.message);
    } finally {
        btnEvaluate.disabled = false;
        btnEvaluate.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            Run ML Risk Inference
        `;
    }
}

// Render Results & XAI Breakdown
function renderAssessmentResults(data) {
    document.getElementById('empty-state').style.display = 'none';
    document.getElementById('results-content').style.display = 'block';
    document.getElementById('btn-print-report').style.display = 'inline-flex';

    const riskScore = data.risk_score;
    const category = data.risk_category;

    const gaugeWrapper = document.querySelector('.risk-gauge-wrapper');
    const riskBadge = document.getElementById('risk-badge');
    const scoreVal = document.getElementById('res-risk-score');
    scoreVal.innerText = `${riskScore}%`;

    gaugeWrapper.className = 'risk-gauge-wrapper';
    riskBadge.className = 'badge';

    if (category === 'High Risk') {
        gaugeWrapper.classList.add('high');
        riskBadge.classList.add('high');
        riskBadge.innerText = 'High Risk Tier';
    } else if (category === 'Medium Risk') {
        gaugeWrapper.classList.add('medium');
        riskBadge.classList.add('medium');
        riskBadge.innerText = 'Medium Risk Tier';
    } else {
        gaugeWrapper.classList.add('low');
        riskBadge.classList.add('low');
        riskBadge.innerText = 'Low Risk Tier';
    }

    document.getElementById('res-patient-id').innerText = data.patient_id;
    document.getElementById('res-patient-name').innerText = data.patient_name;

    // Render XAI Feature Attribution Drivers & Protective Factors
    const xaiList = document.getElementById('xai-drivers-list');
    xaiList.innerHTML = '';

    if (data.top_risk_drivers && data.top_risk_drivers.length > 0) {
        data.top_risk_drivers.forEach(item => {
            const div = document.createElement('div');
            div.className = 'xai-item driver';
            div.innerHTML = `
                <div class="xai-header">
                    <span class="xai-feature">🚨 ${item.feature} (${item.value})</span>
                    <span class="xai-impact">+${item.impact_percentage}% Risk</span>
                </div>
                <div class="xai-explanation">${item.explanation}</div>
            `;
            xaiList.appendChild(div);
        });
    }

    if (data.top_protective_factors && data.top_protective_factors.length > 0) {
        data.top_protective_factors.forEach(item => {
            const div = document.createElement('div');
            div.className = 'xai-item protective';
            div.innerHTML = `
                <div class="xai-header">
                    <span class="xai-feature">🛡️ ${item.feature} (${item.value})</span>
                    <span class="xai-impact">${item.impact_percentage}% Risk</span>
                </div>
                <div class="xai-explanation">${item.explanation}</div>
            `;
            xaiList.appendChild(div);
        });
    }

    const recsList = document.getElementById('clinical-recs-list');
    recsList.innerHTML = '';
    data.clinical_recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.innerText = rec;
        recsList.appendChild(li);
    });
}

// Print Clinical Report
function printClinicalReport() {
    window.print();
}

// Fetch KPI metrics & Analytics data
async function fetchKPIsAndAnalytics() {
    try {
        const res = await fetch('/api/analytics');
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById('kpi-total').innerText = data.total_patients;
        document.getElementById('kpi-high-risk').innerText = data.high_risk_count;
        document.getElementById('kpi-avg-risk').innerText = `${data.avg_risk_score}%`;
        document.getElementById('kpi-low-risk').innerText = data.low_risk_count;

        window.analyticsCache = data;

        if (document.getElementById('tab-analytics').classList.contains('active')) {
            renderPopulationCharts();
        }
    } catch (err) {
        console.error("Error fetching analytics:", err);
    }
}

// Render Plotly Population Analytics Charts (Module E)
function renderPopulationCharts() {
    const data = window.analyticsCache;
    if (!data) return;

    const chartLayoutBase = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8', family: 'Plus Jakarta Sans, sans-serif' },
        margin: { t: 30, r: 20, l: 40, b: 40 },
        autosize: true
    };

    // Chart 1: Donut Tier Breakdown
    const tierData = [{
        values: [data.risk_tier_distribution['High Risk'], data.risk_tier_distribution['Medium Risk'], data.risk_tier_distribution['Low Risk']],
        labels: ['High Risk', 'Medium Risk', 'Low Risk'],
        type: 'pie',
        hole: 0.55,
        marker: { colors: ['#ef4444', '#f59e0b', '#10b981'] },
        textinfo: 'label+percent'
    }];
    Plotly.newPlot('chart-risk-tier', tierData, { ...chartLayoutBase, showlegend: false });

    // Chart 2: Age Cohort Bar Chart
    const ageGroups = data.age_group_risk.map(d => d.age_group);
    const ageScores = data.age_group_risk.map(d => d.avg_risk_score);
    const ageData = [{
        x: ageGroups,
        y: ageScores,
        type: 'bar',
        marker: {
            color: ageScores,
            colorscale: [[0, '#10b981'], [0.5, '#f59e0b'], [1, '#ef4444']]
        }
    }];
    Plotly.newPlot('chart-age-risk', ageData, {
        ...chartLayoutBase,
        xaxis: { title: 'Age Cohort' },
        yaxis: { title: 'Avg Risk Score (%)' }
    });

    // Chart 3: Prior Visits Scatter/Line
    const visits = data.prior_visits_correlation.map(d => d.prior_visits);
    const visitScores = data.prior_visits_correlation.map(d => d.avg_risk_score);
    const visitData = [{
        x: visits,
        y: visitScores,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#0ea5e9', width: 3 },
        marker: { size: 8, color: '#6366f1' }
    }];
    Plotly.newPlot('chart-visits-risk', visitData, {
        ...chartLayoutBase,
        xaxis: { title: 'Prior Inpatient Stays (12 Months)' },
        yaxis: { title: 'Average Risk Score (%)' }
    });

    // Chart 4: Comorbidity Impact Horizontal Bar
    const conditions = data.comorbidity_impact.map(d => d.condition).reverse();
    const presentRisk = data.comorbidity_impact.map(d => d.present_avg_risk).reverse();
    const comorbidityData = [{
        x: presentRisk,
        y: conditions,
        type: 'bar',
        orientation: 'h',
        marker: { color: '#f59e0b' }
    }];
    Plotly.newPlot('chart-comorbidity-risk', comorbidityData, {
        ...chartLayoutBase,
        xaxis: { title: 'Avg Readmission Risk (%)' },
        margin: { t: 30, r: 20, l: 150, b: 40 }
    });

    // Chart 5: Diagnosis Breakdown Stacked / Bar
    if (data.diagnosis_breakdown && data.diagnosis_breakdown.length > 0) {
        const diagLabels = data.diagnosis_breakdown.map(d => d.diagnosis);
        const diagRisks = data.diagnosis_breakdown.map(d => d.avg_risk);
        const diagCounts = data.diagnosis_breakdown.map(d => d.count);

        const diagData = [{
            x: diagLabels,
            y: diagRisks,
            type: 'bar',
            text: diagCounts.map(c => `${c} patients`),
            textposition: 'auto',
            marker: { color: '#6366f1' }
        }];
        Plotly.newPlot('chart-diagnosis-risk', diagData, {
            ...chartLayoutBase,
            xaxis: { title: 'Primary Diagnosis Category' },
            yaxis: { title: 'Average Readmission Risk Score (%)' }
        });
    }
}

// Fetch Patient Registry Table
async function fetchPatientRegistry() {
    try {
        const res = await fetch('/api/patients');
        if (!res.ok) return;
        registryData = await res.json();
        renderRegistryTable(registryData);
    } catch (err) {
        console.error("Error fetching registry:", err);
    }
}

// Render Registry Table Rows
function renderRegistryTable(patients) {
    const tbody = document.getElementById('patients-table-body');
    tbody.innerHTML = '';

    if (!patients || patients.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted);">No patient records found.</td></tr>`;
        return;
    }

    patients.forEach(p => {
        const tr = document.createElement('tr');

        let badgeClass = 'low';
        if (p.risk_category === 'High Risk') badgeClass = 'high';
        else if (p.risk_category === 'Medium Risk') badgeClass = 'medium';

        const dateStr = p.assessed_at ? new Date(p.assessed_at).toLocaleDateString() : 'Recent';

        tr.innerHTML = `
            <td><strong>${p.patient_id}</strong></td>
            <td>${p.name}</td>
            <td>${p.age} y / ${p.gender}</td>
            <td>${p.primary_diagnosis}</td>
            <td>${p.prior_inpatient_visits} visits</td>
            <td><strong>${p.risk_score}%</strong></td>
            <td><span class="badge ${badgeClass}">${p.risk_category}</span></td>
            <td>${dateStr}</td>
            <td>
                <div style="display: flex; gap: 0.3rem;">
                    <button class="btn-chip" onclick='openPatientModal("${p.patient_id}")'>View File</button>
                    <button class="btn-chip" style="color: var(--risk-high);" onclick='deletePatientRecord("${p.patient_id}")'>Delete</button>
                </div>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

// Filter Patient Registry
function filterPatients() {
    const search = document.getElementById('search-input').value.toLowerCase();
    const riskTier = document.getElementById('filter-risk').value;

    const filtered = registryData.filter(p => {
        const matchesSearch = p.name.toLowerCase().includes(search) ||
                              p.patient_id.toLowerCase().includes(search) ||
                              p.primary_diagnosis.toLowerCase().includes(search);

        const matchesRisk = riskTier === 'All' || p.risk_category === riskTier;

        return matchesSearch && matchesRisk;
    });

    renderRegistryTable(filtered);
}

// Open Patient Details & History Trajectory Modal
async function openPatientModal(patientId) {
    const modal = document.getElementById('patient-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');

    modalTitle.innerText = `Clinical File & Trajectory: ${patientId}`;
    modalBody.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 2rem;">Loading clinical history...</div>`;
    modal.classList.add('active');

    try {
        const res = await fetch(`/api/patient/${patientId}/history`);
        if (!res.ok) throw new Error("History fetch failed");
        const history = await res.json();

        const latest = history[history.length - 1];
        const patient = registryData.find(p => p.patient_id === patientId) || latest;

        let badgeClass = 'low';
        if (latest.risk_category === 'High Risk') badgeClass = 'high';
        else if (latest.risk_category === 'Medium Risk') badgeClass = 'medium';

        const drivers = latest.top_risk_drivers || [];
        const recs = latest.clinical_recommendations || [];

        modalBody.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(15,23,42,0.6); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <div>
                    <h4 style="font-size: 1.1rem;">30-Day Readmission Probability: <span style="color: var(--primary);">${latest.risk_score}%</span></h4>
                    <p style="font-size: 0.8rem; color: var(--text-muted);">Assessed on: ${latest.assessed_at ? new Date(latest.assessed_at).toLocaleString() : 'N/A'}</p>
                </div>
                <span class="badge ${badgeClass}" style="font-size: 0.9rem;">${latest.risk_category}</span>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; font-size: 0.85rem;">
                <div>
                    <p><strong>Patient Name:</strong> ${patient.name || 'N/A'}</p>
                    <p><strong>Age / Gender:</strong> ${patient.age} / ${patient.gender}</p>
                    <p><strong>Primary Diagnosis:</strong> ${patient.primary_diagnosis}</p>
                    <p><strong>Discharge Plan:</strong> ${patient.discharge_destination}</p>
                </div>
                <div>
                    <p><strong>Prior Inpatient Stays:</strong> ${patient.prior_inpatient_visits}</p>
                    <p><strong>Emergency ED Visits:</strong> ${patient.emergency_visits}</p>
                    <p><strong>HbA1c / Glucose:</strong> ${patient.hba1c_level}% / ${patient.glucose_level} mg/dL</p>
                    <p><strong>BP / Heart Rate:</strong> ${patient.systolic_bp}/${patient.diastolic_bp} mmHg (${patient.heart_rate} bpm)</p>
                </div>
            </div>

            <div style="margin-bottom: 1rem;">
                <h4 style="font-size: 0.95rem; margin-bottom: 0.5rem; color: var(--primary);">Risk Trajectory Over Time:</h4>
                <div id="patient-trajectory-chart" style="width: 100%; height: 160px;"></div>
            </div>

            <h4 style="font-size: 0.95rem; margin-bottom: 0.5rem; color: var(--primary);">Explainable AI (XAI) Feature Drivers:</h4>
            <ul class="recs-list" style="margin-bottom: 1rem;">
                ${drivers.map(d => `<li><strong>${d.feature} (${d.value}):</strong> ${d.explanation}</li>`).join('')}
            </ul>

            <h4 style="font-size: 0.95rem; margin-bottom: 0.5rem; color: var(--primary);">Tailored Clinical Action Plan:</h4>
            <ul class="recs-list">
                ${recs.map(r => `<li>${r}</li>`).join('')}
            </ul>
        `;

        // Plot Trajectory Sparkline
        const dates = history.map(h => new Date(h.assessed_at).toLocaleTimeString());
        const scores = history.map(h => h.risk_score);

        Plotly.newPlot('patient-trajectory-chart', [{
            x: dates,
            y: scores,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#0ea5e9', width: 3 },
            marker: { size: 6, color: '#6366f1' }
        }], {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#94a3b8', family: 'Plus Jakarta Sans, sans-serif' },
            margin: { t: 10, r: 10, l: 30, b: 30 },
            xaxis: { title: '' },
            yaxis: { title: 'Risk %', range: [0, 100] }
        });

    } catch (err) {
        modalBody.innerHTML = `<div style="color: var(--risk-high); padding: 1rem;">Failed to load patient history file.</div>`;
    }
}

// Delete Patient Record
async function deletePatientRecord(patientId) {
    if (!confirm(`Are you sure you want to delete patient record ${patientId}?`)) return;

    try {
        const res = await fetch(`/api/patients/${patientId}`, { method: 'DELETE' });
        if (res.ok) {
            fetchPatientRegistry();
            fetchKPIsAndAnalytics();
        }
    } catch (err) {
        alert("Error deleting record: " + err.message);
    }
}

// Open ML Model Metrics Modal
async function openModelMetricsModal() {
    const modal = document.getElementById('model-modal');
    const modalBody = document.getElementById('model-modal-body');
    modal.classList.add('active');

    try {
        const res = await fetch('/api/model/info');
        const data = await res.json();

        modalBody.innerHTML = `
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
                <div style="background: rgba(15,23,42,0.6); padding: 1rem; border-radius: 10px; text-align: center;">
                    <span style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Classifier Algorithm</span>
                    <h3 style="color: var(--primary); margin-top: 0.2rem;">${data.model_name || 'GradientBoosting'}</h3>
                </div>
                <div style="background: rgba(15,23,42,0.6); padding: 1rem; border-radius: 10px; text-align: center;">
                    <span style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">ROC-AUC / Accuracy Score</span>
                    <h3 style="color: var(--risk-low); margin-top: 0.2rem;">${(data.roc_auc * 100).toFixed(1)}% / ${(data.accuracy * 100).toFixed(1)}%</h3>
                </div>
            </div>

            <h4 style="font-size: 0.95rem; margin-bottom: 0.5rem; color: var(--primary);">Top ML Feature Importances:</h4>
            <div style="display: flex; flex-direction: column; gap: 0.4rem; max-height: 250px; overflow-y: auto;">
                ${(data.feature_importances || []).slice(0, 8).map(f => `
                    <div style="display: flex; justify-content: space-between; background: rgba(15,23,42,0.4); padding: 0.5rem 0.8rem; border-radius: 6px; font-size: 0.82rem;">
                        <span>${f.feature}</span>
                        <strong style="color: var(--primary);">${(f.importance * 100).toFixed(1)}%</strong>
                    </div>
                `).join('')}
            </div>
        `;
    } catch (err) {
        modalBody.innerHTML = `<div style="color: var(--risk-high);">Failed to load model metrics telemetry.</div>`;
    }
}

// Open Batch Upload Modal
function openBatchUploadModal() {
    document.getElementById('batch-modal').classList.add('active');
    document.getElementById('batch-status').innerText = '';
    document.getElementById('btn-run-batch').disabled = true;
    selectedBatchCSV = null;
}

// Handle CSV File Select
function handleCSVFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        selectedBatchCSV = file;
        document.getElementById('batch-status').innerText = `Selected file: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        document.getElementById('btn-run-batch').disabled = false;
    }
}

// Process CSV Batch Inference
async function processCSVBatch() {
    if (!selectedBatchCSV) return;

    const btn = document.getElementById('btn-run-batch');
    const status = document.getElementById('batch-status');
    btn.disabled = true;
    status.innerText = "Parsing CSV & Running Batch Inference...";

    const reader = new FileReader();
    reader.onload = async (e) => {
        try {
            const text = e.target.result;
            const lines = text.trim().split('\n');
            const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));

            const patientsPayload = [];

            for (let i = 1; i < lines.length; i++) {
                if (!lines[i].trim()) continue;
                const row = lines[i].split(',').map(cell => cell.trim().replace(/^"|"$/g, ''));
                const p = {};
                headers.forEach((h, idx) => {
                    let val = row[idx];
                    if (['age', 'prior_inpatient_visits', 'emergency_visits', 'length_of_stay', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'has_diabetes', 'has_hypertension', 'has_heart_failure', 'has_copd', 'has_ckd', 'has_asthma'].includes(h)) {
                        p[h] = parseInt(val) || 0;
                    } else if (['glucose_level', 'hba1c_level', 'bmi'].includes(h)) {
                        p[h] = parseFloat(val) || 0.0;
                    } else {
                        p[h] = val || '';
                    }
                });
                patientsPayload.push(p);
            }

            const response = await fetch('/api/predict/batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(patientsPayload)
            });

            if (!response.ok) throw new Error("Batch processing failed");
            const result = await response.json();

            status.innerHTML = `✅ Successfully processed ${result.processed_count} patient assessments!`;
            fetchPatientRegistry();
            fetchKPIsAndAnalytics();

        } catch (err) {
            status.innerText = "Error: " + err.message;
        } finally {
            btn.disabled = false;
        }
    };

    reader.readAsText(selectedBatchCSV);
}

// Download Sample CSV Template
function downloadSampleCSVTemplate() {
    const csvContent = `name,age,gender,prior_inpatient_visits,emergency_visits,length_of_stay,glucose_level,hba1c_level,systolic_bp,diastolic_bp,heart_rate,bmi,has_diabetes,has_hypertension,has_heart_failure,has_copd,has_ckd,has_asthma,primary_diagnosis,discharge_destination
Sample Patient A,68,Male,2,1,5,155.0,7.8,142,86,78,28.5,1,1,0,0,1,0,Cardiovascular,Home Health
Sample Patient B,45,Female,0,0,2,98.0,5.5,118,74,68,23.1,0,0,0,0,0,0,General Medicine,Home`;

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = "sample_patients_template.csv";
    a.click();
    URL.revokeObjectURL(url);
}

// Download Cohort CSV
function downloadCohortCSV() {
    window.location.href = '/api/patients/export';
}

// Close Modal helper
function closeModal(event, modalId) {
    if (!event || event.target.classList.contains('modal-overlay') || event.target.classList.contains('close-btn')) {
        document.getElementById(modalId).classList.remove('active');
    }
}

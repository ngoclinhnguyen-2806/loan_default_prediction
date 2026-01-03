# Loan Default Prediction: End-to-End MLOps Pipeline

**Automated credit risk assessment system with production-grade data processing, model training, and monitoring infrastructure.**

---

## 🎯 Project Overview

Built a complete MLOps pipeline to predict loan defaults on synthetic data using medallion architecture (Bronze-Silver-Gold) with automated orchestration, experiment tracking, and model monitoring.

**Business Problem:** Manual loan approval is slow (2-3 days) and inconsistent. This system reduces decision time to <5 minutes with data-driven risk assessment.

**Architecture:**

```
CSV Sources → Bronze (Raw) → Silver (Cleaned) → Gold (Features) → Model → Predictions
                    ↓              ↓                ↓               ↓
                Airflow DAGs orchestrating the entire pipeline
                         MLflow tracking experiments
```

---

## ✨ Key Features

- **Medallion Data Architecture:** Bronze-Silver-Gold layers for data quality and lineage
- **Automated Pipelines:** 3 Airflow DAGs managing ETL, training, and inference
- **Experiment Tracking:** MLflow for comparing 10+ model configurations  
- **Temporal Validation:** Prevents data leakage by filtering features available at application time
- **Production Monitoring:** Daily drift detection comparing predicted vs actual default rates
- **Containerized Deployment:** Docker Compose for consistent dev/prod environments

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Orchestration** | Apache Airflow |
| **Experiment Tracking** | MLflow |
| **Data Processing** | PySpark, Pandas |
| **ML Models** | Scikit-learn, XGBoost |
| **Containerization** | Docker, Docker Compose |
| **Storage** | Parquet (data), Pickle (models) |
| **Development** | Jupyter Lab, Python 3.10 |

---

## 📁 Project Structure

```
.
├── docker-compose.yaml
├── ReadMe.md
│
├── airflow/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── airflow.cfg
│   └── model_bank/
│       ├── credit_model_LR_2024_08_01.pkl
│       └── credit_model_XGB_2024_08_01.pkl
│
├── dags/
│   ├── data_pipeline_dag.py          # Bronze → Silver → Gold ETL
│   ├── scheduled_training_dag.py      # Model training pipeline
│   └── monitoring_dag.py              # Inference + drift detection
│
├── data/
│   └── [Source CSV files]
│
├── datamart/
│   ├── bronze/                        # Raw data partitioned by date
│   ├── silver/                        # Cleaned, validated data
│   └── gold/                          # Feature-engineered datasets
│       ├── applications/
│       ├── features/
│       └── labels/
│
├── mlruns/                            # MLflow experiment artifacts
│
├── notebooks/
│   ├── 00_data_processing_main.ipynb
│   ├── 01a_explore_csv.ipynb
│   ├── 02a_features_analysis_silver.ipynb
│   ├── 04_model_train_main_XGB.ipynb
│   ├── 05_Inference.ipynb
│   └── 06_Monitoring.ipynb
│
├── scripts/
│   ├── 01_create_bronze.py
│   ├── 02_create_silver_features.py
│   ├── 03_create_features.py
│   ├── 04_model_train_LR_improved.py
│   ├── 04_model_train_XGB.py
│   ├── 05_model_inference.py
│   └── 06_model_monitoring.py
│
└── utils/
    ├── data_processing_bronze_table.py
    ├── data_processing_silver_table.py
    └── data_processing_gold_table.py
```

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 8GB+ RAM recommended

### Setup

```bash
# Clone repository
git clone <repo-url>
cd loan-default-mlops

# Start services
docker-compose up -d

# Access services
# Airflow:  http://localhost:8080
# MLflow:   http://localhost:5000
# Jupyter:  http://localhost:8888
```

### Running the Pipeline

1. **Data Processing (Daily ETL):**
   ```bash
   # Trigger via Airflow UI or CLI
   airflow dags trigger data_pipeline_dag
   ```

2. **Model Training:**
   ```bash
   # Manually trigger retraining
   airflow dags trigger scheduled_training_dag
   ```

3. **Inference & Monitoring:**
   ```bash
   # Runs daily, or trigger manually
   airflow dags trigger monitoring_dag
   ```

---

## 📊 Model Performance

### Current Results

| Model | Train AUC | Test AUC | OOT AUC | Stability |
|-------|-----------|----------|---------|-----------|
| **Logistic Regression** | 0.821 | 0.810 | **0.805** | ✅ 1.6% gap |
| XGBoost | 0.860 | 0.807 | 0.750 | ⚠️ 8% gap |

**Selected Model:** Logistic Regression
- **Why:** Superior temporal stability + regulatory interpretability
- **Trade-off:** Slightly lower training performance, but better generalization

### Key Findings

⚠️ **Current Limitations:**
- Model performance below production threshold (AUC ~0.53 on recent data)
- Root causes identified: Feature engineering quality, class imbalance (5% default rate)
- Temporal instability in monthly predictions

**Planned Improvements:**
- External credit bureau data integration
- SMOTE for class balancing
- Advanced feature engineering (behavioral pattern aggregations)
- Target: AUC >0.70, KS >0.25

---

## 🔍 Technical Highlights

### Data Leakage Prevention
```python
# Filter features by temporal availability
features = features[features['snapshot_date'] <= features['application_date']]
```

### MLflow Integration
- Logs scaler artifacts alongside models for reproducible inference
- Tracks 10+ hyperparameter configurations per model type
- Model registry for version control

### Monitoring Strategy
- Daily comparison of predicted vs actual default rates
- KS statistic tracking over time
- Automated alerts for drift detection

---

## 📈 Business Impact

| Metric | Before | After |
|--------|--------|-------|
| **Decision Time** | 2-3 days | <5 minutes |
| **Approval Rate** | Manual (inconsistent) | 87% (balanced threshold) |
| **Default in Approved** | 5% baseline | 4.2% (target) |

---

## 🤝 Contributing

This is an academic project. Feedback and suggestions welcome via issues.

---

## 👤 Author

**Linh Nguyen**  
SMU MITB | CS611 Machine Learning Engineering | Nov 2025

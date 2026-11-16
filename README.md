# Mini Production ML System – Telco Customer Churn

**BITS Pilani WILP · ML Model Engineering · Graded Assignment**
Rakesh Dutta · 2025em1200105

An end-to-end, production-style ML system that predicts which telecom customers are
about to leave (churn), so the retention team can call them first.

```
Kaggle data -> leak-free split -> sklearn Pipeline (clean + impute + scale + one-hot + model)
            -> threshold tuned on business cost -> joblib artifacts
            -> FastAPI REST service  (/health, /predict, /predict/batch)
            -> Batch CLI             (predict_batch.py)
            -> Benchmark (p50/p95/p99, records/sec) + edge cases + drift monitor (PSI / KS)
```

![Architecture](docs/architecture.png)

---

## 1. Quick start (Windows or macOS)

Needs **Python 3.10 – 3.13** (3.12 recommended) and internet access for the Kaggle download.
No Kaggle account or API key is needed.

**macOS / Linux**
```bash
git clone https://github.com/2025em1200105/mini-production-ml-system.git
cd mini-production-ml-system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

**Windows (PowerShell)**
```powershell
git clone https://github.com/2025em1200105/mini-production-ml-system.git
cd mini-production-ml-system
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_all.py
```
> If PowerShell blocks the activate script, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

`run_all.py` runs all 8 steps in order (about 2–3 minutes) and stops on the first error:

| Step | Command it runs | Output |
|---|---|---|
| 1 | `python -m src.data_pipeline` | Kaggle download, stratified 60/20/20 split, leakage audit |
| 2 | `python -m src.train` | baselines + 3 models, best by PR-AUC, cost-tuned threshold, `artifacts/` |
| 3 | `python -m src.evaluate` | test metrics + charts in `reports/` |
| 4 | `python -m serving.predict_batch ...` | `reports/batch_predictions.csv` |
| 5 | `python -m src.drift --input ...` | `reports/drift_report.csv` |
| 6 | `python -m src.drift --input ... --simulate-drift` | `reports/drift_report_simulated.csv` |
| 7 | `python -m serving.benchmark` | `reports/benchmark.json`, latency chart |
| 8 | `python -m pytest -q` | 14 unit / integration tests |

In **VS Code** you can also use the *Run and Debug* panel – three ready-made launch
configurations are in `.vscode/launch.json`.

---

## 2. Run the REST API

```bash
python -m uvicorn serving.app:app --port 8000
```
Open **http://127.0.0.1:8000/docs** for the interactive Swagger page.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `ok` if the model is loaded, `degraded` if the fallback rule is active |
| GET | `/model-info` | model version, threshold, validation scores |
| POST | `/predict` | score one customer |
| POST | `/predict/batch` | score up to 1,000 customers |

**macOS / Linux**
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```
**Windows PowerShell**
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -InFile sample_request.json
```

Example answer:
```json
{"customerID": "7590-VHVEG", "churn_probability": 0.8656, "will_churn": true,
 "risk_level": "high", "decision_threshold": 0.38, "model_used": "ml_model",
 "model_version": "v20261116-...", "warnings": []}
```

Bad input gets a clear **HTTP 422**:
```json
{"error": "invalid_input", "detail": [{"field": "tenure", "problem": "Input should be greater than or equal to 0"}]}
```

## 3. Run batch scoring

```bash
python -m serving.predict_batch --input data/processed/new_customers.csv --output reports/batch_predictions.csv
```
The input CSV needs a `customerID` column plus any of the feature columns.
Exit codes: `0` success, `1` bad input file, `2` scoring failure.

---

## 4. Project layout

```
mini-production-ml-system/
├── README.md
├── requirements.txt          pinned versions
├── run_all.py                one command, whole pipeline
├── sample_request.json       example API payload
├── data/                     filled automatically from Kaggle (not committed)
├── artifacts/                churn_model.joblib, baseline_model.joblib,
│                             model_metadata.json, reference_stats.json
├── src/
│   ├── config.py             all paths, columns, seeds and costs
│   ├── data_pipeline.py      download, clean, stratified split, leakage audit
│   ├── baseline_rule.py      business rule = baseline AND API fallback
│   ├── train.py              pipelines, model selection, threshold tuning, export
│   ├── evaluate.py           test metrics, PR curve, confusion matrix, cost curve
│   ├── metrics_utils.py      business cost, threshold search, recall@precision
│   └── drift.py              PSI + KS drift monitor (+ drift simulation)
├── serving/
│   ├── app.py                FastAPI service
│   ├── schemas.py            Pydantic input/output contracts
│   ├── predictor.py          model loading + scoring + fallback (shared by API and CLI)
│   ├── predict_batch.py      batch CLI
│   └── benchmark.py          latency, throughput, edge cases, fallback check
├── tests/                    pytest: schemas, cleaning, fallback, API end-to-end
├── reports/                  metrics, charts, drift reports, benchmark results
└── docs/architecture.png
```

## 5. Key results (test set, 1,409 customers)

| Model | PR-AUC | ROC-AUC | Recall | Business cost |
|---|---|---|---|---|
| Dummy (always predicts churn rate) | 0.265 | 0.500 | 0.00 | Rs 18.70 lakh |
| Rule: month-to-month & tenure < 12 | 0.466 | 0.767 | 0.55 | Rs 10.24 lakh |
| Logistic regression (plain) | 0.634 | 0.843 | 0.55 | Rs 9.46 lakh |
| **HistGradientBoosting (balanced) @ t=0.38** | **0.657** | **0.843** | **0.86** | **Rs 6.22 lakh** |

Full details are in the PDF report and in `reports/`.
Small differences (third decimal place) are normal on a different OS or CPU.

## 6. Dataset

[Telco Customer Churn – Kaggle (blastchar)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) –
7,043 customers, 19 features, 26.5% churn. If Kaggle is not reachable, the code falls back to
IBM's public copy of the same file.

# Evaluation Datasets

RazorRisk reports two deliberately separate evaluation tracks.

## 1. Project synthetic dataset

Used for:

- user/device/IP graph construction;
- fraud-ring and benign-look-alike scenarios;
- GNN/community evaluation;
- end-to-end risk-policy regression testing.

Latest reproducible run:

| Model | ROC-AUC | PR-AUC | Accuracy | Balanced Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tabular / XGBoost | 0.855892 | 0.637720 | 0.981179 | 0.773541 | 0.944444 | 0.548387 | 0.693878 |
| GraphSAGE / GNN | 0.948412 | 0.696521 | 0.978670 | 0.725806 | 1.000000 | 0.451613 | 0.622222 |
| Learned stacker | 0.938937 | 0.713269 | 0.982434 | 0.774194 | 1.000000 | 0.548387 | 0.708333 |

Run:

```bash
python tests/evaluate_models.py --dataset synthetic
python tests/evaluate_models.py --dataset synthetic --retrain
```

These metrics describe a controlled synthetic benchmark and must not be interpreted as production fraud-detection performance.

## 2. ULB Credit Card Fraud Detection / Kaggle

The supplied `creditcard.csv` contains:

- 284,807 transactions;
- 492 fraud transactions;
- `Time`, `Amount`, `V1`–`V28`, and `Class`.

The dataset does **not** expose stable user/device/IP identities. Therefore the external benchmark evaluates only the tabular model. RazorRisk does not fabricate graph relationships for this benchmark.

### Actual run used in the README

Chronological 80/20 split:

- train: 227,845 rows, 417 fraud;
- test: 56,962 rows, 75 fraud.

| Metric | XGBoost |
|---|---:|
| ROC-AUC | **0.986233** |
| PR-AUC | **0.792616** |
| Accuracy | **0.999579** |
| Balanced Accuracy | **0.873289** |
| Precision | **0.918033** |
| Recall | **0.746667** |
| F1 | **0.823529** |
| TP | 56 |
| FP | 5 |
| FN | 19 |
| TN | 56,882 |

Reproduce:

```bash
python tests/evaluate_models.py --dataset kaggle --csv data/creditcard.csv
```

The evaluator prints the exact train/test counts, metrics, and confusion matrix.

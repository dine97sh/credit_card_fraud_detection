# Credit Card Fraud Detection

End-to-end fraud detection pipeline combining **transaction-level features** and **graph-network features**, built on a BankSim-style dataset. The model is trained with XGBoost, corrected for severe class imbalance with SMOTE, explained with SHAP, and presented through an interactive **Power BI** dashboard.

---

## Problem Statement

Fraudulent transactions make up a very small fraction (~1–2%) of all transactions, but they carry disproportionate financial risk. The goal of this project is to build a model that can reliably flag fraudulent transactions in real time while remaining interpretable enough to explain *why* a transaction was flagged.

---

## Dataset

source : https://www.kaggle.com/datasets/ealaxi/banksim1?select=bsNET140513_032310.csv

Two related views of the same transaction data:

| File | Description |
|---|---|
| `raw_data_1.csv` | Transaction-level table: `customer, age, gender, zipcodeOri, merchant, zipMerchant, category, amount, fraud` |
| `raw_data_2.csv` | The same transactions as a graph edge list: `Source (customer), Target (merchant), Weight (amount), typeTrans, fraud` |

Using both lets the model learn from individual transaction attributes *and* from the structure of the customer–merchant network — something a purely tabular approach misses.

---

## Pipeline

1. **Data Cleaning** — strip stray quote characters, encode unknown ages, fix data types
2. **EDA** — fraud rate by category, amount distribution, gender/age breakdown
3. **Feature Engineering (Tabular)** — customer/merchant transaction frequency, amount deviation from customer average
4. **Feature Engineering (Graph)** — degree and PageRank from a NetworkX customer↔merchant graph
5. **Train/Test Split** — stratified 80/20 split to preserve fraud ratio
6. **Class Imbalance Handling** — SMOTE applied to the training set only
7. **Model Training** — XGBoost classifier tuned for PR-AUC
8. **Evaluation** — precision, recall, F1, PR-AUC, ROC-AUC (accuracy is not used — it's misleading on imbalanced data)
9. **Threshold Tuning** — operating threshold selected for a target recall
10. **Cross-Validation** — 5-fold stratified CV to confirm result stability
11. **Explainability** — SHAP values to identify and visualize the top fraud signals
12. **Export** — predictions, metrics, confusion matrix, and SHAP importances exported as CSVs for the dashboard

Full pipeline with explanations is in [`fraud_detection_credit_card.ipynb`](notebooks/fraud_detection_credit_card.ipynb).

---

## Key Findings

- The strongest fraud signals are **merchant familiarity** (`merchant_txn_count`), **transaction amount**, and the two **graph-based features** (`customer_pagerank`, `merchant_degree`) — confirming that network structure adds real predictive value beyond tabular features alone.
- Transactions at merchants a customer rarely uses, for unusually high amounts, are the model's clearest fraud indicators.
- Model performance on the held-out test set:

| Metric | Score |
|---|---|
| Precision | 0.51 |
| Recall | 0.95 |
| F1 | 0.67|
| ROC-AUC | 1.00 |
| PR-AUC | 0.93 |

---

## Dashboard

An interactive Power BI dashboard presents the results across five pages: Overview, Exploratory Insights, Model Performance, SHAP Feature Importance, and a transaction drill-down table.

<img width="4150" height="2400" alt="cc_pages-to-jpg-0001" src="https://github.com/user-attachments/assets/155ada44-a702-44b5-990a-f49422de9869" />
<img width="4150" height="2400" alt="cc_pages-to-jpg-0002" src="https://github.com/user-attachments/assets/020067cb-98a7-4d48-9666-623bd3ac03ce" />
<img width="4150" height="2400" alt="cc_pages-to-jpg-0003" src="https://github.com/user-attachments/assets/c5e6cbb8-25dc-4ac0-bdf5-d0e01237d862" />






## Tools & Libraries

**Python:** pandas, numpy, scikit-learn, imbalanced-learn (SMOTE), xgboost, networkx, shap, joblib
**Visualization:** seaborn, matplotlib, Power BI


## Author

Dinesh T — [[LinkedIn] : (https://www.linkedin.com/in/dineshtd1997/) · [GitHub](#)

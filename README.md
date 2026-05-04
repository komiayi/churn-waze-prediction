# Waze User Churn Prediction

> A supervised machine-learning pipeline that predicts which Waze users are at risk of churning, based on behavioral features. Three tree-based models are compared (Decision Tree, Random Forest, XGBoost) and the final model is selected through cross-validation and decision-threshold optimization.

[![Status](https://img.shields.io/badge/status-completed-brightgreen)](https://github.com/komiayi/churn-waze-prediction)
[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-006ACC)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Last commit](https://img.shields.io/github/last-commit/komiayi/churn-waze-prediction)

---

## Overview

This project develops a **binary classification model** to identify Waze users who are likely to churn, using behavioral indicators such as activity days, sessions, and navigation patterns. The objective is to provide the business with an early-warning tool that supports targeted retention campaigns.

The work was carried out as part of the **Google Advanced Data Analytics Professional Certificate (Coursera)** capstone project, and extended with additional methodological refinements — most notably an explicit decision-threshold optimization step on the validation set.

>  **[View the full Jupyter notebook →](notebooks/Waze_ML.ipynb)**

---

## ⚖️ Ethical considerations and metric choice

Before any modeling, the consequences of model errors were explicitly evaluated:

- **False negative** (predicted as retained but actually churned): Waze fails to act and loses a user — direct, measurable cost.
- **False positive** (predicted as churned but actually retained): Waze sends an unnecessary retention message — minor cost, mostly reputational.

Given this asymmetry, **recall was selected as the primary evaluation metric**: maximizing the proportion of at-risk users correctly identified takes priority over minimizing false alarms.

This choice also addresses the **moderate class imbalance** of the dataset (18% churners, 82% retained), which would render plain accuracy misleading: a trivial classifier predicting "retained" for everyone would score 82% accuracy while detecting zero churners.

---

## Methodology

### Models compared

Three tree-based classifiers were tuned and compared on the same training data:

1. **Decision Tree** — used as a transparent baseline.
2. **Random Forest** — bagging-based ensemble.
3. **XGBoost** — gradient-boosted trees.

### Class imbalance handling

- Decision Tree and Random Forest: `class_weight="balanced"`.
- XGBoost: `scale_pos_weight = n_negative / n_positive`, computed dynamically from the training labels.

### Hyperparameter tuning

Each model was tuned via **`GridSearchCV` with 5-fold cross-validation**, refitting the best estimator on **recall** to align with the business objective. Search grids covered tree depth, sample-split thresholds, the number of estimators (for ensembles), learning rate, and regularization parameters (XGBoost).

### Decision-threshold optimization

After training, the classification threshold of the best model was re-optimized on the **validation set** to maximize the F1-score, providing a more deliberate trade-off between precision and recall than the default 0.5 cutoff.

### Feature engineering

Several engineered features were derived from raw activity data, including average kilometers per driving day, percentage of sessions in the last month, average sessions per day since onboarding, average speed, and percentage of drives toward favorite locations.

---

## Data sources

The dataset is the **Waze user activity dataset** distributed by Coursera as part of the *Google Advanced Data Analytics Professional Certificate*. It contains anonymized, synthetic-style user records with behavioral indicators (sessions, drives, kilometers driven, activity days, device type) and a binary churn label.

> **Acknowledgment.** This project relies entirely on data made available by Coursera/Google for educational purposes. The data is **not included in this repository** and is not redistributed; any reproduction requires direct access to the certificate's course materials.

---

## Results

Final model: **XGBoost** with classification threshold optimized at **0.49** on the validation set.

| Metric        | Test-set value |
| ------------- | -------------- |
| **Recall**    | **0.64**       |
| Precision     | 0.31           |
| F1-score      | 0.42           |
| Accuracy      | 0.68           |

### Top predictors

The most influential features identified by the model align with intuitive engagement signals:

1. **`activity_days`** — number of days the user was active in the last month.
2. **`total_navigations_fav1`** — navigations to the user's primary favorite location.
3. **`n_days_after_onboarding`** — overall tenure on the platform.
4. **`percent_sessions_in_last_month`** — recency of engagement relative to total history.
5. **`duration_minutes_drives`** and **`km_per_driving_day`** — intensity of recent driving behavior.

Conversely, features such as `professional_driver` and `km_per_drive` carried very low predictive weight, suggesting they could be removed without material loss in performance.

### Interpretation

The model successfully captures roughly two thirds of actual churners, at the cost of a moderate false-positive rate (precision around 0.31). Given the asymmetric cost structure described above — where missing a churner is costly and a false alarm is cheap — this trade-off is acceptable and aligned with the business objective. The model can therefore serve as a screening tool that flags at-risk users for targeted retention campaigns, rather than as a precision instrument for individual decisions.

---

## Technology stack

| Component         | Technology                                |
| ----------------- | ----------------------------------------- |
| Core language     | Python                                    |
| Modeling          | scikit-learn, XGBoost                     |
| Data handling     | pandas, NumPy                             |
| Visualization     | matplotlib                                |
| Notebook          | Jupyter                                   |
| Version control   | Git / GitHub                              |

---

## Repository structure

```
churn-waze-prediction/
├── notebooks/
│   └── Waze_ML.ipynb     # Complete end-to-end analysis
└── README.md
```

---

## Reproduction

**1. Clone the repository:**

```bash
git clone https://github.com/komiayi/churn-waze-prediction.git
cd churn-waze-prediction
```

**2. Install the required Python packages:**

```bash
pip install pandas numpy matplotlib scikit-learn xgboost jupyter
```

**3. Provide the dataset.** Place the `waze_dataset.csv` file (obtained through the Google Advanced Data Analytics Certificate course materials) in a `data/` folder at the root of the repository.

**4. Launch the notebook:**

```bash
jupyter notebook notebooks/Waze_ML.ipynb
```

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for full terms.

---

## Author

**Komi Roger Ayi**
Biostatistician — Data Scientist
Montréal, Québec, Canada

[Portfolio](https://komiayi.github.io) · [LinkedIn](https://www.linkedin.com/in/komi-ayi) · [GitHub](https://github.com/komiayi)

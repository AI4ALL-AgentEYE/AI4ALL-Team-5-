"""
build_agent_security_risk_analysis.py

Generates agent_security_risk_analysis.ipynb — a single, coherent
notebook covering data cleaning, fairness/bias analysis, EDA, and
three modeling approaches for the Agentic AI Security Risk dataset.

Run with:
    python3 build_agent_security_risk_analysis.py

Notebook structure:
    1. Load dataset (kagglehub)
    2. Dataset overview
    3. Data cleaning        (duplicates, missing values, outliers —
                              actually dropped/handled, not just printed)
    4. Fairness / bias analysis (group representation, avg score by role)
    5. Exploratory data analysis (distributions, correlations)
    6. Preprocessing for modeling
    7. Model 1: XGBoost      -> predicts action_risk_score
    8. Model 2: SHAP          -> explains the XGBoost predictions
    9. Model 3: Linear Regression -> predicts access_decision from action_risk_score alone
   10. Conclusions

A cleaned CSV (cleaned_agent_security_risk_scores.csv) is written out
after the cleaning step so it can be reused elsewhere.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ----------------------------------------------------------------
# Title / overview
# ----------------------------------------------------------------

md("""# Agent Security Risk Analysis

This notebook covers the full pipeline for the Agentic AI Security
Risk dataset: loading, cleaning, fairness analysis, exploratory data
analysis, and modeling.

## Research Questions

1. Is the dataset clean and reliable enough to model? (duplicates,
   missing values, outliers)
2. Are there signs of bias — e.g. do certain agent roles get
   systematically higher/lower risk scores, or is any group
   under-represented?
3. Can machine learning accurately predict an AI agent's action risk
   score, and which features drive that prediction?
4. Can we predict whether an action should be **Allowed**, **Require
   Approval**, or **Blocked**?

---

### Pipeline

| Step | Purpose |
|------|---------|
| Data Cleaning | Remove duplicates, handle missing values, handle outliers |
| Bias Analysis | Check group representation and score parity across agent roles |
| EDA | Visualize distributions and correlations |
| XGBoost | Predict the numerical `action_risk_score` |
| SHAP | Explain which features drive the XGBoost predictions |
| Linear Regression | Predict `access_decision` from `action_risk_score` alone |

**Leakage prevention:** `human_approval_required` and `access_decision`
are generated *after* the risk score is calculated, so both are
excluded whenever `action_risk_score` is the prediction target.
""")

# ----------------------------------------------------------------
# Imports
# ----------------------------------------------------------------

code("""# ============================================================
# IMPORTS
# ============================================================

from dotenv import load_dotenv
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    classification_report,
)
from sklearn.linear_model import LinearRegression

import xgboost as xgb
import shap

plt.style.use("ggplot")

RANDOM_STATE = 42""")

# ----------------------------------------------------------------
# Load dataset
# ----------------------------------------------------------------

md("""## 1. Load Dataset""")

code("""# ============================================================
# LOAD DATASET
# ============================================================

load_dotenv()

# Download (or use the cached copy of) the dataset from KaggleHub
path = kagglehub.dataset_download(
    "algozee/agentic-ai-security-risk-dataset"
)

print("Dataset location:", path)
print("Files:", os.listdir(path))

csv_file = os.path.join(path, "agent_security_risk_scores.csv")
df = pd.read_csv(csv_file)

print(df.shape)
df.head()""")

# ----------------------------------------------------------------
# Dataset overview
# ----------------------------------------------------------------

md("""## 2. Dataset Overview""")

code("""print("Columns:")
print(df.columns.tolist())

print("\\nDtypes / non-null counts:")
df.info()

print("\\nSummary statistics:")
df.describe(include="all")""")

# ----------------------------------------------------------------
# Data cleaning
# ----------------------------------------------------------------

md("""## 3. Data Cleaning

This section **modifies**
`df`: duplicate rows are dropped, missing values are handled, and
outliers are flagged and reviewed (capped rather than silently
deleted, since risk scores near the extremes may be legitimate and
informative).""")

code("""# ============================================================
# 3a. DUPLICATES
# ============================================================

n_duplicates = df.duplicated().sum()
print(f"Duplicate rows found: {n_duplicates}")

df = df.drop_duplicates().reset_index(drop=True)

print(f"Shape after dropping duplicates: {df.shape}")""")

code("""# ============================================================
# 3b. MISSING VALUES
# ============================================================

missing = df.isnull().sum()
missing_percent = (missing / len(df) * 100).round(2)

missing_report = pd.DataFrame({
    "missing_values": missing,
    "percent": missing_percent,
}).sort_values("missing_values", ascending=False)

print(missing_report[missing_report["missing_values"] > 0])

# Strategy:
#   - Numeric columns  -> fill with the column median (robust to outliers)
#   - Categorical columns -> fill with "Unknown" so the missingness
#     itself is preserved as information rather than dropping rows
numeric_cols = df.select_dtypes(include="number").columns
categorical_cols = df.select_dtypes(include="object").columns

for col in numeric_cols:
    if df[col].isnull().any():
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        print(f"Filled missing values in '{col}' with median ({median_value})")

for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna("Unknown")
        print(f"Filled missing values in '{col}' with 'Unknown'")

print(f"\\nRemaining missing values: {df.isnull().sum().sum()}")""")

code("""# ============================================================
# 3c. OUTLIERS (IQR method)
# ============================================================

def detect_outliers(data, column):
    \"\"\"Return rows where `column` falls outside 1.5 x IQR.\"\"\"
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] < lower) | (data[column] > upper)], lower, upper

outliers, lower_bound, upper_bound = detect_outliers(df, "action_risk_score")
print(f"Outliers in 'action_risk_score': {len(outliers)}")
print(f"Valid range: [{lower_bound:.2f}, {upper_bound:.2f}]")

# Cap (winsorize) rather than delete, so we keep the row but pull the
# extreme value back to the IQR boundary. This preserves sample size
# while preventing a few extreme points from dominating the models.
df["action_risk_score"] = df["action_risk_score"].clip(
    lower=lower_bound, upper=upper_bound
)

print("Outliers capped to the valid range.")""")

code("""# ============================================================
# 3d. SAVE CLEANED DATASET
# ============================================================

df.to_csv("cleaned_agent_security_risk_scores.csv", index=False)
print("Saved cleaned_agent_security_risk_scores.csv")
print(df.shape)""")

# ----------------------------------------------------------------
# Bias / fairness analysis
# ----------------------------------------------------------------

md("""## 4. Fairness / Bias Analysis

Checks whether any agent role is under/over-represented, and whether
average risk scores differ systematically by role — both signs of
potential bias in the scoring system.""")

code("""# ============================================================
# GROUP REPRESENTATION
# ============================================================

group_representation = df["agent_role"].value_counts(normalize=True)
print("Share of requests by agent role:")
print(group_representation.round(3))

group_representation.plot.bar(figsize=(9, 4), title="Agent Role Representation")
plt.ylabel("Proportion of requests")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()""")

code("""# ============================================================
# AVERAGE RISK SCORE BY AGENT ROLE
# ============================================================

avg_score_by_role = (
    df.groupby("agent_role")["action_risk_score"]
      .mean()
      .sort_values(ascending=False)
)

print("Average action_risk_score by agent role:")
print(avg_score_by_role.round(2))

avg_score_by_role.plot(
    kind="bar", figsize=(9, 4), title="Average Risk Score by Agent Role"
)
plt.ylabel("Average Action Risk Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()""")

code("""# Categorical value counts — useful for spotting rare/typo'd categories
for column in categorical_cols:
    print(f"\\n{column}:")
    print(df[column].value_counts())""")

# ----------------------------------------------------------------
# EDA
# ----------------------------------------------------------------

md("""## 5. Exploratory Data Analysis""")

code("""# ============================================================
# TARGET DISTRIBUTIONS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["action_risk_score"], bins=30, edgecolor="black")
axes[0].set_title("Distribution of Action Risk Scores (cleaned)")
axes[0].set_xlabel("Action Risk Score")
axes[0].set_ylabel("Count")

df["access_decision"].value_counts().plot.bar(ax=axes[1])
axes[1].set_title("Access Decisions")
axes[1].set_xlabel("Decision")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()""")

code("""# ============================================================
# CORRELATION WITH THE TARGET
# ============================================================

corr = (
    df.select_dtypes("number")
      .corr()["action_risk_score"]
      .drop("action_risk_score")
      .sort_values()
)

corr.plot.barh(figsize=(8, 4))
plt.title("Correlation with Action Risk Score")
plt.tight_layout()
plt.show()""")

# ----------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------

md("""## 6. Preprocessing for Modeling

One-hot encode the categorical columns. The feature set used to
predict `action_risk_score` excludes the two "leaky" columns that are
downstream consequences of the score, not causes of it.""")

code("""# ============================================================
# PREPROCESSING FOR THE RISK-SCORE MODEL
# ============================================================

TARGET = "action_risk_score"

LEAKY_COLUMNS = [
    "human_approval_required",
    "access_decision",
]

CATEGORICAL_COLUMNS = [
    "agent_role",
    "user_role",
    "requested_action",
    "tool_requested",
    "resource_type",
]

X = pd.get_dummies(
    df.drop(columns=[TARGET] + LEAKY_COLUMNS),
    columns=CATEGORICAL_COLUMNS,
)

y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)

print(f"{X.shape[1]} features | train={len(X_train)} | test={len(X_test)}")""")

# ----------------------------------------------------------------
# Model 1: XGBoost
# ----------------------------------------------------------------

md("""## 7. Model 1 — XGBoost Risk Score Prediction

Predicts the numerical `action_risk_score` from the request's
security-relevant features.""")

code("""# ============================================================
# MODEL 1: XGBOOST REGRESSOR
# ============================================================

xgb_model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE,
)

xgb_model.fit(X_train, y_train)

predictions = xgb_model.predict(X_test)

print("R^2:", round(r2_score(y_test, predictions), 3))
print("MAE:", round(mean_absolute_error(y_test, predictions), 2))""")

# ----------------------------------------------------------------
# Model 2: SHAP
# ----------------------------------------------------------------

md("""## 8. Model 2 — SHAP Explainability

SHAP values show which features push a given prediction up or down,
and by how much.""")

code("""# ============================================================
# MODEL 2: SHAP EXPLANATIONS
# ============================================================

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, max_display=15)""")

code("""# Rank features by average absolute SHAP value (impact on score)
importance = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=X.columns,
)

print("Top 15 drivers of the risk score:")
print(importance.sort_values(ascending=False).head(15))""")

# ----------------------------------------------------------------
# Model 3: Linear Regression
# ----------------------------------------------------------------

md("""## 9. Model 3 — Linear Regression: `access_decision` from `action_risk_score` Alone

Earlier EDA showed `access_decision` tracks `action_risk_score`
closely (Allowed requests cluster at low scores, Blocked requests at
high scores). This model tests how far a **single input feature**
gets us: `access_decision` is label-encoded as an ordinal target
(`Allowed=0 < Needs_Human_Approval=1 < Blocked=2`, reflecting
increasing risk/restriction) and predicted with ordinary linear
regression using only `action_risk_score` as the predictor.

Because linear regression outputs a continuous number rather than a
class, predictions are rounded and clipped to the nearest valid class
(0, 1, or 2) to produce a decision label.""")

code("""# ============================================================
# MODEL 3: LINEAR REGRESSION (single feature: action_risk_score)
# ============================================================

DECISION_ORDER = ["Allowed", "Needs_Human_Approval", "Blocked"]
decision_to_code = {label: i for i, label in enumerate(DECISION_ORDER)}
code_to_decision = {i: label for label, i in decision_to_code.items()}

X_lin = df[["action_risk_score"]]
y_lin = df["access_decision"].map(decision_to_code)

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_lin, y_lin, test_size=0.25, random_state=RANDOM_STATE, stratify=y_lin
)

lin_model = LinearRegression()
lin_model.fit(X_train_l, y_train_l)

print(f"Coefficient (action_risk_score): {lin_model.coef_[0]:.4f}")
print(f"Intercept: {lin_model.intercept_:.4f}")
print("(Higher risk score -> higher predicted decision code, i.e. more restrictive)")""")

code("""# ============================================================
# EVALUATE: ROUND CONTINUOUS PREDICTIONS INTO DECISION CLASSES
# ============================================================

raw_pred = lin_model.predict(X_test_l)
rounded_pred = np.clip(np.round(raw_pred), 0, 2).astype(int)

pred_labels = pd.Series(rounded_pred).map(code_to_decision)
true_labels = y_test_l.map(code_to_decision)

print("R^2 (continuous):", round(r2_score(y_test_l, raw_pred), 3))
print("MAE (continuous):", round(mean_absolute_error(y_test_l, raw_pred), 3))
print()
print(classification_report(true_labels, pred_labels))""")

code("""# ============================================================
# IMPLIED SCORE THRESHOLDS BETWEEN DECISION CLASSES
# ============================================================
# Solve coef * score + intercept = 0.5 and 1.5 for the score values
# where the rounded prediction crosses from one class to the next.

slope = lin_model.coef_[0]
intercept = lin_model.intercept_

threshold_allowed_to_approval = (0.5 - intercept) / slope
threshold_approval_to_blocked = (1.5 - intercept) / slope

print(f"Implied score threshold, Allowed -> Needs_Human_Approval: {threshold_allowed_to_approval:.1f}")
print(f"Implied score threshold, Needs_Human_Approval -> Blocked: {threshold_approval_to_blocked:.1f}")""")

# ----------------------------------------------------------------
# Conclusions
# ----------------------------------------------------------------

md("""## 10. Conclusions

This notebook covered the full pipeline:

1. **Data Cleaning** — duplicates were removed, missing values were
   imputed (median for numeric, "Unknown" for categorical), and
   outliers in `action_risk_score` were capped using the IQR method.
   The cleaned dataset was saved to
   `cleaned_agent_security_risk_scores.csv`.
2. **Fairness Analysis** — group representation and average risk
   score were compared across agent roles to check for
   under-representation or systematic scoring differences.
3. **XGBoost** predicts an AI agent's numerical risk score with
   reasonable accuracy (see R² / MAE above).
4. **SHAP** explains which features have the greatest influence on
   those predictions, making the risk-scoring behavior interpretable
   rather than a black box.
5. **Linear Regression** shows how well `action_risk_score` alone
   predicts the final access decision (Allowed, Require Approval, or
   Blocked), and surfaces the implied score thresholds between
   decision classes.

Together, these steps provide a cleaned, bias-checked dataset plus
predictive performance and interpretability, supporting an evaluation
of whether an agent's requested action should be trusted.""")

# ----------------------------------------------------------------
# Write notebook
# ----------------------------------------------------------------

nb.cells = cells
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

nbf.write(nb, "agent_security_risk_analysis.ipynb")
print("written")
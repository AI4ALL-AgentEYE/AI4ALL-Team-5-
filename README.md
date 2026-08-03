# AgentEYE: Security Risk Assessment for Agentic AI Models

Auditing an automated **agent security risk-scoring system**. Rather than build a risk score
from scratch, this project *interrogates an existing one*: what drives it, how well it can be
predicted, and whether access decisions actually follow from it.

> AI4ALL — Group 05A · Ruhi Shah · Donovon Mott · Deeksha Vaidyanathan · Dim Zuun · Mannat Kaur · Sangam Subedi

---

## Motivation

As AI agents act with less human oversight — using tools, accessing resources, and making
decisions — they create a new class of security risk. Many systems assign each agent action a
**risk score** and an **access decision** (Allowed / Blocked / Needs Human Approval). This
project treats that scoring system as something to be *audited*, not trusted by default.

## Research question

> **1. Can we predict the security risk score of an AI agent's action** from its autonomy
> level, permissions, requested tools, and resource-access patterns?
> **2. Can we decide whether an agent should be granted access** based on that score?

**Short answers (this project):**
- **Predicting the score — yes.** An XGBoost model reaches **R² ≈ 0.91**; the biggest drivers
  are **permission match** and **data-exfiltration risk**.
- **Deciding access from the score alone — only partly.** The three decision tiers overlap in
  score, so a score-only classifier tops out around **59%** across all three classes.
  `access_decision` depends on more than the score.

## Which notebook is the final one?

This repo keeps **three** notebooks on purpose — together they show the progression of our
understanding and the trials we ran. Here is what each is:

| Notebook | Role |
|---|---|
| **`agent_security_risk_analysis.ipynb`** | ⭐ **FINAL / primary analysis.** The two-model pipeline the presentation is built on. Start here. |
| `ai4all_data_analysis_notebook.ipynb` | **EDA & bias companion** — data cleaning, distributions, fairness checks. |
| `ai4all-team-project.ipynb` | **Additional exploration** — a wider 4-model triangulation experiment (adds a decision tree + isolation forest). Not the final deliverable; kept to show what we tried. |

Supporting scripts: `datacleaninganalysis.py` (a script version of the EDA notebook) and
`build_notebook.py` (a legacy notebook generator, kept pending review with its author).

## Method — the final analysis

The final notebook uses **two supervised approaches**, one per research question:

| Model | Question it answers |
|---|---|
| **XGBoost + SHAP** | What drives the risk score, and can we predict it? (SHAP explains the XGBoost model — it is an *explainer*, not a separate model.) |
| **Logistic Regression** | Do access decisions follow from the score? |

**Leakage note:** `human_approval_required` and `access_decision` are decided *downstream* of
the score, so they are excluded from the models that predict it.

## Key findings

- **The score is highly predictable.** XGBoost reaches **R² ≈ 0.914** (RMSE ≈ 8.4, MAE ≈ 6.2).
  Including `user_role` helped slightly (0.914 vs 0.902), so we kept it.
- **Top drivers:** **permission match** (a mismatch pushes the score up) and
  **data-exfiltration risk**, followed by resource sensitivity.
- **Robust to missing predictors.** Dropping the two hardest-to-obtain features
  (`data_exfiltration_risk`, `permission_match`) only lowers R² to **0.844** (~0.06), with
  RMSE/MAE ~3 points worse — the model degrades gracefully.
- **The score alone can't route decisions.** The three decision tiers overlap in score, so a
  score-only classifier reaches only **59%** across all three. Dropping the rare
  `Needs_Human_Approval` tier improves it: the model catches **Blocked** well (93% recall) but
  over-predicts Blocked for **Allowed** actions (64% recall).
- **Little role-based bias.** Average risk score is similar across the 11 agent roles (51–60).

## Dataset

- **Name:** Agent AI Security Risk Dataset — 2,200 simulated agent requests × 15 columns.
- **Target:** `action_risk_score` (0–100); secondary target `access_decision`.
- **Source & licensing:** originally from Kaggle (`algozee/agentic-ai-security-risk-dataset`),
  **now removed from Kaggle**, so the CSVs are vendored in `data/`. See **[`DATA_NOTICE.md`](DATA_NOTICE.md)**
  for the full explanation of why the data lives in this repo, why we dropped `kagglehub`, and
  the licensing/attribution caveat.

## Repository structure

```
.
├── README.md
├── DATA_NOTICE.md                        # data source, vendoring & licensing note
├── data/
│   ├── agent_security_risk_scores.csv          # raw dataset
│   └── cleaned_agent_security_risk_scores.csv  # cleaned version
├── agent_security_risk_analysis.ipynb    # ⭐ FINAL analysis (start here)
├── ai4all_data_analysis_notebook.ipynb   # EDA & bias companion
├── ai4all-team-project.ipynb             # additional 4-model exploration
├── datacleaninganalysis.py               # script version of the EDA
└── build_notebook.py                     # legacy generator (pending author review)
```

## How to run

The notebooks now read the dataset **locally** from `data/` — no Kaggle account, credentials,
or internet needed.

1. Open **`agent_security_risk_analysis.ipynb`**.
2. Run all cells top to bottom. The load cell reads `data/agent_security_risk_scores.csv`.

Requires Python 3.11–3.12 and the usual data-science stack (`pandas`, `numpy`, `matplotlib`,
`seaborn`, `scikit-learn`, `xgboost`, `shap`). *A pinned `requirements.txt` will be added.*

## Limitations

- **Label subjectivity** — the score and decisions were assigned by a prior process; the
  models learn *that* definition of risk, not ground truth.
- **Rare high-risk actions are under-represented**, so performance on the most dangerous cases
  is the least certain.
- **Simulated, single-source data** — results would need revalidation on real logs.

## References

- Madkour, N., et al. (2025). *Agentic AI Risk-Management Standards Profile.* UC Berkeley CLTC.
- National Institute of Standards and Technology. (2023). *AI Risk Management Framework (AI RMF 1.0).*
- Lynch, A., et al. (2025). *Agentic Misalignment: How LLMs Could Be Insider Threats.* Anthropic.
- Christodorescu, M., et al. (2026). *Agent security is a systems problem.* arXiv.
- Chhabra, A., et al. (2026). *Agentic AI security: Threats, defenses, evaluation, and open challenges.* IEEE Access.
- Evtimov, I., et al. (2025). *WASP: Benchmarking web agent security against prompt injection attacks.* NeurIPS.

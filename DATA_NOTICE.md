# Data Notice — Source, Vendoring & Licensing

## What the data is
This project uses the **Agent AI Security Risk Dataset** — 2,200 simulated AI-agent
requests × 15 columns. It was originally published on Kaggle by **algozee**:
`algozee/agentic-ai-security-risk-dataset`
(`https://www.kaggle.com/datasets/algozee/agentic-ai-security-risk-dataset`).

## Why the data is committed to this repository
**The original Kaggle dataset is no longer available** — the source was removed after we
downloaded it. Because the notebooks can no longer fetch it programmatically, and because
reproducibility matters for this coursework, the CSVs are vendored directly in `data/`:

- `data/agent_security_risk_scores.csv` — the raw dataset as originally downloaded.
- `data/cleaned_agent_security_risk_scores.csv` — the cleaned version produced by our
  data-cleaning step, reused by downstream notebooks.

## Why we dropped `kagglehub`
The notebooks originally downloaded the data at runtime via
`kagglehub.dataset_download(...)`, which required Kaggle credentials (a `.env` /
`kaggle.json`), an internet connection, and — critically — the dataset still existing on
Kaggle. With the source removed, those calls now fail. We replaced them with a plain local
read:

```python
df = pd.read_csv("data/agent_security_risk_scores.csv")
```

This makes every notebook run **offline, with no credentials**, straight from the repo.

> Note: `build_notebook.py` still references `kagglehub`. It is intentionally left unchanged
> pending a check with its original author.

## Licensing / attribution caveat
We do **not** own this dataset. It was created by the original Kaggle author (**algozee**)
and is redistributed here **only** for the purposes of this AI4ALL academic project, with
attribution. Because the original Kaggle page was removed, its exact license terms are no
longer visible to us, so we cannot fully verify redistribution rights. If the dataset owner
or Kaggle requests removal, we will take it down promptly. Anyone reusing this repository
should treat the data as the property of its original author, not of this project.

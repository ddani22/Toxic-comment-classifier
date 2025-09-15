## Purpose

Short, actionable instructions to quickly orient an AI coding agent inside this repository so it can make safe, useful edits.

## Quick summary

This repo is a small end-to-end toxic comment classifier: data exploration and training live in `notebooks/`, model artifacts are in `artifacts/`, and a Streamlit UI lives in `app/app.py`. Models and vectorizers are persisted as pickles (joblib) in `artifacts/` and loaded by the app.

Key locations:
- `notebooks/` — exploration and training pipelines (source of truth for preprocessing).
- `artifacts/` — saved outputs: `model.pkl`, `vectorizer.pkl`, `data_splits.pkl`.
- `app/app.py` — Streamlit entrypoint that loads the vectorizer + model and exposes a text input + prediction UI.
- `data/data.csv` — example dataset (kept small here), local-only.
- `requirements.txt` — present but currently empty; pin required packages before making reproducible environments.

## What to inspect first

- Open `app/app.py` to see how the model and vectorizer are loaded (look for `joblib.load` or `pickle` patterns).
- Inspect `notebooks/02_entrenamiento_modelo.ipynb` and `03_guardar_modelo.ipynb` to extract the exact preprocessing steps and the filenames used when saving artifacts.
- Check `artifacts/` to confirm artifact names: `model.pkl`, `vectorizer.pkl`, `data_splits.pkl`.

## Repo-specific patterns and conventions

- Persistence: models and transformers are persisted with joblib/pickle into `artifacts/`. Example path: `artifacts/model.pkl` and `artifacts/vectorizer.pkl`.
- UI: `app/app.py` implements a minimal Streamlit UI that expects the saved artifacts; common flow is: load vectorizer -> transform input -> predict -> show label + probability.
- Notebooks are canonical: reuse cells from `notebooks/` when re-implementing training code into scripts.
- Datasets: `data/` is local-only (avoid committing large files). Use `data/data.csv` as the canonical sample dataset.

## Developer workflows (Windows PowerShell examples)

Activate virtualenv (PowerShell):

& .\venv\Scripts\Activate.ps1

Install pinned deps (after updating `requirements.txt`):

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app/app.py

Run notebooks:

jupyter notebook notebooks

Smoke test (recommended when changing model-loading code): create a tiny script that does:

- joblib.load('artifacts/vectorizer.pkl')
- joblib.load('artifacts/model.pkl')
- transform a sample string and assert predict_proba runs without exception

## Examples from the codebase

- Model load: joblib.load('artifacts/model.pkl') (see `app/app.py`).
- Vectorizer: joblib.load('artifacts/vectorizer.pkl') and used to transform raw text before prediction.

## Integration points & dependencies

- Required (discoverable): `pandas`, `scikit-learn`, `nltk` (stopwords/tokenizers), `joblib`, `streamlit`.
- `requirements.txt` must be populated and pinned before CI or deployment. Current file is empty — treat this as a blocker for reproducible runs.
- `Dockerfile` exists but may not be up-to-date; prefer testing locally with the venv and streamlit before containerizing.

## Safe edit guidance for agents

- When changing model I/O, update or add a smoke test script under `scripts/` (create if missing). Keep artifact filenames consistent with `artifacts/`.
- When editing `app/app.py`, preserve the public UI contract: a single text box input and an "Analizar"/predict button that returns label + probability.
- Don't commit large data files into `data/`.

## Where to look for more context

- `README.md` — project goals and intended UI flow.
- `notebooks/` — exact preprocessing/tokenization used in training.
- `artifacts/` — example saved artifacts to reproduce predictions locally.

## If something is missing

- If `requirements.txt` is not filled, populate it with minimal pinned packages: e.g. `pandas`, `scikit-learn`, `nltk`, `joblib`, `streamlit` and run `pip install -r requirements.txt` to verify.
- If `app/app.py` fails to load `artifacts/` by name, check `notebooks/03_guardar_modelo.ipynb` for the original save path used by training.

---

If you'd like, I can (pick one):

- add a `scripts/smoke_test.py` that loads `artifacts/*` and runs a sample prediction, or
- populate a suggested `requirements.txt` with pinned versions and validate `streamlit run app/app.py` in this environment.

Tell me which and I'll implement it next.

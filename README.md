# 🧪 Toxic Comment Classifier

## Description

A lightweight, demo-ready application that classifies text comments as "Toxic" or "Not Toxic" using a machine learning model serialized with `joblib`, and a minimal web UI built with Streamlit. Ideal as a portfolio project that demonstrates an end-to-end flow: data exploration → training → packaging → deployment.

This repo emphasizes reproducibility: trained artifacts live in `artifacts/` and the app runs locally or inside a container.

## Quick demo

- Start the app, type or paste a comment into the text area and click "Classify".
- The app returns either "Toxic ⚠️" or "Not Toxic ✅".

> Note: This repository includes `artifacts/model.pkl` and `artifacts/vectorizer.pkl` so you can try the demo immediately.

## Basic usage

### Web UI (Streamlit)

Open the app in your browser, type or paste a comment into the text box and click "Classify". The interface shows the predicted label and basic feedback icons.

### Programmatic example

Load the model and vectorizer from `artifacts/` and run a quick inference from a Python script:

```python
import joblib

vectorizer = joblib.load('artifacts/vectorizer.pkl')
clf = joblib.load('artifacts/model.pkl')

comment = "I don't like your attitude"
X = vectorizer.transform([comment])
pred = clf.predict(X)[0]
print('Toxic' if pred == 1 else 'Not Toxic')
```

This snippet is useful to integrate the classifier into other applications or tests.

## Main features

- Binary comment classification: Toxic / Not Toxic.
- Interactive Streamlit UI for immediate demo and manual testing.
- Pre-saved model and vectorizer artifacts for instant inference.
- Dockerfile for reproducible containerized runs.

## Technologies

- Python 3.11
- scikit-learn
- joblib
- Streamlit
- pandas, numpy
- Docker

Exact dependency versions are listed in `requirements.txt`.

## Project structure

- `app/` — Streamlit app entrypoint (`app.py`)
- `artifacts/` — serialized artifacts: `model.pkl`, `vectorizer.pkl`, `data_splits.pkl`
- `data/` — dataset (local usage; avoid committing large datasets to Git)
- `notebooks/` — Jupyter notebooks for exploration and training
- `requirements.txt` — Python dependencies
- `Dockerfile` — Container build definition
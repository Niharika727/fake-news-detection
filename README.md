# Fake News Detection (ML, Python)

## Problem Statement
Misinformation spreads rapidly across digital platforms, undermining public trust and decision‑making. The goal of this project is to **build and evaluate a machine‑learning model that classifies a news article (headline or full text) as FAKE or REAL**. The system should:
- Ingest a labeled dataset of articles with text and binary labels (`fake`/`real`).
- Clean and normalize text.
- Learn a classifier using TF‑IDF features.
- Report accuracy, precision, recall, F1, ROC‑AUC on a held‑out test set.
- Provide a simple CLI and a Streamlit UI for interactive predictions.

## Project Structure
```
fake-news-detection/
  ├─ README.md
  ├─ requirements.txt
  ├─ data/
  │   └─ sample_fake_news.csv
  ├─ models/
  │   └─ (created at train time)
  ├─ src/
  │   ├─ data.py
  │   ├─ preprocess.py
  │   ├─ train.py
  │   ├─ evaluate.py
  │   ├─ predict.py
  │   └─ app.py
  └─ tests/
      └─ test_preprocess.py
```

## Quickstart
1. **Install**
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
2. **Train**
```bash
python src/train.py --data data/sample_fake_news.csv --text_col text --label_col label --model_dir models
```
3. **Evaluate**
```bash
python src/evaluate.py --data data/sample_fake_news.csv --model_dir models
```
4. **CLI Predict**
```bash
python src/predict.py --model_dir models --text "Breaking: Scientists confirm water is dry"
```
5. **Streamlit App**
```bash
streamlit run src/app.py -- --model_dir models
```

## Data Format
CSV with at least two columns:
- `text`: article text or headline
- `label`: `fake` or `real` (case‑insensitive)

You can replace `data/sample_fake_news.csv` with your dataset (e.g., Kaggle Fake/Real News).

## Notes
- Baseline model: `TfidfVectorizer` + `LinearSVC` (strong linear baseline). Also exports calibrated probabilities with `CalibratedClassifierCV` for ROC‑AUC.
- Clean text via lowercasing, URL, number, punctuation handling, and basic stopword removal.
- Easily swappable model—try `LogisticRegression` or `SGDClassifier`.
```
# World Cup Prediction Dashboard - Streamlit Cloud Deploy

This folder is a standalone deployment copy for Streamlit Community Cloud. It intentionally does not depend on the parent project folder.

## What Is Included

- `app.py` as the Streamlit entrypoint.
- `src/worldcup_prediction/` with the dashboard model, data pipeline, odds, schedule, and evaluation code.
- `data/processed/` with the current processed CSV/JSON snapshot so the app loads immediately.
- Small raw fallback files: historical results, goalscorers, FIFA hub HTML, and StatsBomb competitions metadata.
- `requirements.txt` for Streamlit Cloud dependency installation.

Large raw PDF and StatsBomb event caches are intentionally excluded.

## Run Locally From This Folder

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

## Deploy Summary

1. Create a new GitHub repository from this folder.
2. Push this folder's contents to that repository.
3. In Streamlit Community Cloud, create an app from the repository.
4. Use `app.py` as the main file.

No API key is required for the default public schedule and public ESPN odds flow. If you later use The Odds API, add `ODDS_API_KEY` through Streamlit Cloud secrets rather than committing it.

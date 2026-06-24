# GitHub and Streamlit Cloud Deployment Steps

## 1. Open This Deployment Folder

```bash
cd /Users/kyle/Documents/MyProjects/WorldCupPrediction/streamlit_cloud_deploy
```

## 2. Sanity Check It Locally

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Open the local URL shown by Streamlit. Stop the app with `Ctrl+C`.

## 3. Create A New Git Repo

```bash
git init
git branch -M main
git add .
git commit -m "Initial Streamlit Cloud deployment"
```

## 4. Create A New GitHub Repo

Recommended repo name:

```text
worldcup-prediction-streamlit
```

Create it on GitHub as a new empty repository. Do not add a README, `.gitignore`, or license in the GitHub UI because this folder already contains the needed files.

Then connect and push:

```bash
git remote add origin https://github.com/YOUR_USERNAME/worldcup-prediction-streamlit.git
git push -u origin main
```

If you use the GitHub CLI, this single command can replace the GitHub UI and remote steps:

```bash
gh repo create worldcup-prediction-streamlit --public --source=. --remote=origin --push
```

Use `--private` instead of `--public` if you do not want the repo public. Streamlit Community Cloud can deploy private GitHub repos if your account grants access.

## 5. Deploy On Streamlit Community Cloud

1. Go to `https://share.streamlit.io/`.
2. Sign in with GitHub.
3. Click `Create app`.
4. Choose `Deploy a public app from GitHub` or select your connected repository.
5. Repository: `YOUR_USERNAME/worldcup-prediction-streamlit`.
6. Branch: `main`.
7. Main file path: `app.py`.
8. Click `Deploy`.

Streamlit Cloud should detect `requirements.txt` and install the Python dependencies automatically.

## 6. Optional Secrets

The app works without an odds API key. If you later want optional The Odds API support:

1. Open the deployed app settings in Streamlit Cloud.
2. Go to `Secrets`.
3. Add:

```toml
ODDS_API_KEY = "your-key-here"
```

4. Save and reboot the app.

Do not commit secrets into GitHub.

## 7. Updating The App Later

From this folder:

```bash
git add .
git commit -m "Update dashboard"
git push
```

Streamlit Cloud redeploys after the push.

If you update the parent/local project later, copy the changed files into this folder intentionally. This deployment folder is a snapshot and will not update automatically.

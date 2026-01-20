# Streamlit Cloud Deployment Guide

## Quick Deploy Steps

### 1. Push Code to GitHub
```bash
cd /Users/anushkadhiman/Desktop/work/AI-impactSense
git add -A
git commit -m "Fix model loading paths and add joblib dependency"
git push origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set branch to `main`
6. Set main file path to `app.py`
7. Click "Deploy"

### 3. If Deployment Fails
- On Streamlit Cloud: Settings → Clear cache → Restart
- Or delete and redeploy the app

## Files Required in GitHub
- ✅ `app.py`
- ✅ `earthquake_impact_rf.pkl`
- ✅ `feature_order.pkl`
- ✅ `requirements.txt`

## Troubleshooting Model Error
If you still see "No such file or directory: 'earthquake_impact_rf.pkl'":

1. Check if model files are committed to GitHub:
   ```bash
   git ls-files | grep pkl
   ```
   Should show both `.pkl` files

2. If files are not tracked:
   ```bash
   git add earthquake_impact_rf.pkl feature_order.pkl
   git commit -m "Add model files"
   git push
   ```

3. Redeploy on Streamlit Cloud

## Test Locally
```bash
streamlit run app.py
```


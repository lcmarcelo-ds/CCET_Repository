# National CCET Data Analytics Dashboard

Streamlit dashboard for the National Climate Change Expenditure Tagging (CCET) PAP-level dataset, FY2017–FY2026.

## 1. Project structure

```text
ccet_github_dashboard/
├── app.py
├── requirements.txt
├── data/
│   └── Cleaned_National_CCET_PAPs_FY_2017_to_2026.xlsx
└── .streamlit/
    └── config.toml
```

## 2. Run locally

```bash
cd ccet_github_dashboard
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
streamlit run app.py
```

## 3. Push to GitHub

```bash
git init
git add .
git commit -m "Initial CCET Streamlit dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ccet-dashboard.git
git push -u origin main
```

## 4. Deploy to Streamlit Community Cloud

1. Go to Streamlit Community Cloud.
2. Click **New app**.
3. Select your GitHub repository.
4. Main file path: `app.py`.
5. Click **Deploy**.

## 5. Dashboard pages

- Executive Overview
- Budget Trends
- Agency Explorer
- Typology / NCCAP Analysis
- Compliance / Participation Proxy
- Data Quality Checks

## 6. Notes

The app currently loads the Excel file directly. When the database is ready, replace the `load_data()` function in `app.py` with a SQL query function.

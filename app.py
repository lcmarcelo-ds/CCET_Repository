
import os
import re
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ModuleNotFoundError:
    REPORTLAB_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="National CCET Smart Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_CANDIDATES = [
    "data/Cleaned_National_CCET_PAPs_FY_2017_to_2026.csv"
]

# Cleaned CCC PAP-level dataset stores amounts in thousand pesos.
DATASET_VALUE_MULTIPLIER = 1_000

PRIMARY_BLUE = "#17365D"
DEEP_BLUE = "#1F4E79"
MID_BLUE = "#5B9BD5"
LIGHT_BLUE = "#D9EAF7"
GREEN = "#70AD47"
ORANGE = "#ED7D31"
YELLOW = "#FFC000"
TEAL = "#00A6A6"
PURPLE = "#8064A2"
GRAY = "#A5A5A5"
DARK_GRAY = "#666666"
LIGHT_GRAY = "#E6ECF5"
DARK_TEXT = "#1F1F1F"
CARD_BG = "#F8FBFF"

NCCAP_PRIORITY = {
    "1": "Food Security",
    "2": "Water Sufficiency",
    "3": "Ecosystem & Environmental Stability",
    "4": "Human Security",
    "5": "Climate-Smart Industries & Services",
    "6": "Sustainable Energy",
    "7": "Knowledge & Capacity Development",
    "8": "Cross-Cutting",
}

NCCAP_ORDER = list(NCCAP_PRIORITY.values()) + ["Unclassified"]

PDP_KEYWORDS = [
    "climate", "resilience", "disaster", "risk reduction", "adaptation", "mitigation",
    "flood", "drainage", "water", "irrigation", "food security", "agriculture",
    "renewable", "energy efficiency", "sustainable", "environment", "ecosystem",
    "biodiversity", "green", "carbon", "emission", "hazard", "watershed",
    "low carbon", "loss and damage", "vulnerability", "disaster risk", "sustainable energy",
]

INSTITUTION_COLORS = {
    "NGA": DEEP_BLUE,
    "GOCC": GREEN,
    "SUC": ORANGE,
    "Unclassified": GRAY,
}

PILLAR_COLORS = {
    "Adaptation": DEEP_BLUE,
    "Mitigation": GREEN,
    "Unclassified": GRAY,
}

PRIORITY_COLORS = {
    "Food Security": DEEP_BLUE,
    "Water Sufficiency": MID_BLUE,
    "Ecosystem & Environmental Stability": GREEN,
    "Human Security": YELLOW,
    "Climate-Smart Industries & Services": ORANGE,
    "Sustainable Energy": TEAL,
    "Knowledge & Capacity Development": PURPLE,
    "Cross-Cutting": GRAY,
    "Unclassified": "#C9C9C9",
}

NATIONAL_BUDGET_REFERENCE_B = pd.DataFrame({
    "Fiscal_Year": [2022, 2023, 2024, 2025],
    "Total National Budget (Billion Pesos)": [5023.00, 5268.00, 5768.00, 6326.00],
})

COLUMN_DICTIONARY = {
    "Fiscal_Year": "Fiscal year covered by the climate-tagged PAP record.",
    "Type": "Budget classification such as NEP, GAA, or Actual.",
    "DEPARTMENT": "Parent department or sector of the implementing agency.",
    "GRIT TAGGING": "Institution type used in the assessment: NGA, SUC, GOCC, or Unclassified.",
    "AGENCY": "Implementing or reporting national government institution.",
    "Agency Unit": "Derived combined department-agency-institution identifier used for counting unique NGIs.",
    "Agency Label": "Derived readable label combining department and agency to avoid generic agency names.",
    "PAP ID": "Program, Activity, or Project identifier.",
    "PAP Description": "Name or description of the climate-tagged PAP.",
    "TYPOLOGY ID": "CCET typology code used to classify the PAP.",
    "TYPOLOGY Description": "Description of the assigned CCET typology.",
    "ADAPTATION": "Raw amount tagged for climate change adaptation. Dataset values are treated as thousand pesos.",
    "MITIGATION": "Raw amount tagged for climate change mitigation. Dataset values are treated as thousand pesos.",
    "TOTAL": "Raw total climate-tagged amount. Dataset values are treated as thousand pesos.",
    "Climate Pillar": "Derived from typology ID or amount fields: Adaptation, Mitigation, or Unclassified.",
    "NCCAP Code": "Derived NCCAP priority code from the CCET typology ID.",
    "NCCAP Priority": "Derived NCCAP thematic priority.",
    "PDP / Executive Agenda Alignment": "Keyword-based analytical proxy for possible alignment with national climate/development priorities.",
}

FGD_KII_INSIGHTS = pd.DataFrame([
    {
        "Theme": "Knowledge concentration and continuity",
        "Institution Type": "NGA / GOCC / SUC",
        "Challenge": "CCET knowledge and responsibilities are concentrated among select personnel; turnover weakens continuity.",
        "Smart Dashboard Response": "Add institution-type filters, training/familiarity indicators, and agency-level implementation readiness notes.",
        "Recommendation": "Establish minimum institutional requirements, designate CCET focal teams, and institutionalize onboarding/refresher training.",
        "Priority": "High",
        "Budget Cycle Stage": "Preparation",
    },
    {
        "Theme": "Climate relevance and attribution ambiguity",
        "Institution Type": "NGA / GOCC / SUC",
        "Challenge": "Agencies may differ in tagging whole PAPs versus climate-relevant components; attribution rules remain unevenly understood.",
        "Smart Dashboard Response": "Flag large blanket-tagged PAPs, add attribution-method fields, and provide PAP-level review tables.",
        "Recommendation": "Develop detailed, context-specific tagging and proportional attribution guidance similar to a structured scoring approach.",
        "Priority": "High",
        "Budget Cycle Stage": "Preparation",
    },
    {
        "Theme": "Budget-cycle traceability",
        "Institution Type": "NGA / GOCC / SUC",
        "Challenge": "Users need to trace PAPs from NEP to GAA to Actual, but records are difficult to reconcile across stages.",
        "Smart Dashboard Response": "Add NEP-GAA-Actual pivot tables, variance charts, and downloadable reconciliation outputs.",
        "Recommendation": "Institutionalize feedback and reconciliation mechanisms for PAP-level budget traceability.",
        "Priority": "High",
        "Budget Cycle Stage": "Legislation / Execution",
    },
    {
        "Theme": "Climate results tracking",
        "Institution Type": "NGA / GOCC / SUC",
        "Challenge": "The current data show tagged budgets but provide limited evidence on climate outputs, outcomes, or results.",
        "Smart Dashboard Response": "Add placeholders for climate indicators, accomplishment reports, M&E readiness, and audit linkages.",
        "Recommendation": "Integrate climate indicators into agency planning, M&E, and accomplishment reporting frameworks.",
        "Priority": "High",
        "Budget Cycle Stage": "Execution / Accountability",
    },
    {
        "Theme": "Limited use in budget deliberations",
        "Institution Type": "NGA / GOCC",
        "Challenge": "CCET outputs are perceived to have limited influence in budget review and deliberation processes.",
        "Smart Dashboard Response": "Generate budget-stage summaries, variance insights, and concise policy briefs for deliberation support.",
        "Recommendation": "Strengthen strategic utilization of CCET data during budget preparation, technical budget hearings, and policy review.",
        "Priority": "Medium",
        "Budget Cycle Stage": "Legislation",
    },
    {
        "Theme": "Audit guideline awareness gaps",
        "Institution Type": "NGA / GOCC / SUC",
        "Challenge": "There are gaps in awareness and application of audit guidelines for climate-tagged expenditures.",
        "Smart Dashboard Response": "Add audit-readiness indicators and data quality flags that can guide agency validation.",
        "Recommendation": "Review, clarify, and disseminate climate expenditure audit guidance with CCC, DBM, COA, and agencies.",
        "Priority": "Medium",
        "Budget Cycle Stage": "Accountability",
    },
    {
        "Theme": "Administrative burden and fragmented systems",
        "Institution Type": "NGA / GOCC / SUC",
        "Challenge": "Separate requirements and systems can make tagging, documentation, and validation administratively heavy.",
        "Smart Dashboard Response": "Move toward an integrated dashboard/platform concept with common templates, downloads, and validation checks.",
        "Recommendation": "Streamline administrative requirements through an integrated CCET platform and standardized data templates.",
        "Priority": "Medium",
        "Budget Cycle Stage": "Preparation / Reporting",
    },
])

INTERNATIONAL_LESSONS = pd.DataFrame([
    {"Country / Case": "Nepal", "Dashboard Lesson": "Use clearer tagging methodologies and improve coordination between central and sector agencies.", "Design Option": "Add method notes, attribution fields, and ministry-level ranking."},
    {"Country / Case": "Bangladesh", "Dashboard Lesson": "Use climate budget information strategically in planning and accountability.", "Design Option": "Add executive briefs and budget-cycle interpretation notes."},
    {"Country / Case": "Indonesia", "Dashboard Lesson": "Strengthen transparency and sector-level use of climate budget data.", "Design Option": "Add downloadable tables, open data outputs, and sector/priority summaries."},
    {"Country / Case": "France", "Dashboard Lesson": "Move toward more nuanced tagging, including favorable and unfavorable budget effects.", "Design Option": "Future enhancement: add scoring/weighting and climate relevance confidence level."},
])

BUDGET_CYCLE_STAGES = pd.DataFrame([
    {"Stage": "Preparation", "CCET Focus": "Agency tagging, typology selection, QAR evidence, budget forms", "Dashboard Module": "PAP Explorer, NCCAP Priorities, Data Quality"},
    {"Stage": "Legislation", "CCET Focus": "NEP-to-GAA movement and deliberation changes", "Dashboard Module": "Budget Cycle Variance, Agency Concentration"},
    {"Stage": "Execution", "CCET Focus": "Actual expenditure/utilization where available", "Dashboard Module": "Actual vs GAA Gap, Budget Stage Comparison"},
    {"Stage": "Accountability", "CCET Focus": "Accomplishment, auditability, validation, feedback", "Dashboard Module": "Data Quality, Recommendations, M&E Readiness"},
])


# ============================================================
# CSS FOR A CLEANER STREAMLIT LOOK
# ============================================================

st.markdown(
    f"""
    <style>
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2.0rem;
        max-width: 1500px;
    }}
    div[data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border: 1px solid #DDEAF7;
        padding: 16px 18px;
        border-radius: 16px;
        box-shadow: 0 1px 4px rgba(31, 78, 121, 0.06);
    }}
    div[data-testid="stMetricLabel"] p {{
        color: {PRIMARY_BLUE};
        font-weight: 700;
    }}
    .chart-card {{
        background: #FFFFFF;
        border: 1px solid #DDEAF7;
        border-radius: 18px;
        padding: 18px 18px 12px 18px;
        margin-bottom: 22px;
        box-shadow: 0 1px 6px rgba(31, 78, 121, 0.06);
    }}
    .chart-title {{
        font-size: 1.14rem;
        font-weight: 750;
        color: {PRIMARY_BLUE};
        margin-bottom: 0.12rem;
    }}
    .chart-subtitle {{
        color: #4F5B66;
        font-size: 0.92rem;
        margin-bottom: 0.6rem;
    }}
    .source-note {{
        color: #64707D;
        font-size: 0.78rem;
        line-height: 1.3;
        margin-top: -0.25rem;
    }}
    .interpretation-box {{
        background: #FFF8E8;
        border-left: 5px solid {ORANGE};
        border-radius: 12px;
        padding: 14px 16px;
        margin: 8px 0 16px 0;
        color: #3B3B3B;
    }}
    .method-box {{
        background: #F7FBFF;
        border-left: 5px solid {DEEP_BLUE};
        border-radius: 12px;
        padding: 14px 16px;
        margin: 8px 0 16px 0;
        color: #263238;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BASIC FORMATTERS
# ============================================================


def clean_string(value):
    if pd.isna(value):
        return ""
    return str(value).replace("\ufeff", "").replace("\xa0", " ").strip()


def normalize_colname(col):
    col = clean_string(col)
    col = re.sub(r"\s+", " ", col)
    return col


def as_actual_pesos(raw_dataset_value):
    return float(raw_dataset_value or 0) * DATASET_VALUE_MULTIPLIER


def raw_to_billion(raw_dataset_value):
    return as_actual_pesos(raw_dataset_value) / 1_000_000_000


def raw_to_trillion(raw_dataset_value):
    return as_actual_pesos(raw_dataset_value) / 1_000_000_000_000


def peso_from_raw(raw_dataset_value):
    value = as_actual_pesos(raw_dataset_value)
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000_000_000:
        return f"₱{value / 1_000_000_000_000:,.3f}T"
    if abs(value) >= 1_000_000_000:
        return f"₱{value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"₱{value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"₱{value / 1_000:,.2f}K"
    return f"₱{value:,.0f}"


def peso_billion(value):
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1000:
        return f"₱{value / 1000:,.3f}T"
    return f"₱{value:,.2f}B"


def signed_peso_billion(value):
    if pd.isna(value):
        return "N/A"
    sign = "+" if value > 0 else ""
    if abs(value) >= 1000:
        return f"{sign}₱{value / 1000:,.3f}T"
    return f"{sign}₱{value:,.2f}B"


def pct(value, decimals=1):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"


def safe_divide(numerator, denominator):
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return np.nan
    return numerator / denominator


# ============================================================
# DATA PREPARATION
# ============================================================


def classify_pdp_alignment(text):
    text = str(text).lower()
    hits = sum(1 for kw in PDP_KEYWORDS if kw in text)
    if hits >= 3:
        return "Strongly Aligned"
    if hits >= 1:
        return "Partially Aligned"
    return "Weak / Unclassified"


def detect_and_rename_columns(df):
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]

    rename_map = {}
    for col in df.columns:
        key = col.upper().replace("_", " ").replace("-", " ")
        key = re.sub(r"\s+", " ", key).strip()
        compact = key.replace(" ", "")

        if key in {"FISCAL YEAR", "FISCAL_YEAR", "FY"} or compact == "FISCALYEAR":
            rename_map[col] = "Fiscal_Year"
        elif key == "ADAPTION":
            rename_map[col] = "ADAPTATION"
        elif key == "ADAPTATION":
            rename_map[col] = "ADAPTATION"
        elif key == "MITIGATION":
            rename_map[col] = "MITIGATION"
        elif key == "TOTAL":
            rename_map[col] = "TOTAL"
        elif key == "GRIT TAGGING" or compact in {"GRITTAGGING", "INSTITUTIONTYPE", "INSTITUTIONTAGGING"}:
            rename_map[col] = "GRIT TAGGING"
        elif key == "DEPARTMENT":
            rename_map[col] = "DEPARTMENT"
        elif key == "AGENCY":
            rename_map[col] = "AGENCY"
        elif key in {"PAP ID", "PAPID"} or compact == "PAPID":
            rename_map[col] = "PAP ID"
        elif key in {"PAP DESCRIPTION", "PAPDESC"} or compact == "PAPDESCRIPTION":
            rename_map[col] = "PAP Description"
        elif key in {"TYPOLOGY ID", "TYPOLOGYID"} or compact == "TYPOLOGYID":
            rename_map[col] = "TYPOLOGY ID"
        elif key in {"TYPOLOGY DESCRIPTION", "TYPOLOGYDESC"} or compact == "TYPOLOGYDESCRIPTION":
            rename_map[col] = "TYPOLOGY Description"
        elif key == "TYPE":
            rename_map[col] = "Type"

    df = df.rename(columns=rename_map)
    return df


def standardize_type(value):
    v = clean_string(value).upper()
    if v in {"GAA", "GENERAL APPROPRIATIONS ACT"}:
        return "GAA"
    if v in {"NEP", "NATIONAL EXPENDITURE PROGRAM"}:
        return "NEP"
    if v in {"ACTUAL", "ACTUALS", "ACTUAL EXPENDITURE", "ACTUAL EXPENDITURES"}:
        return "Actual"
    return clean_string(value) if clean_string(value) else "Unclassified"


def standardize_grit(value):
    v = clean_string(value).upper()
    v = v.replace("NATIONAL GOVERNMENT AGENCY", "NGA")
    v = v.replace("NATIONAL GOVERNMENT AGENCIES", "NGA")
    v = v.replace("STATE UNIVERSITIES AND COLLEGES", "SUC")
    v = v.replace("STATE UNIVERSITY AND COLLEGE", "SUC")
    v = v.replace("GOVERNMENT OWNED AND CONTROLLED CORPORATION", "GOCC")
    v = v.replace("GOVERNMENT-OWNED AND CONTROLLED CORPORATION", "GOCC")
    v = v.replace("GOVERNMENT-OWNED OR -CONTROLLED CORPORATION", "GOCC")
    v = v.strip()
    if v in {"", "NAN", "NONE", "NULL", "NA", "N/A"}:
        return "Unclassified"
    if "GOCC" in v:
        return "GOCC"
    if v == "SUC" or "SUC" in v:
        return "SUC"
    if v == "NGA" or "NGA" in v:
        return "NGA"
    return v


def prepare_data(df):
    df = detect_and_rename_columns(df)

    required_text_cols = [
        "Type", "DEPARTMENT", "GRIT TAGGING", "AGENCY", "PAP ID", "PAP Description",
        "TYPOLOGY ID", "TYPOLOGY Description",
    ]
    required_num_cols = ["Fiscal_Year", "ADAPTATION", "MITIGATION", "TOTAL"]

    for col in required_text_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].apply(clean_string)

    for col in required_num_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Fiscal_Year"].notna()].copy()
    df["Fiscal_Year"] = df["Fiscal_Year"].astype(int)

    for col in ["ADAPTATION", "MITIGATION", "TOTAL"]:
        df[col] = df[col].fillna(0)

    df["Type"] = df["Type"].apply(standardize_type)
    df["GRIT TAGGING"] = df["GRIT TAGGING"].apply(standardize_grit)
    df["Institution Type"] = df["GRIT TAGGING"]

    # Strengthen GRIT TAGGING recovery when the field is blank but department/agency indicates SUC or GOCC.
    agency_text = (df["DEPARTMENT"] + " " + df["AGENCY"]).str.upper()
    df.loc[(df["GRIT TAGGING"] == "Unclassified") & agency_text.str.contains("STATE UNIVERSITY|UNIVERSITY|COLLEGE", na=False), "GRIT TAGGING"] = "SUC"
    df.loc[(df["GRIT TAGGING"] == "Unclassified") & agency_text.str.contains("CORPORATION|AUTHORITY|INSURANCE|NATIONAL FOOD AUTHORITY|IRRIGATION", na=False), "GRIT TAGGING"] = "GOCC"
    df["Institution Type"] = df["GRIT TAGGING"]

    for col in ["DEPARTMENT", "AGENCY", "PAP ID", "PAP Description", "TYPOLOGY ID", "TYPOLOGY Description"]:
        df[col] = df[col].fillna("").apply(clean_string)

    typo = df["TYPOLOGY ID"].str.upper().str.replace(" ", "", regex=False)
    df["Climate Pillar"] = np.where(
        typo.str.startswith("M"),
        "Mitigation",
        np.where(typo.str.startswith("A"), "Adaptation", "Unclassified")
    )

    # Where typology is missing but amount split exists, infer pillar only for display.
    df.loc[(df["Climate Pillar"] == "Unclassified") & (df["ADAPTATION"] > 0) & (df["MITIGATION"] <= 0), "Climate Pillar"] = "Adaptation"
    df.loc[(df["Climate Pillar"] == "Unclassified") & (df["MITIGATION"] > 0) & (df["ADAPTATION"] <= 0), "Climate Pillar"] = "Mitigation"

    df["NCCAP Code"] = typo.str.extract(r"^[AM](\d)", expand=False).fillna("")
    df["NCCAP Priority"] = df["NCCAP Code"].map(NCCAP_PRIORITY).fillna("Unclassified")

    combined_text = (
        df["PAP Description"].astype(str) + " " +
        df["TYPOLOGY Description"].astype(str) + " " +
        df["AGENCY"].astype(str) + " " +
        df["DEPARTMENT"].astype(str)
    )
    df["PDP / Executive Agenda Alignment"] = combined_text.apply(classify_pdp_alignment)

    # Combined labels prevent generic agency names from appearing alone.
    df["Agency Label"] = np.where(
        df["DEPARTMENT"].str.strip().ne(""),
        df["DEPARTMENT"] + " - " + df["AGENCY"],
        df["AGENCY"],
    )
    df["Agency Label"] = df["Agency Label"].str.replace(r"\s+-\s+$", "", regex=True).str.strip()
    df["Agency Unit"] = (
        df["GRIT TAGGING"].astype(str) + " | " +
        df["DEPARTMENT"].astype(str) + " | " +
        df["AGENCY"].astype(str)
    )

    df["TOTAL_B"] = df["TOTAL"].apply(raw_to_billion)
    df["ADAPTATION_B"] = df["ADAPTATION"].apply(raw_to_billion)
    df["MITIGATION_B"] = df["MITIGATION"].apply(raw_to_billion)

    return df


@st.cache_data(show_spinner="Reading CSV file...")
def read_csv_cached(source):
    return pd.read_csv(source, encoding="utf-8-sig")


@st.cache_data(show_spinner="Loading and preparing CCET dataset...")
def load_and_prepare_from_path(path):
    return prepare_data(pd.read_csv(path, encoding="utf-8-sig"))


def find_default_data_path():
    for path in DATA_CANDIDATES:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return None


# ============================================================
# DATA AGGREGATION HELPERS
# ============================================================


def filter_options(series):
    vals = sorted([v for v in series.dropna().unique().tolist() if clean_string(v) != ""])
    return ["All"] + vals


def group_sum_b(df, group_cols, value_col="TOTAL"):
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["Amount_B"])
    out = df.groupby(group_cols, as_index=False)[value_col].sum()
    out["Amount_B"] = out[value_col].apply(raw_to_billion)
    return out


def safe_top(dataframe, group_col, value_col="TOTAL"):
    if dataframe.empty or group_col not in dataframe.columns:
        return "No data", 0, np.nan
    d = dataframe.groupby(group_col, as_index=False)[value_col].sum().sort_values(value_col, ascending=False)
    if d.empty or d[value_col].sum() == 0:
        return "No data", 0, np.nan
    row = d.iloc[0]
    return row[group_col], row[value_col], row[value_col] / d[value_col].sum() * 100


def budget_stage_pivot(dataframe, years=None):
    d = dataframe.copy()
    if years is not None:
        d = d[d["Fiscal_Year"].isin(years)]
    d = d[d["Type"].isin(["NEP", "GAA", "Actual"])]
    if d.empty:
        return pd.DataFrame(columns=["Fiscal_Year", "NEP", "GAA", "Actual"])
    p = d.groupby(["Fiscal_Year", "Type"], as_index=False)["TOTAL"].sum()
    p["Amount_B"] = p["TOTAL"].apply(raw_to_billion)
    pivot = p.pivot(index="Fiscal_Year", columns="Type", values="Amount_B").reset_index()
    for col in ["NEP", "GAA", "Actual"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot = pivot[["Fiscal_Year", "NEP", "GAA", "Actual"]].sort_values("Fiscal_Year")
    pivot["GAA minus NEP"] = pivot["GAA"] - pivot["NEP"]
    pivot["Actual minus GAA"] = pivot["Actual"] - pivot["GAA"]
    pivot["Actual vs GAA (%)"] = (pivot["Actual"] - pivot["GAA"]) / pivot["GAA"] * 100
    return pivot


def ranking_base(dataframe, selected_type="GAA"):
    if selected_type == "Use active filters":
        return dataframe.copy()
    if selected_type in ["GAA", "NEP", "Actual"]:
        return dataframe[dataframe["Type"] == selected_type].copy()
    return dataframe.copy()


def build_quality_flags(dataframe):
    f = dataframe.copy()
    flags = {
        "Missing department": f["DEPARTMENT"].eq("").sum(),
        "Missing agency": f["AGENCY"].eq("").sum(),
        "Missing GRIT TAGGING / institution type": f["GRIT TAGGING"].eq("Unclassified").sum(),
        "Missing PAP ID": f["PAP ID"].eq("").sum(),
        "Missing typology ID": f["TYPOLOGY ID"].eq("").sum(),
        "Missing typology description": f["TYPOLOGY Description"].eq("").sum(),
        "Zero or blank total": (f["TOTAL"].fillna(0) == 0).sum(),
        "Negative total": (f["TOTAL"].fillna(0) < 0).sum(),
        "Adaptation + Mitigation ≠ Total": ((f["ADAPTATION"].fillna(0) + f["MITIGATION"].fillna(0) - f["TOTAL"].fillna(0)).abs() > 1).sum(),
        "Possible duplicate PAP-stage records": f.duplicated(
            subset=["Fiscal_Year", "Type", "GRIT TAGGING", "DEPARTMENT", "AGENCY", "PAP ID", "TYPOLOGY ID"],
            keep=False,
        ).sum(),
        "Unclassified NCCAP priority": f["NCCAP Priority"].eq("Unclassified").sum(),
    }
    return pd.DataFrame({"Data Quality Check": list(flags.keys()), "Flagged Records": list(flags.values())})


def quality_mask(dataframe, issue):
    f = dataframe.copy()
    if issue == "Missing department":
        return f["DEPARTMENT"].eq("")
    if issue == "Missing agency":
        return f["AGENCY"].eq("")
    if issue == "Missing GRIT TAGGING / institution type":
        return f["GRIT TAGGING"].eq("Unclassified")
    if issue == "Missing PAP ID":
        return f["PAP ID"].eq("")
    if issue == "Missing typology ID":
        return f["TYPOLOGY ID"].eq("")
    if issue == "Missing typology description":
        return f["TYPOLOGY Description"].eq("")
    if issue == "Zero or blank total":
        return f["TOTAL"].fillna(0) == 0
    if issue == "Negative total":
        return f["TOTAL"].fillna(0) < 0
    if issue == "Adaptation + Mitigation ≠ Total":
        return ((f["ADAPTATION"].fillna(0) + f["MITIGATION"].fillna(0) - f["TOTAL"].fillna(0)).abs() > 1)
    if issue == "Possible duplicate PAP-stage records":
        return f.duplicated(subset=["Fiscal_Year", "Type", "GRIT TAGGING", "DEPARTMENT", "AGENCY", "PAP ID", "TYPOLOGY ID"], keep=False)
    if issue == "Unclassified NCCAP priority":
        return f["NCCAP Priority"].eq("Unclassified")
    return pd.Series(False, index=f.index)


# ============================================================
# CHART STYLE HELPERS
# ============================================================


def polish_fig(fig, height=620, legend="bottom", left=80, right=55, top=25, bottom=80):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=left, r=right, t=top, b=bottom),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12, color=DARK_TEXT),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
    )
    if legend == "bottom":
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.16,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#D9E2F3",
                borderwidth=1,
                font=dict(size=10),
            )
        )
    elif legend == "right":
        fig.update_layout(
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#D9E2F3",
                borderwidth=1,
                font=dict(size=10),
            )
        )
    elif legend == "none":
        fig.update_layout(showlegend=False)

    fig.update_xaxes(automargin=True, showline=True, linewidth=1, linecolor="#BFBFBF", gridcolor=LIGHT_GRAY)
    fig.update_yaxes(automargin=True, showline=True, linewidth=1, linecolor="#BFBFBF", gridcolor=LIGHT_GRAY)
    return fig


def chart_card(title, subtitle, fig, file_stem, source_note=None, height=620, width=1450):
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="chart-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {"format": "png", "filename": file_stem, "height": height, "width": width, "scale": 3},
        },
    )
    if source_note:
        st.markdown(f'<div class="source-note">{source_note}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        st.download_button(
            "HTML",
            data=fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8"),
            file_name=f"{file_stem}.html",
            mime="text/html",
            key=f"{file_stem}_html",
            use_container_width=True,
        )
    with c2:
        try:
            png_bytes = fig.to_image(format="png", width=width, height=height, scale=3)
            st.download_button(
                "PNG",
                data=png_bytes,
                file_name=f"{file_stem}.png",
                mime="image/png",
                key=f"{file_stem}_png",
                use_container_width=True,
            )
        except Exception:
            st.caption("PNG export requires `kaleido`.")
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# FIGURE BUILDERS
# ============================================================


def fig_climate_budget_share(df):
    gaa = df[(df["Type"] == "GAA") & (df["Fiscal_Year"].isin(NATIONAL_BUDGET_REFERENCE_B["Fiscal_Year"]))]
    d = gaa.groupby("Fiscal_Year", as_index=False)["TOTAL"].sum()
    d["Climate-Tagged GAA (Billion Pesos)"] = d["TOTAL"].apply(raw_to_billion)
    d = NATIONAL_BUDGET_REFERENCE_B.merge(d[["Fiscal_Year", "Climate-Tagged GAA (Billion Pesos)"]], on="Fiscal_Year", how="left")
    d["Climate-Tagged GAA (Billion Pesos)"] = d["Climate-Tagged GAA (Billion Pesos)"].fillna(0)
    d["Climate-Tagged GAA (Trillion Pesos)"] = d["Climate-Tagged GAA (Billion Pesos)"] / 1000
    d["Total National Budget (Trillion Pesos)"] = d["Total National Budget (Billion Pesos)"] / 1000
    d["Share of National Budget (%)"] = d["Climate-Tagged GAA (Billion Pesos)"] / d["Total National Budget (Billion Pesos)"] * 100
    d["FY"] = "FY" + d["Fiscal_Year"].astype(str)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=d["FY"], y=d["Climate-Tagged GAA (Trillion Pesos)"], name="Climate-tagged GAA",
            marker=dict(color=DEEP_BLUE, line=dict(color="white", width=1)),
            hovertemplate="<b>%{x}</b><br>Climate-tagged GAA: ₱%{customdata:,.2f}B<extra></extra>",
            customdata=d["Climate-Tagged GAA (Billion Pesos)"],
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=d["FY"], y=d["Total National Budget (Trillion Pesos)"], name="Total national budget",
            marker=dict(color=GRAY, line=dict(color="white", width=1)),
            hovertemplate="<b>%{x}</b><br>Total national budget: ₱%{customdata:,.2f}B<extra></extra>",
            customdata=d["Total National Budget (Billion Pesos)"],
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=d["FY"], y=d["Share of National Budget (%)"], name="Share of national budget",
            mode="lines+markers",
            line=dict(color=ORANGE, width=3), marker=dict(size=9, color=ORANGE),
            hovertemplate="<b>%{x}</b><br>Share: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Amount (trillion pesos)", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="Share (%)", ticksuffix="%", secondary_y=True, rangemode="tozero", showgrid=False)
    fig.update_xaxes(title_text="Fiscal year")
    return polish_fig(fig, height=610, legend="bottom", left=75, right=80, bottom=110)


def fig_participation_by_year(df):
    d = df.groupby(["Fiscal_Year", "GRIT TAGGING"], as_index=False)["Agency Unit"].nunique()
    d = d.rename(columns={"Agency Unit": "Participating NGIs"})
    fig = px.bar(
        d,
        x="Fiscal_Year",
        y="Participating NGIs",
        color="GRIT TAGGING",
        color_discrete_map=INSTITUTION_COLORS,
        barmode="stack",
        hover_data={"Fiscal_Year": True, "GRIT TAGGING": True, "Participating NGIs": ":,.0f"},
    )
    fig.update_xaxes(title_text="Fiscal year", type="category")
    fig.update_yaxes(title_text="Unique participating institutions")
    return polish_fig(fig, height=600, legend="bottom", bottom=115)


def fig_budget_by_institution(df):
    d = df.groupby(["Fiscal_Year", "GRIT TAGGING"], as_index=False)["TOTAL"].sum()
    d["Amount_B"] = d["TOTAL"].apply(raw_to_billion)
    fig = px.bar(
        d,
        x="Fiscal_Year",
        y="Amount_B",
        color="GRIT TAGGING",
        color_discrete_map=INSTITUTION_COLORS,
        barmode="stack",
        hover_data={"Amount_B": ":,.2f", "TOTAL": False},
    )
    fig.update_xaxes(title_text="Fiscal year", type="category")
    fig.update_yaxes(title_text="Climate-tagged amount (billion pesos)")
    return polish_fig(fig, height=600, legend="bottom", bottom=115)


def fig_budget_stage_comparison(pivot):
    d = pivot.copy()
    d["FY"] = "FY" + d["Fiscal_Year"].astype(str)
    fig = go.Figure()
    stage_colors = {"NEP": DEEP_BLUE, "GAA": MID_BLUE, "Actual": GREEN}
    for stage in ["NEP", "GAA", "Actual"]:
        fig.add_trace(go.Bar(
            x=d["FY"], y=d[stage], name=stage,
            marker=dict(color=stage_colors[stage], line=dict(color="white", width=1)),
            hovertemplate=f"<b>%{{x}}</b><br>{stage}: ₱%{{y:,.2f}}B<extra></extra>",
        ))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Fiscal year")
    fig.update_yaxes(title_text="Amount (billion pesos)", rangemode="tozero")
    return polish_fig(fig, height=620, legend="bottom", bottom=115)


def fig_budget_variance(pivot):
    d = pivot.copy()
    d["FY"] = "FY" + d["Fiscal_Year"].astype(str)
    fig = go.Figure()
    for col, color in [("GAA minus NEP", ORANGE), ("Actual minus GAA", GRAY)]:
        fig.add_trace(go.Bar(
            x=d["FY"], y=d[col], name=col,
            marker=dict(color=color, line=dict(color="white", width=1)),
            hovertemplate=f"<b>%{{x}}</b><br>{col}: ₱%{{y:,.2f}}B<extra></extra>",
        ))
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#808080")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Fiscal year")
    fig.update_yaxes(title_text="Variance (billion pesos)", zeroline=True)
    return polish_fig(fig, height=620, legend="bottom", bottom=115)


def fig_actual_vs_gaa(pivot):
    d = pivot.copy()
    d["FY"] = "FY" + d["Fiscal_Year"].astype(str)
    fig = go.Figure(go.Bar(
        x=d["FY"], y=d["Actual vs GAA (%)"], name="Actual vs GAA",
        marker=dict(color=DEEP_BLUE, line=dict(color="white", width=1)),
        hovertemplate="<b>%{x}</b><br>Actual vs GAA: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#808080")
    fig.update_xaxes(title_text="Fiscal year")
    fig.update_yaxes(title_text="Percent difference", ticksuffix="%")
    return polish_fig(fig, height=560, legend="none", bottom=70)


def fig_adaptation_mitigation_share(df):
    d = df.groupby("Fiscal_Year", as_index=False)[["ADAPTATION", "MITIGATION", "TOTAL"]].sum().sort_values("Fiscal_Year")
    d["Adaptation Share"] = np.where(d["TOTAL"] != 0, d["ADAPTATION"] / d["TOTAL"] * 100, np.nan)
    d["Mitigation Share"] = np.where(d["TOTAL"] != 0, d["MITIGATION"] / d["TOTAL"] * 100, np.nan)
    plot = d.melt(id_vars="Fiscal_Year", value_vars=["Adaptation Share", "Mitigation Share"], var_name="Pillar", value_name="Share")
    plot["Pillar"] = plot["Pillar"].str.replace(" Share", "", regex=False)
    fig = px.bar(
        plot,
        x="Fiscal_Year", y="Share", color="Pillar",
        color_discrete_map={"Adaptation": DEEP_BLUE, "Mitigation": GREEN},
        barmode="stack",
        hover_data={"Share": ":.2f"},
    )
    fig.update_xaxes(title_text="Fiscal year", type="category")
    fig.update_yaxes(title_text="Share of climate-tagged budget", ticksuffix="%", range=[0, 100])
    return polish_fig(fig, height=590, legend="bottom", bottom=115)


def fig_top_agencies(df, group_col="Agency Label", top_n=15, title_col=None):
    d = df.groupby([group_col], as_index=False)["TOTAL"].sum().sort_values("TOTAL", ascending=False).head(top_n)
    d["Amount_B"] = d["TOTAL"].apply(raw_to_billion)
    d = d.sort_values("Amount_B", ascending=True)
    fig = px.bar(
        d,
        x="Amount_B",
        y=group_col,
        orientation="h",
        color_discrete_sequence=[DEEP_BLUE],
        hover_data={"Amount_B": ":,.2f", "TOTAL": False},
    )
    fig.update_xaxes(title_text="Amount (billion pesos)")
    fig.update_yaxes(title_text="", tickfont=dict(size=10))
    return polish_fig(fig, height=max(520, 33 * len(d) + 180), legend="none", left=340, right=45, bottom=70)


def fig_agency_pareto(df, top_n=20):
    d = df.groupby("Agency Label", as_index=False)["TOTAL"].sum().sort_values("TOTAL", ascending=False).head(top_n)
    total = df["TOTAL"].sum()
    d["Amount_B"] = d["TOTAL"].apply(raw_to_billion)
    d["Share"] = d["TOTAL"] / total * 100 if total else np.nan
    d["Cumulative Share"] = d["TOTAL"].cumsum() / total * 100 if total else np.nan
    d = d.sort_values("Amount_B", ascending=True)
    fig = px.bar(
        d,
        x="Amount_B", y="Agency Label", orientation="h",
        color="Cumulative Share", color_continuous_scale="Blues",
        hover_data={"Amount_B": ":,.2f", "Share": ":.2f", "Cumulative Share": ":.2f", "TOTAL": False},
    )
    fig.update_coloraxes(colorbar_title="Cumulative share")
    fig.update_xaxes(title_text="Amount (billion pesos)")
    fig.update_yaxes(title_text="", tickfont=dict(size=10))
    return polish_fig(fig, height=max(600, 32 * len(d) + 170), legend="none", left=360, right=90, bottom=70)


def fig_nccap_heatmap(df):
    d = df.groupby(["NCCAP Priority", "Fiscal_Year"], as_index=False)["TOTAL"].sum()
    d["Amount_B"] = d["TOTAL"].apply(raw_to_billion)
    pivot = d.pivot(index="NCCAP Priority", columns="Fiscal_Year", values="Amount_B").fillna(0)
    ordered_index = [p for p in NCCAP_ORDER if p in pivot.index]
    pivot = pivot.loc[ordered_index]
    if pivot.empty:
        fig = go.Figure()
        return polish_fig(fig, height=450)

    col_totals = pivot.sum(axis=0).replace(0, np.nan)
    share = pivot.div(col_totals, axis=1) * 100

    custom = np.dstack([pivot.values, share.fillna(0).values])
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"FY{int(c)}" for c in pivot.columns],
        y=pivot.index,
        customdata=custom,
        colorscale=[[0, "#F7FBFF"], [0.2, "#D7E8F5"], [0.45, "#8CB9D9"], [0.7, "#3D7FAE"], [1, PRIMARY_BLUE]],
        colorbar=dict(title="₱B", len=0.75),
        xgap=3,
        ygap=3,
        hovertemplate="<b>%{y}</b><br>%{x}<br>Amount: ₱%{customdata[0]:,.2f}B<br>Share: %{customdata[1]:.2f}%<extra></extra>",
    ))

    # Keep annotations readable: labels are short and inside heatmap cells only.
    max_val = np.nanmax(pivot.values) if pivot.size else 0
    for i, priority in enumerate(pivot.index):
        for j, year in enumerate(pivot.columns):
            amount = pivot.loc[priority, year]
            pct_share = share.loc[priority, year]
            if amount == 0:
                label = ""
            elif amount >= 1000:
                label = f"₱{amount/1000:.2f}T<br>{pct_share:.1f}%"
            else:
                label = f"₱{amount:.1f}B<br>{pct_share:.1f}%"
            font_color = "white" if amount >= max_val * 0.50 and max_val > 0 else DARK_TEXT
            fig.add_annotation(x=f"FY{int(year)}", y=priority, text=label, showarrow=False, font=dict(size=10, color=font_color))

    fig.update_xaxes(title_text="Fiscal year", side="top")
    fig.update_yaxes(title_text="", autorange="reversed", tickfont=dict(size=10))
    return polish_fig(fig, height=max(600, 55 * len(pivot.index) + 150), legend="none", left=300, right=80, top=35, bottom=65)


def fig_nccap_rank(df, top_n=None):
    d = df.groupby("NCCAP Priority", as_index=False)["TOTAL"].sum().sort_values("TOTAL", ascending=False)
    if top_n:
        d = d.head(top_n)
    d["Amount_B"] = d["TOTAL"].apply(raw_to_billion)
    d = d.sort_values("Amount_B", ascending=True)
    fig = px.bar(
        d,
        x="Amount_B", y="NCCAP Priority", orientation="h",
        color="NCCAP Priority", color_discrete_map=PRIORITY_COLORS,
        hover_data={"Amount_B": ":,.2f", "TOTAL": False},
    )
    fig.update_xaxes(title_text="Amount (billion pesos)")
    fig.update_yaxes(title_text="")
    return polish_fig(fig, height=max(520, 42 * len(d) + 160), legend="none", left=300, bottom=70)


def fig_pdp_alignment(df):
    d = df.groupby("PDP / Executive Agenda Alignment", as_index=False)["TOTAL"].sum().sort_values("TOTAL", ascending=False)
    total = d["TOTAL"].sum()
    d["Amount_B"] = d["TOTAL"].apply(raw_to_billion)
    d["Share"] = np.where(total != 0, d["TOTAL"] / total * 100, np.nan)
    fig = px.bar(
        d.sort_values("Amount_B", ascending=True),
        x="Amount_B", y="PDP / Executive Agenda Alignment", orientation="h",
        color="PDP / Executive Agenda Alignment",
        color_discrete_map={"Strongly Aligned": GREEN, "Partially Aligned": MID_BLUE, "Weak / Unclassified": GRAY},
        hover_data={"Amount_B": ":,.2f", "Share": ":.2f", "TOTAL": False},
    )
    fig.update_xaxes(title_text="Amount (billion pesos)")
    fig.update_yaxes(title_text="")
    return polish_fig(fig, height=440, legend="none", left=230, bottom=70)


def fig_pdp_alignment_trend(df):
    d = df.groupby(["Fiscal_Year", "PDP / Executive Agenda Alignment"], as_index=False)["TOTAL"].sum()
    d["Amount_B"] = d["TOTAL"].apply(raw_to_billion)
    fig = px.bar(
        d,
        x="Fiscal_Year", y="Amount_B", color="PDP / Executive Agenda Alignment",
        color_discrete_map={"Strongly Aligned": GREEN, "Partially Aligned": MID_BLUE, "Weak / Unclassified": GRAY},
        barmode="stack",
        hover_data={"Amount_B": ":,.2f", "TOTAL": False},
    )
    fig.update_xaxes(title_text="Fiscal year", type="category")
    fig.update_yaxes(title_text="Amount (billion pesos)")
    return polish_fig(fig, height=560, legend="bottom", bottom=115)


def fig_quality_flags(qc):
    d = qc.sort_values("Flagged Records", ascending=True)
    fig = px.bar(
        d,
        x="Flagged Records", y="Data Quality Check", orientation="h",
        color="Flagged Records", color_continuous_scale="Oranges",
        hover_data={"Flagged Records": ":,.0f"},
    )
    fig.update_coloraxes(colorbar_title="Records")
    fig.update_xaxes(title_text="Flagged records")
    fig.update_yaxes(title_text="", tickfont=dict(size=10))
    return polish_fig(fig, height=max(560, 35 * len(d) + 160), legend="none", left=310, right=90, bottom=70)


def fig_challenge_priorities():
    d = FGD_KII_INSIGHTS.groupby(["Priority", "Budget Cycle Stage"], as_index=False).size()
    priority_order = ["High", "Medium", "Low"]
    fig = px.bar(
        d,
        x="Priority", y="size", color="Budget Cycle Stage",
        category_orders={"Priority": priority_order},
        labels={"size": "Number of coded issues"},
        hover_data={"size": ":,.0f"},
    )
    fig.update_xaxes(title_text="Priority level")
    fig.update_yaxes(title_text="Number of coded issues", dtick=1)
    return polish_fig(fig, height=500, legend="bottom", bottom=115)


def fig_budget_cycle_map():
    d = pd.DataFrame({
        "Stage": ["Preparation", "Legislation", "Execution", "Accountability"],
        "Score": [4, 3, 3, 4],
        "Focus": ["Tagging & QAR", "NEP→GAA", "Actuals", "Validation & audit"],
    })
    fig = px.bar(
        d,
        x="Stage", y="Score", color="Stage",
        color_discrete_sequence=[DEEP_BLUE, MID_BLUE, ORANGE, GREEN],
        hover_data={"Focus": True, "Score": False},
    )
    fig.update_xaxes(title_text="Budget cycle stage")
    fig.update_yaxes(title_text="Dashboard emphasis", showticklabels=False, range=[0, 5])
    return polish_fig(fig, height=450, legend="none", bottom=70)


def fig_budget_trend(df):
    d = df.groupby("Fiscal_Year", as_index=False)["TOTAL"].sum().sort_values("Fiscal_Year")
    d["Amount_B"] = d["TOTAL"].apply(raw_to_billion)
    fig = px.area(
        d,
        x="Fiscal_Year", y="Amount_B",
        markers=True,
        color_discrete_sequence=[DEEP_BLUE],
        hover_data={"Amount_B": ":,.2f", "TOTAL": False},
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=7))
    fig.update_xaxes(title_text="Fiscal year", type="category")
    fig.update_yaxes(title_text="Amount (billion pesos)", rangemode="tozero")
    return polish_fig(fig, height=560, legend="none", bottom=70)


def fig_yoy_growth(df):
    d = df.groupby("Fiscal_Year", as_index=False)["TOTAL"].sum().sort_values("Fiscal_Year")
    d["Amount_B"] = d["TOTAL"].apply(raw_to_billion)
    d["YoY Growth (%)"] = d["Amount_B"].pct_change() * 100
    fig = px.bar(
        d,
        x="Fiscal_Year", y="YoY Growth (%)",
        color="YoY Growth (%)", color_continuous_scale="Blues",
        hover_data={"YoY Growth (%)": ":.2f"},
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#808080")
    fig.update_xaxes(title_text="Fiscal year", type="category")
    fig.update_yaxes(title_text="Year-on-year growth", ticksuffix="%")
    return polish_fig(fig, height=520, legend="none", right=90, bottom=70)


# ============================================================
# PDF REPORT
# ============================================================


def generate_pdf_report(f, filters_used):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    total_budget = f["TOTAL"].sum()
    adaptation_total = f["ADAPTATION"].sum()
    mitigation_total = f["MITIGATION"].sum()
    adaptation_share = safe_divide(adaptation_total, total_budget) * 100 if total_budget else 0
    mitigation_share = safe_divide(mitigation_total, total_budget) * 100 if total_budget else 0
    top_agency, top_agency_amount, top_agency_share = safe_top(f, "Agency Label")
    top_priority, top_priority_amount, top_priority_share = safe_top(f, "NCCAP Priority")

    story.append(Paragraph("National CCET Smart Analytics Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y %I:%M %p')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Filters Applied", styles["Heading2"]))
    filter_table = [["Filter", "Selected Value"]] + [[k, str(v)] for k, v in filters_used.items()]
    table = Table(filter_table, colWidths=[170, 310])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    summary = f"""
    The filtered dataset covers {peso_from_raw(total_budget)} in climate-tagged PAPs across
    {f['Agency Unit'].nunique()} unique institution units and {f['PAP ID'].replace('', np.nan).nunique()} PAP IDs.
    Adaptation accounts for {adaptation_share:.2f}% while mitigation accounts for {mitigation_share:.2f}% of the filtered total.
    The top institution is {top_agency} with {peso_from_raw(top_agency_amount)} ({top_agency_share:.2f}% of the filtered total).
    The largest NCCAP priority is {top_priority} with {peso_from_raw(top_priority_amount)} ({top_priority_share:.2f}% of the filtered total).
    """
    story.append(Paragraph(summary, styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Interpretation Note", styles["Heading2"]))
    note = """
    The dashboard supports analysis and validation, but it does not replace official government validation.
    Climate-tagged budgets should not be treated as direct proof of actual expenditure, implementation performance, or climate outcomes.
    PDP alignment is a keyword-based proxy, while NCCAP priority is derived from CCET typology codes.
    """
    story.append(Paragraph(note, styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ============================================================
# MAIN APP: DATA LOAD
# ============================================================

st.title("National CCET Smart Policy Analytics Dashboard")
st.caption("PAP-level climate budget analytics, budget-cycle traceability, NCCAP priorities, FGD/KII insights, and data quality validation")

st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CCET CSV dataset",
    type=["csv"],
    help="Upload the cleaned National CCET PAP dataset. If no file is uploaded, the app will look for data/ccet_data.csv or the latest cleaned CSV in the working folder.",
)

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    df = prepare_data(raw_df)
    dataset_source = "Uploaded CSV file"
else:
    default_path = find_default_data_path()
    if default_path is None:
        st.error("No default dataset found. Please upload the cleaned CCET CSV file in the sidebar.")
        st.stop()
    df = load_and_prepare_from_path(default_path)
    dataset_source = default_path

# Defensive sanity check for GRIT TAGGING.
if "GRIT TAGGING" not in df.columns:
    st.error("GRIT TAGGING column was not detected after preparation. Please check the CSV header.")
    st.stop()

st.sidebar.caption(f"Current dataset: {dataset_source}")
st.sidebar.caption("Budget values are treated as thousand pesos and converted for display.")

with st.sidebar.expander("GRIT TAGGING check", expanded=False):
    st.write(df["GRIT TAGGING"].value_counts(dropna=False))


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Filters")

year_filter = st.sidebar.selectbox("Fiscal Year", filter_options(df["Fiscal_Year"]))
budget_filter = st.sidebar.selectbox("Budget Type", filter_options(df["Type"]))
grit_filter = st.sidebar.selectbox("GRIT TAGGING / Institution Type", filter_options(df["GRIT TAGGING"]))
department_filter = st.sidebar.selectbox("Department", filter_options(df["DEPARTMENT"]))
pillar_filter = st.sidebar.selectbox("Climate Pillar", filter_options(df["Climate Pillar"]))
nccap_filter = st.sidebar.selectbox("NCCAP Thematic Priority", filter_options(df["NCCAP Priority"]))
pdp_filter = st.sidebar.selectbox("PDP / Executive Agenda Alignment", filter_options(df["PDP / Executive Agenda Alignment"]))

f = df.copy()
if year_filter != "All":
    f = f[f["Fiscal_Year"] == int(year_filter)]
if budget_filter != "All":
    f = f[f["Type"] == budget_filter]
if grit_filter != "All":
    f = f[f["GRIT TAGGING"] == grit_filter]
if department_filter != "All":
    f = f[f["DEPARTMENT"] == department_filter]
if pillar_filter != "All":
    f = f[f["Climate Pillar"] == pillar_filter]
if nccap_filter != "All":
    f = f[f["NCCAP Priority"] == nccap_filter]
if pdp_filter != "All":
    f = f[f["PDP / Executive Agenda Alignment"] == pdp_filter]

filters_used = {
    "Dataset": dataset_source,
    "Fiscal Year": year_filter,
    "Budget Type": budget_filter,
    "GRIT TAGGING / Institution Type": grit_filter,
    "Department": department_filter,
    "Climate Pillar": pillar_filter,
    "NCCAP Thematic Priority": nccap_filter,
    "PDP / Executive Agenda Alignment": pdp_filter,
}

st.sidebar.divider()
if REPORTLAB_AVAILABLE:
    pdf_buffer = generate_pdf_report(f, filters_used)
    st.sidebar.download_button(
        "Download PDF summary",
        data=pdf_buffer,
        file_name="ccet_smart_analytics_summary.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
else:
    st.sidebar.warning("PDF export requires `reportlab`.")

st.sidebar.download_button(
    "Download filtered CSV",
    data=f.to_csv(index=False).encode("utf-8-sig"),
    file_name="filtered_ccet_data.csv",
    mime="text/csv",
    use_container_width=True,
)


# ============================================================
# TOP KPI CARDS
# ============================================================

total_budget = f["TOTAL"].sum()
adaptation_total = f["ADAPTATION"].sum()
mitigation_total = f["MITIGATION"].sum()
adaptation_share = safe_divide(adaptation_total, total_budget) * 100 if total_budget else 0
mitigation_share = safe_divide(mitigation_total, total_budget) * 100 if total_budget else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Climate-tagged total", peso_from_raw(total_budget))
k2.metric("Adaptation share", pct(adaptation_share, 1))
k3.metric("Mitigation share", pct(mitigation_share, 1))
k4.metric("Unique institutions", f"{f['Agency Unit'].nunique():,}")
k5.metric("PAP IDs", f"{f['PAP ID'].replace('', np.nan).nunique():,}")

if f.empty:
    st.warning("No records match the selected filters. Please adjust the sidebar filters.")
    st.stop()

st.markdown(
    """
    <div class="interpretation-box">
    <b>Interpretation reminder:</b> Climate-tagged budget is evidence of tagging and budget prioritization. It should not be treated by itself as proof of actual expenditure, project completion, or climate outcome. Use the Budget Cycle, PAP Explorer, and Data Quality tabs for validation.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "Guide",
    "Data Profile",
    "Executive Overview",
    "Key Findings",
    "Participation & Coverage",
    "Budget Cycle",
    "Agency Concentration",
    "NCCAP Priorities",
    "FGD/KII Challenges",
    "Recommendations",
    "Policy Alignment",
    "Budget Trends",
    "PAP Explorer",
    "Data Quality",
    "Methods & Rationale",
])

(
    tab_guide,
    tab_profile,
    tab_exec,
    tab_key,
    tab_participation,
    tab_cycle,
    tab_concentration,
    tab_nccap,
    tab_fgd,
    tab_reco,
    tab_policy,
    tab_trends,
    tab_pap,
    tab_quality,
    tab_methods,
) = tabs


# ============================================================
# GUIDE
# ============================================================

with tab_guide:
    st.subheader("Dashboard Guide")
    st.markdown(
        """
        This dashboard is organized according to the user's analytical journey:

        **Data loading → filters → KPI cards → key findings → participation → budget cycle → agency concentration → NCCAP priorities → FGD/KII challenges → recommendations → policy alignment → PAP explorer → data quality.**

        The dashboard is intended for CCET policy review and decision support. It uses the cleaned PAP-level dataset and converts raw values from **thousand pesos** into display units.

        **Main changes in this version:**
        - `GRIT TAGGING` is now retained and cleaned as the institution-type filter.
        - The old `NDC Sector` filter was removed and replaced with `NCCAP Thematic Priority`.
        - Visualizations were redesigned to avoid overlapping titles, legends, and labels.
        - Repetitive charts were removed; each tab now has a clearer analytical purpose.
        - Budget-cycle, participation, agency concentration, FGD/KII, and data quality modules were strengthened.
        """
    )
    st.markdown("### Filter behavior")
    st.dataframe(pd.DataFrame([{"Filter": k, "Selected value": v} for k, v in filters_used.items()]), use_container_width=True)


# ============================================================
# DATA PROFILE
# ============================================================

with tab_profile:
    st.subheader("Data Profile and Schema")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{len(df):,}")
    c2.metric("Filtered records", f"{len(f):,}")
    c3.metric("Fiscal year range", f"{int(df['Fiscal_Year'].min())}–{int(df['Fiscal_Year'].max())}")
    c4.metric("Columns", f"{len(df.columns):,}")

    st.markdown("### Institution type distribution")
    st.dataframe(df["GRIT TAGGING"].value_counts(dropna=False).rename_axis("GRIT TAGGING").reset_index(name="Records"), use_container_width=True)

    st.markdown("### Data dictionary")
    schema_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[col].dtype) for col in df.columns],
        "Missing / Blank Count": [df[col].isna().sum() + (df[col].eq("").sum() if df[col].dtype == "object" else 0) for col in df.columns],
        "Description": [COLUMN_DICTIONARY.get(col, "Column from uploaded dataset or derived by the dashboard.") for col in df.columns],
    })
    st.dataframe(schema_df, use_container_width=True, height=520)

    st.markdown("### Dataset preview")
    st.dataframe(df.head(30), use_container_width=True, height=380)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

with tab_exec:
    st.subheader("Executive Overview")
    top_agency, top_agency_amount, top_agency_share = safe_top(f, "Agency Label")
    top_priority, top_priority_amount, top_priority_share = safe_top(f, "NCCAP Priority")
    top_grit, top_grit_amount, top_grit_share = safe_top(f, "GRIT TAGGING")

    st.markdown(
        f"""
        The filtered dataset covers **{peso_from_raw(total_budget)}** in climate-tagged PAPs across **{f['Agency Unit'].nunique():,} unique institutions** and **{f['PAP ID'].replace('', np.nan).nunique():,} PAP IDs**.

        Adaptation accounts for **{adaptation_share:.2f}%** of the filtered total, while mitigation accounts for **{mitigation_share:.2f}%**. The largest institution-type group is **{top_grit}** with **{peso_from_raw(top_grit_amount)}** or **{top_grit_share:.2f}%** of the filtered amount.

        The top agency/institution is **{top_agency}** with **{peso_from_raw(top_agency_amount)}** or **{top_agency_share:.2f}%** of the filtered total. The largest NCCAP thematic priority is **{top_priority}** with **{peso_from_raw(top_priority_amount)}** or **{top_priority_share:.2f}%**.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        chart_card(
            "Budget trend under active filters",
            "Total climate-tagged amount by fiscal year.",
            fig_budget_trend(f),
            "exec_budget_trend",
            "Source: Dashboard computation from loaded CCET PAP-level dataset.",
            height=560,
        )
    with c2:
        chart_card(
            "NCCAP priority ranking under active filters",
            "Top thematic priorities based on the current sidebar selections.",
            fig_nccap_rank(f),
            "exec_nccap_priority_ranking",
            "Source: Dashboard computation from CCET typology-derived NCCAP priority.",
            height=560,
        )


# ============================================================
# KEY FINDINGS
# ============================================================

with tab_key:
    st.subheader("Key Findings")
    st.markdown(
        """
        This tab presents the headline findings most aligned with the updated assessment report: growth in climate-tagged appropriations, budget-cycle movement, adaptation-heavy composition, and concentration among institutions.
        """
    )

    chart_card(
        "Climate-tagged GAA, total national budget, and share of national budget",
        "Bars show peso amounts; the line shows climate-tagged GAA as a share of the national budget. Values are computed from the loaded dataset for GAA and compared with the national budget reference for FY2022–FY2025.",
        fig_climate_budget_share(df),
        "key_climate_budget_share",
        "Source: Dashboard computation from loaded CCET dataset and national budget reference values used in the assessment report.",
        height=610,
    )

    c1, c2 = st.columns(2)
    with c1:
        pivot_2225 = budget_stage_pivot(df, years=[2022, 2023, 2024, 2025])
        chart_card(
            "NEP, GAA, and Actual CCET budgets",
            "Shows how climate-tagged budgets change from proposal to approval to actual reporting.",
            fig_budget_stage_comparison(pivot_2225),
            "key_nep_gaa_actual",
            "Source: Dashboard computation from loaded CCET dataset, FY2022–FY2025.",
            height=620,
        )
    with c2:
        chart_card(
            "Adaptation and mitigation share",
            "Shows whether climate-tagged allocations are adaptation-heavy or mitigation-heavy.",
            fig_adaptation_mitigation_share(df[df["Fiscal_Year"].isin([2022, 2023, 2024, 2025]) & (df["Type"] == "GAA")]),
            "key_adaptation_mitigation_share",
            "Source: Dashboard computation from GAA records, FY2022–FY2025.",
            height=620,
        )


# ============================================================
# PARTICIPATION & COVERAGE
# ============================================================

with tab_participation:
    st.subheader("Participation and Coverage")
    st.markdown(
        """
        This module responds to the report finding that CCET participation increased over time. It counts unique institution units using `GRIT TAGGING + Department + Agency` to avoid undercounting generic agency names.
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        chart_card(
            "Participating institutions by fiscal year and GRIT TAGGING",
            "Counts unique institution units by fiscal year and institution type.",
            fig_participation_by_year(df),
            "participation_by_year_grit",
            "Source: Dashboard computation from loaded CCET PAP-level dataset.",
            height=600,
        )
    with c2:
        chart_card(
            "Climate-tagged amount by institution type",
            "Shows how total tagged amounts are distributed among NGA, GOCC, SUC, and Unclassified records over time.",
            fig_budget_by_institution(df[df["Type"] == "GAA"] if "GAA" in df["Type"].unique() else df),
            "budget_by_institution_type",
            "Source: Dashboard computation from loaded dataset; GAA records used where available.",
            height=600,
        )

    participation_table = df.groupby(["Fiscal_Year", "GRIT TAGGING"], as_index=False).agg(
        Records=("TOTAL", "size"),
        Unique_Institutions=("Agency Unit", "nunique"),
        Unique_PAP_IDs=("PAP ID", lambda s: s.replace("", np.nan).nunique()),
        Total_B=("TOTAL", lambda s: raw_to_billion(s.sum())),
    )
    st.markdown("### Participation and coverage table")
    st.dataframe(participation_table.sort_values(["Fiscal_Year", "GRIT TAGGING"]), use_container_width=True, height=420)


# ============================================================
# BUDGET CYCLE
# ============================================================

with tab_cycle:
    st.subheader("Budget Cycle Analysis")
    st.markdown(
        """
        This tab follows the budget-cycle logic of CCET: preparation, legislation, execution, and accountability. It helps users avoid treating tagged budgets as final proof of actual spending or climate impact.
        """
    )

    chart_card(
        "Dashboard coverage across the public budget cycle",
        "Shows how the dashboard modules correspond to preparation, legislation, execution, and accountability.",
        fig_budget_cycle_map(),
        "budget_cycle_coverage_map",
        "Source: Dashboard design based on the assessment's public budget-cycle analytical framework.",
        height=450,
    )

    pivot_all = budget_stage_pivot(f)
    if pivot_all.empty:
        st.info("No NEP/GAA/Actual data available under the current filters.")
    else:
        chart_card(
            "NEP, GAA, and Actual under active filters",
            "Compares proposed, approved, and actual reported amounts where available.",
            fig_budget_stage_comparison(pivot_all),
            "cycle_nep_gaa_actual_filtered",
            "Source: Dashboard computation from filtered CCET dataset.",
            height=620,
        )
        c1, c2 = st.columns(2)
        with c1:
            chart_card(
                "Budget-cycle variance",
                "Positive values indicate increases between stages; negative values indicate reductions.",
                fig_budget_variance(pivot_all),
                "cycle_budget_variance_filtered",
                "Formula: GAA variance = GAA - NEP; Actual variance = Actual - GAA.",
                height=620,
            )
        with c2:
            chart_card(
                "Actual compared with GAA",
                "Shows the percentage gap between actual and approved GAA where both values are available.",
                fig_actual_vs_gaa(pivot_all),
                "cycle_actual_vs_gaa_filtered",
                "Formula: Actual vs GAA (%) = (Actual - GAA) / GAA × 100.",
                height=560,
            )

        display_pivot = pivot_all.copy()
        for col in ["NEP", "GAA", "Actual", "GAA minus NEP", "Actual minus GAA"]:
            display_pivot[col] = display_pivot[col].apply(peso_billion)
        display_pivot["Actual vs GAA (%)"] = pivot_all["Actual vs GAA (%)"].apply(lambda x: pct(x, 2))
        st.markdown("### Budget-stage table")
        st.dataframe(display_pivot, use_container_width=True)


# ============================================================
# AGENCY CONCENTRATION
# ============================================================

with tab_concentration:
    st.subheader("Agency Concentration")
    st.markdown(
        """
        This module identifies which institutions drive climate-tagged budgets. It uses combined department-agency labels so generic names such as “Office of the Secretary” are not shown without context.
        """
    )
    ranking_basis = st.radio(
        "Ranking basis",
        ["GAA", "NEP", "Actual", "Use active filters"],
        horizontal=True,
        help="Use GAA for report-style allocation rankings. Use active filters to rank based on your sidebar selections.",
    )
    rbase = ranking_base(f, ranking_basis)
    if rbase.empty:
        st.info("No data available for the selected ranking basis.")
    else:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            chart_card(
                "Top institutions by climate-tagged amount",
                "Color intensity indicates cumulative share among the displayed top institutions. Hover to inspect exact values.",
                fig_agency_pareto(rbase, top_n=20),
                "agency_pareto_top20",
                "Source: Dashboard computation from filtered CCET dataset.",
                height=760,
            )
        with c2:
            top_agency, top_amount, top_share = safe_top(rbase, "Agency Label")
            top10 = rbase.groupby("Agency Label", as_index=False)["TOTAL"].sum().sort_values("TOTAL", ascending=False).head(10)
            top10_share = top10["TOTAL"].sum() / rbase["TOTAL"].sum() * 100 if rbase["TOTAL"].sum() else np.nan
            st.metric("Top institution", top_agency)
            st.metric("Top institution amount", peso_from_raw(top_amount))
            st.metric("Top institution share", pct(top_share, 2))
            st.metric("Top 10 share", pct(top10_share, 2))
            st.markdown(
                """
                <div class="method-box">
                <b>Why this matters:</b> Large aggregate climate budgets may be driven by a small number of infrastructure or agriculture institutions. Concentration analysis helps users see whether the budget is broad-based or dominated by a few agencies.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            nga = rbase[rbase["GRIT TAGGING"] == "NGA"]
            chart_card(
                "Top NGA departments",
                "Aggregates climate-tagged amounts by department for NGA records.",
                fig_top_agencies(nga, group_col="DEPARTMENT", top_n=12) if not nga.empty else go.Figure(),
                "top_nga_departments",
                "Source: Dashboard computation from NGA records.",
                height=620,
            )
        with c2:
            gocc = rbase[rbase["GRIT TAGGING"] == "GOCC"]
            chart_card(
                "Top GOCC agencies",
                "Ranks GOCC agencies under the selected basis.",
                fig_top_agencies(gocc, group_col="Agency Label", top_n=12) if not gocc.empty else go.Figure(),
                "top_gocc_agencies",
                "Source: Dashboard computation from GOCC records.",
                height=620,
            )
        with c3:
            suc = rbase[rbase["GRIT TAGGING"] == "SUC"]
            chart_card(
                "Top SUCs",
                "Ranks SUCs under the selected basis.",
                fig_top_agencies(suc, group_col="Agency Label", top_n=12) if not suc.empty else go.Figure(),
                "top_sucs",
                "Source: Dashboard computation from SUC records.",
                height=620,
            )


# ============================================================
# NCCAP PRIORITIES
# ============================================================

with tab_nccap:
    st.subheader("NCCAP Thematic Priorities")
    st.markdown(
        """
        NCCAP priorities are derived from the CCET typology ID. This module shows both the thematic distribution and the exact PAP-level records behind the classifications.
        """
    )
    chart_card(
        "NCCAP thematic priority allocation matrix",
        "Each cell shows amount and share within that fiscal year. Darker cells indicate larger allocations.",
        fig_nccap_heatmap(f),
        "nccap_priority_heatmap_filtered",
        "Source: Dashboard computation from filtered dataset. Formula: priority share = priority allocation / fiscal-year total × 100.",
        height=720,
        width=1550,
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        chart_card(
            "NCCAP priority ranking",
            "Ranks priorities by total amount under current filters.",
            fig_nccap_rank(f),
            "nccap_priority_ranking_filtered",
            "Source: Dashboard computation from typology-derived NCCAP priority.",
            height=620,
        )
    with c2:
        priority_table = f.groupby(["NCCAP Priority", "Climate Pillar"], as_index=False).agg(
            Records=("TOTAL", "size"),
            Unique_PAP_IDs=("PAP ID", lambda s: s.replace("", np.nan).nunique()),
            Total_B=("TOTAL", lambda s: raw_to_billion(s.sum())),
        ).sort_values("Total_B", ascending=False)
        st.markdown("### NCCAP priority table")
        st.dataframe(priority_table, use_container_width=True, height=560)


# ============================================================
# FGD/KII CHALLENGES
# ============================================================

with tab_fgd:
    st.subheader("FGD/KII Challenges")
    st.markdown(
        """
        This tab translates qualitative findings into dashboard-ready monitoring categories. It links implementation challenges to dashboard features and reform directions.
        """
    )
    chart_card(
        "FGD/KII challenge priorities by budget-cycle stage",
        "Counts coded implementation issues by priority level and budget-cycle stage.",
        fig_challenge_priorities(),
        "fgd_kii_priority_chart",
        "Source: Synthesized FGD/KII challenge themes from the updated assessment direction.",
        height=500,
    )
    st.markdown("### Challenge-to-dashboard response matrix")
    st.dataframe(FGD_KII_INSIGHTS, use_container_width=True, height=480)
    st.download_button(
        "Download FGD/KII challenge matrix",
        data=FGD_KII_INSIGHTS.to_csv(index=False).encode("utf-8-sig"),
        file_name="fgd_kii_challenge_matrix.csv",
        mime="text/csv",
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

with tab_reco:
    st.subheader("Recommendations Tracker")
    st.markdown(
        """
        The tracker operationalizes the assessment recommendations as dashboard features, monitoring actions, and possible next-build enhancements.
        """
    )
    priority_filter = st.selectbox("Recommendation priority", ["All"] + sorted(FGD_KII_INSIGHTS["Priority"].unique().tolist()))
    stage_filter = st.selectbox("Budget-cycle stage", ["All"] + sorted(FGD_KII_INSIGHTS["Budget Cycle Stage"].unique().tolist()))
    reco = FGD_KII_INSIGHTS.copy()
    if priority_filter != "All":
        reco = reco[reco["Priority"] == priority_filter]
    if stage_filter != "All":
        reco = reco[reco["Budget Cycle Stage"] == stage_filter]
    st.dataframe(reco[["Theme", "Budget Cycle Stage", "Challenge", "Recommendation", "Smart Dashboard Response", "Priority"]], use_container_width=True, height=440)

    st.markdown("### International CBT design lessons to reflect in future dashboard builds")
    st.dataframe(INTERNATIONAL_LESSONS, use_container_width=True, height=260)

    st.markdown(
        """
        ### Suggested smart-dashboard enhancements
        - Agency-level CCET maturity scorecard.
        - PAP-level NEP-GAA-Actual reconciliation template.
        - Attribution method field: whole PAP, proportional, component-based, or not specified.
        - QAR completeness tracker.
        - Climate indicator / M&E linkage fields.
        - Audit-readiness field and evidence checklist.
        - Future scoring model for climate relevance confidence.
        """
    )


# ============================================================
# POLICY ALIGNMENT
# ============================================================

with tab_policy:
    st.subheader("PDP / Executive Agenda Proxy Alignment")
    st.markdown(
        """
        This tab uses keyword-based text classification to estimate possible alignment with climate and development priorities. It is exploratory and does not replace official validation.
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        chart_card(
            "PDP / Executive Agenda alignment share",
            "Estimated alignment based on keyword hits in PAP, typology, agency, and department text.",
            fig_pdp_alignment(f),
            "pdp_alignment_share",
            "Method: rule-based NLP proxy. Hits ≥ 3 = Strongly Aligned; hits ≥ 1 = Partially Aligned; hits = 0 = Weak / Unclassified.",
            height=440,
        )
    with c2:
        chart_card(
            "PDP / Executive Agenda alignment trend",
            "Shows how estimated alignment categories move over time under active filters.",
            fig_pdp_alignment_trend(f),
            "pdp_alignment_trend",
            "Source: Dashboard computation from filtered dataset using keyword-based proxy classification.",
            height=560,
        )

    st.markdown("### Keyword basis used by the proxy")
    st.write(", ".join(PDP_KEYWORDS))


# ============================================================
# BUDGET TRENDS
# ============================================================

with tab_trends:
    st.subheader("Budget Trends")
    st.markdown("Trend analytics based on the active sidebar filters.")
    c1, c2 = st.columns(2)
    with c1:
        chart_card(
            "Climate budget trend",
            "Time-series view of total climate-tagged amount under active filters.",
            fig_budget_trend(f),
            "trend_total_budget",
            "Formula: yearly total = sum(TOTAL) by fiscal year.",
            height=560,
        )
    with c2:
        chart_card(
            "Year-on-year growth rate",
            "Highlights unusually sharp increases or declines in the filtered climate budget.",
            fig_yoy_growth(f),
            "trend_yoy_growth",
            "Formula: YoY growth = (current year total - previous year total) / previous year total × 100.",
            height=520,
        )

    by_year = f.groupby("Fiscal_Year", as_index=False).agg(
        Records=("TOTAL", "size"),
        Total_B=("TOTAL", lambda s: raw_to_billion(s.sum())),
        Adaptation_B=("ADAPTATION", lambda s: raw_to_billion(s.sum())),
        Mitigation_B=("MITIGATION", lambda s: raw_to_billion(s.sum())),
        Unique_Institutions=("Agency Unit", "nunique"),
    ).sort_values("Fiscal_Year")
    by_year["YoY Growth (%)"] = by_year["Total_B"].pct_change() * 100
    st.markdown("### Yearly trend table")
    st.dataframe(by_year, use_container_width=True)


# ============================================================
# PAP EXPLORER
# ============================================================

with tab_pap:
    st.subheader("PAP Explorer")
    st.markdown("Search and validate the PAP-level records behind the charts.")
    search = st.text_input("Search PAP description, agency, department, typology, or PAP ID")
    explorer = f.copy()
    if search:
        s = search.lower().strip()
        mask = (
            explorer["PAP Description"].str.lower().str.contains(s, na=False) |
            explorer["AGENCY"].str.lower().str.contains(s, na=False) |
            explorer["DEPARTMENT"].str.lower().str.contains(s, na=False) |
            explorer["TYPOLOGY Description"].str.lower().str.contains(s, na=False) |
            explorer["TYPOLOGY ID"].str.lower().str.contains(s, na=False) |
            explorer["PAP ID"].str.lower().str.contains(s, na=False)
        )
        explorer = explorer[mask]

    explorer_cols = [
        "Fiscal_Year", "Type", "GRIT TAGGING", "DEPARTMENT", "AGENCY", "Agency Label",
        "PAP ID", "PAP Description", "TYPOLOGY ID", "TYPOLOGY Description",
        "Climate Pillar", "NCCAP Priority", "PDP / Executive Agenda Alignment",
        "ADAPTATION", "MITIGATION", "TOTAL", "TOTAL_B",
    ]
    st.dataframe(explorer[explorer_cols].sort_values("TOTAL", ascending=False), use_container_width=True, height=620)
    st.download_button(
        "Download PAP Explorer CSV",
        data=explorer[explorer_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="pap_explorer_filtered.csv",
        mime="text/csv",
    )


# ============================================================
# DATA QUALITY
# ============================================================

with tab_quality:
    st.subheader("Data Quality Checks")
    st.markdown(
        """
        Data quality checks help prevent overclaiming. They identify records that may need validation before they are used for policy conclusions.
        """
    )
    qc = build_quality_flags(f)
    chart_card(
        "Data quality flags under active filters",
        "Counts records that may require review or validation.",
        fig_quality_flags(qc),
        "data_quality_flags",
        "Source: Dashboard rule-based validation checks.",
        height=650,
    )
    st.markdown("### Quality summary table")
    st.dataframe(qc.sort_values("Flagged Records", ascending=False), use_container_width=True)

    issue = st.selectbox("Inspect flagged records for", qc["Data Quality Check"].tolist())
    flagged = f[quality_mask(f, issue)]
    st.write(f"Flagged records for **{issue}**: {len(flagged):,}")
    st.dataframe(flagged.head(1000), use_container_width=True, height=420)
    st.download_button(
        "Download flagged records CSV",
        data=flagged.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"data_quality_{issue.lower().replace(' ', '_').replace('/', '_')}.csv",
        mime="text/csv",
    )


# ============================================================
# METHODS & RATIONALE
# ============================================================

with tab_methods:
    st.subheader("Methods, Rationale, and Data Science Application")
    st.markdown(
        """
        The dashboard operationalizes CCET policy requirements through applied data science. It loads, cleans, classifies, validates, aggregates, visualizes, and interprets climate-tagged public expenditure data across the budget cycle.
        """
    )

    st.markdown("### Dashboard methods by feature")
    methods_feature = pd.DataFrame([
        {"Feature": "Dataset loader", "Policy rationale": "CCET requires tagged expenditure data to be tracked and reported.", "Data science method": "Data ingestion / ETL", "Formula or rule": "Raw CSV → cleaned dataset → analysis-ready dataset"},
        {"Feature": "GRIT TAGGING filter", "Policy rationale": "National CCET covers NGAs, GOCCs, and SUCs.", "Data science method": "Categorical standardization", "Formula or rule": "Normalize GRIT TAGGING to NGA, GOCC, SUC, or Unclassified"},
        {"Feature": "NCCAP Thematic Priority filter", "Policy rationale": "CCET typologies are anchored on NCCAP priorities.", "Data science method": "Feature engineering", "Formula or rule": "Extract priority code from TYPOLOGY ID"},
        {"Feature": "KPI cards", "Policy rationale": "CCET should generate timely climate expenditure statistics.", "Data science method": "Descriptive analytics", "Formula or rule": "sum(TOTAL), sum(ADAPTATION), sum(MITIGATION), count unique agencies/PAPs"},
        {"Feature": "Budget cycle analysis", "Policy rationale": "CCET follows preparation, legislation, execution, and accountability.", "Data science method": "Variance and utilization-gap analysis", "Formula or rule": "GAA - NEP; Actual - GAA; (Actual - GAA) / GAA × 100"},
        {"Feature": "Agency concentration", "Policy rationale": "Climate-tagged budgets may be driven by a few institutions.", "Data science method": "Ranking and concentration analysis", "Formula or rule": "rank agencies by sum(TOTAL); top 10 share"},
        {"Feature": "FGD/KII module", "Policy rationale": "Implementation challenges must inform reforms.", "Data science method": "Qualitative coding", "Formula or rule": "finding → theme → challenge → recommendation → priority"},
        {"Feature": "Data quality", "Policy rationale": "CCET analysis depends on reliable administrative data.", "Data science method": "Rule-based anomaly detection", "Formula or rule": "missing fields, zero total, mismatch, duplicates"},
    ])
    st.dataframe(methods_feature, use_container_width=True, height=360)

    st.markdown("### Methods by visualization")
    methods_viz = pd.DataFrame([
        {"Visualization": "Climate-tagged GAA vs national budget", "Method": "Trend + ratio analysis", "Formula": "Climate share = climate-tagged GAA / total national budget × 100", "Use": "Shows budget visibility relative to the national budget."},
        {"Visualization": "NEP-GAA-Actual comparison", "Method": "Budget-stage comparison", "Formula": "Compare NEP vs GAA vs Actual", "Use": "Shows movement across budget stages."},
        {"Visualization": "Budget-cycle variance", "Method": "Delta analysis", "Formula": "GAA - NEP; Actual - GAA", "Use": "Identifies increases/reductions requiring PAP-level review."},
        {"Visualization": "Actual vs GAA", "Method": "Utilization gap analysis", "Formula": "(Actual - GAA) / GAA × 100", "Use": "Shows whether approved allocations translated into actual reported spending."},
        {"Visualization": "Adaptation vs mitigation share", "Method": "Composition analysis", "Formula": "Pillar amount / total × 100", "Use": "Shows whether the budget is adaptation-heavy or mitigation-heavy."},
        {"Visualization": "Participation by GRIT TAGGING", "Method": "Coverage analysis", "Formula": "count unique Agency Unit by year and GRIT TAGGING", "Use": "Shows participation growth by institution type."},
        {"Visualization": "Agency concentration", "Method": "Pareto/concentration analysis", "Formula": "agency total / overall total × 100", "Use": "Shows whether budgets are concentrated among a few institutions."},
        {"Visualization": "NCCAP heatmap", "Method": "Cross-tabulation + share analysis", "Formula": "priority allocation / fiscal-year total × 100", "Use": "Shows amount and share per priority and fiscal year."},
        {"Visualization": "PDP alignment", "Method": "Rule-based NLP proxy", "Formula": "keyword hits: ≥3 strong, ≥1 partial, 0 weak", "Use": "Explores possible alignment but does not replace official validation."},
        {"Visualization": "Data quality flags", "Method": "Rule-based validation", "Formula": "flag missing, zero, mismatch, duplicate records", "Use": "Identifies records needing validation."},
    ])
    st.dataframe(methods_viz, use_container_width=True, height=420)

    st.markdown(
        """
        ### Legal and research basis to cite in the report text
        - DBM-CCC-DILG JMC No. 2015-01 for Local CCET tagging/tracking in the local budget.
        - DBM-CCC JMC No. 2015-01 and JMC No. 2013-01 for National CCET.
        - Climate Change Act of 2009 for mainstreaming climate change in government policy and planning.
        - NCCAP 2011–2028 for thematic climate priorities.
        - CPEIR / World Bank literature for climate public expenditure and institutional review.
        - Monsod UPSE Discussion Paper for analysis of NCCAP alignment and budget concentration.
        - PEFA-Climate for climate-responsive public financial management and expenditure tracking.
        - Updated CCET Impact Assessment Report for FGD/KII challenges, public budget-cycle framing, and recommendations.
        """
    )


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
    page_title="National CCET Smart Policy Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN CONSTANTS
# ============================================================

DEFAULT_DATA_CANDIDATES = [
    "data/ccet_data.csv",
    "Cleaned_National_CCET_PAPs_FY_2017_to_2026.csv"
]

# The cleaned National CCET PAP-level dataset stores budget values in thousand pesos.
# Example: raw value 1,000,000 = ₱1,000,000,000 = ₱1.00B.
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
RED = "#C00000"
GRAY = "#A5A5A5"
DARK_GRAY = "#595959"
LIGHT_GRAY = "#E6ECF5"
DARK_TEXT = "#1F1F1F"

CHART_COLOR_SEQUENCE = [DEEP_BLUE, MID_BLUE, GREEN, ORANGE, TEAL, PURPLE, YELLOW, GRAY]

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

NCCAP_ORDER = [
    "Food Security",
    "Water Sufficiency",
    "Ecosystem & Environmental Stability",
    "Human Security",
    "Climate-Smart Industries & Services",
    "Sustainable Energy",
    "Knowledge & Capacity Development",
    "Cross-Cutting",
    "Unclassified",
]

PDP_KEYWORDS = [
    "climate", "resilience", "disaster", "risk reduction", "adaptation",
    "mitigation", "flood", "drainage", "water", "irrigation",
    "food security", "agriculture", "renewable", "energy efficiency",
    "sustainable", "environment", "ecosystem", "biodiversity",
    "green", "carbon", "emission", "hazard", "watershed", "low-carbon",
    "climate-smart", "sustainable energy", "human security",
]

NATIONAL_BUDGET_BILLION = pd.DataFrame({
    "Fiscal_Year": [2022, 2023, 2024, 2025],
    "Total National Budget (Billion Pesos)": [5023.00, 5268.00, 5768.00, 6326.00],
})

COLUMN_DICTIONARY = {
    "Fiscal_Year": "Fiscal year covered by the climate-tagged PAP record.",
    "Type": "Budget stage/classification, such as NEP, GAA, or Actual.",
    "DEPARTMENT": "Parent department or sector of the implementing institution.",
    "GRIT TAGGING": "Institution-type classification used in the dataset: NGA, SUC, or GOCC.",
    "AGENCY": "Implementing or reporting agency/institution.",
    "Agency Display": "Derived department-agency label for clearer ranking and search.",
    "PAP ID": "Program, Activity, or Project identifier.",
    "PAP Description": "Name or description of the climate-tagged PAP.",
    "TYPOLOGY ID": "CCET typology code used to classify the PAP.",
    "TYPOLOGY Description": "Description of the assigned CCET typology.",
    "ADAPTATION": "Amount tagged for climate change adaptation, raw dataset value in thousand pesos.",
    "MITIGATION": "Amount tagged for climate change mitigation, raw dataset value in thousand pesos.",
    "TOTAL": "Total climate-tagged amount, raw dataset value in thousand pesos.",
    "Climate Pillar": "Derived climate pillar based on typology code and/or adaptation-mitigation amounts.",
    "NCCAP Code": "Derived NCCAP thematic priority code from the typology ID.",
    "NCCAP Thematic Priority": "Derived NCCAP thematic priority from the typology code.",
    "PDP / Executive Agenda Alignment": "Keyword-based analytical proxy for possible alignment with climate and development priorities.",
    "Text Corpus": "Combined text field used for keyword search and proxy alignment classification.",
}


# ============================================================
# FGD/KII AND RECOMMENDATION REFERENCE DATA
# ============================================================

FGD_KII_INSIGHTS = pd.DataFrame([
    {
        "Theme": "Knowledge concentration and continuity",
        "Budget Cycle Stage": "Preparation",
        "Institution Type": "NGA / GOCC / SUC",
        "Challenge": "CCET knowledge and responsibilities are often concentrated among a few focal persons, creating continuity risks when personnel change.",
        "Dashboard Response": "Add institution-type participation, focal-role notes, and familiarity indicators.",
        "Data Science Response": "Qualitative coding, frequency counts, and institution-type segmentation.",
        "Recommendation": "Establish minimum institutional requirements, internal CCET roles, and regular onboarding/refresher training.",
        "Priority": "High",
        "Smart Indicator": "Familiarity / role / year-involvement tracker",
    },
    {
        "Theme": "Climate relevance and attribution ambiguity",
        "Budget Cycle Stage": "Preparation",
        "Institution Type": "NGA / GOCC / SUC",
        "Challenge": "Agencies face difficulty deciding whether to tag whole PAPs or only climate-relevant components.",
        "Dashboard Response": "Add PAP-level attribution notes and flag large blanket-tagged PAPs for review.",
        "Data Science Response": "Rule-based flags, outlier detection, and PAP-level drill-down.",
        "Recommendation": "Develop more detailed, sector-specific tagging and attribution guidance, including proportional tagging rules.",
        "Priority": "High",
        "Smart Indicator": "Attribution method / large PAP flag",
    },
    {
        "Theme": "Weak climate-results tracking",
        "Budget Cycle Stage": "Execution / Accountability",
        "Institution Type": "NGA / GOCC / SUC",
        "Challenge": "Existing systems provide limited support for linking tagged budgets to climate indicators, accomplishments, or outcomes.",
        "Dashboard Response": "Add M&E readiness fields and future accomplishment/audit linkage placeholders.",
        "Data Science Response": "Data linkage design, completeness scoring, and dashboard maturity indicators.",
        "Recommendation": "Integrate climate indicators into planning, M&E, and reporting frameworks.",
        "Priority": "High",
        "Smart Indicator": "M&E linkage completeness",
    },
    {
        "Theme": "Budget-cycle traceability",
        "Budget Cycle Stage": "Legislation / Execution",
        "Institution Type": "NGA / GOCC / SUC",
        "Challenge": "PAP-level movement from NEP to GAA to Actual is difficult to reconcile consistently.",
        "Dashboard Response": "Add NEP-GAA-Actual variance and PAP-level reconciliation tables.",
        "Data Science Response": "Variance analysis, utilization-gap analysis, and unique-key reconciliation.",
        "Recommendation": "Institutionalize feedback and reconciliation mechanisms across budget stages.",
        "Priority": "High",
        "Smart Indicator": "NEP-GAA-Actual variance flag",
    },
    {
        "Theme": "Limited budget deliberation use",
        "Budget Cycle Stage": "Legislation",
        "Institution Type": "NGA / GOCC",
        "Challenge": "CCET reports are perceived to have limited influence in DBM or congressional budget deliberations.",
        "Dashboard Response": "Add budget-stage use notes and show changes between NEP and GAA.",
        "Data Science Response": "Comparative analytics and policy narrative generation.",
        "Recommendation": "Strengthen use of CCET analytics in budget review, policy briefs, and deliberation support.",
        "Priority": "Medium",
        "Smart Indicator": "NEP-to-GAA movement",
    },
    {
        "Theme": "Audit awareness and accountability",
        "Budget Cycle Stage": "Accountability",
        "Institution Type": "NGA / GOCC / SUC",
        "Challenge": "There are gaps in awareness and application of audit guidelines for climate-related expenditures.",
        "Dashboard Response": "Add audit-readiness notes and data-quality flags for missing typology or inconsistent amounts.",
        "Data Science Response": "Data quality validation and rule-based anomaly detection.",
        "Recommendation": "Review and clarify climate expenditure audit guidelines and agency documentation requirements.",
        "Priority": "Medium",
        "Smart Indicator": "Audit-readiness / data-quality score",
    },
    {
        "Theme": "Administrative burden and platform fragmentation",
        "Budget Cycle Stage": "Preparation / Reporting",
        "Institution Type": "NGA / GOCC / SUC",
        "Challenge": "Administrative processes and requirements can be fragmented across forms, platforms, and reporting tools.",
        "Dashboard Response": "Add integrated platform concept and exportable reconciled tables.",
        "Data Science Response": "Data productization, standardized schema, and automated exports.",
        "Recommendation": "Streamline administrative processes and requirements through an integrated platform.",
        "Priority": "Medium",
        "Smart Indicator": "Integrated platform readiness",
    },
])

RECOMMENDATION_SCORECARD = pd.DataFrame([
    {
        "Reform Area": "Minimum institutional requirements",
        "Purpose": "Ensure each institution has clear CCET roles, focal persons, and continuity arrangements.",
        "Dashboard Feature": "Participation and institution-type tracker",
        "Data Science Method": "Coverage analytics and segmentation",
        "Priority": "High",
    },
    {
        "Reform Area": "Tagging and attribution guidance",
        "Purpose": "Reduce ambiguity on whole-PAP vs component-based tagging.",
        "Dashboard Feature": "PAP Explorer, large-PAP flags, attribution notes",
        "Data Science Method": "Rule-based flags and outlier detection",
        "Priority": "High",
    },
    {
        "Reform Area": "Budget traceability and reconciliation",
        "Purpose": "Trace climate-tagged PAPs across NEP, GAA, and Actual stages.",
        "Dashboard Feature": "Budget Cycle Analysis",
        "Data Science Method": "Variance and utilization-gap analysis",
        "Priority": "High",
    },
    {
        "Reform Area": "Climate indicators and M&E",
        "Purpose": "Move from budget tagging toward results monitoring.",
        "Dashboard Feature": "Future M&E linkage fields",
        "Data Science Method": "Completeness scoring and data linkage",
        "Priority": "High",
    },
    {
        "Reform Area": "Integrated CCET platform",
        "Purpose": "Streamline data submission, review, reporting, and analytics.",
        "Dashboard Feature": "Unified schema, exports, data quality checks",
        "Data Science Method": "ETL, data validation, automated reporting",
        "Priority": "Medium",
    },
    {
        "Reform Area": "Audit guidance clarification",
        "Purpose": "Improve awareness and application of audit/accountability expectations.",
        "Dashboard Feature": "Data Quality and Audit Readiness",
        "Data Science Method": "Rule-based anomaly detection",
        "Priority": "Medium",
    },
])

BUDGET_CYCLE_STAGES = pd.DataFrame([
    {
        "Stage": "Preparation",
        "CCET Activity": "Agency climate planning, PAP identification, typology tagging, QAR preparation.",
        "Dashboard Module": "NCCAP Priorities, PAP Explorer, FGD/KII Challenges",
        "Data Science Method": "Feature engineering, classification, data profiling",
    },
    {
        "Stage": "Legislation",
        "CCET Activity": "NEP-to-GAA changes and budget deliberation movement.",
        "Dashboard Module": "Budget Cycle Analysis, Variance Charts",
        "Data Science Method": "Comparative analytics and variance analysis",
    },
    {
        "Stage": "Execution",
        "CCET Activity": "Actual expenditure / utilization tracking where available.",
        "Dashboard Module": "Actual vs GAA, Trend Analytics",
        "Data Science Method": "Utilization gap analysis and time-series aggregation",
    },
    {
        "Stage": "Accountability",
        "CCET Activity": "Reporting, auditability, data validation, and reconciliation.",
        "Dashboard Module": "Data Quality, Recommendations Tracker",
        "Data Science Method": "Rule-based validation and anomaly detection",
    },
])


# ============================================================
# DATA FUNCTIONS
# ============================================================

def find_default_data_path():
    for path in DEFAULT_DATA_CANDIDATES:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return None


def read_csv(source):
    return pd.read_csv(source, encoding="utf-8-sig")


def normalize_text_value(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def clean_pap_id(value):
    if pd.isna(value):
        return ""
    try:
        num = float(value)
        if num.is_integer():
            return str(int(num))
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def classify_alignment(text):
    text = str(text).lower()
    hits = sum(1 for kw in PDP_KEYWORDS if kw in text)
    if hits >= 3:
        return "Strongly Aligned"
    if hits >= 1:
        return "Partially Aligned"
    return "Weak / Unclassified"


def prepare_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"ADAPTION": "ADAPTATION"})

    required_text_cols = [
        "Type", "DEPARTMENT", "GRIT TAGGING", "AGENCY",
        "PAP ID", "PAP Description", "TYPOLOGY ID", "TYPOLOGY Description"
    ]
    required_num_cols = ["Fiscal_Year", "ADAPTATION", "MITIGATION", "TOTAL"]

    for col in required_text_cols:
        if col not in df.columns:
            df[col] = ""

    for col in required_num_cols:
        if col not in df.columns:
            df[col] = 0

    df["Fiscal_Year"] = pd.to_numeric(df["Fiscal_Year"], errors="coerce")
    df = df[df["Fiscal_Year"].notna()].copy()
    df["Fiscal_Year"] = df["Fiscal_Year"].astype(int)

    for col in ["ADAPTATION", "MITIGATION", "TOTAL"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["Type", "DEPARTMENT", "GRIT TAGGING", "AGENCY", "PAP Description", "TYPOLOGY ID", "TYPOLOGY Description"]:
        df[col] = df[col].apply(normalize_text_value)

    df["PAP ID"] = df["PAP ID"].apply(clean_pap_id)

    df["Type"] = df["Type"].str.strip()
    df["GRIT TAGGING"] = df["GRIT TAGGING"].str.upper().str.strip().replace({"": "Unclassified"})
    df["DEPARTMENT"] = df["DEPARTMENT"].replace({"": "Unspecified Department"})
    df["AGENCY"] = df["AGENCY"].replace({"": "Unspecified Agency"})

    df["Agency Display"] = np.where(
        df["DEPARTMENT"].str.lower().eq(df["AGENCY"].str.lower()),
        df["AGENCY"],
        df["DEPARTMENT"] + " - " + df["AGENCY"],
    )

    typo = df["TYPOLOGY ID"].astype(str).str.upper().str.strip()
    df["NCCAP Code"] = typo.str.extract(r"^[AM](\d)", expand=False).fillna("")
    df["NCCAP Thematic Priority"] = df["NCCAP Code"].map(NCCAP_PRIORITY).fillna("Unclassified")
    df["NCCAP Priority"] = df["NCCAP Thematic Priority"]

    pillar_from_typology = np.select(
        [typo.str.startswith("A"), typo.str.startswith("M")],
        ["Adaptation", "Mitigation"],
        default="",
    )

    pillar_from_amount = np.select(
        [
            (df["ADAPTATION"] > 0) & (df["MITIGATION"] <= 0),
            (df["MITIGATION"] > 0) & (df["ADAPTATION"] <= 0),
            (df["ADAPTATION"] > 0) & (df["MITIGATION"] > 0),
        ],
        ["Adaptation", "Mitigation", "Adaptation + Mitigation"],
        default="Unclassified",
    )

    df["Climate Pillar"] = np.where(pillar_from_typology != "", pillar_from_typology, pillar_from_amount)

    df["Text Corpus"] = (
        df["PAP Description"].astype(str) + " " +
        df["TYPOLOGY Description"].astype(str) + " " +
        df["AGENCY"].astype(str) + " " +
        df["DEPARTMENT"].astype(str)
    )
    df["PDP / Executive Agenda Alignment"] = df["Text Corpus"].apply(classify_alignment)

    df["TOTAL_BILLION"] = df["TOTAL"] / 1_000_000
    df["ADAPTATION_BILLION"] = df["ADAPTATION"] / 1_000_000
    df["MITIGATION_BILLION"] = df["MITIGATION"] / 1_000_000

    return df


@st.cache_data(show_spinner="Loading CCET dataset...")
def load_default_data():
    data_path = find_default_data_path()
    if data_path is None:
        st.error(
            "Default CSV dataset was not found. Please upload the cleaned CCET CSV from the sidebar, "
            "or save it as `data/ccet_data.csv`."
        )
        st.stop()
    return prepare_data(read_csv(data_path)), data_path


# ============================================================
# FORMATTERS
# ============================================================

def raw_to_actual_pesos(raw_value):
    return float(raw_value or 0) * DATASET_VALUE_MULTIPLIER


def raw_to_billion(raw_value):
    return float(raw_value or 0) / 1_000_000


def peso_from_raw(raw_value):
    value = raw_to_actual_pesos(raw_value)
    return peso_from_actual(value)


def peso_from_actual(value):
    if pd.isna(value):
        return "N/A"
    value = float(value)
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
    value = float(value)
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


def percent_label(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"


def pct(numerator, denominator):
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return numerator / denominator * 100


def wrap_label(text, width=44):
    text = str(text)
    if len(text) <= width:
        return text
    words = text.split()
    lines = []
    line = ""
    for word in words:
        if len(line + " " + word) <= width:
            line = (line + " " + word).strip()
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return "<br>".join(lines[:3])


def safe_group_sum(data, group_cols, value_col="TOTAL", top_n=None):
    if data.empty:
        return pd.DataFrame(columns=group_cols + [value_col])
    out = data.groupby(group_cols, as_index=False)[value_col].sum().sort_values(value_col, ascending=False)
    if top_n:
        out = out.head(top_n)
    return out


def safe_top_value(dataframe, group_col, value_col="TOTAL"):
    if dataframe.empty or group_col not in dataframe.columns:
        return "No data", 0
    temp = safe_group_sum(dataframe, [group_col], value_col=value_col, top_n=1)
    if temp.empty:
        return "No data", 0
    return temp.iloc[0][group_col], temp.iloc[0][value_col]


# ============================================================
# UI HELPERS
# ============================================================

def filter_dropdown(label, values, key=None):
    clean_values = sorted([v for v in pd.Series(values).dropna().unique() if str(v).strip() != ""])
    return st.sidebar.selectbox(label, ["All"] + clean_values, key=key)


def apply_report_layout(
    fig,
    title=None,
    subtitle=None,
    height=720,
    left=80,
    right=70,
    top=125,
    bottom=115,
    source_note=None,
    legend_y=1.02,
    showlegend=True,
):
    title_text = title or ""
    if subtitle:
        title_text += f"<br><span style='font-size:14px;color:#4F4F4F'>{subtitle}</span>"

    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=left, r=right, t=top, b=bottom),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", size=12, color=DARK_TEXT),
        title=dict(
            text=title_text,
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=20, color=PRIMARY_BLUE),
        ),
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=legend_y,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="#D9E2F3",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(showline=True, linecolor="#BFBFBF", ticks="outside")
    fig.update_yaxes(showline=True, linecolor="#BFBFBF", ticks="outside", gridcolor=LIGHT_GRAY)
    if source_note:
        fig.add_annotation(
            text=source_note,
            xref="paper",
            yref="paper",
            x=0,
            y=-0.20,
            showarrow=False,
            align="left",
            font=dict(size=10.5, color="#555555"),
        )
    return fig


def render_chart(fig, file_stem, title="Chart", height=720, width=1450):
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": file_stem,
                "height": height,
                "width": width,
                "scale": 3,
            },
        },
    )
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            f"Download {title} HTML",
            fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8"),
            file_name=f"{file_stem}.html",
            mime="text/html",
            key=f"{file_stem}_html",
        )
    with c2:
        try:
            png = fig.to_image(format="png", width=width, height=height, scale=3)
            st.download_button(
                f"Download {title} PNG",
                png,
                file_name=f"{file_stem}.png",
                mime="image/png",
                key=f"{file_stem}_png",
            )
        except Exception:
            st.caption("PNG export requires `kaleido`. Add `kaleido` to requirements.txt.")


def smart_note(text):
    st.markdown(
        f"""
        <div style="background:#F7FBFF;border-left:5px solid {DEEP_BLUE};
        padding:0.9rem 1rem;border-radius:0.45rem;margin:0.5rem 0 1rem 0;">
        {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ANALYTICS FUNCTIONS
# ============================================================

def climate_budget_share_data(data):
    gaa = data[data["Type"].str.upper().eq("GAA")].copy()
    annual = gaa.groupby("Fiscal_Year", as_index=False)["TOTAL"].sum()
    annual["Climate-Tagged GAA (Billion Pesos)"] = annual["TOTAL"].apply(raw_to_billion)
    merged = NATIONAL_BUDGET_BILLION.merge(annual[["Fiscal_Year", "Climate-Tagged GAA (Billion Pesos)"]], on="Fiscal_Year", how="left")
    merged["Climate-Tagged GAA (Billion Pesos)"] = merged["Climate-Tagged GAA (Billion Pesos)"].fillna(0)
    merged["Share of National Budget (%)"] = (
        merged["Climate-Tagged GAA (Billion Pesos)"] / merged["Total National Budget (Billion Pesos)"] * 100
    )
    merged["Fiscal Year"] = "FY" + merged["Fiscal_Year"].astype(str)
    merged["Climate-Tagged GAA (Trillion Pesos)"] = merged["Climate-Tagged GAA (Billion Pesos)"] / 1000
    merged["Total National Budget (Trillion Pesos)"] = merged["Total National Budget (Billion Pesos)"] / 1000
    return merged


def budget_stage_wide(data):
    d = data.copy()
    stage = d.pivot_table(index="Fiscal_Year", columns="Type", values="TOTAL", aggfunc="sum").reset_index()
    for col in ["NEP", "GAA", "Actual"]:
        if col not in stage.columns:
            stage[col] = np.nan
    stage = stage.sort_values("Fiscal_Year")
    for col in ["NEP", "GAA", "Actual"]:
        stage[col + " (Billion Pesos)"] = stage[col] / 1_000_000
    stage["GAA minus NEP (Billion Pesos)"] = stage["GAA (Billion Pesos)"] - stage["NEP (Billion Pesos)"]
    stage["Actual minus GAA (Billion Pesos)"] = stage["Actual (Billion Pesos)"] - stage["GAA (Billion Pesos)"]
    stage["Actual vs GAA (%)"] = (stage["Actual"] - stage["GAA"]) / stage["GAA"] * 100
    stage["Fiscal Year"] = "FY" + stage["Fiscal_Year"].astype(str)
    return stage


def data_quality_masks(data):
    tolerance = 1
    masks = {
        "Missing department": data["DEPARTMENT"].eq("Unspecified Department") | data["DEPARTMENT"].eq(""),
        "Missing agency": data["AGENCY"].eq("Unspecified Agency") | data["AGENCY"].eq(""),
        "Missing GRIT TAGGING": data["GRIT TAGGING"].eq("Unclassified") | data["GRIT TAGGING"].eq(""),
        "Missing PAP ID": data["PAP ID"].eq(""),
        "Missing typology ID": data["TYPOLOGY ID"].eq(""),
        "Missing typology description": data["TYPOLOGY Description"].eq(""),
        "Zero or blank total": data["TOTAL"].fillna(0).eq(0),
        "Negative total": data["TOTAL"].fillna(0).lt(0),
        "Adaptation + Mitigation ≠ Total": (
            (data["ADAPTATION"].fillna(0) + data["MITIGATION"].fillna(0) - data["TOTAL"].fillna(0)).abs() > tolerance
        ),
        "Duplicate PAP-stage records": data.duplicated(
            subset=["Fiscal_Year", "Type", "DEPARTMENT", "AGENCY", "PAP ID", "TYPOLOGY ID"],
            keep=False,
        ),
        "Unclassified NCCAP priority": data["NCCAP Thematic Priority"].eq("Unclassified"),
    }
    return masks


def data_quality_summary(data):
    masks = data_quality_masks(data)
    out = pd.DataFrame([
        {
            "Check": name,
            "Flagged Records": int(mask.sum()),
            "Share of Filtered Records (%)": pct(int(mask.sum()), len(data)) if len(data) else 0,
        }
        for name, mask in masks.items()
    ])
    return out


def quality_score(data):
    if len(data) == 0:
        return 0
    masks = data_quality_masks(data)
    critical = (
        masks["Missing typology ID"] |
        masks["Zero or blank total"] |
        masks["Adaptation + Mitigation ≠ Total"] |
        masks["Duplicate PAP-stage records"]
    )
    return max(0, 100 - critical.mean() * 100)


def apply_filters(df):
    f = df.copy()
    if st.session_state.get("year_filter") != "All":
        f = f[f["Fiscal_Year"] == int(st.session_state["year_filter"])]
    if st.session_state.get("type_filter") != "All":
        f = f[f["Type"] == st.session_state["type_filter"]]
    if st.session_state.get("grit_filter") != "All":
        f = f[f["GRIT TAGGING"] == st.session_state["grit_filter"]]
    if st.session_state.get("dept_filter") != "All":
        f = f[f["DEPARTMENT"] == st.session_state["dept_filter"]]
    if st.session_state.get("pillar_filter") != "All":
        f = f[f["Climate Pillar"] == st.session_state["pillar_filter"]]
    if st.session_state.get("nccap_filter") != "All":
        f = f[f["NCCAP Thematic Priority"] == st.session_state["nccap_filter"]]
    if st.session_state.get("pdp_filter") != "All":
        f = f[f["PDP / Executive Agenda Alignment"] == st.session_state["pdp_filter"]]
    return f


# ============================================================
# CHART BUILDERS
# ============================================================

def build_climate_budget_share_chart(data):
    d = climate_budget_share_data(data)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=d["Fiscal Year"],
            y=d["Climate-Tagged GAA (Trillion Pesos)"],
            name="Climate-Tagged GAA",
            marker=dict(color=DEEP_BLUE, line=dict(color="white", width=1)),
            text=[peso_billion(v) for v in d["Climate-Tagged GAA (Billion Pesos)"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Climate-Tagged GAA: %{text}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=d["Fiscal Year"],
            y=d["Total National Budget (Trillion Pesos)"],
            name="Total National Budget",
            marker=dict(color=GRAY, line=dict(color="white", width=1)),
            text=[f"₱{v:.3f}T" for v in d["Total National Budget (Trillion Pesos)"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Total National Budget: %{text}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=d["Fiscal Year"],
            y=d["Share of National Budget (%)"],
            name="Share of National Budget",
            mode="lines+markers+text",
            line=dict(color=ORANGE, width=4),
            marker=dict(size=10, color=ORANGE),
            text=[percent_label(v, 1) for v in d["Share of National Budget (%)"]],
            textposition="top center",
            hovertemplate="<b>%{x}</b><br>Share: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    apply_report_layout(
        fig,
        title="<b>Climate-Tagged GAA, Total National Budget, and Share of National Budget</b>",
        subtitle="Bars show budget amounts; line shows share of national budget.",
        height=760,
        bottom=135,
        source_note="Method: Climate-tagged GAA ÷ total national budget × 100. When filters are active, climate-tagged GAA reflects the selected subset.",
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Amount (Trillion Pesos)", range=[0, 7.3], secondary_y=False)
    fig.update_yaxes(title_text="Share of National Budget (%)", range=[0, max(22, d["Share of National Budget (%)"].max() * 1.25)], ticksuffix="%", secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="Fiscal Year")
    return fig


def build_budget_by_type_chart(data):
    stage = safe_group_sum(data, ["Fiscal_Year", "Type"], "TOTAL")
    stage["Amount (Billion Pesos)"] = stage["TOTAL"] / 1_000_000
    stage["Fiscal Year"] = "FY" + stage["Fiscal_Year"].astype(str)
    fig = px.line(
        stage.sort_values("Fiscal_Year"),
        x="Fiscal Year",
        y="Amount (Billion Pesos)",
        color="Type",
        markers=True,
        color_discrete_sequence=CHART_COLOR_SEQUENCE,
        category_orders={"Type": ["NEP", "GAA", "Actual"]},
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    apply_report_layout(
        fig,
        title="<b>Climate-Tagged Budget by Budget Stage</b>",
        subtitle="NEP, GAA, and Actual values across available fiscal years.",
        height=680,
        bottom=125,
        source_note="Method: Sum of TOTAL by fiscal year and budget type. Values are expressed in billion pesos.",
    )
    fig.update_yaxes(title_text="Amount (Billion Pesos)")
    fig.update_xaxes(title_text="Fiscal Year")
    return fig


def build_budget_cycle_variance_chart(data):
    w = budget_stage_wide(data)
    fig = go.Figure()
    variance_cols = ["GAA minus NEP (Billion Pesos)", "Actual minus GAA (Billion Pesos)"]
    colors = {"GAA minus NEP (Billion Pesos)": ORANGE, "Actual minus GAA (Billion Pesos)": GRAY}
    labels = {"GAA minus NEP (Billion Pesos)": "GAA minus NEP", "Actual minus GAA (Billion Pesos)": "Actual minus GAA"}
    for col in variance_cols:
        fig.add_trace(
            go.Bar(
                x=w["Fiscal Year"],
                y=w[col],
                name=labels[col],
                marker=dict(color=colors[col], line=dict(color="white", width=1)),
                text=[signed_peso_billion(v) for v in w[col]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{x}</b><br>" + labels[col] + ": %{text}<extra></extra>",
            )
        )
    max_abs = np.nanmax(np.abs(w[variance_cols].values)) if len(w) else 1
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#808080")
    apply_report_layout(
        fig,
        title="<b>Budget-Cycle Variance in CCET</b>",
        subtitle="Difference between NEP and GAA, and between GAA and Actual.",
        height=720,
        bottom=135,
        source_note="Formula: GAA variance = GAA - NEP; Actual variance = Actual - GAA. Positive values indicate increases across budget stages.",
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Billion Pesos", range=[-max_abs * 1.45, max_abs * 1.45])
    fig.update_xaxes(title_text="Fiscal Year")
    return fig


def build_actual_vs_gaa_chart(data):
    w = budget_stage_wide(data)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=w["Fiscal Year"],
            y=w["Actual vs GAA (%)"],
            name="Actual vs GAA",
            marker=dict(color=DEEP_BLUE, line=dict(color="white", width=1)),
            text=[percent_label(v, 2) for v in w["Actual vs GAA (%)"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Actual vs GAA: %{text}<extra></extra>",
        )
    )
    max_pct = np.nanmax(np.abs(w["Actual vs GAA (%)"].values)) if len(w) else 1
    if not np.isfinite(max_pct) or max_pct == 0:
        max_pct = 1
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#808080")
    apply_report_layout(
        fig,
        title="<b>Actual Compared with GAA</b>",
        subtitle="Percentage difference between Actual and approved GAA CCET amounts.",
        height=680,
        bottom=135,
        source_note="Formula: (Actual - GAA) ÷ GAA × 100. Missing Actual data are treated as unavailable, not zero.",
        showlegend=False,
    )
    fig.update_yaxes(title_text="Percent Difference", ticksuffix="%", range=[-max_pct * 1.25, max_pct * 1.25])
    fig.update_xaxes(title_text="Fiscal Year")
    return fig


def build_adaptation_mitigation_share_chart(data):
    d = data.copy()
    annual = d.groupby("Fiscal_Year", as_index=False)[["ADAPTATION", "MITIGATION"]].sum()
    annual["Total Pillar Amount"] = annual["ADAPTATION"] + annual["MITIGATION"]
    annual["Adaptation Share"] = annual["ADAPTATION"] / annual["Total Pillar Amount"] * 100
    annual["Mitigation Share"] = annual["MITIGATION"] / annual["Total Pillar Amount"] * 100
    annual = annual.replace([np.inf, -np.inf], np.nan).fillna(0)
    annual["Fiscal Year"] = "FY" + annual["Fiscal_Year"].astype(str)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=annual["Fiscal Year"],
            y=annual["Adaptation Share"],
            name="Adaptation",
            marker=dict(color=DEEP_BLUE, line=dict(color="white", width=1)),
            text=[percent_label(v, 1) for v in annual["Adaptation Share"]],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="<b>%{x}</b><br>Adaptation: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=annual["Fiscal Year"],
            y=annual["Mitigation Share"],
            name="Mitigation",
            marker=dict(color=GREEN, line=dict(color="white", width=1)),
            text=[percent_label(v, 1) if v >= 3 else "" for v in annual["Mitigation Share"]],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="<b>%{x}</b><br>Mitigation: %{y:.2f}%<extra></extra>",
        )
    )
    apply_report_layout(
        fig,
        title="<b>Adaptation and Mitigation Share</b>",
        subtitle="Distribution of climate-tagged budget by climate pillar.",
        height=700,
        bottom=130,
        source_note="Formula: pillar amount ÷ total adaptation-plus-mitigation amount × 100. Values respond to sidebar filters.",
    )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Share of Climate-Tagged Budget (%)", range=[0, 100], ticksuffix="%")
    fig.update_xaxes(title_text="Fiscal Year")
    return fig


def build_participation_chart(data):
    d = data.drop_duplicates(["Fiscal_Year", "GRIT TAGGING", "Agency Display"])
    d = d.groupby(["Fiscal_Year", "GRIT TAGGING"], as_index=False)["Agency Display"].nunique()
    d = d.rename(columns={"Agency Display": "Participating NGIs"})
    d["Fiscal Year"] = "FY" + d["Fiscal_Year"].astype(str)
    fig = px.bar(
        d.sort_values("Fiscal_Year"),
        x="Fiscal Year",
        y="Participating NGIs",
        color="GRIT TAGGING",
        barmode="stack",
        color_discrete_sequence=CHART_COLOR_SEQUENCE,
        text="Participating NGIs",
    )
    fig.update_traces(textposition="inside")
    apply_report_layout(
        fig,
        title="<b>CCET Participation by Institution Type</b>",
        subtitle="Unique participating institutions by fiscal year and GRIT TAGGING.",
        height=700,
        bottom=130,
        source_note="Method: Count unique department-agency labels per fiscal year and institution type.",
    )
    fig.update_yaxes(title_text="Number of Participating Institutions")
    fig.update_xaxes(title_text="Fiscal Year")
    return fig


def build_institution_type_budget_chart(data):
    d = safe_group_sum(data, ["Fiscal_Year", "GRIT TAGGING"], "TOTAL")
    d["Amount (Billion Pesos)"] = d["TOTAL"] / 1_000_000
    d["Fiscal Year"] = "FY" + d["Fiscal_Year"].astype(str)
    fig = px.area(
        d.sort_values("Fiscal_Year"),
        x="Fiscal Year",
        y="Amount (Billion Pesos)",
        color="GRIT TAGGING",
        color_discrete_sequence=CHART_COLOR_SEQUENCE,
        line_group="GRIT TAGGING",
    )
    apply_report_layout(
        fig,
        title="<b>Climate-Tagged Budget by Institution Type</b>",
        subtitle="Budget composition across NGAs, SUCs, and GOCCs.",
        height=680,
        bottom=130,
        source_note="Method: Sum of TOTAL by fiscal year and GRIT TAGGING. Values are expressed in billion pesos.",
    )
    fig.update_yaxes(title_text="Amount (Billion Pesos)")
    fig.update_xaxes(title_text="Fiscal Year")
    return fig


def build_ranking_chart(data, group_col, title, subtitle, top_n=15, source_note=None):
    d = safe_group_sum(data, [group_col], "TOTAL", top_n=top_n)
    d["Amount (Billion Pesos)"] = d["TOTAL"] / 1_000_000
    d["Amount Label"] = d["Amount (Billion Pesos)"].apply(peso_billion)
    d["Wrapped Label"] = d[group_col].apply(lambda x: wrap_label(x, 58))
    d = d.sort_values("Amount (Billion Pesos)", ascending=True)
    fig = px.bar(
        d,
        x="Amount (Billion Pesos)",
        y="Wrapped Label",
        orientation="h",
        text="Amount Label",
        color_discrete_sequence=[DEEP_BLUE],
    )
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_color="white", marker_line_width=1.2)
    apply_report_layout(
        fig,
        title=title,
        subtitle=subtitle,
        height=max(560, 42 * len(d) + 220),
        left=360,
        right=140,
        bottom=130,
        source_note=source_note or "Method: Sum of TOTAL by selected category. Values are expressed in billion pesos.",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Amount (Billion Pesos)")
    fig.update_yaxes(title_text="")
    return fig


def build_top_ngi_pareto(data, top_n=20):
    d = safe_group_sum(data, ["Agency Display"], "TOTAL", top_n=top_n)
    if d.empty:
        return go.Figure()
    d["Amount (Billion Pesos)"] = d["TOTAL"] / 1_000_000
    d["Share (%)"] = d["TOTAL"] / data["TOTAL"].sum() * 100 if data["TOTAL"].sum() else 0
    d["Cumulative Share (%)"] = d["Share (%)"].cumsum()
    d["Wrapped Label"] = d["Agency Display"].apply(lambda x: wrap_label(x, 34))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=d["Wrapped Label"],
            y=d["Amount (Billion Pesos)"],
            name="Amount",
            marker=dict(color=DEEP_BLUE, line=dict(color="white", width=1)),
            text=[peso_billion(v) for v in d["Amount (Billion Pesos)"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Amount: %{text}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=d["Wrapped Label"],
            y=d["Cumulative Share (%)"],
            name="Cumulative Share",
            mode="lines+markers",
            line=dict(color=ORANGE, width=3),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>Cumulative share: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    apply_report_layout(
        fig,
        title="<b>Top NGI Concentration / Pareto View</b>",
        subtitle=f"Top {top_n} institutions and their cumulative share of the filtered climate budget.",
        height=760,
        left=95,
        right=95,
        bottom=245,
        source_note="Method: Rank institutions by total amount, then compute cumulative share of the filtered total.",
    )
    fig.update_xaxes(title_text="Institution", tickangle=-35)
    fig.update_yaxes(title_text="Amount (Billion Pesos)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Share (%)", ticksuffix="%", range=[0, min(105, max(100, d["Cumulative Share (%)"].max() * 1.05))], secondary_y=True, showgrid=False)
    return fig


def build_nccap_heatmap(data):
    d = safe_group_sum(data, ["NCCAP Thematic Priority", "Fiscal_Year"], "TOTAL")
    d["Amount (Billion Pesos)"] = d["TOTAL"] / 1_000_000
    matrix = d.pivot_table(index="NCCAP Thematic Priority", columns="Fiscal_Year", values="Amount (Billion Pesos)", aggfunc="sum").fillna(0)
    ordered_index = [p for p in NCCAP_ORDER if p in matrix.index] + [p for p in matrix.index if p not in NCCAP_ORDER]
    matrix = matrix.loc[ordered_index]
    years = list(matrix.columns)

    share_matrix = matrix.copy()
    for year in years:
        total = share_matrix[year].sum()
        share_matrix[year] = share_matrix[year] / total * 100 if total else 0

    customdata = np.stack([matrix.values, share_matrix.values], axis=-1)

    fig = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=["FY" + str(y) for y in years],
            y=matrix.index,
            customdata=customdata,
            colorscale=[
                [0.00, "#F7FBFF"],
                [0.20, "#CFE1F2"],
                [0.40, "#8FBBD9"],
                [0.60, "#4F8FC2"],
                [0.80, "#1F5F99"],
                [1.00, PRIMARY_BLUE],
            ],
            colorbar=dict(title="Billion pesos", len=0.78),
            xgap=3,
            ygap=3,
            hovertemplate="<b>%{y}</b><br>%{x}<br>Allocation: ₱%{customdata[0]:,.2f}B<br>Share: %{customdata[1]:.2f}%<extra></extra>",
        )
    )

    max_val = np.nanmax(matrix.values) if matrix.size else 0
    threshold = max_val * 0.45
    for i, priority in enumerate(matrix.index):
        for j, year_label in enumerate(["FY" + str(y) for y in years]):
            amount = matrix.iloc[i, j]
            share = share_matrix.iloc[i, j]
            label = f"{peso_billion(amount)}<br>({share:.1f}%)" if amount > 0 else "—"
            font_color = "white" if amount >= threshold and amount > 0 else DARK_TEXT
            fig.add_annotation(
                x=year_label,
                y=priority,
                text=label,
                showarrow=False,
                align="center",
                font=dict(size=10.5, color=font_color),
            )

    fig.update_layout(
        title=dict(
            text="<b>NCCAP Thematic Priority Allocation Matrix</b><br><span style='font-size:14px;color:#4F4F4F'>Amount and fiscal-year share per thematic priority.</span>",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=20, color=PRIMARY_BLUE),
        ),
        template="plotly_white",
        height=max(760, 72 * len(matrix.index) + 220),
        margin=dict(l=390, r=110, t=155, b=135),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", size=12),
    )
    fig.update_xaxes(title_text="Fiscal Year", side="top")
    fig.update_yaxes(title_text="NCCAP Thematic Priority", autorange="reversed")
    fig.add_annotation(
        text="Formula: Priority share = priority allocation ÷ total fiscal-year allocation × 100. Values respond to sidebar filters.",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.15,
        showarrow=False,
        align="left",
        font=dict(size=10.5, color="#555555"),
    )
    return fig


def build_nccap_stacked(data):
    d = safe_group_sum(data, ["Fiscal_Year", "NCCAP Thematic Priority"], "TOTAL")
    d["Amount (Billion Pesos)"] = d["TOTAL"] / 1_000_000
    d["Fiscal Year"] = "FY" + d["Fiscal_Year"].astype(str)
    order = [p for p in NCCAP_ORDER if p in d["NCCAP Thematic Priority"].unique()]
    fig = px.bar(
        d.sort_values("Fiscal_Year"),
        x="Fiscal Year",
        y="Amount (Billion Pesos)",
        color="NCCAP Thematic Priority",
        category_orders={"NCCAP Thematic Priority": order},
        color_discrete_sequence=CHART_COLOR_SEQUENCE,
    )
    apply_report_layout(
        fig,
        title="<b>NCCAP Thematic Priority Allocation Mix</b>",
        subtitle="Stacked view of climate-tagged allocations by priority.",
        height=820,
        bottom=250,
        legend_y=-0.23,
        source_note="Method: Sum TOTAL by fiscal year and NCCAP thematic priority. Values are expressed in billion pesos.",
    )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Amount (Billion Pesos)")
    fig.update_xaxes(title_text="Fiscal Year")
    return fig


def build_nccap_rank(data):
    return build_ranking_chart(
        data,
        "NCCAP Thematic Priority",
        "<b>NCCAP Thematic Priorities by Allocation</b>",
        "Ranking based on the currently filtered dataset.",
        top_n=20,
        source_note="Method: Sum TOTAL by NCCAP thematic priority. Values are expressed in billion pesos.",
    )


def build_pdp_alignment_chart(data):
    d = safe_group_sum(data, ["PDP / Executive Agenda Alignment"], "TOTAL")
    d["Amount (Billion Pesos)"] = d["TOTAL"] / 1_000_000
    d["Share (%)"] = d["TOTAL"] / d["TOTAL"].sum() * 100 if d["TOTAL"].sum() else 0
    fig = px.bar(
        d.sort_values("Amount (Billion Pesos)"),
        x="Amount (Billion Pesos)",
        y="PDP / Executive Agenda Alignment",
        orientation="h",
        text=d["Share (%)"].map(lambda x: f"{x:.1f}%"),
        color="PDP / Executive Agenda Alignment",
        color_discrete_sequence=CHART_COLOR_SEQUENCE,
    )
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_color="white", marker_line_width=1.2)
    apply_report_layout(
        fig,
        title="<b>PDP / Executive Agenda Alignment Proxy</b>",
        subtitle="Keyword-based exploratory alignment, not official validation.",
        height=560,
        left=210,
        right=140,
        bottom=130,
        source_note="Method: Count climate/development keyword hits in PAP and typology text; classify as strong, partial, or weak/unclassified.",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Amount (Billion Pesos)")
    fig.update_yaxes(title_text="")
    return fig


def build_pdp_alignment_trend(data):
    d = safe_group_sum(data, ["Fiscal_Year", "PDP / Executive Agenda Alignment"], "TOTAL")
    d["Amount (Billion Pesos)"] = d["TOTAL"] / 1_000_000
    d["Fiscal Year"] = "FY" + d["Fiscal_Year"].astype(str)
    fig = px.bar(
        d.sort_values("Fiscal_Year"),
        x="Fiscal Year",
        y="Amount (Billion Pesos)",
        color="PDP / Executive Agenda Alignment",
        barmode="stack",
        color_discrete_sequence=CHART_COLOR_SEQUENCE,
    )
    apply_report_layout(
        fig,
        title="<b>Estimated National Plan Alignment Trend</b>",
        subtitle="Budget movement by keyword-based alignment category.",
        height=650,
        bottom=160,
        source_note="Method: Sum TOTAL by fiscal year and keyword-based alignment category.",
    )
    fig.update_yaxes(title_text="Amount (Billion Pesos)")
    fig.update_xaxes(title_text="Fiscal Year")
    return fig


def build_fgd_priority_chart():
    d = FGD_KII_INSIGHTS.groupby(["Priority", "Budget Cycle Stage"], as_index=False).size()
    d = d.rename(columns={"size": "Number of Issues"})
    order = ["High", "Medium", "Low"]
    fig = px.bar(
        d,
        x="Priority",
        y="Number of Issues",
        color="Budget Cycle Stage",
        category_orders={"Priority": order},
        text="Number of Issues",
        color_discrete_sequence=CHART_COLOR_SEQUENCE,
    )
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_color="white", marker_line_width=1.2)
    apply_report_layout(
        fig,
        title="<b>FGD/KII Challenges by Priority and Budget-Cycle Stage</b>",
        subtitle="Structured coding of implementation issues for dashboard design.",
        height=620,
        bottom=135,
        source_note="Method: Qualitative findings are coded by theme, budget-cycle stage, institution type, and priority level.",
    )
    fig.update_yaxes(title_text="Number of Coded Issues", dtick=1)
    fig.update_xaxes(title_text="Priority")
    return fig


def build_recommendation_chart():
    d = RECOMMENDATION_SCORECARD.groupby(["Priority"], as_index=False).size()
    d = d.rename(columns={"size": "Number of Reform Areas"})
    fig = px.bar(
        d,
        x="Priority",
        y="Number of Reform Areas",
        color="Priority",
        text="Number of Reform Areas",
        category_orders={"Priority": ["High", "Medium", "Low"]},
        color_discrete_map={"High": ORANGE, "Medium": MID_BLUE, "Low": GREEN},
    )
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_color="white", marker_line_width=1.2)
    apply_report_layout(
        fig,
        title="<b>Recommendation Priority Tracker</b>",
        subtitle="Reform areas translated into dashboard monitoring actions.",
        height=540,
        bottom=115,
        source_note="Method: Challenge-to-recommendation mapping from FGD/KII synthesis and report recommendations.",
        showlegend=False,
    )
    fig.update_yaxes(title_text="Number of Reform Areas", dtick=1)
    fig.update_xaxes(title_text="Priority")
    return fig


def build_quality_chart(data):
    q = data_quality_summary(data).sort_values("Flagged Records", ascending=True)
    fig = px.bar(
        q,
        x="Flagged Records",
        y="Check",
        orientation="h",
        text="Flagged Records",
        color="Share of Filtered Records (%)",
        color_continuous_scale="Blues",
    )
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_color="white", marker_line_width=1.2)
    apply_report_layout(
        fig,
        title="<b>Data Quality Flags</b>",
        subtitle="Records that may need validation before policy interpretation.",
        height=max(650, 38 * len(q) + 220),
        left=280,
        right=150,
        bottom=135,
        source_note="Method: Rule-based validation flags missing fields, zero totals, inconsistent amounts, duplicates, and unclassified priorities.",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Flagged Records")
    fig.update_yaxes(title_text="")
    return fig


# ============================================================
# REPORT EXPORT
# ============================================================

def generate_pdf_report(f, filters_used):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    total_budget = f["TOTAL"].sum()
    adaptation_total = f["ADAPTATION"].sum()
    mitigation_total = f["MITIGATION"].sum()
    adaptation_share = pct(adaptation_total, total_budget)
    mitigation_share = pct(mitigation_total, total_budget)
    top_agency, top_agency_amount = safe_top_value(f, "Agency Display")
    top_priority, top_priority_amount = safe_top_value(f, "NCCAP Thematic Priority")

    story.append(Paragraph("National CCET Smart Policy Analytics Report", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y %I:%M %p')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Filters Applied", styles["Heading2"]))
    filter_table = [["Filter", "Selected Value"]] + [[k, str(v)] for k, v in filters_used.items()]
    table = Table(filter_table, colWidths=[170, 330])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    summary = f"""
    The filtered dataset covers {peso_from_raw(total_budget)} in climate-tagged PAPs,
    involving {f['Agency Display'].nunique()} institutions and {f['PAP ID'].replace('', np.nan).nunique()} unique PAP IDs.
    Adaptation accounts for {adaptation_share:.2f}% of the filtered total, while mitigation accounts for {mitigation_share:.2f}%.
    The top institution is {top_agency} with {peso_from_raw(top_agency_amount)}.
    The largest NCCAP thematic priority is {top_priority} with {peso_from_raw(top_priority_amount)}.
    """
    story.append(Paragraph(summary, styles["Normal"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Interpretation Note", styles["Heading2"]))
    note = """
    This dashboard supports policy analysis but does not replace official agency validation, QAR documentation,
    audit review, or climate impact evaluation. Climate-tagged budgets indicate tagging and budget visibility,
    not automatically actual expenditure, implementation performance, or climate outcomes.
    """
    story.append(Paragraph(note, styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ============================================================
# MAIN APP
# ============================================================

st.title("National CCET Smart Policy Analytics Platform")
st.caption(
    "Dynamic PAP-level analytics for National Climate Change Expenditure Tagging: budget trends, participation, "
    "NCCAP priorities, budget-cycle traceability, FGD/KII challenges, recommendations, and data quality."
)

st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload updated CCET CSV",
    type=["csv"],
    help="Upload the cleaned National CCET PAP-level dataset. If no file is uploaded, the dashboard uses the default data path.",
)

if uploaded_file is not None:
    df = prepare_data(read_csv(uploaded_file))
    dataset_source = "Uploaded CSV file"
else:
    df, default_path = load_default_data()
    dataset_source = f"Default dataset: {default_path}"

st.sidebar.caption(f"Current dataset: {dataset_source}")
st.sidebar.caption("Budget values are interpreted as thousand pesos, then converted for display.")

st.sidebar.header("Filters")
filter_dropdown("Fiscal Year", df["Fiscal_Year"].unique(), key="year_filter")
filter_dropdown("Budget Type", df["Type"].unique(), key="type_filter")
filter_dropdown("GRIT TAGGING / Institution Type", df["GRIT TAGGING"].unique(), key="grit_filter")
filter_dropdown("Department", df["DEPARTMENT"].unique(), key="dept_filter")
filter_dropdown("Climate Pillar", df["Climate Pillar"].unique(), key="pillar_filter")
filter_dropdown("NCCAP Thematic Priority", df["NCCAP Thematic Priority"].unique(), key="nccap_filter")
filter_dropdown("PDP / Executive Agenda Alignment", df["PDP / Executive Agenda Alignment"].unique(), key="pdp_filter")

f = apply_filters(df)

filters_used = {
    "Dataset": dataset_source,
    "Fiscal Year": st.session_state["year_filter"],
    "Budget Type": st.session_state["type_filter"],
    "GRIT TAGGING / Institution Type": st.session_state["grit_filter"],
    "Department": st.session_state["dept_filter"],
    "Climate Pillar": st.session_state["pillar_filter"],
    "NCCAP Thematic Priority": st.session_state["nccap_filter"],
    "PDP / Executive Agenda Alignment": st.session_state["pdp_filter"],
}

st.sidebar.header("Downloads")
st.sidebar.download_button(
    "Download Filtered CSV",
    data=f.to_csv(index=False).encode("utf-8-sig"),
    file_name="filtered_ccet_pap_data.csv",
    mime="text/csv",
)

if REPORTLAB_AVAILABLE:
    st.sidebar.download_button(
        "Download PDF Summary",
        data=generate_pdf_report(f, filters_used),
        file_name="ccet_smart_policy_analytics_summary.pdf",
        mime="application/pdf",
    )
else:
    st.sidebar.caption("PDF export requires `reportlab`.")

# ============================================================
# KPI RIBBON
# ============================================================

total_budget = f["TOTAL"].sum()
adaptation_total = f["ADAPTATION"].sum()
mitigation_total = f["MITIGATION"].sum()
adaptation_share = pct(adaptation_total, total_budget)
mitigation_share = pct(mitigation_total, total_budget)
unique_institutions = f["Agency Display"].nunique()
unique_paps = f["PAP ID"].replace("", np.nan).nunique()
q_score = quality_score(f)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Climate Budget", peso_from_raw(total_budget))
k2.metric("Adaptation Share", percent_label(adaptation_share, 1))
k3.metric("Mitigation Share", percent_label(mitigation_share, 1))
k4.metric("Institutions", f"{unique_institutions:,}")
k5.metric("Unique PAP IDs", f"{unique_paps:,}")
k6.metric("Data Quality Score", f"{q_score:.1f}/100")

if f.empty:
    st.warning("No records match the selected filters. Please adjust the sidebar filters.")
    st.stop()

smart_note(
    "<b>Interpretation rule:</b> Climate-tagged budget shows budget visibility and tagging coverage. "
    "It should not be interpreted automatically as actual expenditure, implementation performance, or climate impact."
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
    tab_agency,
    tab_nccap,
    tab_fgd,
    tab_reco,
    tab_alignment,
    tab_pap,
    tab_quality,
    tab_methods,
) = tabs


# ============================================================
# TAB: GUIDE
# ============================================================

with tab_guide:
    st.subheader("Dashboard Guide")
    st.markdown("""
    ### Purpose

    This dashboard is a smart policy analytics platform for the National Climate Change Expenditure Tagging system.
    It supports examination of climate-tagged Programs, Activities, and Projects across fiscal years, budget stages,
    institutions, climate pillars, and NCCAP thematic priorities.

    ### What changed in this version

    - Added **GRIT TAGGING / Institution Type** as a dashboard-wide filter.
    - Replaced the **NDC Sector** filter with **NCCAP Thematic Priority**.
    - Removed repetitive chart placement and reorganized visuals into clearer analytical modules.
    - Made core findings more dynamic using the latest PAP-level dataset.
    - Added participation tracking, agency concentration, budget-cycle traceability, FGD/KII challenge mapping,
      recommendations tracker, and stronger data quality checks.

    ### Interpretation rule

    The dashboard supports analysis. It does not replace official CCC/DBM validation, agency QAR documentation,
    COA audit, or climate impact evaluation. Climate-tagged amounts are budget-tagging indicators, not automatic
    proof of climate outcomes.
    """)

    st.markdown("### Current Filters")
    st.dataframe(pd.DataFrame([filters_used]).T.rename(columns={0: "Selected Value"}), use_container_width=True)


# ============================================================
# TAB: DATA PROFILE
# ============================================================

with tab_profile:
    st.subheader("Data Profile and Dataset Schema")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Fiscal Year Coverage", f"{df['Fiscal_Year'].min()}–{df['Fiscal_Year'].max()}")
    c3.metric("Budget Types", f"{df['Type'].nunique():,}")
    c4.metric("Institution Types", f"{df['GRIT TAGGING'].nunique():,}")

    schema = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[col].dtype) for col in df.columns],
        "Missing / Blank Count": [
            int(df[col].isna().sum() + (df[col].astype(str).str.strip().eq("").sum() if df[col].dtype == "object" else 0))
            for col in df.columns
        ],
        "Description": [COLUMN_DICTIONARY.get(col, "Derived or source field from the uploaded dataset.") for col in df.columns],
    })
    st.dataframe(schema, use_container_width=True, height=430)

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(30), use_container_width=True, height=430)

    st.download_button(
        "Download Data Dictionary CSV",
        schema.to_csv(index=False).encode("utf-8-sig"),
        "ccet_data_dictionary.csv",
        "text/csv",
    )


# ============================================================
# TAB: EXECUTIVE OVERVIEW
# ============================================================

with tab_exec:
    st.subheader("Executive Overview")

    top_agency, top_agency_amount = safe_top_value(f, "Agency Display")
    top_priority, top_priority_amount = safe_top_value(f, "NCCAP Thematic Priority")
    top_dept, top_dept_amount = safe_top_value(f, "DEPARTMENT")

    st.markdown(f"""
    The current filtered dataset covers **{peso_from_raw(total_budget)}** in climate-tagged PAPs, involving
    **{unique_institutions:,} institutions** and **{unique_paps:,} unique PAP IDs**.

    Adaptation accounts for **{percent_label(adaptation_share, 2)}**, while mitigation accounts for
    **{percent_label(mitigation_share, 2)}** of the filtered total.

    The top institution is **{top_agency}** with **{peso_from_raw(top_agency_amount)}**.
    The top department is **{top_dept}** with **{peso_from_raw(top_dept_amount)}**.
    The largest NCCAP thematic priority is **{top_priority}** with **{peso_from_raw(top_priority_amount)}**.
    """)

    c1, c2 = st.columns(2)
    with c1:
        render_chart(build_budget_by_type_chart(f), "overview_budget_by_type", "Budget by Type", height=680)
    with c2:
        render_chart(build_pdp_alignment_chart(f), "overview_pdp_alignment", "PDP Alignment", height=560, width=900)


# ============================================================
# TAB: KEY FINDINGS
# ============================================================

with tab_key:
    st.subheader("Key Findings")

    smart_note(
        "This tab presents the core quantitative findings from the dashboard: budget visibility, "
        "budget-stage movement, adaptation orientation, and concentration. Use the specialized tabs "
        "for deeper drill-down."
    )

    render_chart(
        build_climate_budget_share_chart(f),
        "key_climate_budget_share",
        "Climate Budget Share",
        height=760,
        width=1450,
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    gaa_current = f[f["Type"].str.upper().eq("GAA")]["TOTAL"].sum()
    actual_current = f[f["Type"].str.upper().eq("Actual")]["TOTAL"].sum()
    nep_current = f[f["Type"].str.upper().eq("NEP")]["TOTAL"].sum()
    c1.metric("Filtered NEP", peso_from_raw(nep_current))
    c2.metric("Filtered GAA", peso_from_raw(gaa_current))
    c3.metric("Filtered Actual", peso_from_raw(actual_current))
    c4.metric("Top NCCAP Priority", top_priority if len(str(top_priority)) < 24 else str(top_priority)[:24] + "…")

    st.divider()
    render_chart(
        build_adaptation_mitigation_share_chart(f),
        "key_adaptation_mitigation_share",
        "Adaptation-Mitigation Share",
        height=700,
        width=1450,
    )


# ============================================================
# TAB: PARTICIPATION & COVERAGE
# ============================================================

with tab_participation:
    st.subheader("Participation and Coverage")

    smart_note(
        "This module responds to the report finding that CCET participation increased over time, "
        "but participation alone does not guarantee substantive compliance or results tracking."
    )

    render_chart(
        build_participation_chart(f),
        "participation_by_institution_type",
        "Participation by Institution Type",
        height=700,
        width=1450,
    )

    st.divider()

    render_chart(
        build_institution_type_budget_chart(f),
        "budget_by_institution_type",
        "Budget by Institution Type",
        height=680,
        width=1450,
    )

    st.markdown("### Participation Table")
    participation_table = (
        f.drop_duplicates(["Fiscal_Year", "GRIT TAGGING", "Agency Display"])
        .groupby(["Fiscal_Year", "GRIT TAGGING"], as_index=False)["Agency Display"]
        .nunique()
        .rename(columns={"Agency Display": "Participating Institutions"})
        .sort_values(["Fiscal_Year", "GRIT TAGGING"])
    )
    st.dataframe(participation_table, use_container_width=True, height=420)


# ============================================================
# TAB: BUDGET CYCLE
# ============================================================

with tab_cycle:
    st.subheader("Budget Cycle Analysis")

    smart_note(
        "This module follows the CCET budget-cycle logic: preparation, legislation, execution, and accountability. "
        "It highlights whether tagged amounts changed from NEP to GAA to Actual."
    )

    st.markdown("### Budget Cycle Stage Map")
    st.dataframe(BUDGET_CYCLE_STAGES, use_container_width=True, height=230)

    render_chart(
        build_budget_by_type_chart(f),
        "budget_cycle_nep_gaa_actual_trend",
        "NEP-GAA-Actual Trend",
        height=680,
        width=1450,
    )

    st.divider()

    render_chart(
        build_budget_cycle_variance_chart(f),
        "budget_cycle_variance",
        "Budget-Cycle Variance",
        height=720,
        width=1450,
    )

    st.divider()

    render_chart(
        build_actual_vs_gaa_chart(f),
        "actual_vs_gaa_percent",
        "Actual vs GAA",
        height=680,
        width=1450,
    )

    st.markdown("### Budget Stage Table")
    w = budget_stage_wide(f)
    display = w.copy()
    for col in ["NEP (Billion Pesos)", "GAA (Billion Pesos)", "Actual (Billion Pesos)", "GAA minus NEP (Billion Pesos)", "Actual minus GAA (Billion Pesos)"]:
        if col in display:
            display[col] = display[col].apply(peso_billion if "minus" not in col else signed_peso_billion)
    if "Actual vs GAA (%)" in display:
        display["Actual vs GAA (%)"] = display["Actual vs GAA (%)"].apply(lambda x: percent_label(x, 2))
    keep_cols = [
        "Fiscal Year", "NEP (Billion Pesos)", "GAA (Billion Pesos)", "Actual (Billion Pesos)",
        "GAA minus NEP (Billion Pesos)", "Actual minus GAA (Billion Pesos)", "Actual vs GAA (%)"
    ]
    st.dataframe(display[[c for c in keep_cols if c in display.columns]], use_container_width=True, height=400)


# ============================================================
# TAB: AGENCY CONCENTRATION
# ============================================================

with tab_agency:
    st.subheader("Agency and NGI Concentration")

    smart_note(
        "This module identifies which institutions drive the climate-tagged budget. It uses department-agency labels "
        "to avoid confusion among agencies with similar names such as 'Office of the Secretary'."
    )

    top_n = st.slider("Number of institutions to show", 5, 40, 15)

    render_chart(
        build_top_ngi_pareto(f, top_n=top_n),
        "agency_concentration_pareto",
        "NGI Pareto Concentration",
        height=760,
        width=1550,
    )

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        render_chart(
            build_ranking_chart(
                f[f["GRIT TAGGING"].eq("NGA")],
                "DEPARTMENT",
                "<b>Top NGA Departments</b>",
                "Filtered climate-tagged allocation by parent department.",
                top_n=12,
            ),
            "top_nga_departments",
            "Top NGA Departments",
            height=680,
            width=900,
        )
    with c2:
        render_chart(
            build_ranking_chart(
                f[f["GRIT TAGGING"].eq("GOCC")],
                "Agency Display",
                "<b>Top GOCC Agencies</b>",
                "Filtered climate-tagged allocation by GOCC.",
                top_n=12,
            ),
            "top_gocc_agencies",
            "Top GOCC Agencies",
            height=680,
            width=900,
        )

    st.divider()

    c3, c4 = st.columns(2)
    with c3:
        render_chart(
            build_ranking_chart(
                f[f["GRIT TAGGING"].eq("SUC")],
                "Agency Display",
                "<b>Top SUCs</b>",
                "Filtered climate-tagged allocation by SUC.",
                top_n=12,
            ),
            "top_sucs",
            "Top SUCs",
            height=680,
            width=900,
        )
    with c4:
        render_chart(
            build_ranking_chart(
                f,
                "Agency Display",
                "<b>Top Institutions Overall</b>",
                "Filtered climate-tagged allocation by institution.",
                top_n=12,
            ),
            "top_institutions_overall",
            "Top Institutions Overall",
            height=680,
            width=900,
        )


# ============================================================
# TAB: NCCAP PRIORITIES
# ============================================================

with tab_nccap:
    st.subheader("NCCAP Thematic Priorities")

    smart_note(
        "This module replaces the previous NDC-sector filter emphasis with the NCCAP thematic priority lens. "
        "NCCAP classification is derived from the CCET typology code."
    )

    render_chart(
        build_nccap_heatmap(f),
        "nccap_priority_matrix",
        "NCCAP Priority Matrix",
        height=850,
        width=1550,
    )

    st.divider()

    c1, c2 = st.columns([1, 1])
    with c1:
        render_chart(
            build_nccap_rank(f),
            "nccap_priority_ranking",
            "NCCAP Priority Ranking",
            height=680,
            width=900,
        )
    with c2:
        render_chart(
            build_nccap_stacked(f),
            "nccap_priority_mix",
            "NCCAP Priority Mix",
            height=820,
            width=900,
        )

    st.markdown("### NCCAP Priority Table")
    nccap_table = safe_group_sum(f, ["Fiscal_Year", "NCCAP Thematic Priority"], "TOTAL")
    nccap_table["Amount"] = nccap_table["TOTAL"].apply(peso_from_raw)
    total_by_year = nccap_table.groupby("Fiscal_Year")["TOTAL"].transform("sum")
    nccap_table["Share of Fiscal Year (%)"] = nccap_table["TOTAL"] / total_by_year * 100
    st.dataframe(
        nccap_table.sort_values(["Fiscal_Year", "TOTAL"], ascending=[True, False])[
            ["Fiscal_Year", "NCCAP Thematic Priority", "Amount", "Share of Fiscal Year (%)"]
        ],
        use_container_width=True,
        height=460,
    )


# ============================================================
# TAB: FGD/KII CHALLENGES
# ============================================================

with tab_fgd:
    st.subheader("FGD/KII Challenges and Dashboard Responses")

    smart_note(
        "This module turns FGD/KII implementation issues into structured dashboard requirements. "
        "It connects qualitative findings with budget-cycle stages, recommendations, and data science responses."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Coded Themes", FGD_KII_INSIGHTS["Theme"].nunique())
    c2.metric("High-Priority Issues", int((FGD_KII_INSIGHTS["Priority"] == "High").sum()))
    c3.metric("Budget Stages Covered", FGD_KII_INSIGHTS["Budget Cycle Stage"].nunique())

    render_chart(
        build_fgd_priority_chart(),
        "fgd_kii_priority_chart",
        "FGD/KII Challenge Priorities",
        height=620,
        width=1450,
    )

    st.markdown("### Challenge-to-Dashboard Matrix")
    st.dataframe(FGD_KII_INSIGHTS, use_container_width=True, height=520)

    st.download_button(
        "Download FGD/KII Challenge Matrix CSV",
        FGD_KII_INSIGHTS.to_csv(index=False).encode("utf-8-sig"),
        "fgd_kii_challenge_dashboard_matrix.csv",
        "text/csv",
    )


# ============================================================
# TAB: RECOMMENDATIONS
# ============================================================

with tab_reco:
    st.subheader("Recommendations Tracker")

    smart_note(
        "This module translates recommendations into dashboard modules, data science methods, and implementation monitoring areas."
    )

    render_chart(
        build_recommendation_chart(),
        "recommendation_priority_tracker",
        "Recommendation Priority Tracker",
        height=540,
        width=1100,
    )

    priority_filter = st.selectbox(
        "Filter recommendation table by priority",
        ["All"] + sorted(RECOMMENDATION_SCORECARD["Priority"].unique().tolist()),
    )
    reco = RECOMMENDATION_SCORECARD.copy()
    if priority_filter != "All":
        reco = reco[reco["Priority"] == priority_filter]
    st.dataframe(reco, use_container_width=True, height=430)

    st.markdown("### Proposed Smart Add-ons for the Next Build")
    st.write("""
    - Agency-level CCET maturity scorecard.
    - PAP-level NEP-GAA-Actual reconciliation upload template.
    - Attribution method field: whole PAP, component-based, proportional, or not specified.
    - QAR completeness tracker.
    - Climate indicator and accomplishment linkage fields.
    - Audit-readiness tracker connected to data-quality flags.
    - International CBT benchmarking page for Nepal, Bangladesh, Indonesia, and France.
    """)


# ============================================================
# TAB: POLICY ALIGNMENT
# ============================================================

with tab_alignment:
    st.subheader("PDP / Executive Agenda Alignment")

    smart_note(
        "This module is exploratory. Alignment is estimated through keyword matching in PAP descriptions, typology descriptions, agencies, and departments. "
        "It is not official validation."
    )

    c1, c2 = st.columns(2)
    with c1:
        render_chart(
            build_pdp_alignment_chart(f),
            "policy_alignment_share",
            "Policy Alignment Share",
            height=560,
            width=900,
        )
    with c2:
        render_chart(
            build_pdp_alignment_trend(f),
            "policy_alignment_trend",
            "Policy Alignment Trend",
            height=650,
            width=900,
        )

    st.markdown("### Keyword-Based Classification Rules")
    st.code(
        "Hits = count of matched climate/development keywords in PAP text\n"
        "If Hits ≥ 3 → Strongly Aligned\n"
        "If Hits ≥ 1 → Partially Aligned\n"
        "If Hits = 0 → Weak / Unclassified",
        language="text",
    )


# ============================================================
# TAB: PAP EXPLORER
# ============================================================

with tab_pap:
    st.subheader("PAP Explorer")

    search = st.text_input("Search PAP Description, Agency, Department, Typology, or NCCAP Priority")
    explorer = f.copy()
    if search:
        s = search.lower()
        explorer = explorer[
            explorer["PAP Description"].str.lower().str.contains(s, na=False) |
            explorer["AGENCY"].str.lower().str.contains(s, na=False) |
            explorer["DEPARTMENT"].str.lower().str.contains(s, na=False) |
            explorer["TYPOLOGY Description"].str.lower().str.contains(s, na=False) |
            explorer["NCCAP Thematic Priority"].str.lower().str.contains(s, na=False)
        ]

    show_cols = [
        "Fiscal_Year", "Type", "GRIT TAGGING", "DEPARTMENT", "AGENCY",
        "PAP ID", "PAP Description", "TYPOLOGY ID", "TYPOLOGY Description",
        "Climate Pillar", "NCCAP Thematic Priority", "PDP / Executive Agenda Alignment",
        "ADAPTATION", "MITIGATION", "TOTAL",
    ]
    explorer_display = explorer[show_cols].sort_values("TOTAL", ascending=False).copy()
    explorer_display["ADAPTATION"] = explorer_display["ADAPTATION"].apply(peso_from_raw)
    explorer_display["MITIGATION"] = explorer_display["MITIGATION"].apply(peso_from_raw)
    explorer_display["TOTAL"] = explorer_display["TOTAL"].apply(peso_from_raw)

    st.dataframe(explorer_display, use_container_width=True, height=620)

    st.download_button(
        "Download PAP Explorer CSV",
        explorer[show_cols].to_csv(index=False).encode("utf-8-sig"),
        "pap_explorer_filtered.csv",
        "text/csv",
    )


# ============================================================
# TAB: DATA QUALITY
# ============================================================

with tab_quality:
    st.subheader("Data Quality and Validation")

    smart_note(
        "This module checks whether the filtered dataset contains records that may need validation. "
        "It supports cautious interpretation before using dashboard outputs for policy conclusions."
    )

    c1, c2, c3 = st.columns(3)
    q_summary = data_quality_summary(f)
    total_flags = int(q_summary["Flagged Records"].sum())
    critical_flags = int(
        (
            data_quality_masks(f)["Missing typology ID"] |
            data_quality_masks(f)["Zero or blank total"] |
            data_quality_masks(f)["Adaptation + Mitigation ≠ Total"] |
            data_quality_masks(f)["Duplicate PAP-stage records"]
        ).sum()
    )
    c1.metric("Data Quality Score", f"{q_score:.1f}/100")
    c2.metric("Total Flag Events", f"{total_flags:,}")
    c3.metric("Critical Flagged Rows", f"{critical_flags:,}")

    render_chart(
        build_quality_chart(f),
        "data_quality_flags",
        "Data Quality Flags",
        height=720,
        width=1450,
    )

    st.markdown("### Quality Summary Table")
    q_table = q_summary.copy()
    q_table["Share of Filtered Records (%)"] = q_table["Share of Filtered Records (%)"].map(lambda x: f"{x:.2f}%")
    st.dataframe(q_table, use_container_width=True, height=380)

    issue = st.selectbox("Inspect flagged records", list(data_quality_masks(f).keys()))
    mask = data_quality_masks(f)[issue]
    st.dataframe(f.loc[mask], use_container_width=True, height=470)

    st.download_button(
        "Download Selected Data Quality Flags CSV",
        f.loc[mask].to_csv(index=False).encode("utf-8-sig"),
        f"data_quality_{issue.lower().replace(' ', '_').replace('/', '_')}.csv",
        "text/csv",
    )


# ============================================================
# TAB: METHODS & RATIONALE
# ============================================================

with tab_methods:
    st.subheader("Methods, Rationale, and Data Science Application")

    st.markdown("""
    ### Policy and research rationale

    The dashboard operationalizes the Climate Change Expenditure Tagging logic as a public financial management
    and policy analytics system. It supports the identification, classification, tracking, monitoring, and reporting
    of climate-related public expenditures.

    The dashboard is aligned with the National CCET assessment logic: CCET is treated as a system-level tool for
    climate budgeting, not as a direct measure of project-level climate impact. Therefore, all outputs should be read
    as evidence of budget tagging, expenditure visibility, and institutional process quality.

    ### Data science pipeline

    1. **Data ingestion** — load default or uploaded CSV.
    2. **Data cleaning** — standardize fields, convert numbers, clean text, and fix column names.
    3. **Feature engineering** — derive Climate Pillar, NCCAP Code, NCCAP Thematic Priority, and policy-alignment proxy.
    4. **Interactive filtering** — allow users to slice by fiscal year, budget type, GRIT TAGGING, department, pillar, NCCAP priority, and alignment.
    5. **Descriptive analytics** — compute totals, shares, counts, rankings, and participation indicators.
    6. **Diagnostic analytics** — compare NEP, GAA, Actual, variance, and utilization gaps.
    7. **Qualitative coding** — structure FGD/KII challenges into themes, recommendations, and dashboard responses.
    8. **Data validation** — flag missing, inconsistent, zero, duplicate, and unclassified records.
    9. **Data productization** — export charts, filtered data, and summary reports.

    ### Core formulas

    ```text
    Total Climate Budget = sum(TOTAL)

    Adaptation Share (%) = sum(ADAPTATION) / sum(TOTAL) × 100

    Mitigation Share (%) = sum(MITIGATION) / sum(TOTAL) × 100

    Climate Budget Share (%) = Climate-Tagged GAA / Total National Budget × 100

    GAA Variance = GAA - NEP

    Actual Variance = Actual - GAA

    Actual vs GAA (%) = (Actual - GAA) / GAA × 100

    Priority Share (%) = NCCAP Priority Allocation / Fiscal Year Total × 100

    YoY Growth (%) = (Current Year Total - Previous Year Total) / Previous Year Total × 100

    Data Quality Score = 100 - critical flagged row rate
    ```

    ### Important caveat

    Climate-tagged budget is not the same as climate impact. It is a budget-tagging and public finance tracking indicator.
    The dashboard must therefore be used with QAR documents, agency reports, accomplishment data, audit information,
    and policy interpretation.
    """)

    method_table = pd.DataFrame([
        ["Budget visibility", "Trend and ratio analysis", "Climate-tagged GAA vs national budget share"],
        ["Budget-cycle traceability", "Variance and utilization-gap analysis", "NEP-GAA-Actual charts"],
        ["Climate objective mix", "Composition analysis", "Adaptation vs mitigation share"],
        ["Institutional concentration", "Ranking and Pareto analysis", "Top NGI / agency concentration"],
        ["NCCAP alignment", "Typology-based classification", "NCCAP matrix and priority ranking"],
        ["Implementation challenges", "Qualitative coding", "FGD/KII challenge tracker"],
        ["Recommendation management", "Decision-support mapping", "Recommendations tracker"],
        ["Reliability checks", "Rule-based anomaly detection", "Data quality flags"],
    ], columns=["Analytical Need", "Data Science Method", "Dashboard Module"])

    st.dataframe(method_table, use_container_width=True, height=360)

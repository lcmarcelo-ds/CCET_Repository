import os
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
    page_title="National CCET Policy Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONSTANTS AND REFERENCE DATA
# ============================================================

DATA_PATH = "data/ccet_data.csv"

# The cleaned CCC PAP-level dataset used in this dashboard stores budget values in thousand pesos.
# This multiplier converts raw dataset values into actual pesos for dashboard display.
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
LIGHT_GRAY = "#E6ECF5"
DARK_TEXT = "#1F1F1F"

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

PDP_KEYWORDS = [
    "climate", "resilience", "disaster", "risk reduction", "adaptation",
    "mitigation", "flood", "drainage", "water", "irrigation",
    "food security", "agriculture", "renewable", "energy efficiency",
    "sustainable", "environment", "ecosystem", "biodiversity",
    "green", "carbon", "emission", "hazard", "watershed"
]

NDC_SECTORS = {
    "Energy": ["renewable", "solar", "wind", "hydro", "energy efficiency", "power", "electricity"],
    "Transport": ["transport", "railway", "road", "public transport", "mobility", "traffic"],
    "Agriculture": ["agriculture", "irrigation", "farm", "fisheries", "livestock", "food"],
    "Waste": ["waste", "solid waste", "sewerage", "sanitation"],
    "Industry": ["industry", "industrial", "manufacturing"],
    "Water / Flood Control": ["flood", "drainage", "river", "water", "dam", "irrigation"],
    "Ecosystems": ["forest", "biodiversity", "ecosystem", "mangrove", "watershed"],
}

COLUMN_DICTIONARY = {
    "Fiscal_Year": "Fiscal year covered by the climate-tagged PAP record.",
    "Type": "Budget classification such as NEP, GAA, or Actual.",
    "DEPARTMENT": "Parent department or sector of the implementing agency.",
    "AGENCY": "Implementing or reporting national government agency/instrumentality.",
    "PAP ID": "Program, Activity, or Project identifier.",
    "PAP Description": "Name or description of the climate-tagged PAP.",
    "TYPOLOGY ID": "CCET typology code used to classify the PAP.",
    "TYPOLOGY Description": "Description of the assigned CCET typology.",
    "ADAPTATION": "Amount tagged for climate change adaptation.",
    "MITIGATION": "Amount tagged for climate change mitigation.",
    "TOTAL": "Total climate-tagged amount.",
    "Climate Pillar": "Derived field based on Typology ID: Adaptation, Mitigation, or Unclassified.",
    "NCCAP Code": "Derived NCCAP priority code from the Typology ID.",
    "NCCAP Priority": "Derived NCCAP thematic priority.",
    "PDP / Executive Agenda Alignment": "Keyword-based analytical proxy for possible alignment with national climate and development priorities.",
    "NDC Sector Alignment": "Keyword-based analytical proxy for estimated sectoral climate alignment.",
}

# Report-grade key findings tables used for fixed dashboard modules.
# Values are expressed in billion pesos unless otherwise noted.
TABLE_5_6 = pd.DataFrame({
    "Fiscal Year": ["FY2022", "FY2023", "FY2024", "FY2025"],
    "Climate-Tagged GAA (Billion Pesos)": [289.73, 464.50, 457.41, 1156.00],
    "Total National Budget (Billion Pesos)": [5023.00, 5268.00, 5768.00, 6326.00],
    "Share of National Budget (%)": [5.8, 8.8, 7.9, 18.3],
})

NEP_GAA_ACTUAL = pd.DataFrame({
    "Fiscal Year": ["FY2022", "FY2023", "FY2024", "FY2025"],
    "NEP": [np.nan, 453.11, 543.45, 1020.00],
    "GAA": [289.73, 464.50, 457.41, 1156.00],
    "Actual": [444.86, 568.94, 581.73, np.nan],
})
NEP_GAA_ACTUAL["GAA minus NEP"] = NEP_GAA_ACTUAL["GAA"] - NEP_GAA_ACTUAL["NEP"]
NEP_GAA_ACTUAL["Actual minus GAA"] = NEP_GAA_ACTUAL["Actual"] - NEP_GAA_ACTUAL["GAA"]
NEP_GAA_ACTUAL["Actual vs GAA (%)"] = (
    (NEP_GAA_ACTUAL["Actual"] - NEP_GAA_ACTUAL["GAA"]) / NEP_GAA_ACTUAL["GAA"] * 100
)

ADAPT_MITIG_SHARE = pd.DataFrame({
    "Fiscal Year": ["FY2022", "FY2023", "FY2024", "FY2025"],
    "Adaptation Share": [90.92, 88.53, 96.61, 97.18],
    "Mitigation Share": [9.08, 11.47, 3.39, 2.82],
})

NGI_TOP10 = pd.DataFrame({
    "Rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "NGI / Agency": [
        "DPWH - Office of the Secretary",
        "DA - Office of the Secretary",
        "DOTr - Office of the Secretary",
        "DENR - Office of the Secretary",
        "MMDA",
        "Philippine Crop Insurance Corporation",
        "National Food Authority",
        "DSWD - Office of the Secretary",
        "National Irrigation Administration",
        "Environmental Management Bureau",
    ],
    "Institution Type": [
        "NGA", "NGA", "NGA", "NGA", "NGA / ALGU allocation",
        "GOCC", "GOCC", "NGA", "GOCC", "NGA"
    ],
    "Cumulative GAA FY2022-FY2025 (Billion Pesos)": [
        1983.00, 120.03, 94.12, 35.63, 25.14,
        18.00, 18.00, 6.16, 4.46, 3.84
    ],
})
NGI_TOP10["Agency Label"] = NGI_TOP10["Rank"].astype(str) + ". " + NGI_TOP10["NGI / Agency"]

NCCAP_ALLOCATIONS = pd.DataFrame({
    "NCCAP Priority": [
        "Food Security",
        "Water Sufficiency",
        "Ecosystem and Environmental Stability",
        "Human Security",
        "Climate Smart Industries and Services",
        "Sustainable Energy",
        "Knowledge and Capacity Development",
        "Cross-cutting",
    ],
    "FY2022": [28.77, 217.47, 7.29, 1.74, 6.47, 24.80, 2.27, 0.92],
    "FY2023": [36.24, 357.30, 5.65, 3.76, 5.43, 54.44, 0.95, 0.73],
    "FY2024": [36.59, 373.45, 6.48, 2.83, 6.13, 18.19, 13.14, 0.60],
    "FY2025": [56.88, 366.27, 6.32, 10.96, 338.37, 364.79, 11.84, 0.67],
})
NCCAP_YEARS = ["FY2022", "FY2023", "FY2024", "FY2025"]

FGD_KII_INSIGHTS = pd.DataFrame([
    {
        "Theme": "Personnel familiarity and continuity",
        "Institution Type": "NGA / GOCC / SUC",
        "Challenge": "Uneven familiarity with CCET processes; some focal persons are newly assigned or have limited exposure.",
        "Dashboard Response": "Add a training-needs and familiarity tracker by institution type.",
        "Recommendation": "Institutionalize regular CCET orientation, onboarding, and refresher training for focal persons.",
        "Priority": "High",
    },
    {
        "Theme": "Institutional ownership",
        "Institution Type": "NGA",
        "Challenge": "Implementation can become compliance-driven rather than strategically owned across planning, budgeting, and technical units.",
        "Dashboard Response": "Add challenge-to-budget-cycle mapping and agency responsibility matrix.",
        "Recommendation": "Designate clear internal CCET roles through office orders or memoranda.",
        "Priority": "High",
    },
    {
        "Theme": "Attribution and tagging methodology",
        "Institution Type": "NGA / GOCC / SUC",
        "Challenge": "Agencies differ in whether they tag whole PAPs or only climate-relevant components; HGDG is repeatedly cited as a possible model.",
        "Dashboard Response": "Add attribution-method notes and flag PAPs with large blanket-tagged allocations.",
        "Recommendation": "Develop more operational guidance on proportional attribution, thresholds, and supporting evidence.",
        "Priority": "High",
    },
    {
        "Theme": "PAP-level reconciliation",
        "Institution Type": "NGA / GOCC / SUC",
        "Challenge": "Difficulty tracing a PAP from NEP to GAA to Actual to accomplishment and audit findings.",
        "Dashboard Response": "Add NEP-GAA-Actual variance module and PAP-level reconciliation table.",
        "Recommendation": "Create a standard PAP-level reconciliation template across the full budget cycle.",
        "Priority": "High",
    },
    {
        "Theme": "Monitoring and evaluation",
        "Institution Type": "NGA / GOCC / SUC",
        "Challenge": "Monitoring remains decentralized; many agencies rely on general accomplishment reports rather than CCET-specific M&E.",
        "Dashboard Response": "Add M&E readiness indicators and documentation completeness fields.",
        "Recommendation": "Develop a CCET-specific monitoring and reporting module tied to QAR, accomplishment reports, and audit data.",
        "Priority": "High",
    },
    {
        "Theme": "Budget deliberation use",
        "Institution Type": "GOCC / NGA",
        "Challenge": "CCET reports are perceived to be used more at the agency level than in DBM or congressional budget deliberations.",
        "Dashboard Response": "Add budget-stage use indicators and show NEP-to-GAA changes.",
        "Recommendation": "Strengthen the use of CCET analytics in budget review, deliberation, and policy briefs.",
        "Priority": "Medium",
    },
    {
        "Theme": "GOCC applicability",
        "Institution Type": "GOCC",
        "Challenge": "Some GOCCs have limited familiarity where funding does not come from GAA or where attribution is seen as less applicable.",
        "Dashboard Response": "Add institution-type filter and GOCC-specific interpretation notes.",
        "Recommendation": "Clarify CCET applicability rules for GOCCs, especially those with mixed funding sources.",
        "Priority": "Medium",
    },
    {
        "Theme": "Audit and accountability",
        "Institution Type": "NGA / GOCC / SUC",
        "Challenge": "Accountability linkages remain weak when audit findings and accomplishment data are not connected to tagged PAPs.",
        "Dashboard Response": "Add accountability-stage placeholders and future COA/audit-link fields.",
        "Recommendation": "Link CCET data with accomplishment reporting and audit review, where available.",
        "Priority": "Medium",
    },
])

BUDGET_CYCLE_STAGES = pd.DataFrame([
    {"Stage": "Preparation", "Focus": "Agency tagging, QAR forms, BP/DBM forms", "Dashboard Module": "PAP Explorer, NCCAP Matrix, FGD/KII Insights"},
    {"Stage": "Legislation", "Focus": "NEP to GAA movement and deliberation changes", "Dashboard Module": "Budget Cycle Analysis, Variance Charts"},
    {"Stage": "Execution", "Focus": "Actual expenditure / utilization where available", "Dashboard Module": "NEP-GAA-Actual, Actual vs GAA"},
    {"Stage": "Accountability", "Focus": "Accomplishment reports, audit findings, data validation", "Dashboard Module": "Data Quality, Recommendations Tracker"},
])

# ============================================================
# DATA PREPARATION
# ============================================================


def classify_alignment(text):
    text = str(text).lower()
    hits = sum(1 for kw in PDP_KEYWORDS if kw in text)
    if hits >= 3:
        return "Strongly Aligned"
    if hits >= 1:
        return "Partially Aligned"
    return "Weak / Unclassified"


def classify_ndc_sector(text):
    text = str(text).lower()
    for sector, keywords in NDC_SECTORS.items():
        if any(kw in text for kw in keywords):
            return sector
    return "Unclassified"


def read_csv(source):
    return pd.read_csv(source, encoding="utf-8-sig")


def prepare_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"ADAPTION": "ADAPTATION"})

    for col in ["Fiscal_Year", "ADAPTATION", "MITIGATION", "TOTAL"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df[df["Fiscal_Year"].notna()].copy()
    df["Fiscal_Year"] = df["Fiscal_Year"].astype(int)

    text_cols = [
        "Type", "DEPARTMENT", "AGENCY",
        "PAP ID", "PAP Description", "TYPOLOGY ID", "TYPOLOGY Description"
    ]
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = (
            df[col].astype(str)
            .replace({"nan": "", "None": ""})
            .str.strip()
        )

    typo = df["TYPOLOGY ID"].str.upper()
    df["Climate Pillar"] = np.where(
        typo.str.startswith("M"),
        "Mitigation",
        np.where(typo.str.startswith("A"), "Adaptation", "Unclassified")
    )
    df["NCCAP Code"] = typo.str.extract(r"^[AM](\d)", expand=False).fillna("")
    df["NCCAP Priority"] = df["NCCAP Code"].map(NCCAP_PRIORITY).fillna("Unclassified")

    combined_text = (
        df["PAP Description"].astype(str) + " " +
        df["TYPOLOGY Description"].astype(str) + " " +
        df["AGENCY"].astype(str) + " " +
        df["DEPARTMENT"].astype(str)
    )
    df["PDP / Executive Agenda Alignment"] = combined_text.apply(classify_alignment)
    df["NDC Sector Alignment"] = combined_text.apply(classify_ndc_sector)
    return df


@st.cache_data(show_spinner="Loading default CCET dataset...")
def load_default_data():
    if not os.path.exists(DATA_PATH):
        st.error("Default CSV dataset not found. Please upload it as `data/ccet_data.csv`.")
        st.stop()
    if os.path.getsize(DATA_PATH) == 0:
        st.error("`data/ccet_data.csv` is empty. Please re-upload the real CSV file.")
        st.stop()
    return prepare_data(read_csv(DATA_PATH))

# ============================================================
# FORMATTERS AND CHART EXPORT HELPERS
# ============================================================


def as_actual_pesos(raw_dataset_value):
    return float(raw_dataset_value or 0) * DATASET_VALUE_MULTIPLIER


def peso(value):
    """Format raw dataset amount, assuming raw values are in thousand pesos."""
    value = as_actual_pesos(value)
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
    """Format amount already expressed in billion pesos."""
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1000:
        return f"₱{value / 1000:.3f}T"
    return f"₱{value:,.2f}B"


def signed_peso_billion(value):
    if pd.isna(value):
        return "N/A"
    sign = "+" if value > 0 else ""
    if abs(value) >= 1000:
        return f"{sign}₱{value / 1000:.3f}T"
    return f"{sign}₱{value:,.2f}B"


def percent_label(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"


def filter_dropdown(label, values):
    opts = sorted([v for v in values if str(v).strip() != ""])
    return st.sidebar.selectbox(label, ["All"] + opts)


def safe_top_value(dataframe, group_col, value_col="TOTAL"):
    if dataframe.empty or group_col not in dataframe.columns:
        return "No data", 0
    temp = (
        dataframe.groupby(group_col, as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
    )
    if temp.empty:
        return "No data", 0
    return temp.iloc[0][group_col], temp.iloc[0][value_col]


def apply_report_layout(fig, height=720, title=None, source_note=None, legend_y=1.02, bottom_margin=100):
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor="center",
                font=dict(size=21, color=PRIMARY_BLUE),
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=75, r=70, t=135, b=bottom_margin),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=legend_y,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#D9E2F3",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    if source_note:
        fig.add_annotation(
            text=source_note,
            xref="paper",
            yref="paper",
            x=0,
            y=-0.18,
            showarrow=False,
            align="left",
            font=dict(size=11, color="#555555"),
        )
    return fig


def render_chart(fig, title, file_stem, height=720, width=1450):
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
    html_bytes = fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label=f"Download {title} HTML",
            data=html_bytes,
            file_name=f"{file_stem}.html",
            mime="text/html",
            key=f"{file_stem}_html",
        )
    with c2:
        try:
            png_bytes = fig.to_image(format="png", width=width, height=height, scale=3)
            st.download_button(
                label=f"Download {title} PNG",
                data=png_bytes,
                file_name=f"{file_stem}.png",
                mime="image/png",
                key=f"{file_stem}_png",
            )
        except Exception:
            st.caption("PNG export requires `kaleido`. Add `kaleido` to requirements.txt.")

# ============================================================
# REPORT-GRADE FIGURE BUILDERS
# ============================================================


def build_climate_budget_share_chart():
    d = TABLE_5_6.copy()
    d["Climate-Tagged GAA (Trillion Pesos)"] = d["Climate-Tagged GAA (Billion Pesos)"] / 1000
    d["Total National Budget (Trillion Pesos)"] = d["Total National Budget (Billion Pesos)"] / 1000

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
            line=dict(color=ORANGE, width=3),
            marker=dict(size=10, color=ORANGE),
            text=[percent_label(v, 1) for v in d["Share of National Budget (%)"]],
            textposition="top center",
            hovertemplate="<b>%{x}</b><br>Share: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    apply_report_layout(
        fig,
        height=760,
        title="<b>Climate-Tagged GAA, Total National Budget, and Share of National Budget</b><br><span style='font-size:15px;'>FY2022-FY2025</span>",
        source_note="The tabulation is based on key findings table. Amounts shown in trillion pesos; share shown as percent of national budget.",
        bottom_margin=125,
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Amount (Trillion Pesos)", range=[0, 7.2], gridcolor=LIGHT_GRAY, secondary_y=False)
    fig.update_yaxes(title_text="Share of National Budget (%)", range=[0, 22], ticksuffix="%", showgrid=False, secondary_y=True)
    fig.update_xaxes(title_text="Fiscal Year")
    return fig


def build_nep_gaa_actual_chart():
    d = NEP_GAA_ACTUAL.copy()
    fig = go.Figure()
    colors = {"NEP": DEEP_BLUE, "GAA": MID_BLUE, "Actual": GREEN}
    for stage in ["NEP", "GAA", "Actual"]:
        fig.add_trace(
            go.Bar(
                x=d["Fiscal Year"],
                y=d[stage],
                name=stage,
                marker=dict(color=colors[stage], line=dict(color="white", width=1)),
                text=[peso_billion(v) for v in d[stage]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=f"<b>%{{x}}</b><br>{stage}: %{{text}}<extra></extra>",
            )
        )
    max_y = np.nanmax(d[["NEP", "GAA", "Actual"]].values)
    fig.add_annotation(x="FY2022", y=max_y * 0.08, text="<i>NEP not available</i>", showarrow=False, font=dict(size=11, color="#555555"))
    fig.add_annotation(x="FY2025", y=max_y * 0.08, text="<i>Actual not available</i>", showarrow=False, font=dict(size=11, color="#555555"))
    apply_report_layout(
        fig,
        height=720,
        title="<b>NEP, GAA, and Actual CCET Budgets</b><br><span style='font-size:15px;'>FY2022-FY2025</span>",
        source_note="The tabulation is based on key findings table. Values are expressed in billion pesos.",
        bottom_margin=125,
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Billion Pesos", range=[0, max_y * 1.35], gridcolor=LIGHT_GRAY)
    fig.update_xaxes(title_text="Fiscal Year")
    return fig


def build_budget_cycle_variance_chart():
    d = NEP_GAA_ACTUAL.copy()
    fig = go.Figure()
    variance_cols = ["GAA minus NEP", "Actual minus GAA"]
    colors = {"GAA minus NEP": ORANGE, "Actual minus GAA": GRAY}
    for col in variance_cols:
        fig.add_trace(
            go.Bar(
                x=d["Fiscal Year"],
                y=d[col],
                name=col,
                marker=dict(color=colors[col], line=dict(color="white", width=1)),
                text=[signed_peso_billion(v) for v in d[col]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{text}}<extra></extra>",
            )
        )
    variance_max = np.nanmax(np.abs(d[variance_cols].values))
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#808080")
    apply_report_layout(
        fig,
        height=720,
        title="<b>Budget-Cycle Variance in CCET</b><br><span style='font-size:15px;'>NEP to GAA and GAA to Actual changes</span>",
        source_note="The computation is based on NEP, GAA, and Actual CCET figures. Positive values indicate increases; negative values indicate reductions.",
        bottom_margin=125,
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Billion Pesos", range=[-variance_max * 1.45, variance_max * 1.45], gridcolor=LIGHT_GRAY)
    fig.update_xaxes(title_text="Fiscal Year")
    return fig


def build_actual_vs_gaa_percent_chart():
    d = NEP_GAA_ACTUAL.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=d["Fiscal Year"],
            y=d["Actual vs GAA (%)"],
            name="Actual vs GAA (%)",
            marker=dict(color=DEEP_BLUE, line=dict(color="white", width=1)),
            text=[percent_label(v, 2) for v in d["Actual vs GAA (%)"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Actual vs GAA: %{text}<extra></extra>",
        )
    )
    max_pct = np.nanmax(np.abs(d["Actual vs GAA (%)"].values))
    fig.add_annotation(x="FY2025", y=max_pct * 0.08, text="<i>No Actual data</i>", showarrow=False, font=dict(size=11, color="#555555"))
    apply_report_layout(
        fig,
        height=700,
        title="<b>Actual Compared with GAA</b><br><span style='font-size:15px;'>Percentage difference between Actual and approved GAA CCET budgets</span>",
        source_note="The computation is based on Actual and GAA CCET figures. FY2025 is excluded because Actual data are not available.",
        bottom_margin=125,
    )
    fig.update_yaxes(title_text="Percent Difference", range=[0, max_pct * 1.35], ticksuffix="%", gridcolor=LIGHT_GRAY)
    fig.update_xaxes(title_text="Fiscal Year")
    return fig


def build_adaptation_mitigation_share_chart():
    d = ADAPT_MITIG_SHARE.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=d["Fiscal Year"],
            y=d["Adaptation Share"],
            name="Adaptation",
            marker=dict(color=DEEP_BLUE, line=dict(color="white", width=1)),
            text=[percent_label(v, 2) for v in d["Adaptation Share"]],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="<b>%{x}</b><br>Adaptation: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=d["Fiscal Year"],
            y=d["Mitigation Share"],
            name="Mitigation",
            marker=dict(color=GREEN, line=dict(color="white", width=1)),
            text=[percent_label(v, 2) for v in d["Mitigation Share"]],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="<b>%{x}</b><br>Mitigation: %{y:.2f}%<extra></extra>",
        )
    )
    apply_report_layout(
        fig,
        height=720,
        title="<b>Adaptation and Mitigation Share</b><br><span style='font-size:15px;'>Distribution of climate-tagged budget by climate pillar, FY2022-FY2025</span>",
        source_note="The tabulation is based on adaptation and mitigation shares. Values are percentage shares of climate-tagged budget.",
        bottom_margin=125,
    )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Share of Climate-Tagged Budget (%)", range=[0, 100], ticksuffix="%", gridcolor=LIGHT_GRAY)
    fig.update_xaxes(title_text="Fiscal Year")
    return fig


def build_ngi_top10_chart(exclude_dpwh=False):
    d = NGI_TOP10.sort_values("Cumulative GAA FY2022-FY2025 (Billion Pesos)", ascending=False).copy()
    if exclude_dpwh:
        d = d[d["NGI / Agency"] != "DPWH - Office of the Secretary"].copy()
    color_map = {"NGA": DEEP_BLUE, "GOCC": GREEN, "NGA / ALGU allocation": ORANGE}
    d["Value Label"] = d["Cumulative GAA FY2022-FY2025 (Billion Pesos)"].apply(peso_billion)
    fig = px.bar(
        d,
        x="Cumulative GAA FY2022-FY2025 (Billion Pesos)",
        y="Agency Label",
        color="Institution Type",
        orientation="h",
        text="Value Label",
        color_discrete_map=color_map,
        hover_data={
            "Rank": True,
            "NGI / Agency": True,
            "Institution Type": True,
            "Cumulative GAA FY2022-FY2025 (Billion Pesos)": ":,.2f",
            "Agency Label": False,
            "Value Label": False,
        },
    )
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_color="white", marker_line_width=1.2)
    title = "<b>NGIs with Highest Climate-Tagged Allocations</b>"
    if exclude_dpwh:
        title = "<b>NGIs with Highest Climate-Tagged Allocations, Excluding DPWH</b>"
    apply_report_layout(
        fig,
        height=760 if not exclude_dpwh else 700,
        title=title,
        source_note="The tabulation is based on cumulative GAA FY2022-FY2025. Values are expressed in billion pesos.",
        bottom_margin=125,
    )
    fig.update_layout(margin=dict(l=340, r=120, t=140, b=125))
    fig.update_yaxes(title_text="", categoryorder="array", categoryarray=d["Agency Label"].tolist(), autorange="reversed")
    fig.update_xaxes(title_text="Cumulative GAA FY2022-FY2025 (Billion Pesos)", gridcolor=LIGHT_GRAY)
    return fig


def build_ngi_institution_type_donut():
    d = NGI_TOP10.groupby("Institution Type", as_index=False)["Cumulative GAA FY2022-FY2025 (Billion Pesos)"].sum()
    color_map = {"NGA": DEEP_BLUE, "GOCC": GREEN, "NGA / ALGU allocation": ORANGE}
    fig = px.pie(
        d,
        values="Cumulative GAA FY2022-FY2025 (Billion Pesos)",
        names="Institution Type",
        hole=0.55,
        color="Institution Type",
        color_discrete_map=color_map,
    )
    fig.update_traces(textinfo="percent+label", hovertemplate="<b>%{label}</b><br>Amount: ₱%{value:,.2f}B<br>Share: %{percent}<extra></extra>")
    apply_report_layout(
        fig,
        height=610,
        title="<b>Share of Top 10 NGI Allocations by Institution Type</b>",
        source_note="The tabulation is based on cumulative GAA FY2022-FY2025.",
        bottom_margin=105,
    )
    fig.add_annotation(text="Top 10<br>NGIs", x=0.5, y=0.5, showarrow=False, font=dict(size=18, color=PRIMARY_BLUE))
    return fig


def build_nccap_matrix_chart():
    df_alloc = NCCAP_ALLOCATIONS.copy()
    df_percent = df_alloc.copy()
    for year in NCCAP_YEARS:
        df_percent[year + " Share (%)"] = df_percent[year] / df_percent[year].sum() * 100

    heatmap_values = df_alloc[NCCAP_YEARS].values
    customdata = []
    for i, priority in enumerate(df_alloc["NCCAP Priority"]):
        row_data = []
        for year in NCCAP_YEARS:
            amount = df_alloc.loc[i, year]
            share = df_percent.loc[i, year + " Share (%)"]
            row_data.append([amount, share])
        customdata.append(row_data)
    customdata = np.array(customdata)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=heatmap_values,
            x=NCCAP_YEARS,
            y=df_alloc["NCCAP Priority"],
            customdata=customdata,
            colorscale=[
                [0.00, "#F7FBFF"],
                [0.20, "#CFE1F2"],
                [0.40, "#8FBBD9"],
                [0.60, "#4F8FC2"],
                [0.80, "#1F5F99"],
                [1.00, PRIMARY_BLUE],
            ],
            colorbar=dict(title="Billion pesos", len=0.85),
            xgap=3,
            ygap=3,
            hovertemplate=(
                "<b>%{y}</b><br>Fiscal Year: %{x}<br>"
                "Allocation: ₱%{customdata[0]:,.2f}B<br>"
                "Share: %{customdata[1]:.2f}%<extra></extra>"
            ),
        )
    )
    for i, priority in enumerate(df_alloc["NCCAP Priority"]):
        for year in NCCAP_YEARS:
            amount = df_alloc.loc[i, year]
            share = df_percent.loc[i, year + " Share (%)"]
            font_color = "white" if amount >= 150 else DARK_TEXT
            fig.add_annotation(
                x=year,
                y=priority,
                text=f"{peso_billion(amount)}<br>({share:.2f}%)",
                showarrow=False,
                align="center",
                font=dict(size=11, color=font_color),
            )
    fig.update_layout(
        title=dict(
            text="<b>NCCAP Thematic Priority Allocations and Shares</b><br><span style='font-size:15px;'>Matrix of climate-tagged GAA allocations and percentage share by priority, FY2022-FY2025</span>",
            x=0.5,
            xanchor="center",
            font=dict(size=21, color=PRIMARY_BLUE),
        ),
        template="plotly_white",
        height=850,
        margin=dict(l=390, r=110, t=155, b=135),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(title_text="Fiscal Year", side="top", tickfont=dict(size=13))
    fig.update_yaxes(title_text="NCCAP Thematic Priority", autorange="reversed", tickfont=dict(size=12))
    fig.add_annotation(
        text="The tabulation is based on NCCAP thematic priority allocations, FY2022-FY2025. Amounts are in billion pesos; percentages indicate each priority's share of fiscal-year total.",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.15,
        showarrow=False,
        align="left",
        font=dict(size=11, color="#555555"),
    )
    return fig


def build_nccap_stacked_chart():
    df_alloc = NCCAP_ALLOCATIONS.copy()
    df_percent = df_alloc.copy()
    for year in NCCAP_YEARS:
        df_percent[year + " Share (%)"] = df_percent[year] / df_percent[year].sum() * 100

    color_map = {
        "Food Security": DEEP_BLUE,
        "Water Sufficiency": MID_BLUE,
        "Ecosystem and Environmental Stability": GREEN,
        "Human Security": YELLOW,
        "Climate Smart Industries and Services": ORANGE,
        "Sustainable Energy": TEAL,
        "Knowledge and Capacity Development": PURPLE,
        "Cross-cutting": GRAY,
    }
    fig = go.Figure()
    for priority in df_alloc["NCCAP Priority"]:
        row = df_alloc[df_alloc["NCCAP Priority"] == priority].iloc[0]
        values = [row[year] for year in NCCAP_YEARS]
        shares = [df_percent.loc[df_percent["NCCAP Priority"] == priority, year + " Share (%)"].iloc[0] for year in NCCAP_YEARS]
        fig.add_trace(
            go.Bar(
                x=NCCAP_YEARS,
                y=values,
                name=priority,
                customdata=np.array(shares),
                marker=dict(color=color_map[priority], line=dict(color="white", width=1)),
                text=[f"{peso_billion(v)}<br>{s:.1f}%" if v >= 20 else "" for v, s in zip(values, shares)],
                textposition="inside",
                hovertemplate=f"<b>{priority}</b><br>Fiscal Year: %{{x}}<br>Allocation: ₱%{{y:,.2f}}B<br>Share: %{{customdata:.2f}}%<extra></extra>",
            )
        )
    apply_report_layout(
        fig,
        height=830,
        title="<b>NCCAP Thematic Priority Allocation Mix</b><br><span style='font-size:15px;'>Stacked view of climate-tagged GAA allocations by priority, FY2022-FY2025</span>",
        legend_y=-0.24,
        bottom_margin=295,
    )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Allocation (Billion Pesos)", gridcolor=LIGHT_GRAY)
    fig.update_xaxes(title_text="Fiscal Year")
    return fig


def build_challenge_priority_chart():
    order = ["High", "Medium", "Low"]
    d = FGD_KII_INSIGHTS.groupby(["Priority", "Institution Type"], as_index=False).size()
    fig = px.bar(
        d,
        x="Priority",
        y="size",
        color="Institution Type",
        category_orders={"Priority": order},
        title="FGD/KII Challenge and Recommendation Priorities",
        labels={"size": "Number of dashboard-relevant issues"},
    )
    apply_report_layout(
        fig,
        height=580,
        title="<b>FGD/KII Challenge and Recommendation Priorities</b>",
        source_note="Source: Synthesized FGD/KII implementation themes from the updated CCET assessment materials.",
        bottom_margin=110,
    )
    fig.update_yaxes(title_text="Number of Issues", dtick=1, gridcolor=LIGHT_GRAY)
    fig.update_xaxes(title_text="Priority")
    return fig


def build_budget_cycle_map_chart():
    d = pd.DataFrame({
        "Budget Stage": ["Preparation", "Legislation", "Execution", "Accountability"],
        "Dashboard Emphasis Score": [4, 3, 3, 4],
        "Main Dashboard Module": ["NCCAP Matrix / PAP Explorer", "NEP-GAA Variance", "Actual vs GAA", "Data Quality / Recommendations"],
    })
    fig = px.bar(
        d,
        x="Budget Stage",
        y="Dashboard Emphasis Score",
        text="Main Dashboard Module",
        color="Budget Stage",
        color_discrete_sequence=[DEEP_BLUE, MID_BLUE, ORANGE, GREEN],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    apply_report_layout(
        fig,
        height=560,
        title="<b> Coverage Across the Public Budget Cycle</b>",
        source_note="Source: The visualization is based on the assessment's public budget cycle analytical framework.",
        bottom_margin=110,
    )
    fig.update_yaxes(title_text="Dashboard Emphasis", range=[0, 5], showticklabels=False, gridcolor=LIGHT_GRAY)
    fig.update_xaxes(title_text="Budget Cycle Stage")
    return fig

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
    agencies = f["AGENCY"].nunique()
    paps = f["PAP ID"].nunique()

    adaptation_share = (adaptation_total / total_budget * 100) if total_budget else 0
    mitigation_share = (mitigation_total / total_budget * 100) if total_budget else 0
    top_agency, top_agency_amount = safe_top_value(f, "AGENCY")
    top_priority, top_priority_amount = safe_top_value(f, "NCCAP Priority")
    top_sector, top_sector_amount = safe_top_value(f, "NDC Sector Alignment")

    story.append(Paragraph("National CCET Policy Analytics Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y %I:%M %p')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Filters Applied", styles["Heading2"]))
    filter_table = [["Filter", "Selected Value"]] + [[k, str(v)] for k, v in filters_used.items()]
    table = Table(filter_table, colWidths=[180, 300])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    summary = f"""
    The filtered dataset covers {peso(total_budget)} in climate-tagged PAPs,
    involving {agencies} agencies and {paps} PAP records. Adaptation accounts
    for {adaptation_share:.2f}% of the filtered raw dataset total, while mitigation accounts for
    {mitigation_share:.2f}%. The top spending agency is {top_agency}
    with {peso(top_agency_amount)}. The largest NCCAP priority is {top_priority}
    with {peso(top_priority_amount)}. The largest estimated NDC sector is
    {top_sector} with {peso(top_sector_amount)}.
    """
    story.append(Paragraph(summary, styles["Normal"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Interpretation Note", styles["Heading2"]))
    disclaimer = """
    This report is generated from the dataset currently loaded in the dashboard.
    NCCAP classifications are derived from CCET typology codes. PDP / Executive
    Agenda Alignment and NDC Sector Alignment are keyword-based analytical proxies.
    The dashboard is intended to support analysis and does not replace official validation.
    """
    story.append(Paragraph(disclaimer, styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ============================================================
# MAIN APP
# ============================================================

st.title("National CCET Policy Analytics Platform")
st.caption("Climate Change Expenditure Tagging PAP-level analytics, key findings, and FGD/KII implementation insights")

st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload new CSV dataset",
    type=["csv"],
    help="Optional. If no file is uploaded, the dashboard uses the default `data/ccet_data.csv` dataset.",
)

if uploaded_file is not None:
    raw_df = read_csv(uploaded_file)
    df = prepare_data(raw_df)
    dataset_source = "Uploaded CSV file"
else:
    df = load_default_data()
    dataset_source = "Default GitHub dataset"

st.sidebar.caption(f"Current dataset: {dataset_source}")
st.sidebar.caption("Budget values in the default cleaned dataset are treated as thousand pesos for display.")

st.sidebar.header("Filters")
year = filter_dropdown("Fiscal Year", df["Fiscal_Year"].unique())
budget_type = filter_dropdown("Budget Type", df["Type"].unique())
department = filter_dropdown("Department", df["DEPARTMENT"].unique())
pillar = filter_dropdown("Climate Pillar", df["Climate Pillar"].unique())
pdp_alignment = filter_dropdown("PDP / Executive Agenda Alignment", df["PDP / Executive Agenda Alignment"].unique())
ndc_sector = filter_dropdown("NDC Sector", df["NDC Sector Alignment"].unique())

f = df.copy()
if year != "All":
    f = f[f["Fiscal_Year"] == int(year)]
if budget_type != "All":
    f = f[f["Type"] == budget_type]
if department != "All":
    f = f[f["DEPARTMENT"] == department]
if pillar != "All":
    f = f[f["Climate Pillar"] == pillar]
if pdp_alignment != "All":
    f = f[f["PDP / Executive Agenda Alignment"] == pdp_alignment]
if ndc_sector != "All":
    f = f[f["NDC Sector Alignment"] == ndc_sector]

filters_used = {
    "Dataset": dataset_source,
    "Fiscal Year": year,
    "Budget Type": budget_type,
    "Department": department,
    "Climate Pillar": pillar,
    "PDP / Executive Agenda Alignment": pdp_alignment,
    "NDC Sector": ndc_sector,
}

if REPORTLAB_AVAILABLE:
    pdf_buffer = generate_pdf_report(f, filters_used)
    st.sidebar.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="ccet_policy_analytics_report.pdf",
        mime="application/pdf",
    )
else:
    st.sidebar.warning("PDF export requires `reportlab`.")

st.sidebar.download_button(
    label="Download Filtered CSV",
    data=f.to_csv(index=False).encode("utf-8-sig"),
    file_name="filtered_ccet_data.csv",
    mime="text/csv",
)

# KPI CARDS
k1, k2, k3, k4, k5 = st.columns(5)
total_budget = f["TOTAL"].sum()
adaptation_total = f["ADAPTATION"].sum()
mitigation_total = f["MITIGATION"].sum()

k1.metric("Total Climate Budget", peso(total_budget))
k2.metric("Adaptation", peso(adaptation_total))
k3.metric("Mitigation", peso(mitigation_total))
k4.metric("Agencies", f["AGENCY"].nunique())
k5.metric("PAP Records", f["PAP ID"].nunique())

# TABS
tabs = st.tabs([
    "User Manual",
    "Dataset Schema",
    "Executive Brief",
    "Key Findings",
    "Budget Cycle Analysis",
    "NGI Highest Allocations",
    "NCCAP Priority Matrix",
    "FGD/KII Insights",
    "Recommendations Tracker",
    "Budget Trends",
    "Agency Ranking",
    "NCCAP Alignment",
    "PDP / Executive Agenda",
    "NDC Sector Alignment",
    "Policy Insights",
    "PAP Explorer",
    "Data Quality",
])

(
    tab_user,
    tab_schema,
    tab_exec,
    tab_key,
    tab_cycle,
    tab_ngi,
    tab_matrix,
    tab_fgd,
    tab_reco,
    tab_trends,
    tab_agency,
    tab_nccap,
    tab_pdp,
    tab_ndc,
    tab_policy,
    tab_pap,
    tab_quality,
) = tabs

with tab_user:
    st.subheader("User Manual and Guide")
    st.markdown("""
    ## Purpose of the Dashboard

    This dashboard is an interactive policy analytics platform for examining
    Climate Change Expenditure Tagging data. It helps to understand how
    climate-tagged Programs, Activities, and Projects are distributed across
    fiscal years, departments, agencies, climate pillars, NCCAP priorities,
    and estimated policy alignment areas.

    The platform is intended to support evidence-based discussion, planning,
    budgeting, monitoring, and policy review.
 ## 2. Intended Users

    This dashboard may be used by not limited to the following:

    - Climate Change Commission
    - Department of Budget and Management
    - NEDA / DEPDev
    - National Government Agencies
    - Policy analysts
    - Researchers
    - Academic institutions
    - Development partners


    ## 3. Dataset Options

    The dashboard automatically loads the default dataset from the GitHub
    repository. Users may also upload a new CSV file using the sidebar.

    If a CSV file is uploaded, all charts, tables, and indicators will update
    based on the uploaded dataset.

    ## 4. Sidebar Filters

    The filters control the entire dashboard.

    **Fiscal Year** filters records by year.

    **Budget Type** filters by available budget classification, such as GAA,
    NEP, or other dataset values.

    **Department** filters the dashboard to a selected government department.

    **Climate Pillar** filters records into Adaptation, Mitigation, or
    Unclassified.

    **PDP / Executive Agenda Alignment** filters records based on keyword-based
    estimated alignment.

    **NDC Sector** filters records based on estimated sector classification.

    ## 5. KPI Cards

    The five cards at the top summarize:

    - Total Climate Budget
    - Adaptation Budget
    - Mitigation Budget
    - Number of Agencies
    - Number of PAP Records

    These indicators change depending on selected filters.

    ## 6. Dashboard Sections

    **Dataset Schema** explains the structure of the dataset and the meaning
    of each column.

    **Executive Brief** provides a high-level summary for policy audiences.

    **Budget Trends** shows changes in climate-tagged budgets across fiscal
    years.

    **Agency Ranking** identifies the agencies with the largest climate-tagged
    budgets.

    **NCCAP Alignment** analyzes allocations according to NCCAP priorities and
    climate pillars.

    **PDP / Executive Agenda** provides a keyword-based estimate of alignment
    with national climate and development priorities.

    **NDC Sector Alignment** estimates sectoral climate alignment.

    **Policy Insights** summarizes major policy observations and suggested
    questions.

    **PAP Explorer** allows users to search and inspect individual records.

    **Data Quality** checks for missing values, duplicate records, and budget
    inconsistencies.

    ## 7. How to Use the Dashboard

    1. Review this User Manual.
    2. Open Dataset Schema to understand the data structure.
    3. Use the sidebar filters.
    4. Review the KPI cards.
    5. Explore the charts and tables.
    6. Validate findings using PAP Explorer.
    7. Check Data Quality before making policy conclusions.
    8. Download the filtered CSV or PDF report if needed.

    ## 8. Important Interpretation Rule

    The dashboard supports analysis, but it does not replace official
    validation by government agencies.

    NCCAP classification is derived from CCET typology codes.

    PDP / Executive Agenda Alignment and NDC Sector Alignment are analytical
    proxies based on keyword matching. They should not be treated as official
    government classifications.


    ## 9. Disclaimer

    The dashboard generates analytics based on the uploaded dataset. Users
    should validate findings using official CCET submissions, QAR forms,
    budget documents, agency reports, and official government publications.
    """)

with tab_schema:
    st.subheader("Dataset Schema and Data Dictionary")
    st.markdown(f"""
    **Current dataset source:** {dataset_source}  
    **Number of records:** {len(df):,}  
    **Number of columns:** {len(df.columns):,}  
    **Fiscal year coverage:** {int(df['Fiscal_Year'].min())} to {int(df['Fiscal_Year'].max())}
    """)
    schema_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[col].dtype) for col in df.columns],
        "Missing Values": [df[col].isna().sum() for col in df.columns],
        "Description": [COLUMN_DICTIONARY.get(col, "Column from uploaded dataset.") for col in df.columns],
    })
    st.dataframe(schema_df, use_container_width=True, height=500)
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

with tab_exec:
    st.subheader("Executive Brief")
    adaptation_share = (f["ADAPTATION"].sum() / total_budget * 100) if total_budget else 0
    mitigation_share = (f["MITIGATION"].sum() / total_budget * 100) if total_budget else 0
    top_agency, top_agency_amount = safe_top_value(f, "AGENCY")
    top_priority, top_priority_amount = safe_top_value(f, "NCCAP Priority")
    top_sector, top_sector_amount = safe_top_value(f, "NDC Sector Alignment")

    st.markdown(f"""
    ### Filtered Dataset Summary

    The filtered dataset covers **{peso(total_budget)}** in climate-tagged PAPs across **{f['AGENCY'].nunique()} agencies** and **{f['PAP ID'].nunique()} PAP records**.

    Adaptation accounts for **{adaptation_share:.2f}%**, while mitigation accounts for **{mitigation_share:.2f}%** of the filtered dataset total.

    The top spending agency is **{top_agency}** with **{peso(top_agency_amount)}**.

    The largest NCCAP priority is **{top_priority}** with **{peso(top_priority_amount)}**.

    The largest estimated NDC sector is **{top_sector}** with **{peso(top_sector_amount)}**.
    """)
    c1, c2 = st.columns(2)
    with c1:
        by_year = f.groupby("Fiscal_Year", as_index=False)["TOTAL"].sum().sort_values("Fiscal_Year")
        by_year["Display Amount"] = by_year["TOTAL"].apply(as_actual_pesos)
        fig = px.line(by_year, x="Fiscal_Year", y="Display Amount", markers=True, title="Total Climate Budget by Fiscal Year")
        fig.update_yaxes(title_text="Amount (Pesos)", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        alignment_summary = f.groupby("PDP / Executive Agenda Alignment", as_index=False)["TOTAL"].sum()
        alignment_summary["Display Amount"] = alignment_summary["TOTAL"].apply(as_actual_pesos)
        fig = px.pie(alignment_summary, names="PDP / Executive Agenda Alignment", values="Display Amount", title="National Plan Alignment Share")
        st.plotly_chart(fig, use_container_width=True)

with tab_key:
    st.subheader("Key Findings Dashboard")
    ("This tab mirrors the report's key findings visuals for FY2022-FY2025. These figures are report-grade reference tables, separate from the sidebar-filtered PAP explorer dataset.")
    render_chart(build_climate_budget_share_chart(), "Figure 5.6", "Figure_5_6_Climate_Tagged_Budget_Share", height=760, width=1450)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("FY2025 Climate-Tagged GAA", "₱1.156T")
        st.metric("FY2025 Share of National Budget", "18.3%")
    with c2:
        st.metric("FY2025 Adaptation Share", "97.18%")
        st.metric("FY2025 Mitigation Share", "2.82%")
    st.divider()
    render_chart(build_nep_gaa_actual_chart(), "NEP-GAA-Actual", "Figure_NEP_GAA_Actual_CCET", height=720, width=1450)
    st.divider()
    render_chart(build_budget_cycle_variance_chart(), "Budget-Cycle Variance", "Figure_Budget_Cycle_Variance_CCET", height=720, width=1450)
    st.divider()
    render_chart(build_adaptation_mitigation_share_chart(), "Adaptation-Mitigation", "Figure_Adaptation_Mitigation_Share", height=720, width=1450)

with tab_cycle:
    st.subheader("Budget Cycle Analysis")
    st.markdown("This section aligns dashboard outputs with the CCET assessment framework: preparation, legislation, execution, and accountability.")
    render_chart(build_budget_cycle_map_chart(), "Budget Cycle Map", "Figure_Budget_Cycle_Map", height=560, width=1400)
    st.markdown("### Budget Cycle Stages and Dashboard Modules")
    st.dataframe(BUDGET_CYCLE_STAGES, use_container_width=True)
    st.markdown("### NEP-GAA-Actual Reference Table")
    display_table = NEP_GAA_ACTUAL.copy()
    for col in ["NEP", "GAA", "Actual", "GAA minus NEP", "Actual minus GAA"]:
        display_table[col] = display_table[col].apply(peso_billion)
    display_table["Actual vs GAA (%)"] = NEP_GAA_ACTUAL["Actual vs GAA (%)"].apply(lambda x: percent_label(x, 2))
    st.dataframe(display_table, use_container_width=True)

with tab_ngi:
    st.subheader("NGIs with Highest Climate-Tagged Allocations")
    st.info("Use the main chart for the report. Use the zoomed chart when DPWH visually compresses the remaining agencies.")
    render_chart(build_ngi_top10_chart(exclude_dpwh=False), "NGI Top 10", "Figure_NGI_Top10_Climate_Tagged_Allocations", height=760, width=1450)
    st.divider()
    render_chart(build_ngi_top10_chart(exclude_dpwh=True), "NGI Top 10 Excluding DPWH", "Figure_NGI_Top10_Excluding_DPWH", height=700, width=1450)
    st.divider()
    render_chart(build_ngi_institution_type_donut(), "Institution Type Share", "Figure_NGI_Top10_Institution_Type_Share", height=610, width=950)
    st.markdown("### Reference Table")
    table = NGI_TOP10.copy()
    table["Cumulative GAA FY2022-FY2025"] = table["Cumulative GAA FY2022-FY2025 (Billion Pesos)"].apply(peso_billion)
    st.dataframe(table[["Rank", "NGI / Agency", "Institution Type", "Cumulative GAA FY2022-FY2025"]], use_container_width=True)

with tab_matrix:
    st.subheader("NCCAP Priority Matrix")
    st.info("Each matrix cell shows both the absolute allocation and the percentage share of that fiscal year's total NCCAP allocation.")
    render_chart(build_nccap_matrix_chart(), "NCCAP Matrix", "Figure_NCCAP_Allocations_and_Shares_Matrix", height=850, width=1550)
    st.divider()
    render_chart(build_nccap_stacked_chart(), "NCCAP Stacked Bar", "Figure_NCCAP_Allocation_Mix", height=830, width=1550)
    st.markdown("### NCCAP Allocation Reference Table")
    st.dataframe(NCCAP_ALLOCATIONS, use_container_width=True)

with tab_fgd:
    st.subheader("FGD/KII Insights")
    st.markdown("This tab translates qualitative FGD/KII themes into dashboard monitoring and reform design features.")
    render_chart(build_challenge_priority_chart(), "FGD/KII Priorities", "Figure_FGD_KII_Challenge_Priorities", height=580, width=1400)
    st.markdown("### FGD/KII Challenge-to-Dashboard Matrix")
    st.dataframe(FGD_KII_INSIGHTS, use_container_width=True, height=520)
    st.download_button(
        "Download FGD/KII Insights CSV",
        FGD_KII_INSIGHTS.to_csv(index=False).encode("utf-8-sig"),
        "fgd_kii_insights_dashboard_matrix.csv",
        "text/csv",
    )

with tab_reco:
    st.subheader("Recommendations Tracker")
    st.markdown("This section operationalizes the recommendations as dashboard features and monitoring actions.")
    priority_filter = st.selectbox("Filter by priority", ["All"] + sorted(FGD_KII_INSIGHTS["Priority"].unique().tolist()))
    reco_df = FGD_KII_INSIGHTS.copy()
    if priority_filter != "All":
        reco_df = reco_df[reco_df["Priority"] == priority_filter]
    st.dataframe(reco_df[["Theme", "Institution Type", "Challenge", "Recommendation", "Dashboard Response", "Priority"]], use_container_width=True, height=520)
    st.markdown("### Recommended Dashboard Add-ons for Next Build")
    st.write("""
    - Add agency-level CCET maturity scorecard.
    - Add PAP-level NEP-GAA-Actual reconciliation upload template.
    - Add CCET attribution method field: whole PAP, proportional, component-based, or not specified.
    - Add QAR completeness tracker.
    - Add M&E linkage fields: accomplishment report available, actual expenditure available, audit finding available.
    - Add FGD/KII coded findings dataset once final qualitative coding is completed.
    """)

with tab_trends:
    st.subheader("Budget Trends")
    by_year = f.groupby("Fiscal_Year", as_index=False)["TOTAL"].sum().sort_values("Fiscal_Year")
    by_year["Display Amount"] = by_year["TOTAL"].apply(as_actual_pesos)
    by_year["YoY Growth %"] = by_year["Display Amount"].pct_change() * 100
    c1, c2 = st.columns(2)
    with c1:
        fig = px.area(by_year, x="Fiscal_Year", y="Display Amount", title="Climate Budget Trend")
        fig.update_yaxes(title_text="Amount (Pesos)", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(by_year, x="Fiscal_Year", y="YoY Growth %", title="Year-on-Year Growth Rate")
        fig.update_yaxes(title_text="YoY Growth (%)", ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)
    pillar_year = f.groupby("Fiscal_Year", as_index=False)[["ADAPTATION", "MITIGATION"]].sum()
    pillar_year["ADAPTATION"] = pillar_year["ADAPTATION"].apply(as_actual_pesos)
    pillar_year["MITIGATION"] = pillar_year["MITIGATION"].apply(as_actual_pesos)
    pillar_year = pillar_year.melt(id_vars="Fiscal_Year", value_vars=["ADAPTATION", "MITIGATION"], var_name="Pillar", value_name="Amount")
    fig = px.bar(pillar_year, x="Fiscal_Year", y="Amount", color="Pillar", barmode="group", title="Adaptation vs Mitigation by Fiscal Year")
    fig.update_yaxes(title_text="Amount (Pesos)", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(by_year, use_container_width=True)

with tab_agency:
    st.subheader("Agency Ranking")
    top_n = st.slider("Number of agencies to show", 5, 50, 20)
    agency_sum = f.groupby(["DEPARTMENT", "AGENCY"], as_index=False)["TOTAL"].sum().sort_values("TOTAL", ascending=False).head(top_n)
    agency_sum["Display Amount"] = agency_sum["TOTAL"].apply(as_actual_pesos)
    fig = px.bar(agency_sum.sort_values("Display Amount"), x="Display Amount", y="AGENCY", orientation="h", color="DEPARTMENT", title="Top Agencies by Climate Budget")
    fig.update_xaxes(title_text="Amount (Pesos)", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(agency_sum, use_container_width=True)

with tab_nccap:
    st.subheader("NCCAP Alignment")
    c1, c2 = st.columns(2)
    priority = f.groupby("NCCAP Priority", as_index=False)["TOTAL"].sum().sort_values("TOTAL", ascending=False)
    priority["Display Amount"] = priority["TOTAL"].apply(as_actual_pesos)
    with c1:
        fig = px.bar(priority, x="Display Amount", y="NCCAP Priority", orientation="h", title="Budget by NCCAP Priority")
        fig.update_xaxes(title_text="Amount (Pesos)", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pillar_df = f.groupby("Climate Pillar", as_index=False)["TOTAL"].sum()
        pillar_df["Display Amount"] = pillar_df["TOTAL"].apply(as_actual_pesos)
        fig = px.pie(pillar_df, values="Display Amount", names="Climate Pillar", title="Budget Share by Climate Pillar")
        st.plotly_chart(fig, use_container_width=True)
    typology = f.groupby(["TYPOLOGY ID", "TYPOLOGY Description", "NCCAP Priority", "Climate Pillar"], as_index=False)["TOTAL"].sum().sort_values("TOTAL", ascending=False)
    st.dataframe(typology, use_container_width=True, height=500)

with tab_pdp:
    st.subheader("PDP / Executive Agenda Alignment")
    st.info("This module uses keyword-based classification. It is an analytical proxy, not official validation.")
    alignment = f.groupby("PDP / Executive Agenda Alignment", as_index=False)["TOTAL"].sum().sort_values("TOTAL", ascending=False)
    alignment["Display Amount"] = alignment["TOTAL"].apply(as_actual_pesos)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(alignment, names="PDP / Executive Agenda Alignment", values="Display Amount", title="Budget Share by National Plan Alignment")
        st.plotly_chart(fig, use_container_width=True)
    alignment_year = f.groupby(["Fiscal_Year", "PDP / Executive Agenda Alignment"], as_index=False)["TOTAL"].sum()
    alignment_year["Display Amount"] = alignment_year["TOTAL"].apply(as_actual_pesos)
    with c2:
        fig = px.bar(alignment_year, x="Fiscal_Year", y="Display Amount", color="PDP / Executive Agenda Alignment", barmode="stack", title="National Plan Alignment Trend")
        fig.update_yaxes(title_text="Amount (Pesos)", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

with tab_ndc:
    st.subheader("NDC Sector Alignment")
    st.info("This module estimates sector alignment using keyword matching. It is not an official NDC classification.")
    sector = f.groupby("NDC Sector Alignment", as_index=False)["TOTAL"].sum().sort_values("TOTAL", ascending=False)
    sector["Display Amount"] = sector["TOTAL"].apply(as_actual_pesos)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(sector, x="Display Amount", y="NDC Sector Alignment", orientation="h", title="Climate Budget by NDC Sector")
        fig.update_xaxes(title_text="Amount (Pesos)", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(sector, names="NDC Sector Alignment", values="Display Amount", title="NDC Sector Share")
        st.plotly_chart(fig, use_container_width=True)

with tab_policy:
    st.subheader("Policy Insights")
    adaptation_share = (f["ADAPTATION"].sum() / total_budget * 100) if total_budget else 0
    mitigation_share = (f["MITIGATION"].sum() / total_budget * 100) if total_budget else 0
    top_agency, top_agency_amount = safe_top_value(f, "AGENCY")
    top_priority, top_priority_amount = safe_top_value(f, "NCCAP Priority")
    top_sector, top_sector_amount = safe_top_value(f, "NDC Sector Alignment")
    st.write(f"""
    **1. Total climate-tagged budget:** {peso(total_budget)}

    **2. Adaptation share:** {adaptation_share:.2f}%

    **3. Mitigation share:** {mitigation_share:.2f}%

    **4. Top spending agency:** {top_agency} — {peso(top_agency_amount)}

    **5. Largest NCCAP priority:** {top_priority} — {peso(top_priority_amount)}

    **6. Largest estimated NDC sector:** {top_sector} — {peso(top_sector_amount)}
    """)
    st.markdown("### Suggested Policy Questions")
    st.write("""
    - Are climate-tagged PAPs concentrated in a few agencies?
    - Is the portfolio too adaptation-heavy?
    - Which NCCAP priorities remain underfunded?
    - Which agencies have large budgets but weak policy alignment?
    - Which PAPs changed materially from NEP to GAA to Actual?
    - Which FGD/KII implementation challenges explain data or tagging limitations?
    """)

with tab_pap:
    st.subheader("PAP Explorer")
    search = st.text_input("Search PAP Description, Agency, Department, or Typology")
    explorer = f.copy()
    if search:
        s = search.lower()
        explorer = explorer[
            explorer["PAP Description"].str.lower().str.contains(s, na=False) |
            explorer["AGENCY"].str.lower().str.contains(s, na=False) |
            explorer["DEPARTMENT"].str.lower().str.contains(s, na=False) |
            explorer["TYPOLOGY Description"].str.lower().str.contains(s, na=False)
        ]
    cols = [
        "Fiscal_Year", "Type", "DEPARTMENT", "AGENCY",
        "PAP ID", "PAP Description", "TYPOLOGY ID", "TYPOLOGY Description",
        "NCCAP Priority", "Climate Pillar",
        "PDP / Executive Agenda Alignment", "NDC Sector Alignment",
        "ADAPTATION", "MITIGATION", "TOTAL",
    ]
    st.dataframe(explorer[cols].sort_values("TOTAL", ascending=False), use_container_width=True, height=600)
    st.download_button(
        "Download PAP Explorer CSV",
        explorer[cols].to_csv(index=False).encode("utf-8-sig"),
        "pap_explorer_filtered.csv",
        "text/csv",
    )

with tab_quality:
    st.subheader("Data Quality Checks")
    checks = {
        "Missing agency": f["AGENCY"].eq("").sum(),
        "Missing typology ID": f["TYPOLOGY ID"].eq("").sum(),
        "Zero or blank total": (f["TOTAL"].fillna(0) == 0).sum(),
        "Adaptation + Mitigation ≠ Total": ((f["ADAPTATION"].fillna(0) + f["MITIGATION"].fillna(0) - f["TOTAL"].fillna(0)).abs() > 1).sum(),
        "Duplicate PAP ID records": f.duplicated(subset=["Fiscal_Year", "Type", "AGENCY", "PAP ID", "TYPOLOGY ID"], keep=False).sum(),
    }
    qc = pd.DataFrame([{"Check": k, "Flagged Records": v} for k, v in checks.items()])
    st.dataframe(qc, use_container_width=True)
    issue_filter = st.selectbox("Show records for", list(checks.keys()))
    mask = pd.Series(False, index=f.index)
    if issue_filter == "Missing agency":
        mask = f["AGENCY"].eq("")
    elif issue_filter == "Missing typology ID":
        mask = f["TYPOLOGY ID"].eq("")
    elif issue_filter == "Zero or blank total":
        mask = f["TOTAL"].fillna(0) == 0
    elif issue_filter == "Adaptation + Mitigation ≠ Total":
        mask = ((f["ADAPTATION"].fillna(0) + f["MITIGATION"].fillna(0) - f["TOTAL"].fillna(0)).abs() > 1)
    elif issue_filter == "Duplicate PAP ID records":
        mask = f.duplicated(subset=["Fiscal_Year", "Type", "AGENCY", "PAP ID", "TYPOLOGY ID"], keep=False)
    st.dataframe(f.loc[mask], use_container_width=True, height=400)

import os
from io import BytesIO
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


st.set_page_config(
    page_title="National CCET Policy Analytics Platform",
    layout="wide"
)

DATA_PATH = "data/ccet_data.csv"

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
    "Fiscal_Year": "Fiscal year of the climate-tagged PAP.",
    "Type": "Budget classification, such as GAA, NEP, or Actual, depending on the dataset.",
    "DEPARTMENT": "Parent department or government sector of the implementing agency.",
    "AGENCY": "Government agency implementing or reporting the PAP.",
    "PAP ID": "Program, Activity, or Project identifier.",
    "PAP Description": "Description or title of the climate-tagged PAP.",
    "TYPOLOGY ID": "CCET typology code used to classify the PAP.",
    "TYPOLOGY Description": "Description of the CCET typology used.",
    "ADAPTATION": "Amount tagged for climate change adaptation.",
    "MITIGATION": "Amount tagged for climate change mitigation.",
    "TOTAL": "Total climate-tagged amount.",
    "Climate Pillar": "Derived classification: Adaptation, Mitigation, or Unclassified.",
    "NCCAP Code": "Derived NCCAP priority code from the typology ID.",
    "NCCAP Priority": "Derived NCCAP thematic priority.",
    "PDP / Executive Agenda Alignment": "Keyword-based proxy classification for national plan alignment.",
    "NDC Sector Alignment": "Keyword-based proxy classification for estimated climate sector alignment.",
}


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
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"ADAPTION": "ADAPTATION"})

    for col in ["Fiscal_Year", "ADAPTATION", "MITIGATION", "TOTAL"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Fiscal_Year"] = df["Fiscal_Year"].astype(int)

    text_cols = [
        "Type", "DEPARTMENT", "AGENCY",
        "PAP ID", "PAP Description", "TYPOLOGY ID", "TYPOLOGY Description"
    ]

    for col in text_cols:
        if col not in df.columns:
            df[col] = ""

        df[col] = (
            df[col]
            .astype(str)
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

    df = read_csv(DATA_PATH)
    return prepare_data(df)


def peso(value):
    value = float(value or 0)

    if abs(value) >= 1_000_000_000:
        return f"₱{value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"₱{value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"₱{value / 1_000:,.2f}K"

    return f"₱{value:,.0f}"


def filter_dropdown(label, values):
    opts = sorted([v for v in values if str(v).strip() != ""])
    return st.sidebar.selectbox(label, ["All"] + opts)


def safe_top_value(dataframe, group_col, value_col="TOTAL"):
    if dataframe.empty:
        return "No data", 0

    temp = (
        dataframe.groupby(group_col, as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
    )

    if temp.empty:
        return "No data", 0

    return temp.iloc[0][group_col], temp.iloc[0][value_col]


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
    for {adaptation_share:.2f}% of the total, while mitigation accounts for
    {mitigation_share:.2f}%. The top spending agency is {top_agency}
    with {peso(top_agency_amount)}. The largest NCCAP priority is {top_priority}
    with {peso(top_priority_amount)}. The largest estimated NDC sector is
    {top_sector} with {peso(top_sector_amount)}.
    """
    story.append(Paragraph(summary, styles["Normal"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Key Indicators", styles["Heading2"]))
    kpi_table = [
        ["Indicator", "Value"],
        ["Total Climate Budget", peso(total_budget)],
        ["Adaptation", peso(adaptation_total)],
        ["Mitigation", peso(mitigation_total)],
        ["Agencies", str(agencies)],
        ["PAP Records", str(paps)],
    ]
    table = Table(kpi_table, colWidths=[220, 260])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Top Agencies", styles["Heading2"]))
    top_agencies = (
        f.groupby("AGENCY", as_index=False)["TOTAL"]
        .sum()
        .sort_values("TOTAL", ascending=False)
        .head(10)
    )
    agency_table = [["Agency", "Total"]] + [[r["AGENCY"], peso(r["TOTAL"])] for _, r in top_agencies.iterrows()]
    table = Table(agency_table, colWidths=[350, 130])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Disclaimer", styles["Heading2"]))
    disclaimer = """
    This report is generated from the dataset currently loaded in the dashboard.
    NCCAP classifications are derived from CCET typology codes. PDP / Executive
    Agenda Alignment and NDC Sector Alignment are keyword-based analytical proxies
    and should not be treated as official validation by CCC, DBM, or NEDA/DEPDev.
    Findings must be validated using official documents and agency submissions.
    """
    story.append(Paragraph(disclaimer, styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


st.title("National CCET Policy Analytics Platform")
st.caption(
    "Climate Change Expenditure Tagging PAP-level analytics | "
    "FY2017–FY2026 | For DBM, CCC, NEDA/DEPDev, and Executive Policy Review"
)

st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload new CSV dataset",
    type=["csv"],
    help="Optional. If no file is uploaded, the dashboard will use the default GitHub dataset."
)

if uploaded_file is not None:
    raw_df = read_csv(uploaded_file)
    df = prepare_data(raw_df)
    dataset_source = "Uploaded CSV file"
else:
    df = load_default_data()
    dataset_source = "Default GitHub dataset"

st.sidebar.caption(f"Current dataset: {dataset_source}")

st.sidebar.header("Filters")

year = filter_dropdown("Fiscal Year", df["Fiscal_Year"].unique())
budget_type = filter_dropdown("Budget Type", df["Type"].unique())
department = filter_dropdown("Department", df["DEPARTMENT"].unique())
pillar = filter_dropdown("Climate Pillar", df["Climate Pillar"].unique())
pdp_alignment = filter_dropdown(
    "PDP / Executive Agenda Alignment",
    df["PDP / Executive Agenda Alignment"].unique()
)
ndc_sector = filter_dropdown("NDC Sector", df["NDC Sector Alignment"].unique())

f = df.copy()

if year != "All":
    f = f[f["Fiscal_Year"] == year]

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

pdf_buffer = generate_pdf_report(f, filters_used)

st.sidebar.download_button(
    label="Download PDF Report",
    data=pdf_buffer,
    file_name="ccet_policy_analytics_report.pdf",
    mime="application/pdf"
)

k1, k2, k3, k4, k5 = st.columns(5)

total_budget = f["TOTAL"].sum()
adaptation_total = f["ADAPTATION"].sum()
mitigation_total = f["MITIGATION"].sum()

k1.metric("Total Climate Budget", peso(total_budget))
k2.metric("Adaptation", peso(adaptation_total))
k3.metric("Mitigation", peso(mitigation_total))
k4.metric("Agencies", f["AGENCY"].nunique())
k5.metric("PAP Records", f["PAP ID"].nunique())

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "Executive Brief",
    "Budget Trends",
    "Agency Ranking",
    "NCCAP Alignment",
    "PDP / Executive Agenda",
    "NDC Sector Alignment",
    "Policy Insights",
    "PAP Explorer",
    "Data Quality",
    "Dataset Schema",
    "User Manual"
])

with tab1:
    st.subheader("Executive Brief")

    total_budget = f["TOTAL"].sum()
    adaptation_share = (f["ADAPTATION"].sum() / total_budget * 100) if total_budget else 0
    mitigation_share = (f["MITIGATION"].sum() / total_budget * 100) if total_budget else 0

    top_agency, top_agency_amount = safe_top_value(f, "AGENCY")
    top_priority, top_priority_amount = safe_top_value(f, "NCCAP Priority")
    top_sector, top_sector_amount = safe_top_value(f, "NDC Sector Alignment")

    st.markdown(
        f"""
        ### Key Findings

        The dashboard covers **{peso(total_budget)}** in climate-tagged PAPs across
        **{f["AGENCY"].nunique()} agencies** and **{f["PAP ID"].nunique()} PAP records**.

        The portfolio is **adaptation-heavy**, with **{adaptation_share:.2f}%**
        for adaptation and **{mitigation_share:.2f}%** for mitigation.

        The top spending agency is **{top_agency}** with **{peso(top_agency_amount)}**.

        The largest NCCAP priority is **{top_priority}** with **{peso(top_priority_amount)}**.

        The largest estimated NDC sector is **{top_sector}** with **{peso(top_sector_amount)}**.
        """
    )

    by_year = f.groupby("Fiscal_Year", as_index=False)[
        ["TOTAL", "ADAPTATION", "MITIGATION"]
    ].sum()

    c1, c2 = st.columns(2)

    with c1:
        fig = px.line(
            by_year,
            x="Fiscal_Year",
            y="TOTAL",
            markers=True,
            title="Total Climate Budget by Fiscal Year"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        alignment_summary = (
            f.groupby("PDP / Executive Agenda Alignment", as_index=False)["TOTAL"]
            .sum()
            .sort_values("TOTAL", ascending=False)
        )

        fig = px.pie(
            alignment_summary,
            names="PDP / Executive Agenda Alignment",
            values="TOTAL",
            title="National Plan Alignment Share"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Budget Trends")

    by_year = (
        f.groupby("Fiscal_Year", as_index=False)["TOTAL"]
        .sum()
        .sort_values("Fiscal_Year")
    )

    by_year["YoY Growth %"] = by_year["TOTAL"].pct_change() * 100

    c1, c2 = st.columns(2)

    with c1:
        fig = px.area(
            by_year,
            x="Fiscal_Year",
            y="TOTAL",
            title="Climate Budget Trend"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            by_year,
            x="Fiscal_Year",
            y="YoY Growth %",
            title="Year-on-Year Growth Rate"
        )
        st.plotly_chart(fig, use_container_width=True)

    pillar_year = (
        f.groupby("Fiscal_Year", as_index=False)[["ADAPTATION", "MITIGATION"]]
        .sum()
        .melt(
            id_vars="Fiscal_Year",
            value_vars=["ADAPTATION", "MITIGATION"],
            var_name="Pillar",
            value_name="Amount"
        )
    )

    fig = px.bar(
        pillar_year,
        x="Fiscal_Year",
        y="Amount",
        color="Pillar",
        barmode="group",
        title="Adaptation vs Mitigation by Fiscal Year"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(by_year, use_container_width=True)

with tab3:
    st.subheader("Agency Ranking")

    top_n = st.slider("Number of agencies to show", 5, 50, 20)

    agency_sum = (
        f.groupby(["DEPARTMENT", "AGENCY"], as_index=False)["TOTAL"]
        .sum()
        .sort_values("TOTAL", ascending=False)
        .head(top_n)
    )

    fig = px.bar(
        agency_sum.sort_values("TOTAL"),
        x="TOTAL",
        y="AGENCY",
        orientation="h",
        color="DEPARTMENT",
        title="Top Agencies by Climate Budget"
    )

    st.plotly_chart(fig, use_container_width=True)

    agency_alignment = (
        f.groupby(["AGENCY", "PDP / Executive Agenda Alignment"], as_index=False)["TOTAL"]
        .sum()
        .sort_values("TOTAL", ascending=False)
        .head(40)
    )

    fig = px.bar(
        agency_alignment,
        x="TOTAL",
        y="AGENCY",
        color="PDP / Executive Agenda Alignment",
        orientation="h",
        title="Top Agencies by National Plan Alignment"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("NCCAP Alignment")

    c1, c2 = st.columns(2)

    priority = (
        f.groupby("NCCAP Priority", as_index=False)["TOTAL"]
        .sum()
        .sort_values("TOTAL", ascending=False)
    )

    with c1:
        fig = px.bar(
            priority,
            x="TOTAL",
            y="NCCAP Priority",
            orientation="h",
            title="Budget by NCCAP Priority"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        pillar_df = f.groupby("Climate Pillar", as_index=False)["TOTAL"].sum()

        fig = px.pie(
            pillar_df,
            values="TOTAL",
            names="Climate Pillar",
            title="Budget Share by Climate Pillar"
        )
        st.plotly_chart(fig, use_container_width=True)

    typology = (
        f.groupby(
            ["TYPOLOGY ID", "TYPOLOGY Description", "NCCAP Priority", "Climate Pillar"],
            as_index=False
        )["TOTAL"]
        .sum()
        .sort_values("TOTAL", ascending=False)
    )

    st.dataframe(typology, use_container_width=True, height=500)

with tab5:
    st.subheader("PDP / Executive Agenda Alignment")

    st.info(
        "This module uses keyword-based classification to estimate whether PAPs "
        "are aligned with national climate, resilience, disaster risk reduction, "
        "sustainable development, food security, water, energy, and ecosystem priorities."
    )

    alignment = (
        f.groupby("PDP / Executive Agenda Alignment", as_index=False)["TOTAL"]
        .sum()
        .sort_values("TOTAL", ascending=False)
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.pie(
            alignment,
            names="PDP / Executive Agenda Alignment",
            values="TOTAL",
            title="Budget Share by National Plan Alignment"
        )
        st.plotly_chart(fig, use_container_width=True)

    alignment_year = (
        f.groupby(["Fiscal_Year", "PDP / Executive Agenda Alignment"], as_index=False)["TOTAL"]
        .sum()
    )

    with c2:
        fig = px.bar(
            alignment_year,
            x="Fiscal_Year",
            y="TOTAL",
            color="PDP / Executive Agenda Alignment",
            barmode="stack",
            title="National Plan Alignment Trend"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        f[[
            "Fiscal_Year", "Type", "DEPARTMENT", "AGENCY",
            "PAP ID", "PAP Description", "NCCAP Priority",
            "PDP / Executive Agenda Alignment", "TOTAL"
        ]].sort_values("TOTAL", ascending=False),
        use_container_width=True,
        height=500
    )

with tab6:
    st.subheader("NDC Sector Alignment")

    st.info(
        "This module estimates alignment with key climate sectors such as energy, "
        "transport, agriculture, waste, industry, water/flood control, and ecosystems."
    )

    sector = (
        f.groupby("NDC Sector Alignment", as_index=False)["TOTAL"]
        .sum()
        .sort_values("TOTAL", ascending=False)
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            sector,
            x="TOTAL",
            y="NDC Sector Alignment",
            orientation="h",
            title="Climate Budget by NDC Sector"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.pie(
            sector,
            names="NDC Sector Alignment",
            values="TOTAL",
            title="NDC Sector Share"
        )
        st.plotly_chart(fig, use_container_width=True)

    sector_year = (
        f.groupby(["Fiscal_Year", "NDC Sector Alignment"], as_index=False)["TOTAL"]
        .sum()
    )

    fig = px.area(
        sector_year,
        x="Fiscal_Year",
        y="TOTAL",
        color="NDC Sector Alignment",
        title="NDC Sector Budget Trend"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab7:
    st.subheader("Policy Insights for DBM, CCC, and NEDA/DEPDev")

    total_budget = f["TOTAL"].sum()
    adaptation_share = (f["ADAPTATION"].sum() / total_budget * 100) if total_budget else 0
    mitigation_share = (f["MITIGATION"].sum() / total_budget * 100) if total_budget else 0

    top_agency, top_agency_amount = safe_top_value(f, "AGENCY")
    top_priority, top_priority_amount = safe_top_value(f, "NCCAP Priority")
    top_sector, top_sector_amount = safe_top_value(f, "NDC Sector Alignment")

    st.markdown("### Key Policy Findings")

    st.write(
        f"""
        **1. Climate budget concentration:** Total climate-tagged budget covered by the dashboard is **{peso(total_budget)}**.

        **2. Adaptation-heavy portfolio:** Around **{adaptation_share:.2f}%** of the tagged budget is for adaptation, while **{mitigation_share:.2f}%** is for mitigation.

        **3. Top spending agency:** The largest climate-tagged allocation is from **{top_agency}**, with **{peso(top_agency_amount)}**.

        **4. Main NCCAP priority:** The biggest thematic allocation is under **{top_priority}**, with **{peso(top_priority_amount)}**.

        **5. Main NDC sector:** The largest estimated NDC sector allocation is **{top_sector}**, with **{peso(top_sector_amount)}**.
        """
    )

    st.markdown("### Suggested Policy Questions")

    st.write(
        """
        - Are climate-tagged PAPs concentrated in a few agencies or broadly mainstreamed across government?
        - Is the national climate budget overly adaptation-heavy compared with mitigation needs?
        - Are PAPs aligned with national climate action and disaster resilience priorities?
        - Which agencies have high climate allocations but weak policy alignment?
        - Which NCCAP priorities remain underfunded?
        - Which NDC sectors require stronger budget support?
        """
    )

    participation = (
        f.groupby("Fiscal_Year", as_index=False)["AGENCY"]
        .nunique()
        .rename(columns={"AGENCY": "Participating Agencies"})
    )

    fig = px.line(
        participation,
        x="Fiscal_Year",
        y="Participating Agencies",
        markers=True,
        title="Agency Participation Trend"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(participation, use_container_width=True)

with tab8:
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
        "ADAPTATION", "MITIGATION", "TOTAL"
    ]

    st.dataframe(
        explorer[cols].sort_values("TOTAL", ascending=False),
        use_container_width=True,
        height=600
    )

with tab9:
    st.subheader("Data Quality Checks")

    checks = {
        "Missing agency": f["AGENCY"].eq("").sum(),
        "Missing typology ID": f["TYPOLOGY ID"].eq("").sum(),
        "Zero or blank total": (f["TOTAL"].fillna(0) == 0).sum(),
        "Adaptation + Mitigation ≠ Total": (
            (
                f["ADAPTATION"].fillna(0)
                + f["MITIGATION"].fillna(0)
                - f["TOTAL"].fillna(0)
            ).abs() > 1
        ).sum(),
        "Duplicate PAP ID records": f.duplicated(
            subset=["Fiscal_Year", "Type", "AGENCY", "PAP ID", "TYPOLOGY ID"],
            keep=False
        ).sum(),
    }

    qc = pd.DataFrame([
        {"Check": k, "Flagged Records": v}
        for k, v in checks.items()
    ])

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
        mask = (
            (
                f["ADAPTATION"].fillna(0)
                + f["MITIGATION"].fillna(0)
                - f["TOTAL"].fillna(0)
            ).abs() > 1
        )
    elif issue_filter == "Duplicate PAP ID records":
        mask = f.duplicated(
            subset=["Fiscal_Year", "Type", "AGENCY", "PAP ID", "TYPOLOGY ID"],
            keep=False
        )

    st.dataframe(f.loc[mask], use_container_width=True, height=400)

with tab10:
    st.subheader("Dataset Schema and Data Dictionary")

    st.markdown(f"""
    **Current dataset source:** {dataset_source}  
    **Number of records:** {len(df):,}  
    **Number of columns:** {len(df.columns):,}  
    **Fiscal year coverage:** {int(df["Fiscal_Year"].min())} to {int(df["Fiscal_Year"].max())}
    """)

    schema_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[col].dtype) for col in df.columns],
        "Missing Values": [df[col].isna().sum() for col in df.columns],
        "Description": [COLUMN_DICTIONARY.get(col, "Column from uploaded dataset.") for col in df.columns]
    })

    st.dataframe(schema_df, use_container_width=True, height=500)

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

with tab11:
    st.subheader("User Manual and Guide")

    st.markdown("""
    ## Introduction

    The National CCET Policy Analytics Platform is an interactive dashboard for analyzing
    climate-tagged Programs, Activities, and Projects using CCET data.

    ## How to Use the Dashboard

    1. Use the sidebar filters to select a fiscal year, budget type, department, climate pillar,
       PDP alignment, or NDC sector.
    2. Review the KPI cards at the top.
    3. Open each tab to explore trends, agencies, NCCAP alignment, policy insights, and PAP-level details.
    4. Use the PAP Explorer to validate specific records.
    5. Use the Data Quality tab to check possible data issues.
    6. Download a PDF report from the sidebar.

    ## Dashboard Pages

    **Executive Brief** gives a quick policy summary.

    **Budget Trends** shows climate budget movement across fiscal years.

    **Agency Ranking** identifies top spending agencies.

    **NCCAP Alignment** shows allocations by NCCAP priority and climate pillar.

    **PDP / Executive Agenda** estimates national plan alignment using keywords.

    **NDC Sector Alignment** estimates sectoral alignment.

    **Policy Insights** provides policy questions and summary findings.

    **PAP Explorer** allows record-level search and validation.

    **Data Quality** checks missing values, duplicates, and budget inconsistencies.

    **Dataset Schema** explains the dataset columns and structure.

    ## Important Disclaimer

    NCCAP classification is derived from CCET typology codes.

    PDP / Executive Agenda Alignment and NDC Sector Alignment are analytical proxies
    based on keyword matching. They should not be treated as official validation by
    CCC, DBM, or NEDA/DEPDev.

    Users should validate findings with official CCET submissions, QAR forms,
    agency documents, and government publications before making policy conclusions.
    """)

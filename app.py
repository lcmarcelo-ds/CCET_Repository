import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="National CCET Analytics Dashboard",
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


@st.cache_data(show_spinner="Loading CCET CSV dataset...")
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        st.error("CSV dataset not found. Please upload it as `data/ccet_data.csv`.")
        st.stop()

    if os.path.getsize(DATA_PATH) == 0:
        st.error("`data/ccet_data.csv` is empty. Please re-upload the real CSV file.")
        st.stop()

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"ADAPTION": "ADAPTATION"})

    for col in ["Fiscal_Year", "ADAPTATION", "MITIGATION", "TOTAL"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Fiscal_Year"] = df["Fiscal_Year"].astype(int)

    text_cols = [
        "Type", "DEPARTMENT", "GRIT TAGGING", "AGENCY",
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


st.title("National CCET Data Analytics Dashboard")
st.caption(
    "Climate Change Expenditure Tagging PAP-level analytics | "
    "FY2017–FY2026 | With National Plan and NDC Alignment Layer"
)

df = load_data()

st.sidebar.header("Filters")

year = filter_dropdown("Fiscal Year", df["Fiscal_Year"].unique())
budget_type = filter_dropdown("Budget Type", df["Type"].unique())
tagging = filter_dropdown("Institution Type", df["GRIT TAGGING"].unique())
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

if tagging != "All":
    f = f[f["GRIT TAGGING"] == tagging]

if department != "All":
    f = f[f["DEPARTMENT"] == department]

if pillar != "All":
    f = f[f["Climate Pillar"] == pillar]

if pdp_alignment != "All":
    f = f[f["PDP / Executive Agenda Alignment"] == pdp_alignment]

if ndc_sector != "All":
    f = f[f["NDC Sector Alignment"] == ndc_sector]

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Total Climate Budget", peso(f["TOTAL"].sum()))
k2.metric("Adaptation", peso(f["ADAPTATION"].sum()))
k3.metric("Mitigation", peso(f["MITIGATION"].sum()))
k4.metric("Agencies", f["AGENCY"].nunique())
k5.metric("PAP Records", f["PAP ID"].nunique())

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Executive Overview",
    "Budget Trends",
    "Agency Explorer",
    "Typology / NCCAP",
    "National Plan Alignment",
    "NDC Sector Alignment",
    "Compliance",
    "Data Quality"
])

with tab1:
    st.subheader("Executive Overview")

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
        long = by_year.melt(
            id_vars="Fiscal_Year",
            value_vars=["ADAPTATION", "MITIGATION"],
            var_name="Pillar",
            value_name="Amount"
        )

        fig = px.bar(
            long,
            x="Fiscal_Year",
            y="Amount",
            color="Pillar",
            barmode="group",
            title="Adaptation vs Mitigation"
        )
        st.plotly_chart(fig, use_container_width=True)

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

    st.dataframe(by_year.sort_values("Fiscal_Year"), use_container_width=True)

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

    st.dataframe(by_year, use_container_width=True)

with tab3:
    st.subheader("Agency Explorer")

    top_n = st.slider("Number of agencies to show", 5, 50, 15)

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

    selected_agency = st.selectbox(
        "Inspect agency",
        ["All"] + sorted(f["AGENCY"].dropna().unique().tolist())
    )

    table = f if selected_agency == "All" else f[f["AGENCY"] == selected_agency]

    cols = [
        "Fiscal_Year", "Type", "DEPARTMENT", "GRIT TAGGING", "AGENCY",
        "PAP ID", "PAP Description", "TYPOLOGY ID", "NCCAP Priority",
        "PDP / Executive Agenda Alignment", "NDC Sector Alignment",
        "ADAPTATION", "MITIGATION", "TOTAL"
    ]

    st.dataframe(
        table[cols].sort_values(["Fiscal_Year", "TOTAL"], ascending=[True, False]),
        use_container_width=True,
        height=500
    )

with tab4:
    st.subheader("Typology / NCCAP Analysis")

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
    st.subheader("National Plan / Executive Agenda Alignment")

    st.info(
        "This module uses keyword-based classification to estimate whether PAPs "
        "are aligned with national climate, resilience, disaster risk reduction, "
        "sustainable development, food security, water, energy, and ecosystem priorities."
    )

    c1, c2 = st.columns(2)

    alignment = (
        f.groupby("PDP / Executive Agenda Alignment", as_index=False)["TOTAL"]
        .sum()
        .sort_values("TOTAL", ascending=False)
    )

    with c1:
        fig = px.pie(
            alignment,
            names="PDP / Executive Agenda Alignment",
            values="TOTAL",
            title="Budget Share by National Plan Alignment"
        )
        st.plotly_chart(fig, use_container_width=True)

    agency_alignment = (
        f.groupby(["AGENCY", "PDP / Executive Agenda Alignment"], as_index=False)["TOTAL"]
        .sum()
        .sort_values("TOTAL", ascending=False)
        .head(30)
    )

    with c2:
        fig = px.bar(
            agency_alignment,
            x="TOTAL",
            y="AGENCY",
            color="PDP / Executive Agenda Alignment",
            orientation="h",
            title="Top Agencies by National Plan Alignment"
        )
        st.plotly_chart(fig, use_container_width=True)

    alignment_year = (
        f.groupby(["Fiscal_Year", "PDP / Executive Agenda Alignment"], as_index=False)["TOTAL"]
        .sum()
    )

    fig = px.bar(
        alignment_year,
        x="Fiscal_Year",
        y="TOTAL",
        color="PDP / Executive Agenda Alignment",
        barmode="stack",
        title="National Plan Alignment Trend by Fiscal Year"
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

    st.dataframe(
        f[[
            "Fiscal_Year", "Type", "DEPARTMENT", "AGENCY",
            "PAP ID", "PAP Description", "NDC Sector Alignment", "TOTAL"
        ]].sort_values("TOTAL", ascending=False),
        use_container_width=True,
        height=500
    )

with tab7:
    st.subheader("Compliance / Participation Proxy")

    participation = (
        f.groupby(["Fiscal_Year", "GRIT TAGGING"], as_index=False)["AGENCY"]
        .nunique()
        .rename(columns={"AGENCY": "Participating Agencies"})
    )

    fig = px.bar(
        participation,
        x="Fiscal_Year",
        y="Participating Agencies",
        color="GRIT TAGGING",
        barmode="stack",
        title="Participating Agencies by Institution Type"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(participation, use_container_width=True)

with tab8:
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

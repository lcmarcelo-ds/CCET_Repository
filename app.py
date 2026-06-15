import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="National CCET Analytics Dashboard",
    layout="wide"
)

DATA_PATH = "data/Cleaned_National_CCET_PAPs_FY_2017_to_2026.xlsx"

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

@st.cache_data(show_spinner="Loading CCET dataset...")
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Dataset not found: {path}")
        st.stop()

    xls = pd.ExcelFile(path, engine="openpyxl")
    sheet = "Tagged" if "Tagged" in xls.sheet_names else xls.sheet_names[0]

    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    df = df.rename(columns={"ADAPTION": "ADAPTATION"})

    for col in ["Fiscal_Year", "ADAPTATION", "MITIGATION", "TOTAL"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "Fiscal_Year" in df.columns:
        df["Fiscal_Year"] = df["Fiscal_Year"].astype(int)

    text_cols = [
        "Type", "DEPARTMENT", "GRIT TAGGING", "AGENCY",
        "PAP ID", "PAP Description", "TYPOLOGY ID", "TYPOLOGY Description"
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .replace({"nan": "", "None": ""})
                .str.strip()
            )

    if "TYPOLOGY ID" in df.columns:
        typo = df["TYPOLOGY ID"].str.upper()

        df["Climate Pillar"] = np.where(
            typo.str.startswith("M"), "Mitigation",
            np.where(typo.str.startswith("A"), "Adaptation", "Unclassified")
        )

        df["NCCAP Code"] = typo.str.extract(r"^[AM](\d)", expand=False).fillna("")
        df["NCCAP Priority"] = df["NCCAP Code"].map(NCCAP_PRIORITY).fillna("Unclassified")
    else:
        df["Climate Pillar"] = "Unclassified"
        df["NCCAP Priority"] = "Unclassified"

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


def filter_multiselect(label, values, default_all=True):
    opts = sorted([v for v in values if str(v).strip() != ""])
    default = opts if default_all else []
    return st.sidebar.multiselect(label, opts, default=default)


st.title("National CCET Data Analytics Dashboard")
st.caption("Climate Change Expenditure Tagging PAP-level analytics | FY2017–FY2026")

df = load_data()

st.sidebar.header("Filters")

years = filter_multiselect("Fiscal Year", df["Fiscal_Year"].unique())
types = filter_multiselect("Budget Type", df["Type"].unique()) if "Type" in df.columns else []
tagging = filter_multiselect("Institution Type", df["GRIT TAGGING"].unique()) if "GRIT TAGGING" in df.columns else []
departments = filter_multiselect("Department", df["DEPARTMENT"].unique()) if "DEPARTMENT" in df.columns else []
pillars = filter_multiselect("Climate Pillar", df["Climate Pillar"].unique())

f = df.copy()

f = f[f["Fiscal_Year"].isin(years)]

if types:
    f = f[f["Type"].isin(types)]

if tagging:
    f = f[f["GRIT TAGGING"].isin(tagging)]

if departments:
    f = f[f["DEPARTMENT"].isin(departments)]

if pillars:
    f = f[f["Climate Pillar"].isin(pillars)]

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Total Climate Budget", peso(f["TOTAL"].sum()))
k2.metric("Adaptation", peso(f["ADAPTATION"].sum()))
k3.metric("Mitigation", peso(f["MITIGATION"].sum()))
k4.metric("Agencies", f["AGENCY"].nunique() if "AGENCY" in f.columns else 0)
k5.metric("PAP Records", f["PAP ID"].nunique() if "PAP ID" in f.columns else len(f))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Executive Overview",
    "Budget Trends",
    "Agency Explorer",
    "Typology / NCCAP",
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
        c for c in [
            "Fiscal_Year", "Type", "DEPARTMENT", "GRIT TAGGING", "AGENCY",
            "PAP ID", "PAP Description", "TYPOLOGY ID", "NCCAP Priority",
            "ADAPTATION", "MITIGATION", "TOTAL"
        ]
        if c in table.columns
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
        pillar = f.groupby("Climate Pillar", as_index=False)["TOTAL"].sum()

        fig = px.pie(
            pillar,
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
    st.subheader("Compliance / Participation Proxy")

    if "GRIT TAGGING" in f.columns:
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
    else:
        st.info("GRIT TAGGING column not found.")

with tab6:
    st.subheader("Data Quality Checks")

    checks = {
        "Missing agency": f["AGENCY"].eq("").sum() if "AGENCY" in f.columns else 0,
        "Missing typology ID": f["TYPOLOGY ID"].eq("").sum() if "TYPOLOGY ID" in f.columns else 0,
        "Zero or blank total": (f["TOTAL"].fillna(0) == 0).sum(),
        "Adaptation + Mitigation ≠ Total": (
            (f["ADAPTATION"].fillna(0) + f["MITIGATION"].fillna(0) - f["TOTAL"].fillna(0)).abs() > 1
        ).sum(),
        "Duplicate PAP ID records": f.duplicated(
            subset=["Fiscal_Year", "Type", "AGENCY", "PAP ID", "TYPOLOGY ID"],
            keep=False
        ).sum() if "PAP ID" in f.columns else 0,
    }

    qc = pd.DataFrame([
        {"Check": k, "Flagged Records": v}
        for k, v in checks.items()
    ])

    st.dataframe(qc, use_container_width=True)

    issue_filter = st.selectbox("Show records for", list(checks.keys()))

    mask = pd.Series(False, index=f.index)

    if issue_filter == "Missing agency" and "AGENCY" in f.columns:
        mask = f["AGENCY"].eq("")
    elif issue_filter == "Missing typology ID" and "TYPOLOGY ID" in f.columns:
        mask = f["TYPOLOGY ID"].eq("")
    elif issue_filter == "Zero or blank total":
        mask = f["TOTAL"].fillna(0) == 0
    elif issue_filter == "Adaptation + Mitigation ≠ Total":
        mask = (
            (f["ADAPTATION"].fillna(0) + f["MITIGATION"].fillna(0) - f["TOTAL"].fillna(0)).abs() > 1
        )
    elif issue_filter == "Duplicate PAP ID records" and "PAP ID" in f.columns:
        mask = f.duplicated(
            subset=["Fiscal_Year", "Type", "AGENCY", "PAP ID", "TYPOLOGY ID"],
            keep=False
        )

    st.dataframe(f.loc[mask], use_container_width=True, height=400)

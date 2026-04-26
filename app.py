import os
import pathlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Resolve DATA_DIR: try the directory containing this file, then cwd.
# Both work locally and on Streamlit Cloud as long as CSVs live next to app.py.
try:
    DATA_DIR = pathlib.Path(__file__).parent.resolve()
except NameError:
    DATA_DIR = pathlib.Path.cwd()

def _find_file(name):
    """
    Find `name` (a CSV filename) in the repo. Checks in order:
    1. The exact CSV file
    2. A .zip with the same stem as the CSV
    3. Any .zip in the search directories (picks first one containing a .csv)
    """
    import zipfile
    bases = (DATA_DIR, pathlib.Path.cwd(), DATA_DIR / "data", pathlib.Path.cwd() / "data")
    for base in bases:
        # Exact CSV match
        p = base / name
        if p.exists():
            return str(p)
        # Same-stem zip
        p = base / (pathlib.Path(name).stem + ".zip")
        if p.exists():
            return str(p)
    # Any zip in any base that contains a CSV
    for base in bases:
        if not base.exists():
            continue
        for zp in sorted(base.glob("*.zip")):
            try:
                with zipfile.ZipFile(zp) as zf:
                    if any(n.endswith(".csv") for n in zf.namelist()):
                        return str(zp)
            except Exception:
                continue
    return None

st.set_page_config(page_title="USDA Digital Service Dashboard", layout="wide")

AGENCY_MAP = {
    "fns.usda.gov": "Food and Nutrition Service",
    "ams.usda.gov": "Agricultural Marketing Service",
    "nrcs.usda.gov": "Natural Resources Conservation Service",
}

# ─────────────────────────────────────────────────────────────────────────────
# System-wide data loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_system_data():
    def p(name):
        path = _find_file(name)
        if path is None:
            raise FileNotFoundError(f"Cannot find '{name}'. Make sure all CSV files are committed to the repository alongside app.py.")
        return path

    device = pd.read_csv(p("device-1-2024.csv"))
    device["date"] = pd.to_datetime(device["date"])
    device["month"] = device["date"].dt.to_period("M").dt.to_timestamp()

    domain = pd.read_csv(p("domain-1-2024.csv"))
    domain["date"] = pd.to_datetime(domain["date"])
    domain["month"] = domain["date"].dt.to_period("M").dt.to_timestamp()

    downloads = pd.read_csv(p("download-1-2024.csv"))
    downloads["date"] = pd.to_datetime(downloads["date"])
    downloads["month"] = downloads["date"].dt.to_period("M").dt.to_timestamp()

    language = pd.read_csv(p("language-1-2024.csv"))
    language["date"] = pd.to_datetime(language["date"])
    language["month"] = language["date"].dt.to_period("M").dt.to_timestamp()

    traffic = pd.read_csv(p("traffic-source-1-2024.csv"))
    traffic["date"] = pd.to_datetime(traffic["date"])
    traffic["month"] = traffic["date"].dt.to_period("M").dt.to_timestamp()

    os_browser = pd.read_csv(p("os-browser-1-2024.csv"))
    win_browser = pd.read_csv(p("windows-browser-1-2024.csv"))

    return device, domain, downloads, language, traffic, os_browser, win_browser


# ─────────────────────────────────────────────────────────────────────────────
# Rural Development data loader / preprocessor
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_rd_data(file_source):
    """file_source is a file path string (csv or zip) or an UploadedFile object."""
    import zipfile, io
    if isinstance(file_source, str):
        if file_source.endswith(".zip"):
            with zipfile.ZipFile(file_source) as zf:
                csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
                with zf.open(csv_name) as f:
                    raw = pd.read_csv(f, header=None, low_memory=False)
        else:
            raw = pd.read_csv(file_source, header=None, low_memory=False)
    else:
        # UploadedFile — check if it's a zip by filename
        if file_source.name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(file_source.read())) as zf:
                csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
                with zf.open(csv_name) as f:
                    raw = pd.read_csv(f, header=None, low_memory=False)
            file_source.seek(0)
        else:
            raw = pd.read_csv(file_source, header=None, low_memory=False)

    # Step 1: skip first 6 metadata rows (rows 0-5); headers at rows 6-7
    h1 = list(raw.iloc[6])
    h2 = list(raw.iloc[7])

    # Step 2: flatten two-row header
    cols = []
    current_cat = ""
    for a, b in zip(h1, h2):
        if pd.notna(a) and str(a).strip():
            current_cat = str(a).strip()
        b_str = str(b).strip() if pd.notna(b) else ""
        if b_str:
            if current_cat and current_cat.split()[0].lower() not in b_str.lower():
                cols.append(current_cat + "_" + b_str)
            else:
                cols.append(b_str)
        else:
            cols.append(current_cat if current_cat else "unnamed")

    # Step 3: drop last empty column
    df = raw.iloc[9:].copy()  # data starts at row 9 (skip header rows 6-7 and totals row 8)
    df.columns = cols[: len(df.columns)]
    df = df.iloc[:, :-1]  # drop last column

    # Identify key columns
    title_col = cols[0]   # "Column1_Page title"
    section_col = cols[1] # "Other" (section label)
    month_col = cols[2]   # "Column2_Month"
    day_col = cols[3]     # "Column3_Day"
    path_col = cols[5]    # "Device category_Page path and screen class"

    # Step 4: remove rows where Page title is null or empty
    df = df[df[title_col].notna() & (df[title_col].astype(str).str.strip() != "")]

    # Step 5: rename page path column
    df = df.rename(columns={path_col: "Page path", title_col: "Page title", section_col: "Section",
                             month_col: "Month", day_col: "Day"})

    # Step 6: convert numeric columns to float
    non_numeric = {"Page title", "Section", "Month", "Day", "Page path", cols[4]}
    for c in df.columns:
        if c not in non_numeric:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Step 7: extract Section from page path using first path segment
    def extract_section(path):
        if pd.isna(path):
            return "Other"
        path = str(path).strip("/")
        seg = path.split("/")[0] if "/" in path else path
        if not seg:
            return "Home"
        return seg.replace("-", " ").title() if seg else "Other"

    # Use Section column if filled, else derive from path
    df["Section"] = df["Section"].where(
        df["Section"].notna() & (df["Section"].astype(str).str.strip() != "") & (df["Section"] != "Other"),
        df["Page path"].apply(extract_section)
    )

    # Build clean column references after rename
    # device columns: desktop cols 6-14, mobile 15-23, tablet 24-32, totals 42-50
    # Index offsets (0-based in original): desktop starts at col 6
    def get_col(name):
        return name if name in df.columns else None

    # Map to actual column names by position
    original_cols = cols  # before renames applied
    def col_at(idx):
        return df.columns[idx] if idx < len(df.columns) else None

    # After renames, let's use positional access:
    # Col 0 -> Page title, 1 -> Section, 2 -> Month, 3 -> Day, 4 -> Country, 5 -> Page path
    # 6..14 -> desktop, 15..23 -> mobile, 24..32 -> tablet, 33..41 -> smart tv, 42..50 -> totals

    def safe_col(idx):
        c = df.columns[idx] if idx < len(df.columns) else None
        return c

    desktop_bounce = safe_col(11)   # desktop Bounce rate
    mobile_bounce  = safe_col(20)   # mobile Bounce rate
    totals_exits   = safe_col(48)   # Totals Exits
    totals_sessions= safe_col(44)   # Totals Sessions
    totals_return  = safe_col(49)   # Totals Returning users
    totals_users   = safe_col(50)   # Totals Total users
    totals_bounce  = safe_col(47)   # Totals Bounce rate
    totals_duration= safe_col(46)   # Totals Average session duration
    totals_views   = safe_col(45)   # Totals Views per session
    totals_active  = safe_col(42)   # Totals Active users

    # Step 8: Device Gap Score
    df["Device Gap Score"] = pd.to_numeric(df[mobile_bounce], errors="coerce") - \
                              pd.to_numeric(df[desktop_bounce], errors="coerce")

    # Step 9: Exit Pressure Index
    exits_s = pd.to_numeric(df[totals_exits], errors="coerce")
    sess_s  = pd.to_numeric(df[totals_sessions], errors="coerce")
    df["Exit Pressure Index"] = exits_s / sess_s.replace(0, np.nan)

    # Step 10: Stickiness Ratio
    ret_s  = pd.to_numeric(df[totals_return], errors="coerce")
    usr_s  = pd.to_numeric(df[totals_users], errors="coerce")
    df["Stickiness Ratio"] = ret_s / usr_s.replace(0, np.nan)

    # Rename key totals columns for clarity
    df = df.rename(columns={
        totals_bounce:   "Bounce Rate",
        totals_duration: "Session Duration",
        totals_views:    "Views per Session",
        totals_active:   "Active Users",
        totals_exits:    "Totals Exits",
        totals_sessions: "Totals Sessions",
        totals_return:   "Returning Users",
        totals_users:    "Total Users",
    })

    # Also rename Month/Day for time series
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df["Day"]   = pd.to_numeric(df["Day"], errors="coerce")

    # Step 11: aggregate to page level
    rate_cols = ["Bounce Rate", "Session Duration", "Views per Session",
                 "Exit Pressure Index", "Stickiness Ratio", "Device Gap Score"]
    count_cols = ["Active Users", "Returning Users", "Total Users"]

    agg_dict = {c: "mean" for c in rate_cols if c in df.columns}
    agg_dict.update({c: "sum" for c in count_cols if c in df.columns})
    agg_dict["Section"] = "first"
    agg_dict["Month"] = "min"

    page_df = df.groupby("Page title").agg(agg_dict).reset_index()

    # Step 12: remove outliers
    page_df = page_df[page_df["Session Duration"] <= 1000]

    # Step 13: Underserved Score
    page_df["Underserved Score"] = (
        page_df["Bounce Rate"].fillna(0) * 0.4 +
        page_df["Exit Pressure Index"].fillna(0) * 0.3 +
        (1 - page_df["Stickiness Ratio"].fillna(0)) * 0.3
    )

    # Step 14: Behavioral Signature
    def assign_signature(row):
        br  = row.get("Bounce Rate", np.nan)
        dur = row.get("Session Duration", np.nan)
        st  = row.get("Stickiness Ratio", np.nan)
        ep  = row.get("Exit Pressure Index", np.nan)
        vps = row.get("Views per Session", np.nan)
        if pd.isna(br) or pd.isna(dur):
            return "Unknown"
        if br > 0.55 and dur < 60:
            return "High Abandonment"
        if ep > 0.6 and dur > 300:
            return "Frustrated Navigation"
        if br < 0.30 and dur < 120:
            return "Quick Task Completion"
        if pd.notna(st) and st > 0.3 and 60 <= dur <= 300:
            return "Repeat Reference"
        if 0.30 <= br <= 0.55 and dur >= 120 and (pd.isna(vps) or vps > 2):
            return "Deep Engagement"
        return "Quick Task Completion"

    page_df["Behavioral Signature"] = page_df.apply(assign_signature, axis=1)

    # Step 15: Standardize 6 features
    feature_cols = ["Bounce Rate", "Session Duration", "Views per Session",
                    "Exit Pressure Index", "Stickiness Ratio", "Device Gap Score"]
    feature_data = page_df[feature_cols].fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_data)

    return page_df, scaled, feature_cols, df  # df = daily-level data


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("USDA Digital Dashboard")
st.sidebar.markdown("---")

RD_FILENAME = "(Rural Development) Edited USDA data base.csv"
rd_default_path = _find_file(RD_FILENAME)

rd_file = st.sidebar.file_uploader(
    "Override Rural Development CSV (optional)",
    type=["csv", "zip"],
    help="Only needed if the bundled file is not in the repository. Accepts .csv or a .zip containing the CSV."
)

if rd_file is not None:
    rd_source = rd_file
    rd_available = True
    st.sidebar.success("Using uploaded file.")
elif rd_default_path is not None:
    rd_source = rd_default_path
    rd_available = True
    st.sidebar.info("RD data loaded from repository.")
else:
    rd_source = None
    rd_available = False
    st.sidebar.error(f"'{RD_FILENAME}' not found in the repository. Please upload it above.")

tabs = st.tabs([
    "Layer 1 · System-Wide Analysis",
    "Layer 2 · Rural Development Baseline",
    "Layer 3 · K-Means Clustering",
    "Layer 4 · Behavioral Signatures",
])

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — System-Wide Descriptive Analysis
# ─────────────────────────────────────────────────────────────────────────────

with tabs[0]:
    st.header("Layer 1: System-Wide Descriptive Analysis")
    st.caption("What is happening across USDA's web presence?")

    try:
        device_df, domain_df, downloads_df, language_df, traffic_df, os_df, win_df = load_system_data()

        subtabs = st.tabs(["Overview", "Agency Traffic", "Content Demand", "Language Reach", "Friction Flags"])

        # ── Overview ──────────────────────────────────────────────────────────
        with subtabs[0]:
            st.subheader("Overview")

            total_visits = device_df["visits"].sum()
            mobile_visits = device_df[device_df["device"] == "mobile"]["visits"].sum()
            mobile_pct = mobile_visits / total_visits * 100 if total_visits else 0

            k1, k2, k3 = st.columns(3)
            k1.metric("Total Visits (Jan–Jun)", f"{total_visits:,.0f}")
            k2.metric("Mobile Share", f"{mobile_pct:.1f}%")
            k3.metric("Unique Dates", f"{device_df['date'].nunique():,}")

            # Daily visit trend
            daily_visits = device_df.groupby("date")["visits"].sum().reset_index()
            fig_daily = go.Figure()
            fig_daily.add_trace(go.Scatter(
                x=daily_visits["date"], y=daily_visits["visits"],
                mode="lines", name="Daily Visits", line=dict(color="#1f77b4")
            ))
            fig_daily.update_layout(
                title="Daily Total Visits Across All USDA Agencies",
                xaxis_title="Date", yaxis_title="Visits",
                height=320
            )
            st.plotly_chart(fig_daily, use_container_width=True)
            st.caption("This chart shows overall daily traffic volume. Peaks may correspond to policy announcements, seasonal programs, or news events affecting USDA services.")

            # Mobile share monthly trend
            monthly_device = device_df.groupby(["month", "device"])["visits"].sum().reset_index()
            monthly_total = monthly_device.groupby("month")["visits"].sum().rename("total")
            monthly_mobile = monthly_device[monthly_device["device"] == "mobile"].set_index("month")["visits"]
            mobile_monthly_pct = (monthly_mobile / monthly_total * 100).reset_index()
            mobile_monthly_pct.columns = ["month", "mobile_pct"]
            mobile_monthly_pct = mobile_monthly_pct.sort_values("month")

            fig_mob = go.Figure()
            fig_mob.add_trace(go.Scatter(
                x=mobile_monthly_pct["month"],
                y=mobile_monthly_pct["mobile_pct"],
                mode="lines+markers", name="Mobile %", line=dict(color="#ff7f0e")
            ))
            fig_mob.update_layout(
                title="Monthly Mobile Traffic Share (%)",
                xaxis_title="Month", yaxis_title="Mobile Share (%)",
                height=300
            )
            st.plotly_chart(fig_mob, use_container_width=True)
            st.caption("Mobile share percentage by month. Rising mobile usage signals the need for mobile-optimized content and services.")

            # Mobile trend flag
            pcts = mobile_monthly_pct["mobile_pct"].tolist()
            rising_streak = 0
            max_streak = 0
            for i in range(1, len(pcts)):
                if pcts[i] > pcts[i - 1]:
                    rising_streak += 1
                    max_streak = max(max_streak, rising_streak)
                else:
                    rising_streak = 0
            if max_streak >= 3:
                st.warning("Trend flag: Mobile share has risen for 3 or more consecutive months. Prioritize mobile-first improvements.")

        # ── Agency Traffic ─────────────────────────────────────────────────────
        with subtabs[1]:
            st.subheader("Agency Traffic")

            # Top 15 hostnames
            total_by_host = domain_df.groupby("domain")["visits"].sum().reset_index()
            total_by_host["Agency"] = total_by_host["domain"].map(
                lambda d: f"{d} — {AGENCY_MAP[d]}" if d in AGENCY_MAP else d
            )
            top15 = total_by_host.nlargest(15, "visits")

            fig_host = go.Figure(go.Bar(
                x=top15["visits"], y=top15["Agency"],
                orientation="h", marker_color="#1f77b4",
                text=top15["visits"].apply(lambda v: f"{v/1e6:.1f}M"),
                textposition="outside"
            ))
            fig_host.update_layout(
                title="Top 15 USDA Hostnames by Total Visits (Jan–Jun)",
                xaxis_title="Total Visits", yaxis_title="",
                yaxis=dict(autorange="reversed"),
                height=500, margin=dict(l=300)
            )
            st.plotly_chart(fig_host, use_container_width=True)
            st.caption("Agencies with the highest total visits represent core public-facing services. Food and Nutrition Service and Forest Service consistently draw the most traffic.")

            # MoM momentum table
            monthly_host = domain_df.groupby(["domain", "month"])["visits"].sum().reset_index()
            month_counts = monthly_host.groupby("domain")["month"].nunique()
            domains_3plus = month_counts[month_counts >= 3].index
            monthly_host = monthly_host[monthly_host["domain"].isin(domains_3plus)]

            pivot = monthly_host.pivot_table(index="domain", columns="month", values="visits", fill_value=0)
            pivot.columns = [c.strftime("%b") for c in pivot.columns]
            months_present = list(pivot.columns)

            if len(months_present) >= 2:
                pivot["% Change"] = ((pivot[months_present[-1]] - pivot[months_present[0]]) /
                                     pivot[months_present[0]].replace(0, np.nan) * 100).round(1)
            pivot = pivot.reset_index()
            pivot["Agency"] = pivot["domain"].map(
                lambda d: f"{d} — {AGENCY_MAP[d]}" if d in AGENCY_MAP else d
            )
            pivot = pivot.drop(columns=["domain"])
            pivot = pivot.set_index("Agency")

            def highlight_change(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return "background-color: #c6efce; color: #276221"
                    elif val < 0:
                        return "background-color: #ffc7ce; color: #9c0006"
                return ""

            # .map() replaces deprecated .applymap() in pandas ≥ 2.2
            _styler_map = getattr(pivot.style, "map", None) or getattr(pivot.style, "applymap")
            styled = _styler_map(highlight_change, subset=["% Change"]).format(
                {c: "{:,.0f}" for c in pivot.columns if c != "% Change"} | {"% Change": "{:.1f}%"}
            )
            st.markdown("**Month-over-Month Momentum (Jan → Jun)**")
            st.dataframe(styled, use_container_width=True)
            st.caption("Green % change = growing audience; red = declining. Agencies with consistent decline warrant content strategy review.")

            # Stacked monthly traffic source bar
            st.subheader("Monthly Traffic by Source")

            def classify_source(row):
                src = str(row["source"]).lower()
                if row.get("has_social_referral", "No") == "Yes":
                    return "Social Referral"
                if src in ("google", "bing", "yahoo", "duckduckgo"):
                    return "Organic Search"
                if src == "(direct)":
                    return "Direct"
                return "Other"

            traffic_df["category"] = traffic_df.apply(classify_source, axis=1)
            monthly_src = traffic_df.groupby(["month", "category"])["visits"].sum().reset_index()

            src_colors = {
                "Organic Search": "#2196F3",
                "Direct": "#4CAF50",
                "Social Referral": "#FF9800",
                "Other": "#9E9E9E",
            }

            fig_src = go.Figure()
            for cat in ["Organic Search", "Direct", "Social Referral", "Other"]:
                sub = monthly_src[monthly_src["category"] == cat]
                fig_src.add_trace(go.Bar(
                    x=sub["month"].dt.strftime("%b %Y"), y=sub["visits"],
                    name=cat, marker_color=src_colors.get(cat, "#999")
                ))
            fig_src.update_layout(
                barmode="stack",
                title="Monthly Visits by Traffic Source",
                xaxis_title="Month", yaxis_title="Visits",
                height=380
            )
            st.plotly_chart(fig_src, use_container_width=True)
            st.caption("Traffic source breakdown shows how users find USDA services. Over-reliance on Organic Search (Google) creates vulnerability if search rankings change.")

            # Google dependency KPI
            total_traffic = traffic_df["visits"].sum()
            google_visits = traffic_df[traffic_df["source"].str.lower() == "google"]["visits"].sum()
            g_pct = google_visits / total_traffic * 100 if total_traffic else 0
            g_label = "Low" if g_pct < 40 else ("Moderate" if g_pct <= 60 else "High")
            g_color = "green" if g_pct < 40 else ("orange" if g_pct <= 60 else "red")

            social_visits = traffic_df[traffic_df.get("has_social_referral", pd.Series(["No"] * len(traffic_df))) == "Yes"]["visits"].sum()
            social_pct = social_visits / total_traffic * 100 if total_traffic else 0

            col_g, col_s = st.columns(2)
            col_g.markdown(f"""
**Google Dependency**
<div style='font-size:2em; font-weight:bold; color:{g_color}'>{g_pct:.1f}% — {g_label}</div>
<small>Below 40% = Low | 40–60% = Moderate | Above 60% = High</small>
""", unsafe_allow_html=True)
            col_s.markdown(f"""
**Social Referral Share**
<div style='font-size:2em; font-weight:bold'>{social_pct:.2f}%</div>
<small>Of all traffic arrives via social platforms</small>
""", unsafe_allow_html=True)

        # ── Content Demand ─────────────────────────────────────────────────────
        with subtabs[2]:
            st.subheader("Content Demand — Top Downloads")

            downloads_df["filename"] = downloads_df["event_label"].apply(
                lambda x: str(x).split("/")[-1] if pd.notna(x) else x
            )
            downloads_df["hostname"] = downloads_df["page"].apply(
                lambda x: str(x).split("/")[0] if pd.notna(x) else x
            )

            top20 = (downloads_df.groupby(["filename", "hostname"])["total_events"]
                     .sum().reset_index()
                     .nlargest(20, "total_events"))

            fig_dl = go.Figure(go.Bar(
                x=top20["total_events"],
                y=top20["filename"] + "  (" + top20["hostname"] + ")",
                orientation="h",
                marker_color="#5e35b1",
                text=top20["total_events"].apply(lambda v: f"{v:,.0f}"),
                textposition="outside"
            ))
            fig_dl.update_layout(
                title="Top 20 Downloaded Files (Jan–Jun)",
                xaxis_title="Total Download Events", yaxis_title="",
                yaxis=dict(autorange="reversed"),
                height=600, margin=dict(l=420)
            )
            st.plotly_chart(fig_dl, use_container_width=True)
            st.caption("These files represent the highest-demand documents across USDA. High download volume indicates strong user need — ensure these files are current, accessible, and easily found.")

            # Monthly downloads bar
            monthly_dl = downloads_df.groupby("month")["total_events"].sum().reset_index()
            fig_mdl = go.Figure(go.Bar(
                x=monthly_dl["month"].dt.strftime("%b %Y"),
                y=monthly_dl["total_events"],
                marker_color="#7b1fa2"
            ))
            fig_mdl.update_layout(
                title="Monthly Total Download Events",
                xaxis_title="Month", yaxis_title="Download Events",
                height=300
            )
            st.plotly_chart(fig_mdl, use_container_width=True)
            st.caption("Monthly download volume shows seasonal demand for USDA documents. Spikes often coincide with program deadlines, reporting periods, or policy updates.")

            # Searchable table
            peak_month = (downloads_df.groupby(["filename", "hostname", "month"])["total_events"]
                          .sum().reset_index()
                          .sort_values("total_events", ascending=False)
                          .groupby(["filename", "hostname"])
                          .first()
                          .reset_index()[["filename", "hostname", "month"]])
            peak_month["month"] = peak_month["month"].dt.strftime("%b %Y")

            table_df = (downloads_df.groupby(["filename", "hostname"])["total_events"]
                        .sum().reset_index()
                        .merge(peak_month, on=["filename", "hostname"])
                        .nlargest(20, "total_events")
                        .rename(columns={"total_events": "Total Events", "month": "Peak Month",
                                         "filename": "Filename", "hostname": "Hostname"}))

            search_q = st.text_input("Search downloads table", placeholder="Filter by filename or hostname…")
            filtered = table_df[
                table_df["Filename"].str.contains(search_q, case=False, na=False) |
                table_df["Hostname"].str.contains(search_q, case=False, na=False)
            ] if search_q else table_df
            st.dataframe(filtered, use_container_width=True, hide_index=True)

        # ── Language Reach ─────────────────────────────────────────────────────
        with subtabs[3]:
            st.subheader("Language Reach")
            st.info("Note: Browser language is a proxy for language preference, not geography. Users may use a non-English browser while located in the United States.")

            def is_english(lang):
                return str(lang).lower().startswith("en")

            language_df["is_english"] = language_df["language"].apply(is_english)
            total_lang = language_df["visits"].sum()
            non_eng = language_df[~language_df["is_english"]]["visits"].sum()
            non_eng_pct = non_eng / total_lang * 100 if total_lang else 0

            st.metric("Non-English Browser Share (Full Period)", f"{non_eng_pct:.1f}%")

            # Monthly non-English trend
            monthly_lang = language_df.groupby(["month", "is_english"])["visits"].sum().reset_index()
            monthly_total_lang = monthly_lang.groupby("month")["visits"].sum().rename("total")
            monthly_non_eng = monthly_lang[~monthly_lang["is_english"]].set_index("month")["visits"]
            non_eng_monthly = (monthly_non_eng / monthly_total_lang * 100).reset_index()
            non_eng_monthly.columns = ["month", "pct"]
            non_eng_monthly = non_eng_monthly.sort_values("month")

            fig_lang_trend = go.Figure(go.Scatter(
                x=non_eng_monthly["month"], y=non_eng_monthly["pct"],
                mode="lines+markers", line=dict(color="#e53935")
            ))
            fig_lang_trend.update_layout(
                title="Monthly Non-English Browser Share (%)",
                xaxis_title="Month", yaxis_title="Non-English Share (%)",
                height=300
            )
            st.plotly_chart(fig_lang_trend, use_container_width=True)
            st.caption("Rising non-English share signals growing demand for multilingual content. USDA should ensure key program pages are available in Spanish, Chinese, and other high-demand languages.")

            # Top 10 non-English
            top10_lang = (language_df[~language_df["is_english"]]
                          .groupby("language")["visits"].sum()
                          .reset_index()
                          .nlargest(10, "visits"))

            fig_lang_bar = go.Figure(go.Bar(
                x=top10_lang["visits"],
                y=top10_lang["language"],
                orientation="h",
                marker_color="#e53935"
            ))
            fig_lang_bar.update_layout(
                title="Top 10 Non-English Browser Languages by Total Visits",
                xaxis_title="Visits", yaxis_title="Language Code",
                yaxis=dict(autorange="reversed"),
                height=350
            )
            st.plotly_chart(fig_lang_bar, use_container_width=True)
            st.caption("Spanish-language variants (es-*) and Chinese (zh-*) dominate non-English usage. These communities should be prioritized in multilingual content strategies.")

        # ── Friction Flags ─────────────────────────────────────────────────────
        with subtabs[4]:
            st.subheader("Friction Flags")

            traffic_df_c = traffic_df.copy()
            traffic_df_c["category"] = traffic_df_c.apply(classify_source, axis=1)
            t_total = traffic_df_c["visits"].sum()
            g_v = traffic_df_c[traffic_df_c["source"].str.lower() == "google"]["visits"].sum()
            g_p = g_v / t_total * 100 if t_total else 0
            g_lbl = "Low" if g_p < 40 else ("Moderate" if g_p <= 60 else "High")

            st.markdown(f"""
### Google Dependency Risk
**{g_p:.1f}% of all USDA web traffic originates from Google Search — rated: {g_lbl}**

{"This level of Google dependency represents a HIGH RISK. Any change in Google's algorithm or ranking could significantly disrupt public access to USDA services." if g_p > 60 else "Moderate dependency. Diversifying traffic sources through direct links, email newsletters, and partnerships would reduce reliance on search rankings." if g_p > 40 else "Low dependency. Traffic is well-diversified across sources."}
""")
            st.markdown("---")

            pcts = mobile_monthly_pct["mobile_pct"].tolist()
            rising = 0
            for i in range(1, len(pcts)):
                if pcts[i] > pcts[i - 1]:
                    rising += 1
                else:
                    rising = 0
            st.markdown("### Mobile Trend")
            if rising >= 3:
                st.warning(f"Mobile share has risen for {rising} consecutive months. USDA services must prioritize mobile-responsive design and faster page load times for mobile users.")
            else:
                st.success("Mobile share is not showing a sustained upward streak at this time.")

            st.markdown("---")
            st.markdown("""
### Legacy Browser Note
Analysis of browser data (os-browser CSV) includes traffic from older Windows versions and Internet Explorer/Edge Legacy. These users may experience degraded functionality on modern web applications. USDA web teams should audit critical service pages for compatibility with legacy environments.
""")

    except Exception as e:
        st.error(f"Error loading system-wide data: {e}")
        st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — Rural Development Baseline
# ─────────────────────────────────────────────────────────────────────────────

with tabs[1]:
    st.header("Layer 2: Rural Development Descriptive Foundation")
    st.caption("How is the Rural Development site performing, and for whom is it performing worst?")

    if not rd_available:
        st.warning("Please upload the Rural Development CSV file in the sidebar to view this analysis.")
    else:
        try:
            page_df, scaled, feature_cols, daily_df = load_rd_data(rd_source)

            # KPI row
            total_users_rd = page_df["Total Users"].sum() if "Total Users" in page_df else 0
            mean_bounce = page_df["Bounce Rate"].mean()
            mean_dur = page_df["Session Duration"].mean()
            mean_views = page_df["Views per Session"].mean()

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total RD Users", f"{total_users_rd:,.0f}")
            k2.metric("Mean Bounce Rate", f"{mean_bounce:.1%}")
            k3.metric("Mean Session Duration", f"{mean_dur:.0f}s")
            k4.metric("Mean Views per Session", f"{mean_views:.2f}")

            st.markdown("---")

            # Heatmap: section × metric
            st.subheader("Section-Level Performance Heatmap")
            heatmap_metrics = ["Bounce Rate", "Session Duration", "Views per Session",
                               "Exit Pressure Index", "Stickiness Ratio"]
            heatmap_data = page_df.groupby("Section")[heatmap_metrics].mean().dropna(how="all")
            heatmap_data = heatmap_data[heatmap_data.index.notna()]

            # Normalize each column 0-1 for coloring (invert where lower = better)
            norm = heatmap_data.copy()
            # For these metrics: lower bounce/exit = better; higher duration/views/stickiness = better
            invert = {"Bounce Rate": True, "Exit Pressure Index": True}
            for col in heatmap_metrics:
                mn, mx = norm[col].min(), norm[col].max()
                if mx > mn:
                    norm[col] = (norm[col] - mn) / (mx - mn)
                    if col in invert:
                        norm[col] = 1 - norm[col]
                else:
                    norm[col] = 0.5

            annotations = []
            for i, row_idx in enumerate(heatmap_data.index):
                for j, col in enumerate(heatmap_metrics):
                    val = heatmap_data.loc[row_idx, col]
                    fmt = f"{val:.2f}" if abs(val) < 10 else f"{val:.0f}"
                    annotations.append(dict(x=j, y=i, text=fmt, showarrow=False,
                                            font=dict(size=9, color="black")))

            fig_heat = go.Figure(go.Heatmap(
                z=norm.values,
                x=heatmap_metrics,
                y=list(heatmap_data.index),
                colorscale="RdYlGn",
                zmin=0, zmax=1,
                showscale=True,
                colorbar=dict(title="Performance<br>(Green = Good)")
            ))
            fig_heat.update_layout(
                title="Section Performance Across Five Engagement Metrics",
                xaxis_title="Metric", yaxis_title="Site Section",
                height=max(400, len(heatmap_data) * 28),
                annotations=annotations
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption("Green cells indicate strong performance; red cells flag areas of concern. Sections with red across multiple metrics require prioritized content or UX intervention.")

            st.markdown("---")

            # Grouped bar: bounce rate & session duration by device
            st.subheader("Engagement by Device Type")

            # Use daily_df which has device-level columns
            device_metrics = {}
            for dev, col_offset in [("Desktop", 11), ("Mobile", 20), ("Tablet", 29)]:
                br_col = daily_df.columns[col_offset] if col_offset < len(daily_df.columns) else None
                dur_col = daily_df.columns[col_offset - 1] if col_offset - 1 < len(daily_df.columns) else None
                if br_col and dur_col:
                    device_metrics[dev] = {
                        "Bounce Rate": pd.to_numeric(daily_df[br_col], errors="coerce").mean(),
                        "Session Duration": pd.to_numeric(daily_df[dur_col], errors="coerce").mean(),
                    }

            if device_metrics:
                devices = list(device_metrics.keys())
                br_vals  = [device_metrics[d]["Bounce Rate"] for d in devices]
                dur_vals = [device_metrics[d]["Session Duration"] for d in devices]

                fig_dev = go.Figure()
                fig_dev.add_trace(go.Bar(name="Mean Bounce Rate", x=devices, y=br_vals, marker_color="#ef5350"))
                fig_dev.add_trace(go.Bar(name="Mean Session Duration (s)", x=devices, y=dur_vals, marker_color="#42a5f5"))
                fig_dev.update_layout(
                    barmode="group",
                    title="Mean Bounce Rate and Session Duration by Device Type",
                    xaxis_title="Device", yaxis_title="Value",
                    height=380
                )
                st.plotly_chart(fig_dev, use_container_width=True)
                st.caption("Mobile users often show higher bounce rates and shorter sessions — indicating friction in mobile content delivery. Gaps between desktop and mobile performance highlight priority areas for mobile optimization.")

            st.markdown("---")

            # Monthly engagement trend line chart
            st.subheader("Monthly Engagement Trends")
            if "Month" in daily_df.columns and "Bounce Rate" in daily_df.columns:
                monthly_trend = daily_df.groupby("Month").agg(
                    Bounce_Rate=("Bounce Rate", "mean"),
                    Session_Duration=("Session Duration", "mean"),
                    Active_Users=("Active Users", "sum")
                ).reset_index().dropna(subset=["Month"])
                monthly_trend = monthly_trend[monthly_trend["Month"].between(1, 12)]
                month_labels = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May",
                                6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct",
                                11: "Nov", 12: "Dec"}
                monthly_trend["Month Label"] = monthly_trend["Month"].map(month_labels)
                monthly_trend = monthly_trend.sort_values("Month")

                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=monthly_trend["Month Label"], y=monthly_trend["Bounce_Rate"],
                                               mode="lines+markers", name="Bounce Rate", line=dict(color="#ef5350")))
                fig_trend.add_trace(go.Scatter(x=monthly_trend["Month Label"], y=monthly_trend["Session_Duration"],
                                               mode="lines+markers", name="Session Duration (s)", yaxis="y2",
                                               line=dict(color="#42a5f5")))
                fig_trend.add_trace(go.Scatter(x=monthly_trend["Month Label"], y=monthly_trend["Active_Users"],
                                               mode="lines+markers", name="Active Users", yaxis="y3",
                                               line=dict(color="#66bb6a", dash="dot")))
                fig_trend.update_layout(
                    title="Monthly Engagement Trends — Bounce Rate, Session Duration, Active Users",
                    xaxis_title="Month",
                    yaxis=dict(title="Bounce Rate", side="left"),
                    yaxis2=dict(title="Session Duration (s)", overlaying="y", side="right"),
                    yaxis3=dict(title="Active Users", overlaying="y", side="right", anchor="free", position=1.0),
                    height=420,
                    legend=dict(x=0, y=1.15, orientation="h")
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                st.caption("Monthly trends reveal whether engagement is improving or declining over the program year. Declining session duration alongside rising bounce rates suggests worsening content relevance or site friction.")

            st.markdown("---")

            # Service tier donut
            st.subheader("Service Tier Classification")
            pct75 = page_df["Underserved Score"].quantile(0.75)
            pct25 = page_df["Underserved Score"].quantile(0.25)

            def tier(score):
                if score >= pct75:
                    return "Underserved"
                elif score <= pct25:
                    return "Well-Served"
                else:
                    return "Moderately Served"

            page_df["Service Tier"] = page_df["Underserved Score"].apply(tier)
            tier_counts = page_df["Service Tier"].value_counts()

            fig_donut = go.Figure(go.Pie(
                labels=tier_counts.index.tolist(),
                values=tier_counts.values.tolist(),
                hole=0.45,
                marker_colors=["#ef5350", "#ffb300", "#66bb6a"]
            ))
            fig_donut.update_layout(
                title="Page Service Tier Distribution (Underserved Score)",
                height=380
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            tier_pct = (tier_counts / tier_counts.sum() * 100).round(1)
            for t in ["Well-Served", "Moderately Served", "Underserved"]:
                if t in tier_counts:
                    st.markdown(f"- **{t}**: {tier_counts[t]} pages ({tier_pct[t]}%)")
            st.caption("Underserved pages have high bounce rates, high exit pressure, and low return visitor engagement. These pages represent the highest-priority targets for content improvement and UX redesign.")

        except Exception as e:
            st.error(f"Error processing Rural Development data: {e}")
            st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — K-Means Clustering
# ─────────────────────────────────────────────────────────────────────────────

with tabs[2]:
    st.header("Layer 3: K-Means Clustering — Page-Level Behavioral Segmentation")
    st.caption("Identifies distinct patterns of page performance to surface both high-friction and high-performing content areas.")

    if not rd_available:
        st.warning("Please upload the Rural Development CSV file in the sidebar to view this analysis.")
    else:
        try:
            page_df, scaled, feature_cols, _ = load_rd_data(rd_source)

            # Section A — Controls
            st.subheader("Clustering Controls")
            st.info("All six features (Bounce Rate, Session Duration, Views per Session, Exit Pressure Index, Stickiness Ratio, Device Gap Score) are standardized using Z-score normalization before clustering.")

            col_k, col_btn = st.columns([3, 1])
            with col_k:
                k_val = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=4)
            with col_btn:
                run_btn = st.button("Run Clustering", type="primary")

            if run_btn or "cluster_labels" in st.session_state:
                if run_btn:
                    # Compute diagnostics
                    wcss, sil_scores = [], []
                    k_range = range(2, 11)
                    for k in k_range:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        km.fit(scaled)
                        wcss.append(km.inertia_)
                        sil_scores.append(silhouette_score(scaled, km.labels_))

                    km_final = KMeans(n_clusters=k_val, random_state=42, n_init=10)
                    labels = km_final.fit_predict(scaled)
                    page_df = page_df.copy()
                    page_df["Cluster"] = labels

                    # Map clusters to service tier labels by mean underserved score
                    cluster_scores = page_df.groupby("Cluster")["Underserved Score"].mean().sort_values()
                    tier_names = ["Power User", "Core Program", "Discovery", "High Friction"]
                    cluster_name_map = {}
                    sorted_clusters = list(cluster_scores.index)
                    for i, c in enumerate(sorted_clusters):
                        cluster_name_map[c] = tier_names[i] if i < len(tier_names) else f"Cluster {c}"
                    page_df["Cluster Label"] = page_df["Cluster"].map(cluster_name_map)

                    st.session_state["cluster_labels"] = labels
                    st.session_state["page_df_clustered"] = page_df
                    st.session_state["wcss"] = wcss
                    st.session_state["sil_scores"] = sil_scores
                    st.session_state["k_val"] = k_val
                    st.session_state["sil_at_k"] = sil_scores[k_val - 2]

                page_df = st.session_state["page_df_clustered"]
                wcss = st.session_state["wcss"]
                sil_scores = st.session_state["sil_scores"]
                k_val = st.session_state["k_val"]
                sil_at_k = st.session_state["sil_at_k"]
                k_range = list(range(2, 11))

                # Section B — Diagnostics
                st.subheader("Clustering Diagnostics")
                dc1, dc2 = st.columns(2)

                with dc1:
                    fig_elbow = go.Figure()
                    fig_elbow.add_trace(go.Scatter(x=k_range, y=wcss, mode="lines+markers", name="WCSS"))
                    fig_elbow.add_vline(x=k_val, line_dash="dash", line_color="red",
                                        annotation_text=f"k={k_val}", annotation_position="top right")
                    fig_elbow.update_layout(title="Elbow Method (WCSS vs k)",
                                            xaxis_title="Number of Clusters (k)",
                                            yaxis_title="Within-Cluster Sum of Squares",
                                            height=320)
                    st.plotly_chart(fig_elbow, use_container_width=True)

                with dc2:
                    peak_k = k_range[int(np.argmax(sil_scores))]
                    peak_sil = max(sil_scores)
                    fig_sil = go.Figure()
                    fig_sil.add_trace(go.Scatter(x=k_range, y=sil_scores, mode="lines+markers", name="Silhouette"))
                    fig_sil.add_vline(x=k_val, line_dash="dash", line_color="red",
                                      annotation_text=f"k={k_val}", annotation_position="top right")
                    fig_sil.add_annotation(x=peak_k, y=peak_sil,
                                           text=f"Peak: {peak_sil:.3f} at k={peak_k}",
                                           showarrow=True, arrowhead=2)
                    fig_sil.update_layout(title="Silhouette Score vs k",
                                          xaxis_title="Number of Clusters (k)",
                                          yaxis_title="Silhouette Score",
                                          height=320)
                    st.plotly_chart(fig_sil, use_container_width=True)

                st.info(f"Based on the diagnostics, k={k_val} was selected. Silhouette score at selected k: {sil_at_k:.4f}")

                # Section C — Cluster Summary Table
                st.subheader("Cluster Summary")
                CLUSTER_COLORS = {
                    "High Friction": "#ffcccc",
                    "Discovery": "#ffe0b2",
                    "Core Program": "#bbdefb",
                    "Power User": "#c8e6c9",
                }

                summary = page_df.groupby("Cluster Label").agg(
                    Pages=("Page title", "count"),
                    Bounce_Rate=("Bounce Rate", "mean"),
                    Session_Duration=("Session Duration", "mean"),
                    Views_per_Session=("Views per Session", "mean"),
                    Exit_Pressure=("Exit Pressure Index", "mean"),
                    Stickiness=("Stickiness Ratio", "mean"),
                    Device_Gap=("Device Gap Score", "mean"),
                ).reset_index().rename(columns={
                    "Cluster Label": "Cluster", "Bounce_Rate": "Bounce Rate",
                    "Session_Duration": "Session Duration (s)",
                    "Views_per_Session": "Views/Session",
                    "Exit_Pressure": "Exit Pressure",
                    "Stickiness": "Stickiness Ratio",
                    "Device_Gap": "Device Gap"
                })

                def color_row(row):
                    color = CLUSTER_COLORS.get(row["Cluster"], "#ffffff")
                    return [f"background-color: {color}"] * len(row)

                styled_sum = summary.style.apply(color_row, axis=1).format({
                    "Bounce Rate": "{:.2%}",
                    "Session Duration (s)": "{:.0f}",
                    "Views/Session": "{:.2f}",
                    "Exit Pressure": "{:.3f}",
                    "Stickiness Ratio": "{:.3f}",
                    "Device Gap": "{:.3f}",
                })
                st.dataframe(styled_sum, use_container_width=True, hide_index=True)
                st.caption("Each cluster represents a distinct behavioral pattern. High Friction pages need immediate attention; Power User pages represent best-in-class content worth replicating.")

                # Section D — 2D PCA
                st.subheader("2D PCA Visualization")
                pca2 = PCA(n_components=2, random_state=42)
                coords2 = pca2.fit_transform(scaled)
                page_df["PC1"] = coords2[:, 0]
                page_df["PC2"] = coords2[:, 1]

                color_seq = px.colors.qualitative.Set2
                cluster_labels_list = page_df["Cluster Label"].unique().tolist()
                color_map = {lbl: color_seq[i % len(color_seq)] for i, lbl in enumerate(cluster_labels_list)}

                fig_2d = px.scatter(
                    page_df, x="PC1", y="PC2", color="Cluster Label",
                    hover_data={"Page title": True, "Section": True,
                                "Bounce Rate": ":.2%", "Session Duration": ":.0f", "PC1": False, "PC2": False},
                    title="Page Clusters in 2D PCA Space",
                    color_discrete_map=color_map,
                    height=500
                )
                fig_2d.update_layout(xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")
                st.plotly_chart(fig_2d, use_container_width=True)
                st.caption("Note: PCA is used for visualization only. Clustering was performed on the full 6-feature space. Each point represents a unique page; nearby points share similar behavioral characteristics.")

                # Section E — 3D PCA
                st.subheader("3D PCA Visualization (Interactive — click and drag to rotate)")
                pca3 = PCA(n_components=3, random_state=42)
                coords3 = pca3.fit_transform(scaled)
                page_df["PC3"] = coords3[:, 2]
                var_exp = pca3.explained_variance_ratio_

                fig_3d = px.scatter_3d(
                    page_df, x="PC1", y="PC2", z="PC3", color="Cluster Label",
                    hover_data={"Page title": True, "Section": True,
                                "Bounce Rate": ":.2%", "Session Duration": ":.0f"},
                    title="Page Clusters in 3D PCA Space",
                    color_discrete_map=color_map,
                    height=600
                )
                fig_3d.update_layout(scene=dict(
                    xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"
                ))
                st.plotly_chart(fig_3d, use_container_width=True)
                st.caption(
                    f"Variance explained — PC1: {var_exp[0]:.1%} | PC2: {var_exp[1]:.1%} | PC3: {var_exp[2]:.1%}. "
                    "Rotate the chart to explore cluster separation from multiple angles."
                )

                # Section F — Radar Chart
                st.subheader("Cluster Behavioral Profiles — Radar Chart")
                radar_features = feature_cols
                cluster_means = page_df.groupby("Cluster Label")[radar_features].mean()

                # Normalize to 0-1 for radar
                cm_norm = cluster_means.copy()
                for col in radar_features:
                    mn, mx = cm_norm[col].min(), cm_norm[col].max()
                    if mx > mn:
                        cm_norm[col] = (cm_norm[col] - mn) / (mx - mn)

                radar_labels = ["Bounce Rate", "Session Duration", "Views/Session",
                                "Exit Pressure", "Stickiness", "Device Gap"]

                fig_radar = go.Figure()
                for i, (lbl, row) in enumerate(cm_norm.iterrows()):
                    vals = list(row.values) + [row.values[0]]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals,
                        theta=radar_labels + [radar_labels[0]],
                        fill="toself",
                        name=lbl,
                        line_color=color_seq[i % len(color_seq)]
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Normalized Cluster Profiles Across All 6 Features",
                    height=500,
                    showlegend=True
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                st.caption("Each line traces a cluster's relative behavioral signature. Clusters that spike on Bounce Rate and Exit Pressure while low on Stickiness are the highest-priority improvement targets.")

                # Section G — Underserved Page Inventory
                st.subheader("Underserved Page Inventory")
                pct75_score = page_df["Underserved Score"].quantile(0.75)
                underserved = page_df[page_df["Underserved Score"] >= pct75_score].copy()
                underserved = underserved.sort_values("Underserved Score", ascending=False)

                sections = ["All"] + sorted(underserved["Section"].dropna().unique().tolist())
                sel_section = st.selectbox("Filter by section", sections, key="underserved_section")
                if sel_section != "All":
                    underserved = underserved[underserved["Section"] == sel_section]

                display_cols = ["Page title", "Section", "Cluster Label", "Bounce Rate",
                                "Session Duration", "Exit Pressure Index", "Stickiness Ratio",
                                "Device Gap Score", "Underserved Score"]
                display_cols = [c for c in display_cols if c in underserved.columns]
                st.dataframe(
                    underserved[display_cols].rename(columns={"Cluster Label": "Cluster"}),
                    use_container_width=True, hide_index=True
                )
                st.caption(f"Showing {len(underserved)} pages with Underserved Score above the 75th percentile ({pct75_score:.3f}). These pages are the highest-priority targets for content improvement.")

        except Exception as e:
            st.error(f"Error in clustering analysis: {e}")
            st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4 — Behavioral Signatures
# ─────────────────────────────────────────────────────────────────────────────

with tabs[3]:
    st.header("Layer 4: Behavioral Signatures")
    st.caption(
        "This tab characterizes pages by their aggregate behavioral signature. "
        "**Signatures are page-level patterns derived from aggregate metrics — they do not represent individual user journeys or infer personal behavior.**"
    )

    if not rd_available:
        st.warning("Please upload the Rural Development CSV file in the sidebar to view this analysis.")
    else:
        try:
            page_df, scaled, feature_cols, _ = load_rd_data(rd_source)

            # Section A — Signature Definitions
            st.subheader("Signature Definitions")
            sig_defs = {
                "Deep Engagement": "Moderate bounce rate (30–55%), healthy session duration (>120s), multi-page visits. Users are exploring content with intent.",
                "Repeat Reference": "High stickiness ratio (>0.3), moderate duration (60–300s). Professional or returning users who regularly access the same content.",
                "High Abandonment": "Bounce rate above 55%, short session duration (<60s). Indicates a content-user mismatch — visitors are not finding what they need.",
                "Quick Task Completion": "Low bounce rate (<30%), short session duration. Users successfully completed a specific task and left — a positive signal.",
                "Frustrated Navigation": "Long session duration (>300s) combined with high exit pressure (>0.6). Users are searching for something they cannot find.",
            }
            sig_colors_health = {
                "Deep Engagement": "#c8e6c9",
                "Repeat Reference": "#bbdefb",
                "High Abandonment": "#ffcdd2",
                "Quick Task Completion": "#dcedc8",
                "Frustrated Navigation": "#ffe0b2",
            }

            cols_sig = st.columns(3)
            for i, (sig, desc) in enumerate(sig_defs.items()):
                with cols_sig[i % 3]:
                    bg = sig_colors_health[sig]
                    st.markdown(f"""
<div style="background:{bg}; border-radius:8px; padding:12px; margin-bottom:10px">
<strong>{sig}</strong><br><small>{desc}</small>
</div>""", unsafe_allow_html=True)

            st.markdown("---")

            # Section B — Signature Scatter Plot
            st.subheader("Session Duration vs Views per Session by Signature")
            avg_dur = page_df["Session Duration"].mean()
            avg_views = page_df["Views per Session"].mean()

            sig_color_map = {
                "Deep Engagement": "#4CAF50",
                "Repeat Reference": "#2196F3",
                "High Abandonment": "#F44336",
                "Quick Task Completion": "#8BC34A",
                "Frustrated Navigation": "#FF9800",
                "Unknown": "#9E9E9E",
            }

            fig_scatter = px.scatter(
                page_df,
                x="Session Duration", y="Views per Session",
                color="Behavioral Signature",
                color_discrete_map=sig_color_map,
                hover_data={"Page title": True, "Section": True,
                            "Bounce Rate": ":.2%", "Session Duration": ":.0f",
                            "Behavioral Signature": True},
                title="Behavioral Signature Map: Session Duration vs Views per Session",
                height=520
            )
            fig_scatter.add_vline(x=avg_dur, line_dash="dash", line_color="gray",
                                  annotation_text=f"Avg Duration: {avg_dur:.0f}s")
            fig_scatter.add_hline(y=avg_views, line_dash="dash", line_color="gray",
                                  annotation_text=f"Avg Views: {avg_views:.2f}")
            fig_scatter.update_layout(xaxis_title="Session Duration (seconds)",
                                       yaxis_title="Views per Session")
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("Pages in the upper-right quadrant (long sessions, many views) represent deep engagement — content that holds attention. Pages in the lower-left may indicate either quick task success or abandonment depending on bounce rate.")

            st.markdown("---")

            # Section C — Signature Distribution
            st.subheader("Signature Distribution")
            sig_counts = page_df["Behavioral Signature"].value_counts().reset_index()
            sig_counts.columns = ["Signature", "Count"]
            sig_counts = sig_counts.sort_values("Count", ascending=True)

            health_colors = {
                "Deep Engagement": "#4CAF50",
                "Repeat Reference": "#2196F3",
                "High Abandonment": "#F44336",
                "Quick Task Completion": "#8BC34A",
                "Frustrated Navigation": "#FF9800",
                "Unknown": "#9E9E9E",
            }

            fig_dist = go.Figure(go.Bar(
                x=sig_counts["Count"],
                y=sig_counts["Signature"],
                orientation="h",
                marker_color=[health_colors.get(s, "#9E9E9E") for s in sig_counts["Signature"]]
            ))
            fig_dist.update_layout(
                title="Number of Pages per Behavioral Signature",
                xaxis_title="Page Count", yaxis_title="",
                height=350
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            st.caption("The distribution of signatures reveals the overall health of the Rural Development site. A large proportion of High Abandonment or Frustrated Navigation pages signals systemic content or navigation issues requiring structural improvements.")

            st.markdown("---")

            # Section D — Signature × Section Breakdown
            st.subheader("Signature Composition by Section")
            sig_section = page_df.groupby(["Section", "Behavioral Signature"]).size().reset_index(name="Count")

            fig_stack = px.bar(
                sig_section, x="Section", y="Count", color="Behavioral Signature",
                color_discrete_map=health_colors,
                title="Behavioral Signatures Within Each Site Section",
                height=480,
                barmode="stack"
            )
            fig_stack.update_layout(xaxis_title="Site Section", yaxis_title="Number of Pages",
                                     xaxis=dict(tickangle=-35))
            st.plotly_chart(fig_stack, use_container_width=True)
            st.caption("Sections dominated by red (High Abandonment) or orange (Frustrated Navigation) bars are structurally underperforming. These sections should be prioritized for content audits and information architecture review.")

            st.markdown("---")

            # Section E — Signature Detail Table
            st.subheader("Signature Detail Table")
            sel_sig = st.selectbox(
                "Select a behavioral signature to view its pages",
                options=sorted(page_df["Behavioral Signature"].unique().tolist()),
                key="sig_selectbox"
            )
            sig_detail = page_df[page_df["Behavioral Signature"] == sel_sig]
            sig_display_cols = ["Page title", "Section", "Bounce Rate", "Session Duration",
                                 "Views per Session", "Exit Pressure Index", "Stickiness Ratio",
                                 "Device Gap Score"]
            sig_display_cols = [c for c in sig_display_cols if c in sig_detail.columns]
            st.dataframe(
                sig_detail[sig_display_cols].sort_values("Bounce Rate", ascending=False),
                use_container_width=True, hide_index=True
            )
            st.caption(f"Showing all {len(sig_detail)} pages classified as '{sel_sig}'. Use this table to identify specific pages for content review or UX improvement initiatives.")

        except Exception as e:
            st.error(f"Error in behavioral signatures analysis: {e}")
            st.exception(e)

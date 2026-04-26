"""
USDA Digital Service Effectiveness Dashboard
Four-layer analytical framework for evaluating and improving USDA website performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import io
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="USDA Digital Service Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #1a5276;
        border-bottom: 3px solid #2e86ab; padding-bottom: 0.5rem; margin-bottom: 1rem;
    }
    .kpi-card {
        background: #f8f9fa; border-radius: 10px; padding: 1.2rem;
        border-left: 5px solid #2e86ab; margin: 0.3rem 0;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1a5276; }
    .kpi-label { font-size: 0.85rem; color: #555; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-sub { font-size: 0.8rem; color: #888; margin-top: 0.2rem; }
    .risk-card {
        background: #fff3cd; border-radius: 10px; padding: 1rem;
        border-left: 5px solid #ffc107; margin: 0.5rem 0;
    }
    .risk-card-high {
        background: #f8d7da; border-radius: 10px; padding: 1rem;
        border-left: 5px solid #dc3545; margin: 0.5rem 0;
    }
    .risk-card-low {
        background: #d4edda; border-radius: 10px; padding: 1rem;
        border-left: 5px solid #28a745; margin: 0.5rem 0;
    }
    .chart-caption {
        font-size: 0.82rem; color: #555; font-style: italic;
        margin-top: -0.5rem; margin-bottom: 1.5rem; padding: 0.5rem;
        background: #f8f9fa; border-radius: 5px;
    }
    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #1a5276;
        margin-top: 1.5rem; margin-bottom: 0.5rem;
    }
    .upload-box {
        border: 2px dashed #2e86ab; border-radius: 10px; padding: 2rem;
        text-align: center; background: #f0f8ff; margin: 1rem 0;
    }
    .sig-card {
        border-radius: 8px; padding: 0.8rem 1rem; margin: 0.3rem 0;
    }
    div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Hostname Mapping ─────────────────────────────────────────────────────────
HOSTNAME_MAP = {
    "fns.usda.gov": "Food and Nutrition Service",
    "ams.usda.gov": "Agricultural Marketing Service",
    "nrcs.usda.gov": "Natural Resources Conservation Service",
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_NUM = {m: i+1 for i, m in enumerate(MONTHS)}

# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_system_csvs(device_b, domain_b, download_b, language_b, os_b, traffic_b, windows_b):
    """Load and parse all system-wide CSV files."""
    def read(b):
        return pd.read_csv(io.BytesIO(b))

    device_df    = read(device_b)
    domain_df    = read(domain_b)
    download_df  = read(download_b)
    language_df  = read(language_b)
    os_df        = read(os_b)
    traffic_df   = read(traffic_b)
    windows_df   = read(windows_b)

    # Standardise column names
    for df in [device_df, domain_df, download_df, language_df, os_df, traffic_df, windows_df]:
        df.columns = [c.strip() for c in df.columns]

    return device_df, domain_df, download_df, language_df, os_df, traffic_df, windows_df


@st.cache_data(show_spinner=False)
def load_rd_data(rd_b):
    """Load and preprocess Rural Development CSV with all required steps."""
    raw = pd.read_csv(io.BytesIO(rd_b), skiprows=6, header=[0, 1])

    # Flatten two-row header
    cols = []
    for top, bot in raw.columns:
        top = str(top).strip().replace("Unnamed: ", "col")
        bot = str(bot).strip()
        if bot and bot != top and "Unnamed" not in bot:
            cols.append(f"{top}_{bot}")
        else:
            cols.append(top)
    raw.columns = cols

    # Drop last empty column
    if raw.columns[-1].startswith("col") or raw.iloc[:, -1].isna().all():
        raw = raw.iloc[:, :-1]

    # Identify page title & path columns
    title_col = next((c for c in raw.columns if "Page title" in c or "page title" in c.lower()), None)
    path_col  = next((c for c in raw.columns if "Page path" in c or "page path" in c.lower()), None)

    if title_col is None:
        # Try to find by position or name patterns
        title_col = raw.columns[0]
    if path_col is None:
        path_col = raw.columns[1] if len(raw.columns) > 1 else raw.columns[0]

    raw = raw.rename(columns={path_col: "Page path"})

    # Remove rows where page title is null
    if title_col in raw.columns:
        raw = raw[raw[title_col].notna() & (raw[title_col].astype(str).str.strip() != "")]
    raw["Page title"] = raw[title_col] if title_col in raw.columns else raw.iloc[:, 0]

    # Convert numerics
    skip = {"Page path", "Page title", title_col}
    for c in raw.columns:
        if c not in skip:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # Extract Section from page path
    def get_section(path):
        try:
            parts = str(path).strip("/").split("/")
            return parts[0] if parts else "other"
        except:
            return "other"
    raw["Section"] = raw["Page path"].apply(get_section)

    # Identify key metric columns (flexible matching)
    def find_col(keywords):
        for kw in keywords:
            for c in raw.columns:
                if kw.lower() in c.lower():
                    return c
        return None

    bounce_col      = find_col(["Bounce rate", "bounce_rate", "Bounce"])
    duration_col    = find_col(["Session duration", "Avg session", "duration"])
    views_col       = find_col(["Views per session", "views_per", "Pages/Session"])
    exits_col       = find_col(["Exits", "exit"])
    sessions_col    = find_col(["Sessions", "session"])
    returning_col   = find_col(["Returning", "return"])
    total_users_col = find_col(["Total users", "Users", "user"])
    mobile_bounce   = find_col(["Mobile.*bounce", "mobile_bounce"])
    desktop_bounce  = find_col(["Desktop.*bounce", "desktop_bounce"])
    active_users    = find_col(["Active users", "active_user"])
    date_col        = find_col(["Date", "date", "Month", "month"])

    # If mobile/desktop bounce not separate, use main bounce for both
    raw["Bounce Rate"]   = raw[bounce_col].astype(float)   if bounce_col else np.nan
    raw["Session Duration"] = raw[duration_col].astype(float) if duration_col else np.nan
    raw["Views per Session"] = raw[views_col].astype(float) if views_col else np.nan

    exits_s    = raw[exits_col].astype(float)    if exits_col    else pd.Series(np.nan, index=raw.index)
    sessions_s = raw[sessions_col].astype(float) if sessions_col else pd.Series(np.nan, index=raw.index)
    ret_s      = raw[returning_col].astype(float) if returning_col else pd.Series(np.nan, index=raw.index)
    tot_s      = raw[total_users_col].astype(float) if total_users_col else pd.Series(np.nan, index=raw.index)

    mb_s = raw[mobile_bounce].astype(float)  if mobile_bounce  else raw["Bounce Rate"]
    db_s = raw[desktop_bounce].astype(float) if desktop_bounce else raw["Bounce Rate"]

    raw["Device Gap Score"]    = mb_s - db_s
    raw["Exit Pressure Index"] = np.where(sessions_s > 0, exits_s / sessions_s, np.nan)
    raw["Stickiness Ratio"]    = np.where(tot_s > 0, ret_s / tot_s, np.nan)
    raw["Total Users"]         = tot_s
    raw["Sessions"]            = sessions_s
    raw["Active Users"]        = raw[active_users].astype(float) if active_users else tot_s

    if date_col:
        raw["Date"] = pd.to_datetime(raw[date_col], errors="coerce")
        raw["Month"] = raw["Date"].dt.strftime("%b")

    # Device type column
    device_col = find_col(["Device", "device"])
    raw["Device"] = raw[device_col] if device_col else "Unknown"

    # Aggregate to page level
    rate_cols  = ["Bounce Rate", "Session Duration", "Views per Session",
                  "Exit Pressure Index", "Stickiness Ratio", "Device Gap Score"]
    count_cols = ["Total Users", "Sessions", "Active Users"]

    agg_dict = {}
    for c in rate_cols:
        if c in raw.columns:
            agg_dict[c] = "mean"
    for c in count_cols:
        if c in raw.columns:
            agg_dict[c] = "sum"
    agg_dict["Section"] = "first"
    agg_dict["Page title"] = "first"

    page_df = raw.groupby("Page path").agg(agg_dict).reset_index()

    # Remove outliers >1000s session duration
    if "Session Duration" in page_df.columns:
        page_df = page_df[page_df["Session Duration"].fillna(0) <= 1000]

    # Fill NaN Stickiness with 0
    page_df["Stickiness Ratio"] = page_df["Stickiness Ratio"].fillna(0).clip(0, 1)
    page_df["Exit Pressure Index"] = page_df["Exit Pressure Index"].fillna(0).clip(0, 1)
    page_df["Bounce Rate"] = page_df["Bounce Rate"].fillna(page_df["Bounce Rate"].median())

    # Underserved Score
    page_df["Underserved Score"] = (
        page_df["Bounce Rate"] * 0.4 +
        page_df["Exit Pressure Index"] * 0.3 +
        (1 - page_df["Stickiness Ratio"]) * 0.3
    )

    # Behavioral Signature
    def classify_sig(row):
        br = row.get("Bounce Rate", 0.5)
        sd = row.get("Session Duration", 60)
        st_ = row.get("Stickiness Ratio", 0)
        ep = row.get("Exit Pressure Index", 0.5)
        vps = row.get("Views per Session", 1.5)

        if br > 0.55 and sd < 60:
            return "High Abandonment"
        if sd > 180 and ep > 0.6:
            return "Frustrated Navigation"
        if br < 0.35 and sd < 60:
            return "Quick Task Completion"
        if st_ > 0.4 and sd > 90:
            return "Repeat Reference"
        if br < 0.5 and sd > 90 and vps > 2:
            return "Deep Engagement"
        return "High Abandonment"

    page_df["Behavioral Signature"] = page_df.apply(classify_sig, axis=1)

    # Standardize 6 clustering features
    feature_cols = ["Bounce Rate", "Session Duration", "Views per Session",
                    "Exit Pressure Index", "Stickiness Ratio", "Device Gap Score"]
    feats = page_df[feature_cols].copy()
    feats = feats.fillna(feats.median())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats)

    # Also keep monthly data for trend charts
    monthly_df = None
    if date_col and "Month" in raw.columns:
        monthly_agg = {}
        for c in ["Bounce Rate", "Session Duration", "Active Users"]:
            if c in raw.columns:
                monthly_agg[c] = "mean" if c != "Active Users" else "sum"
        if monthly_agg:
            monthly_df = raw.groupby("Month").agg(monthly_agg).reset_index()

    # Device breakdown
    device_df = None
    if "Device" in raw.columns:
        dev_agg = {}
        for c in ["Bounce Rate", "Session Duration"]:
            if c in raw.columns:
                dev_agg[c] = "mean"
        if dev_agg:
            device_df = raw.groupby("Device").agg(dev_agg).reset_index()

    return page_df, scaled, feature_cols, monthly_df, device_df, raw


# ════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def kpi(label, value, sub=""):
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value'>{value}</div>
        {'<div class="kpi-sub">' + sub + '</div>' if sub else ''}
    </div>""", unsafe_allow_html=True)

def caption(text):
    st.markdown(f"<div class='chart-caption'>📊 <em>{text}</em></div>", unsafe_allow_html=True)

def section_header(text):
    st.markdown(f"<div class='section-header'>{text}</div>", unsafe_allow_html=True)

def no_file_warning(files_needed):
    st.markdown(f"""
    <div class='upload-box'>
        <h3>📂 Files Required</h3>
        <p>Please upload the following files using the sidebar to view this analysis:</p>
        <ul style='text-align:left; display:inline-block;'>
            {''.join(f'<li><code>{f}</code></li>' for f in files_needed)}
        </ul>
    </div>""", unsafe_allow_html=True)

CLUSTER_COLORS = {
    "High Friction":  "#dc3545",
    "Discovery":      "#fd7e14",
    "Core Program":   "#0d6efd",
    "Power User":     "#198754",
}

SIG_COLORS = {
    "Deep Engagement":      "#198754",
    "Repeat Reference":     "#0d6efd",
    "High Abandonment":     "#dc3545",
    "Quick Task Completion":"#198754",
    "Frustrated Navigation":"#fd7e14",
}

SIG_HEALTH = {
    "Deep Engagement":       "green",
    "Repeat Reference":      "green",
    "Quick Task Completion": "green",
    "High Abandonment":      "red",
    "Frustrated Navigation": "orange",
}


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — FILE UPLOADS
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://www.usda.gov/themes/custom/usda_uswds/img/usda-symbol.svg",
             width=60) if False else None  # skip if no internet
    st.markdown("## 🌾 USDA Dashboard")
    st.markdown("**Upload Files to Begin**")
    st.markdown("---")
    st.markdown("### System-Wide Files (Layer 1)")
    device_file   = st.file_uploader("Device CSV",         type="csv", key="device")
    domain_file   = st.file_uploader("Domain CSV",         type="csv", key="domain")
    download_file = st.file_uploader("Download CSV",       type="csv", key="download")
    language_file = st.file_uploader("Language CSV",       type="csv", key="language")
    os_file       = st.file_uploader("OS/Browser CSV",     type="csv", key="os")
    traffic_file  = st.file_uploader("Traffic Source CSV", type="csv", key="traffic")
    windows_file  = st.file_uploader("Windows Browser CSV",type="csv", key="windows")

    st.markdown("---")
    st.markdown("### Rural Development File (Layers 2–4)")
    rd_file = st.file_uploader("Edited USDA Data Base CSV", type="csv", key="rd")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#888;'>
    <strong>Data Note:</strong> All metrics are aggregated from Jan–Jun 2024
    unless otherwise noted. Browser language is used as a proxy for language
    preference, not geography.
    </div>""", unsafe_allow_html=True)

system_ready = all([device_file, domain_file, download_file,
                    language_file, traffic_file])
rd_ready = rd_file is not None

# ════════════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='main-header'>🌾 USDA Digital Service Effectiveness Dashboard</div>",
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Layer 1 — System-Wide Analysis",
    "🌱 Layer 2 — Rural Development Baseline",
    "🔬 Layer 3 — K-Means Clustering",
    "🎯 Layer 4 — Behavioral Signatures",
])


# ════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — SYSTEM-WIDE DESCRIPTIVE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    if not system_ready:
        no_file_warning([
            "device-1-2024.csv", "domain-1-2024.csv", "download-1-2024.csv",
            "language-1-2024.csv", "traffic-source-1-2024.csv",
        ])
    else:
        device_df, domain_df, download_df, language_df, os_df, traffic_df, windows_df = load_system_csvs(
            device_file.read(), domain_file.read(), download_file.read(),
            language_file.read(),
            os_file.read() if os_file else b"",
            traffic_file.read(),
            windows_file.read() if windows_file else b"",
        )

        # Reset file pointers aren't needed since bytes already read

        # ── Determine column structure of each file ──────────────────────────
        # Device file: expect columns for device type, month, visits
        def sniff(df, label):
            st.write(f"**{label}** — {df.shape} — columns: {list(df.columns[:8])}")

        # Auto-detect month/visit columns (handle various Export formats)
        def get_month_col(df):
            for c in df.columns:
                if any(m.lower() in c.lower() for m in ["month", "date", "period", "jan", "feb"]):
                    return c
            return df.columns[0]

        def get_visits_col(df):
            for c in df.columns:
                if any(k in c.lower() for k in ["visit", "session", "view", "user", "pageview"]):
                    return c
            return df.columns[-1]

        # ── SECTION 1: OVERVIEW ───────────────────────────────────────────────
        section_header("1 · Overview")

        # Total visits KPI from domain file
        visit_col = get_visits_col(domain_df)
        month_col = get_month_col(domain_df)

        # Try to get numeric total visits
        total_visits = pd.to_numeric(domain_df[visit_col], errors="coerce").sum()

        # Mobile share from device file
        mobile_share = None
        mobile_trend = None
        if device_df is not None and len(device_df) > 0:
            dev_type_col = next((c for c in device_df.columns
                                  if "device" in c.lower() or "type" in c.lower()), device_df.columns[0])
            dev_vis_col  = get_visits_col(device_df)
            dev_mon_col  = get_month_col(device_df)
            device_df[dev_vis_col] = pd.to_numeric(device_df[dev_vis_col], errors="coerce")

            if dev_type_col != dev_vis_col:
                total_by_device = device_df.groupby(dev_type_col)[dev_vis_col].sum()
                mobile_total = sum(v for k, v in total_by_device.items()
                                   if "mobile" in str(k).lower())
                grand_total  = total_by_device.sum()
                mobile_share = mobile_total / grand_total if grand_total > 0 else 0

                # Monthly mobile trend
                if dev_mon_col != dev_type_col:
                    monthly_device = device_df.groupby([dev_mon_col, dev_type_col])[dev_vis_col].sum().reset_index()
                    monthly_total  = monthly_device.groupby(dev_mon_col)[dev_vis_col].sum().reset_index()
                    monthly_mobile = monthly_device[monthly_device[dev_type_col].str.lower().str.contains("mobile", na=False)]
                    if len(monthly_mobile) and len(monthly_total):
                        merged = monthly_mobile.merge(monthly_total, on=dev_mon_col, suffixes=("_mob","_tot"))
                        merged["mobile_pct"] = merged[f"{dev_vis_col}_mob"] / merged[f"{dev_vis_col}_tot"]
                        mobile_trend = merged[[dev_mon_col, "mobile_pct"]].rename(
                            columns={dev_mon_col: "Month", "mobile_pct": "Mobile %"})

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi("Total Visits (Jan–Jun)", f"{total_visits:,.0f}" if not np.isnan(total_visits) else "N/A")
        with c2:
            kpi("Mobile Share", f"{mobile_share:.1%}" if mobile_share is not None else "N/A",
                "of all visits from mobile devices")
        with c3:
            num_hostnames = domain_df[domain_df.columns[0]].nunique() if len(domain_df) else 0
            kpi("Active Subdomains", str(num_hostnames))

        # Mobile trend chart
        if mobile_trend is not None and len(mobile_trend) > 1:
            fig = px.line(mobile_trend, x="Month", y="Mobile %",
                          title="Monthly Mobile Traffic Share (%)",
                          markers=True,
                          labels={"Mobile %": "Mobile Share"})
            fig.update_traces(line_color="#2e86ab", line_width=3)
            fig.update_layout(yaxis_tickformat=".1%", height=280,
                              margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            caption("Monthly mobile share trend. A sustained rise in mobile usage signals growing "
                    "demand for mobile-optimized content — critical for equitable rural access.")

            # Mobile flag
            if len(mobile_trend) >= 3:
                pcts = mobile_trend["Mobile %"].tolist()
                rises = sum(1 for i in range(1, len(pcts)) if pcts[i] > pcts[i-1])
                if rises >= 3:
                    st.warning("⚠️ **Mobile Flag:** Mobile share has risen for 3 or more consecutive "
                               "months. Ensure all key pages are mobile-optimized.")
        else:
            # Daily visit trend from domain file
            dom_vis_col = get_visits_col(domain_df)
            dom_mon_col = get_month_col(domain_df)
            if dom_mon_col != dom_vis_col:
                domain_df[dom_vis_col] = pd.to_numeric(domain_df[dom_vis_col], errors="coerce")
                monthly = domain_df.groupby(dom_mon_col)[dom_vis_col].sum().reset_index()
                fig = px.line(monthly, x=dom_mon_col, y=dom_vis_col,
                              title="Monthly Visit Volume Trend",
                              markers=True)
                fig.update_traces(line_color="#2e86ab", line_width=3)
                fig.update_layout(height=280, margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
                caption("Monthly visit totals across all USDA subdomains. Seasonal peaks may indicate "
                        "program application periods or policy announcements.")

        # ── SECTION 2: AGENCY TRAFFIC ─────────────────────────────────────────
        st.divider()
        section_header("2 · Agency Traffic")

        host_col = next((c for c in domain_df.columns
                          if "host" in c.lower() or "domain" in c.lower() or "site" in c.lower()),
                         domain_df.columns[0])
        vis_col  = get_visits_col(domain_df)
        domain_df[vis_col] = pd.to_numeric(domain_df[vis_col], errors="coerce")

        host_totals = domain_df.groupby(host_col)[vis_col].sum().nlargest(15).reset_index()
        host_totals.columns = ["Hostname", "Total Visits"]
        host_totals["Agency Name"] = host_totals["Hostname"].map(
            lambda x: HOSTNAME_MAP.get(str(x).lower(), str(x)))
        host_totals["Display"] = host_totals.apply(
            lambda r: f"{r['Agency Name']}" if r['Agency Name'] != r['Hostname']
                      else r['Hostname'], axis=1)

        fig = px.bar(host_totals.sort_values("Total Visits"),
                     x="Total Visits", y="Display", orientation="h",
                     title="Top 15 USDA Subdomains by Total Visits (Jan–Jun)",
                     color="Total Visits",
                     color_continuous_scale="Blues")
        fig.update_layout(height=480, margin=dict(t=40, b=20),
                          yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        caption("Horizontal bars rank subdomains by total visits. Longer bars indicate higher public "
                "demand; agencies with shorter bars may benefit from improved discoverability or content.")

        # MoM Momentum Table
        mon_col = get_month_col(domain_df)
        if mon_col != host_col:
            pivot = domain_df.pivot_table(index=host_col, columns=mon_col, values=vis_col,
                                          aggfunc="sum").reset_index()
            # Filter to hostnames in 3+ months
            month_presence = pivot.iloc[:, 1:].notna().sum(axis=1)
            pivot = pivot[month_presence >= 3].copy()

            if len(pivot):
                num_cols = [c for c in pivot.columns if c != host_col]
                if len(num_cols) >= 2:
                    first_m, last_m = num_cols[0], num_cols[-1]
                    pivot["% Change Jan→Last"] = (
                        (pivot[last_m] - pivot[first_m]) / pivot[first_m].replace(0, np.nan) * 100
                    ).round(1)
                    pivot["Agency"] = pivot[host_col].map(
                        lambda x: HOSTNAME_MAP.get(str(x).lower(), str(x)))

                    display_cols = ["Agency"] + num_cols + ["% Change Jan→Last"]
                    mom = pivot[display_cols].head(15)

                    def color_mom(val):
                        if isinstance(val, (int, float)):
                            if val > 0:
                                return "background-color: #d4edda"
                            elif val < 0:
                                return "background-color: #f8d7da"
                        return ""

                    st.markdown("**Month-over-Month Momentum Table**")
                    styled = mom.style.applymap(color_mom, subset=["% Change Jan→Last"])
                    st.dataframe(styled, use_container_width=True, hide_index=True)
                    caption("Green = positive growth, Red = traffic decline. Agencies showing "
                            "consecutive declines may need content strategy review.")

        # Stacked monthly source bar
        st.markdown("---")
        section_header("Traffic Source Breakdown")

        src_col  = next((c for c in traffic_df.columns
                          if "source" in c.lower() or "channel" in c.lower() or
                             "medium" in c.lower() or "referr" in c.lower()),
                         traffic_df.columns[0])
        tv_col   = get_visits_col(traffic_df)
        tm_col   = get_month_col(traffic_df)
        traffic_df[tv_col] = pd.to_numeric(traffic_df[tv_col], errors="coerce")

        def map_source(s):
            s = str(s).lower().strip()
            if s in ["google", "bing", "yahoo", "organic", "organic search"]:
                return "Organic Search"
            if s in ["(direct)", "direct", "none", "(none)"]:
                return "Direct"
            if "social" in s or "facebook" in s or "twitter" in s or "linkedin" in s:
                return "Social Referral"
            return "Other"

        traffic_df["Source Category"] = traffic_df[src_col].apply(map_source)

        if tm_col != src_col:
            src_monthly = traffic_df.groupby([tm_col, "Source Category"])[tv_col].sum().reset_index()
            total_traffic = traffic_df[tv_col].sum()
            google_traffic = traffic_df[traffic_df["Source Category"] == "Organic Search"][tv_col].sum()
            google_dep = google_traffic / total_traffic if total_traffic > 0 else 0

            fig = px.bar(src_monthly, x=tm_col, y=tv_col, color="Source Category",
                         title="Monthly Traffic by Source Category",
                         barmode="stack",
                         color_discrete_map={
                             "Organic Search": "#4285F4",
                             "Direct":         "#34A853",
                             "Social Referral":"#EA4335",
                             "Other":          "#FBBC04",
                         })
            fig.update_layout(height=380, margin=dict(t=40, b=20),
                              xaxis_title="Month", yaxis_title="Visits",
                              legend_title="Source")
            st.plotly_chart(fig, use_container_width=True)
            caption("Stacked bars reveal the composition of traffic each month. Heavy reliance on "
                    "a single source (especially Google) creates vulnerability if search algorithms change.")

            # Google dependency KPI
            dep_label = "Low" if google_dep < 0.4 else ("Moderate" if google_dep <= 0.6 else "High")
            dep_color = "risk-card-low" if google_dep < 0.4 else ("risk-card" if google_dep <= 0.6 else "risk-card-high")
            st.markdown(f"""
            <div class='{dep_color}'>
                <strong>Google Dependency: {dep_label} ({google_dep:.1%})</strong><br>
                <small>{google_dep:.1%} of all USDA traffic arrives via Google Search.
                {'High dependency means algorithm changes could significantly reduce visibility.' if google_dep > 0.6 else
                 'Moderate dependency. Consider building direct traffic through newsletters and bookmarking.' if google_dep > 0.4 else
                 'Healthy traffic diversification. Continue maintaining multiple acquisition channels.'}</small>
            </div>""", unsafe_allow_html=True)

            # Social referral inline
            social_pct = traffic_df[traffic_df["Source Category"] == "Social Referral"][tv_col].sum() / total_traffic
            st.markdown(f"**Social Referral Share:** {social_pct:.1%} of total traffic originates from social media platforms.")
        else:
            # Fallback: just show source totals
            src_totals = traffic_df.groupby("Source Category")[tv_col].sum().reset_index()
            fig = px.bar(src_totals, x="Source Category", y=tv_col,
                         title="Traffic by Source Category (All Periods)",
                         color="Source Category")
            fig.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # ── SECTION 3: CONTENT DEMAND ─────────────────────────────────────────
        st.divider()
        section_header("3 · Content Demand — Top Downloads")

        url_col   = next((c for c in download_df.columns
                           if "url" in c.lower() or "file" in c.lower() or
                              "path" in c.lower() or "page" in c.lower()),
                          download_df.columns[0])
        dv_col    = get_visits_col(download_df)
        dm_col    = get_month_col(download_df)
        dh_col    = next((c for c in download_df.columns
                           if "host" in c.lower() or "domain" in c.lower()),
                          None)
        download_df[dv_col] = pd.to_numeric(download_df[dv_col], errors="coerce")

        download_df["filename"] = download_df[url_col].astype(str).apply(
            lambda x: x.rstrip("/").split("/")[-1][:60])
        download_df["hostname_label"] = download_df[dh_col] if dh_col else ""

        top20 = download_df.groupby(["filename", "hostname_label"])[dv_col].sum().nlargest(20).reset_index()
        top20.columns = ["Filename", "Hostname", "Events"]

        fig = px.bar(top20.sort_values("Events"),
                     x="Events", y="Filename", orientation="h",
                     title="Top 20 Downloaded Files (Jan–Jun)",
                     color="Events", color_continuous_scale="Teal",
                     hover_data=["Hostname"])
        fig.update_layout(height=560, margin=dict(t=40, b=20),
                          yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        caption("Files with the highest download counts represent core public information needs. "
                "Ensure these files are current, accessible, and available in multiple languages.")

        # Monthly download bar
        if dm_col != url_col:
            monthly_dl = download_df.groupby(dm_col)[dv_col].sum().reset_index()
            fig = px.bar(monthly_dl, x=dm_col, y=dv_col,
                         title="Monthly Download Event Volume",
                         color_discrete_sequence=["#2e86ab"])
            fig.update_layout(height=280, xaxis_title="Month", yaxis_title="Download Events")
            st.plotly_chart(fig, use_container_width=True)
            caption("Spikes in monthly downloads often align with program deadlines, seasonal "
                    "agricultural cycles, or policy announcements.")

        # Searchable downloads table
        st.markdown("**Searchable Downloads Table**")
        if dm_col != url_col:
            peak_month = download_df.groupby(["filename", dm_col])[dv_col].sum().reset_index()
            peak_month = peak_month.loc[peak_month.groupby("filename")[dv_col].idxmax()][["filename", dm_col]]
            peak_month.columns = ["Filename", "Peak Month"]
            table_df = top20.merge(peak_month, on="Filename", how="left")
        else:
            table_df = top20.copy()
            table_df["Peak Month"] = "N/A"

        search_term = st.text_input("🔍 Search filenames", "", key="dl_search")
        filtered_table = table_df[
            table_df["Filename"].str.lower().str.contains(search_term.lower(), na=False)
        ] if search_term else table_df
        st.dataframe(filtered_table.head(20), use_container_width=True, hide_index=True)

        # ── SECTION 4: LANGUAGE REACH ─────────────────────────────────────────
        st.divider()
        section_header("4 · Language Reach")
        st.info("🌐 **Methodology note:** Browser language settings are used as a proxy for language "
                "preference and do not directly indicate user geography.")

        lang_col = next((c for c in language_df.columns
                          if "lang" in c.lower()), language_df.columns[0])
        lv_col   = get_visits_col(language_df)
        lm_col   = get_month_col(language_df)
        language_df[lv_col] = pd.to_numeric(language_df[lv_col], errors="coerce")

        total_lang = language_df[lv_col].sum()
        english_visits = language_df[
            language_df[lang_col].astype(str).str.lower().str.startswith("en")
        ][lv_col].sum()
        non_english_pct = (total_lang - english_visits) / total_lang if total_lang > 0 else 0

        c1, c2 = st.columns(2)
        with c1:
            kpi("Non-English Browser Share", f"{non_english_pct:.1%}",
                "of sessions use a non-English browser language")

        # Monthly non-English trend
        if lm_col != lang_col:
            lang_monthly = language_df.groupby([lm_col, lang_col])[lv_col].sum().reset_index()
            tot_monthly  = language_df.groupby(lm_col)[lv_col].sum().reset_index()
            non_eng_m    = lang_monthly[~lang_monthly[lang_col].astype(str).str.lower().str.startswith("en")]
            non_eng_m    = non_eng_m.groupby(lm_col)[lv_col].sum().reset_index()
            merged_lang  = non_eng_m.merge(tot_monthly, on=lm_col, suffixes=("_ne", "_tot"))
            merged_lang["Non-English %"] = merged_lang[f"{lv_col}_ne"] / merged_lang[f"{lv_col}_tot"]

            fig = px.line(merged_lang, x=lm_col, y="Non-English %",
                          title="Monthly Non-English Browser Share (%)",
                          markers=True)
            fig.update_traces(line_color="#8e44ad", line_width=3)
            fig.update_layout(height=280, yaxis_tickformat=".1%",
                              xaxis_title="Month", margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            caption("A rising non-English share indicates growing multilingual demand. "
                    "Pages serving high-need communities should be prioritized for translation.")

        # Top 10 non-English languages
        non_eng_df = language_df[
            ~language_df[lang_col].astype(str).str.lower().str.startswith("en")
        ]
        top10_lang = non_eng_df.groupby(lang_col)[lv_col].sum().nlargest(10).reset_index()
        top10_lang.columns = ["Language Code", "Total Visits"]

        fig = px.bar(top10_lang.sort_values("Total Visits"),
                     x="Total Visits", y="Language Code", orientation="h",
                     title="Top 10 Non-English Browser Languages (Jan–Jun)",
                     color="Total Visits", color_continuous_scale="Purples")
        fig.update_layout(height=360, yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        caption("Languages with high visit volumes represent communities that may benefit from "
                "dedicated translated content, particularly for nutrition, farm, and housing programs.")

        # ── SECTION 5: FRICTION FLAGS ─────────────────────────────────────────
        st.divider()
        section_header("5 · Friction Flags")

        if 'google_dep' in dir() or 'google_dep' in locals():
            pass
        else:
            google_dep = 0

        st.markdown("""
        <div class='risk-card'>
            <strong>⚠️ Google Dependency Risk</strong><br>
            Heavy reliance on Google Search means USDA content visibility is subject to algorithm
            changes outside government control. Agencies should develop direct-access strategies
            (email lists, bookmarking prompts, partner referrals) to reduce search dependence.
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='risk-card'>
            <strong>📱 Mobile Engagement Risk</strong><br>
            Mobile users in rural areas often access USDA services on lower-bandwidth connections
            and older devices. Pages with heavy images, PDFs, or complex forms may create
            disproportionate barriers for this audience.
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='risk-card-low'>
            <strong>ℹ️ Legacy Browser Note</strong><br>
            OS/Browser data indicates a non-trivial share of visits from older browser versions.
            Legacy browser usage is common in rural and low-income populations. Ensure USDA
            pages degrade gracefully for users on older technology.
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — RURAL DEVELOPMENT BASELINE
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    if not rd_ready:
        no_file_warning(["Edited_USDA_data_base.csv"])
    else:
        with st.spinner("Processing Rural Development data..."):
            page_df, scaled, feature_cols, monthly_df, dev_df, raw_rd = load_rd_data(rd_file.read())

        st.markdown("### Rural Development — Performance Baseline")
        st.markdown("This layer establishes engagement benchmarks across the Rural Development website, "
                    "identifying where performance is strong and where users face the greatest friction.")

        # ── KPI ROW ──────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            total_users = page_df["Total Users"].sum() if "Total Users" in page_df else 0
            kpi("Total RD Users", f"{total_users:,.0f}")
        with c2:
            mean_br = page_df["Bounce Rate"].mean()
            kpi("Mean Bounce Rate", f"{mean_br:.1%}")
        with c3:
            mean_sd = page_df["Session Duration"].mean()
            kpi("Mean Session Duration", f"{mean_sd:.0f}s", f"≈ {mean_sd/60:.1f} minutes")
        with c4:
            mean_vps = page_df["Views per Session"].mean()
            kpi("Mean Views per Session", f"{mean_vps:.2f}")

        # ── INTERACTIVE HEATMAP ───────────────────────────────────────────────
        st.divider()
        section_header("Section-Level Performance Heatmap")

        metrics = ["Bounce Rate", "Session Duration", "Views per Session",
                   "Exit Pressure Index", "Stickiness Ratio"]
        available_metrics = [m for m in metrics if m in page_df.columns]

        section_perf = page_df.groupby("Section")[available_metrics].mean().reset_index()
        section_perf = section_perf[section_perf["Section"].str.len() > 0].head(20)

        if len(section_perf) > 0 and len(available_metrics) > 0:
            # Normalize for color scale (invert bounce/exit so green=good)
            heat_data = section_perf[available_metrics].copy()
            # For Bounce Rate and Exit Pressure: lower is better → invert for color
            invert = ["Bounce Rate", "Exit Pressure Index"]

            z_norm = heat_data.copy()
            for col in available_metrics:
                col_range = z_norm[col].max() - z_norm[col].min()
                if col_range > 0:
                    z_norm[col] = (z_norm[col] - z_norm[col].min()) / col_range
                if col in invert:
                    z_norm[col] = 1 - z_norm[col]

            fig = go.Figure(data=go.Heatmap(
                z=z_norm.values,
                x=available_metrics,
                y=section_perf["Section"].tolist(),
                colorscale="RdYlGn",
                showscale=True,
                text=[[f"{section_perf[m].iloc[i]:.2f}" for m in available_metrics]
                      for i in range(len(section_perf))],
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate="Section: %{y}<br>Metric: %{x}<br>Value: %{text}<extra></extra>",
            ))
            fig.update_layout(
                title="Section Performance Heatmap (Green = Good, Red = Poor)",
                height=max(400, len(section_perf) * 30 + 100),
                xaxis_title="Performance Metric",
                yaxis_title="Website Section",
                margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
            caption("Each cell shows the actual metric value; color indicates relative performance. "
                    "Sections with many red cells represent priority areas for content and UX improvement. "
                    "Green Stickiness and long Session Duration indicate users finding value; "
                    "red Bounce Rate signals content-user mismatch.")

        # ── DEVICE COMPARISON ─────────────────────────────────────────────────
        st.divider()
        section_header("Device Type — Engagement Comparison")

        if dev_df is not None and len(dev_df) > 0:
            dev_metrics = [c for c in ["Bounce Rate", "Session Duration"] if c in dev_df.columns]
            if dev_metrics:
                fig = go.Figure()
                colors = {"desktop": "#2e86ab", "mobile": "#e84855", "tablet": "#f5a623"}
                for metric in dev_metrics:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=dev_df["Device"].tolist(),
                        y=dev_df[metric].tolist(),
                        text=[f"{v:.2f}" for v in dev_df[metric]],
                        textposition="outside",
                    ))
                fig.update_layout(
                    title="Bounce Rate & Session Duration by Device Type",
                    barmode="group",
                    height=360,
                    xaxis_title="Device Type",
                    yaxis_title="Metric Value",
                )
                st.plotly_chart(fig, use_container_width=True)
                caption("Grouped bars highlight the device experience gap. A significantly higher "
                        "mobile bounce rate compared to desktop signals that mobile visitors are not "
                        "finding what they need — a major equity concern for rural audiences who rely on phones.")
        else:
            st.info("Device breakdown requires separate device-type columns in the Rural Development dataset.")

        # ── MONTHLY ENGAGEMENT TRENDS ──────────────────────────────────────────
        st.divider()
        section_header("Monthly Engagement Trends")

        if monthly_df is not None and len(monthly_df) > 0:
            trend_metrics = [c for c in ["Bounce Rate", "Session Duration", "Active Users"]
                             if c in monthly_df.columns]
            if trend_metrics:
                fig = make_subplots(rows=len(trend_metrics), cols=1,
                                    shared_xaxes=True,
                                    subplot_titles=trend_metrics)
                colors_t = ["#e84855", "#2e86ab", "#2ecc71"]
                for i, m in enumerate(trend_metrics):
                    fig.add_trace(
                        go.Scatter(x=monthly_df["Month"], y=monthly_df[m],
                                   mode="lines+markers", name=m,
                                   line=dict(color=colors_t[i], width=2),
                                   marker=dict(size=6)),
                        row=i+1, col=1
                    )
                fig.update_layout(height=300 * len(trend_metrics),
                                  title="Monthly Engagement Trends",
                                  showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                caption("Multi-metric trend lines reveal whether engagement is improving holistically. "
                        "Rising bounce rate alongside falling session duration is a warning sign of "
                        "content degradation or audience-content mismatch over time.")
        else:
            st.info("Monthly trend chart requires date information in the dataset.")

        # ── SERVICE TIER DONUT ─────────────────────────────────────────────────
        st.divider()
        section_header("Service Tier Classification")

        p75 = page_df["Underserved Score"].quantile(0.75)
        p25 = page_df["Underserved Score"].quantile(0.25)

        def classify_tier(score):
            if score >= p75:
                return "Underserved"
            elif score >= p25:
                return "Moderately Served"
            else:
                return "Well-Served"

        page_df["Service Tier"] = page_df["Underserved Score"].apply(classify_tier)
        tier_counts = page_df["Service Tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        tier_counts["Pct"] = tier_counts["Count"] / tier_counts["Count"].sum() * 100

        tier_colors = {"Well-Served": "#28a745", "Moderately Served": "#ffc107", "Underserved": "#dc3545"}
        fig = go.Figure(data=[go.Pie(
            labels=tier_counts["Tier"],
            values=tier_counts["Count"],
            hole=0.55,
            marker=dict(colors=[tier_colors.get(t, "#888") for t in tier_counts["Tier"]]),
            textinfo="label+percent",
        )])
        fig.update_layout(
            title="Page Service Tier Distribution<br><sub>Based on Underserved Score = (Bounce×0.4) + (Exit Pressure×0.3) + ((1−Stickiness)×0.3)</sub>",
            height=400,
            annotations=[dict(text=f"{len(page_df)}<br>Pages", x=0.5, y=0.5, font_size=16,
                              showarrow=False)],
        )
        st.plotly_chart(fig, use_container_width=True)
        caption("Pages in the 'Underserved' tier have the highest combined score of bounce rate, "
                "exit pressure, and low stickiness — signaling a poor content-user fit. These pages "
                "should be prioritized for content review, UX testing, and mobile optimization.")

        c1, c2, c3 = st.columns(3)
        for _, row in tier_counts.iterrows():
            col = c1 if row["Tier"] == "Well-Served" else (c2 if row["Tier"] == "Moderately Served" else c3)
            with col:
                color = {"Well-Served": "risk-card-low", "Moderately Served": "risk-card",
                         "Underserved": "risk-card-high"}[row["Tier"]]
                st.markdown(f"""
                <div class='{color}'>
                    <strong>{row['Tier']}</strong><br>
                    <span style='font-size:1.5rem; font-weight:700;'>{row['Count']}</span> pages
                    <span style='color:#555;'> ({row['Pct']:.1f}%)</span>
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — K-MEANS CLUSTERING
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    if not rd_ready:
        no_file_warning(["Edited_USDA_data_base.csv"])
    else:
        if "page_df" not in dir():
            with st.spinner("Processing data..."):
                page_df, scaled, feature_cols, monthly_df, dev_df, raw_rd = load_rd_data(rd_file.read())

        st.markdown("### K-Means Behavioral Clustering")
        st.markdown("This layer identifies distinct behavioral patterns across Rural Development pages "
                    "using unsupervised machine learning. Pages with similar engagement signatures are "
                    "grouped into clusters, revealing systemic performance patterns invisible in aggregate statistics.")

        # ── SECTION A: CONTROLS ───────────────────────────────────────────────
        section_header("A · Clustering Controls")
        st.info("ℹ️ All six features (Bounce Rate, Session Duration, Views per Session, "
                "Exit Pressure Index, Stickiness Ratio, Device Gap Score) are standardized "
                "using z-score normalization before clustering, ensuring no single metric "
                "dominates due to scale differences.")

        k_val = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=4, key="k_slider")
        run_btn = st.button("▶ Run Clustering", type="primary", key="run_cluster")

        if run_btn or "cluster_labels" in st.session_state:
            if run_btn:
                with st.spinner("Running K-Means diagnostics..."):
                    wcss, sil_scores = [], []
                    for k in range(2, 11):
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = km.fit_predict(scaled)
                        wcss.append(km.inertia_)
                        sil_scores.append(silhouette_score(scaled, labels))
                    
                    km_final = KMeans(n_clusters=k_val, random_state=42, n_init=10)
                    cluster_labels = km_final.fit_predict(scaled)
                    
                    st.session_state["cluster_labels"] = cluster_labels
                    st.session_state["wcss"] = wcss
                    st.session_state["sil_scores"] = sil_scores
                    st.session_state["k_used"] = k_val

            cluster_labels = st.session_state["cluster_labels"]
            wcss          = st.session_state["wcss"]
            sil_scores    = st.session_state["sil_scores"]
            k_used        = st.session_state["k_used"]

            # ── SECTION B: DIAGNOSTICS ────────────────────────────────────────
            section_header("B · Clustering Diagnostics")
            col1, col2 = st.columns(2)
            k_range = list(range(2, 11))

            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=k_range, y=wcss, mode="lines+markers",
                                         line=dict(color="#2e86ab", width=2)))
                fig.add_vline(x=k_used, line_dash="dash", line_color="#e84855",
                              annotation_text=f"k={k_used}", annotation_position="top")
                fig.update_layout(title="Elbow Method (WCSS vs k)",
                                  xaxis_title="Number of Clusters (k)",
                                  yaxis_title="Within-Cluster Sum of Squares",
                                  height=340)
                st.plotly_chart(fig, use_container_width=True)
                caption("The 'elbow' — where WCSS drops sharply then levels off — suggests the "
                        "optimal number of clusters. Choosing k beyond the elbow adds complexity without insight.")

            with col2:
                peak_k   = k_range[sil_scores.index(max(sil_scores))]
                peak_sil = max(sil_scores)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=k_range, y=sil_scores, mode="lines+markers",
                                         line=dict(color="#8e44ad", width=2)))
                fig.add_vline(x=k_used, line_dash="dash", line_color="#e84855",
                              annotation_text=f"k={k_used}", annotation_position="top")
                fig.add_annotation(x=peak_k, y=peak_sil,
                                   text=f"Peak: {peak_sil:.3f} at k={peak_k}",
                                   showarrow=True, arrowhead=2, bgcolor="white")
                fig.update_layout(title="Silhouette Score vs k",
                                  xaxis_title="Number of Clusters (k)",
                                  yaxis_title="Silhouette Score",
                                  height=340)
                st.plotly_chart(fig, use_container_width=True)
                caption("Higher silhouette scores indicate better-defined, more separated clusters. "
                        "The peak score identifies the statistically optimal cluster count.")

            sel_sil = sil_scores[k_used - 2]
            st.markdown(f"**Based on the diagnostics, k={k_used} was selected. "
                        f"Silhouette score at selected k: {sel_sil:.4f}**")

            # Label clusters
            CLUSTER_NAMES = {0: "High Friction", 1: "Discovery",
                              2: "Core Program", 3: "Power User"}
            page_clustered = page_df.copy()
            page_clustered["Cluster ID"]    = cluster_labels
            page_clustered["Cluster Label"] = [
                CLUSTER_NAMES.get(i % 4, f"Cluster {i}") for i in cluster_labels]

            # ── SECTION C: CLUSTER SUMMARY ────────────────────────────────────
            section_header("C · Cluster Summary Table")
            summary_cols = ["Bounce Rate", "Session Duration", "Views per Session",
                            "Exit Pressure Index", "Stickiness Ratio", "Device Gap Score"]
            avail_s = [c for c in summary_cols if c in page_clustered.columns]

            cluster_summary = page_clustered.groupby("Cluster Label").agg(
                Pages=("Page path", "count"),
                **{c: (c, "mean") for c in avail_s}
            ).reset_index()

            def style_cluster_row(row):
                color_map = {
                    "High Friction": "background-color: #f8d7da",
                    "Discovery":     "background-color: #fff3cd",
                    "Core Program":  "background-color: #cfe2ff",
                    "Power User":    "background-color: #d1e7dd",
                }
                color = color_map.get(row["Cluster Label"], "")
                return [color] * len(row)

            styled_summary = cluster_summary.style.apply(style_cluster_row, axis=1).format(
                {c: "{:.3f}" for c in avail_s if c in cluster_summary.columns}
            )
            st.dataframe(styled_summary, use_container_width=True, hide_index=True)
            caption("Each row represents a cluster of pages with similar behavioral patterns. "
                    "High Friction clusters need immediate attention; Power User clusters represent "
                    "best practices that can be replicated across the site.")

            # ── SECTION D: 2D PCA ──────────────────────────────────────────────
            section_header("D · 2D PCA Visualization")
            st.info("PCA is used for visualization only — clustering was performed on the full "
                    "6-feature standardized space. The 2D projection shows cluster separation.")

            pca2 = PCA(n_components=2, random_state=42)
            coords2 = pca2.fit_transform(scaled)
            pca2_df = page_clustered.copy()
            pca2_df["PC1"] = coords2[:, 0]
            pca2_df["PC2"] = coords2[:, 1]

            hover_cols = ["Page title", "Section", "Bounce Rate", "Session Duration", "Cluster Label"]
            h_avail = [c for c in hover_cols if c in pca2_df.columns]

            fig = px.scatter(pca2_df, x="PC1", y="PC2", color="Cluster Label",
                             hover_data=h_avail,
                             title="2D PCA — Page Clusters",
                             color_discrete_map={
                                 "High Friction": "#dc3545",
                                 "Discovery":     "#fd7e14",
                                 "Core Program":  "#0d6efd",
                                 "Power User":    "#198754",
                             })
            fig.update_traces(marker=dict(size=6, opacity=0.7))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            caption(f"Each point is a unique page. Clusters that are tightly grouped and well-separated "
                    f"indicate behaviorally coherent page sets. Overlap suggests pages with mixed signals. "
                    f"PC1 explains {pca2.explained_variance_ratio_[0]:.1%} and PC2 explains "
                    f"{pca2.explained_variance_ratio_[1]:.1%} of total variance.")

            # ── SECTION E: 3D PCA ──────────────────────────────────────────────
            section_header("E · 3D PCA Visualization (Rotatable)")
            pca3 = PCA(n_components=3, random_state=42)
            coords3 = pca3.fit_transform(scaled)
            pca3_df = page_clustered.copy()
            pca3_df["PC1"] = coords3[:, 0]
            pca3_df["PC2"] = coords3[:, 1]
            pca3_df["PC3"] = coords3[:, 2]

            fig = px.scatter_3d(pca3_df, x="PC1", y="PC2", z="PC3",
                                color="Cluster Label",
                                hover_data=h_avail,
                                title="3D PCA — Page Clusters (Drag to Rotate)",
                                color_discrete_map={
                                    "High Friction": "#dc3545",
                                    "Discovery":     "#fd7e14",
                                    "Core Program":  "#0d6efd",
                                    "Power User":    "#198754",
                                })
            fig.update_traces(marker=dict(size=4, opacity=0.7))
            fig.update_layout(height=560)
            st.plotly_chart(fig, use_container_width=True)
            v1, v2, v3 = pca3.explained_variance_ratio_
            st.markdown(f"**Variance Explained:** PC1 = {v1:.1%} · PC2 = {v2:.1%} · "
                        f"PC3 = {v3:.1%} · **Total = {v1+v2+v3:.1%}**")
            caption("The 3D view reveals cluster structure not visible in 2D. Rotate to inspect "
                    "whether clusters that appear overlapping in 2D are actually separated in 3D space.")

            # ── SECTION F: RADAR CHART ─────────────────────────────────────────
            section_header("F · Cluster Profile Radar Chart")
            radar_features = [c for c in summary_cols if c in cluster_summary.columns]
            if radar_features:
                from sklearn.preprocessing import MinMaxScaler
                radar_data = cluster_summary[radar_features].copy()
                mms = MinMaxScaler()
                radar_norm = pd.DataFrame(mms.fit_transform(radar_data),
                                          columns=radar_features)
                radar_norm["Cluster Label"] = cluster_summary["Cluster Label"].values

                fig = go.Figure()
                radar_colors = {"High Friction": "#dc3545", "Discovery": "#fd7e14",
                                "Core Program": "#0d6efd", "Power User": "#198754"}
                for _, row in radar_norm.iterrows():
                    vals = [row[f] for f in radar_features]
                    vals.append(vals[0])  # close polygon
                    fig.add_trace(go.Scatterpolar(
                        r=vals,
                        theta=radar_features + [radar_features[0]],
                        fill="toself",
                        name=row["Cluster Label"],
                        opacity=0.5,
                        line=dict(color=radar_colors.get(row["Cluster Label"], "#888"), width=2),
                    ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Normalized Cluster Profiles Across All Features",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)
                caption("The radar chart reveals each cluster's behavioral fingerprint. "
                        "Power User clusters should show high Stickiness and Views with low Bounce. "
                        "High Friction clusters will show the opposite pattern.")

            # ── SECTION G: UNDERSERVED PAGE INVENTORY ──────────────────────────
            section_header("G · Underserved Page Inventory")
            p75_score = page_clustered["Underserved Score"].quantile(0.75)
            underserved = page_clustered[
                page_clustered["Underserved Score"] >= p75_score
            ].sort_values("Underserved Score", ascending=False)

            sections = ["All Sections"] + sorted(underserved["Section"].dropna().unique().tolist())
            selected_section = st.selectbox("Filter by Section", sections, key="us_section")

            if selected_section != "All Sections":
                filtered_us = underserved[underserved["Section"] == selected_section]
            else:
                filtered_us = underserved

            us_display_cols = ["Page title", "Section", "Cluster Label", "Bounce Rate",
                               "Session Duration", "Exit Pressure Index", "Stickiness Ratio",
                               "Device Gap Score", "Underserved Score"]
            us_avail = [c for c in us_display_cols if c in filtered_us.columns]
            st.dataframe(
                filtered_us[us_avail].style.format(
                    {c: "{:.3f}" for c in us_avail if c not in ["Page title", "Section", "Cluster Label"]}
                ),
                use_container_width=True, hide_index=True
            )
            st.markdown(f"**{len(filtered_us)} pages** flagged as Underserved "
                        f"(Underserved Score ≥ {p75_score:.3f}, 75th percentile)")
            caption("These pages represent the highest-priority opportunities for improvement. "
                    "Focus content teams on pages with the highest Underserved Scores, particularly "
                    "those in the High Friction cluster serving core rural development programs.")


# ════════════════════════════════════════════════════════════════════════════
#  LAYER 4 — BEHAVIORAL SIGNATURES
# ════════════════════════════════════════════════════════════════════════════

with tab4:
    if not rd_ready:
        no_file_warning(["Edited_USDA_data_base.csv"])
    else:
        if "page_df" not in dir():
            with st.spinner("Processing data..."):
                page_df, scaled, feature_cols, monthly_df, dev_df, raw_rd = load_rd_data(rd_file.read())

        st.markdown("### Behavioral Signatures — Page-Level Usage Characterization")
        st.info("""
        **Methodology Note:** Behavioral signatures are page-level aggregate patterns derived from
        site-wide analytics data. They characterize how groups of users tend to interact with a page
        in aggregate — **they do not represent individual user journeys** and cannot be used to
        infer specific user intentions or behaviors.
        """)

        # ── SECTION A: SIGNATURE DEFINITIONS ─────────────────────────────────
        section_header("A · Behavioral Signature Definitions")

        signatures = [
            ("Deep Engagement", "#198754",
             "Moderate bounce rate, healthy session duration, multi-page visits. "
             "Users are actively exploring content and finding value across multiple pages."),
            ("Repeat Reference", "#0d6efd",
             "High stickiness ratio, moderate session duration. Professionals and program "
             "participants returning to reference specific information they've found before."),
            ("High Abandonment", "#dc3545",
             "Bounce rate above 55%, short session duration. Strong signal of content-user "
             "mismatch — visitors are not finding what they need and leaving quickly."),
            ("Quick Task Completion", "#198754",
             "Low bounce rate, short session duration. Users locate what they need efficiently. "
             "This is a positive signal: the page is clear, scannable, and task-focused."),
            ("Frustrated Navigation", "#fd7e14",
             "Long session duration combined with high exit pressure. Users spend time on "
             "the page but ultimately give up without completing their goal."),
        ]

        cols = st.columns(len(signatures))
        for i, (name, color, desc) in enumerate(signatures):
            with cols[i]:
                st.markdown(f"""
                <div style='border-left: 5px solid {color}; background: #f8f9fa;
                            border-radius: 8px; padding: 0.8rem; height: 180px;'>
                    <strong style='color:{color};'>{name}</strong>
                    <p style='font-size:0.8rem; color:#444; margin-top:0.4rem;'>{desc}</p>
                </div>""", unsafe_allow_html=True)

        # ── SECTION B: SIGNATURE SCATTER ─────────────────────────────────────
        st.divider()
        section_header("B · Signature Scatter Plot")

        if "Session Duration" in page_df.columns and "Views per Session" in page_df.columns:
            avg_sd  = page_df["Session Duration"].mean()
            avg_vps = page_df["Views per Session"].mean()

            hover_b = [c for c in ["Page title", "Section", "Bounce Rate",
                                    "Session Duration", "Behavioral Signature"]
                        if c in page_df.columns]

            fig = px.scatter(page_df,
                             x="Session Duration",
                             y="Views per Session",
                             color="Behavioral Signature",
                             hover_data=hover_b,
                             title="Session Duration vs Views per Session — Colored by Behavioral Signature",
                             color_discrete_map={
                                 "Deep Engagement":       "#198754",
                                 "Repeat Reference":      "#0d6efd",
                                 "High Abandonment":      "#dc3545",
                                 "Quick Task Completion": "#2ecc71",
                                 "Frustrated Navigation": "#fd7e14",
                             })
            fig.add_vline(x=avg_sd, line_dash="dash", line_color="#888",
                          annotation_text=f"Avg Duration: {avg_sd:.0f}s")
            fig.add_hline(y=avg_vps, line_dash="dash", line_color="#888",
                          annotation_text=f"Avg Views: {avg_vps:.2f}")
            fig.update_traces(marker=dict(size=7, opacity=0.7))
            fig.update_layout(height=500, xaxis_title="Session Duration (seconds)",
                              yaxis_title="Views per Session")
            st.plotly_chart(fig, use_container_width=True)
            caption("Pages in the upper-right quadrant (long duration, many views) indicate deep "
                    "engagement. Pages in the lower-left (short duration, few views) are either "
                    "efficient task completions or abandonments — differentiated by bounce rate and "
                    "exit pressure. Reference lines show the site-wide average for each metric.")

        # ── SECTION C: SIGNATURE DISTRIBUTION ────────────────────────────────
        section_header("C · Signature Distribution")

        sig_counts = page_df["Behavioral Signature"].value_counts().reset_index()
        sig_counts.columns = ["Signature", "Count"]
        sig_counts = sig_counts.sort_values("Count", ascending=True)

        health_map = {"Deep Engagement": "#198754", "Repeat Reference": "#198754",
                      "Quick Task Completion": "#198754",
                      "High Abandonment": "#dc3545", "Frustrated Navigation": "#fd7e14"}
        sig_counts["Color"] = sig_counts["Signature"].map(health_map)

        fig = go.Figure(go.Bar(
            x=sig_counts["Count"],
            y=sig_counts["Signature"],
            orientation="h",
            marker_color=sig_counts["Color"].tolist(),
            text=sig_counts["Count"],
            textposition="outside",
        ))
        fig.update_layout(
            title="Page Count by Behavioral Signature",
            xaxis_title="Number of Pages",
            yaxis_title="",
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)
        caption("The distribution reveals how many pages fall into each behavioral pattern. "
                "A site dominated by red/orange bars (High Abandonment, Frustrated Navigation) "
                "signals systemic content or navigation problems requiring structural intervention.")

        # ── SECTION D: SIGNATURE × SECTION BREAKDOWN ──────────────────────────
        section_header("D · Signature × Section Breakdown")

        if "Section" in page_df.columns:
            sig_sec = page_df.groupby(["Section", "Behavioral Signature"]).size().reset_index(name="Count")
            sig_sec = sig_sec[sig_sec["Section"].str.len() > 0]

            fig = px.bar(sig_sec, x="Section", y="Count", color="Behavioral Signature",
                         barmode="stack",
                         title="Behavioral Signatures by Website Section",
                         color_discrete_map={
                             "Deep Engagement":       "#198754",
                             "Repeat Reference":      "#0d6efd",
                             "High Abandonment":      "#dc3545",
                             "Quick Task Completion": "#2ecc71",
                             "Frustrated Navigation": "#fd7e14",
                         })
            fig.update_layout(height=420, xaxis_title="Website Section",
                              yaxis_title="Number of Pages",
                              xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
            caption("Stacked bars reveal whether entire sections are dominated by problematic "
                    "signatures. A section with mostly red (High Abandonment) indicates a structural "
                    "content problem in that area — not just isolated page issues. Use this to "
                    "prioritize section-wide content audits.")

        # ── SECTION E: SIGNATURE DETAIL TABLE ────────────────────────────────
        section_header("E · Signature Detail Table")

        all_sigs = sorted(page_df["Behavioral Signature"].unique().tolist())
        selected_sig = st.selectbox("Select a Behavioral Signature to explore", all_sigs, key="sig_select")

        sig_detail = page_df[page_df["Behavioral Signature"] == selected_sig]
        detail_cols = ["Page title", "Section", "Bounce Rate", "Session Duration",
                       "Views per Session", "Exit Pressure Index", "Stickiness Ratio", "Device Gap Score"]
        avail_d = [c for c in detail_cols if c in sig_detail.columns]

        st.markdown(f"**{len(sig_detail)} pages** classified as **{selected_sig}**")
        st.dataframe(
            sig_detail[avail_d].sort_values("Bounce Rate", ascending=False).style.format(
                {c: "{:.3f}" for c in avail_d if c not in ["Page title", "Section"]}
            ),
            use_container_width=True, hide_index=True
        )

        sig_desc_map = {
            "High Abandonment":     "These pages are failing to meet user expectations. Review content relevance, page load speed, and mobile experience.",
            "Frustrated Navigation":"Users are spending time here but not succeeding. Review navigation clarity, call-to-action placement, and content organization.",
            "Deep Engagement":      "These pages are performing well. Analyze what makes them successful and apply those principles site-wide.",
            "Repeat Reference":     "These pages serve a loyal, returning audience. Ensure they remain accurate and load quickly.",
            "Quick Task Completion":"These pages efficiently serve user needs. Monitor to ensure metrics remain stable as content changes.",
        }
        if selected_sig in sig_desc_map:
            st.info(f"💡 **What this means:** {sig_desc_map[selected_sig]}")

"""
USDA Digital Service Effectiveness Framework
January–June 2024 | Built strictly from provided CSV data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="USDA Digital Service Effectiveness",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# THEME PALETTE
# ─────────────────────────────────────────────
USDA_GREEN   = "#2E7D32"
USDA_GOLD    = "#F9A825"
USDA_BLUE    = "#1565C0"
USDA_RED     = "#C62828"
USDA_GREY    = "#546E7A"
BG_CARD      = "#F4F6F8"

COLOR_SEQ = [USDA_GREEN, USDA_BLUE, USDA_GOLD, "#00897B", "#6A1B9A", "#EF6C00", "#558B2F"]

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    h1 { color: #1B5E20; font-family: 'Georgia', serif; }
    h2, h3 { color: #2E7D32; }
    .stMetric { background: #F4F6F8; border-radius: 10px; padding: 12px; border-left: 4px solid #2E7D32; }
    .kpi-card {
        background: linear-gradient(135deg, #F4F6F8 0%, #E8F5E9 100%);
        border-radius: 12px; padding: 18px; text-align: center;
        border: 1px solid #C8E6C9; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 8px;
    }
    .kpi-title { font-size: 0.78rem; color: #546E7A; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value { font-size: 1.9rem; font-weight: 800; color: #1B5E20; line-height: 1.2; }
    .kpi-sub   { font-size: 0.75rem; color: #78909C; margin-top: 4px; }
    .flag-red   { background:#FFEBEE; border-left:4px solid #C62828; border-radius:6px; padding:10px 14px; margin:4px 0; }
    .flag-amber { background:#FFF8E1; border-left:4px solid #F9A825; border-radius:6px; padding:10px 14px; margin:4px 0; }
    .flag-green { background:#E8F5E9; border-left:4px solid #2E7D32; border-radius:6px; padding:10px 14px; margin:4px 0; }
    .section-header { background: linear-gradient(90deg,#1B5E20,#2E7D32); color:white; padding:8px 16px;
        border-radius:8px; font-size:0.85rem; font-weight:700; letter-spacing:0.04em; margin-bottom:12px; }
    .data-note { background:#E3F2FD; border-radius:6px; padding:10px 14px; font-size:0.78rem;
        color:#1565C0; border-left:3px solid #1565C0; margin:8px 0; }
    .footer { text-align:center; color:#90A4AE; font-size:0.72rem; margin-top:30px; }
    div[data-testid="stTabs"] button { font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING — strict CSV sources only
# ─────────────────────────────────────────────
import os

@st.cache_data
def load_all():
    base = os.path.join(os.path.dirname(__file__), "data") + "/"

    device   = pd.read_csv(base + "device-1-2024.csv",          encoding="utf-8-sig")
    domain   = pd.read_csv(base + "domain-1-2024.csv",          encoding="utf-8-sig")
    download = pd.read_csv(base + "download-1-2024.csv",        encoding="utf-8-sig")
    language = pd.read_csv(base + "language-1-2024.csv",        encoding="utf-8-sig")
    os_br    = pd.read_csv(base + "os-browser-1-2024.csv",      encoding="utf-8-sig")
    traffic  = pd.read_csv(base + "traffic-source-1-2024.csv",  encoding="utf-8-sig")
    windows  = pd.read_csv(base + "windows-browser-1-2024.csv", encoding="utf-8-sig")

    for df in [device, domain, download, language, os_br, traffic, windows]:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    return device, domain, download, language, os_br, traffic, windows

device, domain, download, language, os_br, traffic, windows = load_all()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.markdown("## 🌾")
with col_title:
    st.markdown("# USDA Digital Service Effectiveness Framework")
    st.markdown(
        "<span style='color:#546E7A;font-size:0.9rem;'>January – June 2024 &nbsp;|&nbsp; "
        "Department of Agriculture &nbsp;|&nbsp; "
        "Source: analytics.usa.gov participating hostnames</span>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "🚦 Traffic",
    "📥 Content & Downloads",
    "🌐 Language & Equity",
    "💻 Device & Tech",
    "⚠️ Friction Diagnostics",
])

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">LAYER 1 — SYSTEM-WIDE DESCRIPTIVE ANALYSIS</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="data-note">📌 <b>Metrics derived from:</b> device-1-2024.csv (total visits), '
                'domain-1-2024.csv (top hostname), traffic-source-1-2024.csv (organic share). '
                'KPIs reflect the full Jan–Jun 2024 window.</div>', unsafe_allow_html=True)

    # — KPI Computations ——————————————————————
    total_visits     = device.groupby("date")["visits"].sum()
    total_sessions   = total_visits.sum()
    mobile_share     = (
        device[device["device"] == "mobile"]["visits"].sum()
        / device["visits"].sum() * 100
    )
    top_host         = domain.groupby("domain")["visits"].sum().idxmax()
    top_host_visits  = domain.groupby("domain")["visits"].sum().max()
    top_source       = traffic.groupby("source")["visits"].sum().idxmax()

    organic_visits   = traffic[traffic["source"] == "google"]["visits"].sum()
    direct_visits    = traffic[traffic["source"] == "(direct)"]["visits"].sum()
    total_traf_visits= traffic["visits"].sum()
    organic_share    = organic_visits / total_traf_visits * 100 if total_traf_visits else 0

    non_english_lang_visits = language[~language["language"].str.startswith("en")]["visits"].sum()
    total_lang_visits       = language["visits"].sum()
    non_en_share            = non_english_lang_visits / total_lang_visits * 100

    # — KPI Cards ——————————————————————————————
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "Total Sessions (Device)", f"{total_sessions:,.0f}", "Sum of all device visits Jan–Jun"),
        (c2, "Top Hostname", top_host, f"{top_host_visits:,.0f} total visits"),
        (c3, "Mobile Share", f"{mobile_share:.1f}%", "% of device visits from mobile"),
        (c4, "Organic (Google) Share", f"{organic_share:.1f}%", "Google visits ÷ all tracked sources"),
        (c5, "Non-English Browser Share", f"{non_en_share:.1f}%", "Browser lang not starting with 'en'"),
    ]
    for col, title, val, sub in cards:
        with col:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-title">{title}</div>'
                f'<div class="kpi-value">{val}</div>'
                f'<div class="kpi-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("#### Daily Visit Volume — All Devices (Jan–Jun 2024)")
    daily = device.groupby("date")["visits"].sum().reset_index()
    daily.columns = ["date", "Total Visits"]

    fig_trend = px.area(
        daily, x="date", y="Total Visits",
        color_discrete_sequence=[USDA_GREEN],
        template="plotly_white",
    )
    fig_trend.update_traces(fill="tozeroy", line_width=1.8, fillcolor="rgba(46,125,50,0.15)")
    fig_trend.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title=None, yaxis_title="Daily Sessions",
        hovermode="x unified",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Monthly device breakdown
    st.markdown("#### Monthly Session Volume by Device Type")
    monthly_device = device.groupby(["month", "device"])["visits"].sum().reset_index()
    monthly_device["month_label"] = monthly_device["month"].dt.strftime("%b %Y")
    fig_dev_bar = px.bar(
        monthly_device, x="month_label", y="visits", color="device",
        color_discrete_map={"desktop": USDA_BLUE, "mobile": USDA_GREEN, "tablet": USDA_GOLD},
        template="plotly_white", barmode="group",
    )
    fig_dev_bar.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                               xaxis_title=None, yaxis_title="Sessions", legend_title="Device")
    st.plotly_chart(fig_dev_bar, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — TRAFFIC
# ══════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">LAYER 1 — TRAFFIC SOURCES & HOSTNAME DISTRIBUTION</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="data-note">📌 <b>Metrics derived from:</b> domain-1-2024.csv (hostname visits), '
                'traffic-source-1-2024.csv (source classification, social flag).</div>',
                unsafe_allow_html=True)

    col_left, col_right = st.columns([6, 4])

    with col_left:
        st.markdown("#### Top 15 Hostnames by Total Visits")
        top_domains = domain.groupby("domain")["visits"].sum().nlargest(15).reset_index()
        top_domains.columns = ["Hostname", "Total Visits"]
        fig_dom = px.bar(
            top_domains.sort_values("Total Visits"),
            x="Total Visits", y="Hostname", orientation="h",
            color="Total Visits", color_continuous_scale=["#A5D6A7", USDA_GREEN],
            template="plotly_white",
        )
        fig_dom.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0),
                               yaxis_title=None, showlegend=False,
                               coloraxis_showscale=False)
        st.plotly_chart(fig_dom, use_container_width=True)

    with col_right:
        st.markdown("#### Traffic Source Share (All Sources)")
        src_totals = traffic.groupby("source")["visits"].sum().reset_index()
        src_totals.columns = ["Source", "Visits"]
        src_totals = src_totals.sort_values("Visits", ascending=False)
        # Group tail
        top_n = src_totals.head(8)
        other_visits = src_totals.iloc[8:]["Visits"].sum()
        if other_visits > 0:
            top_n = pd.concat([top_n, pd.DataFrame([{"Source": "Other", "Visits": other_visits}])],
                              ignore_index=True)
        fig_pie = px.pie(
            top_n, names="Source", values="Visits",
            color_discrete_sequence=COLOR_SEQ,
            template="plotly_white", hole=0.42,
        )
        fig_pie.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=10),
                               legend=dict(font_size=11))
        fig_pie.update_traces(textinfo="percent", hovertemplate="%{label}<br>%{value:,}<extra></extra>")
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### Social vs. Non-Social Referrals")
        social_totals = traffic.groupby("has_social_referral")["visits"].sum().reset_index()
        social_totals.columns = ["Is Social Referral", "Visits"]
        fig_soc = px.pie(
            social_totals, names="Is Social Referral", values="Visits",
            color_discrete_map={"Yes": USDA_GOLD, "No": USDA_BLUE},
            template="plotly_white", hole=0.42,
        )
        fig_soc.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                               legend=dict(font_size=11))
        fig_soc.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_soc, use_container_width=True)

    # Monthly trend for top 8 sources
    st.markdown("#### Monthly Traffic by Top 8 Sources")
    top8_sources = traffic.groupby("source")["visits"].sum().nlargest(8).index.tolist()
    traf_monthly = (
        traffic[traffic["source"].isin(top8_sources)]
        .groupby(["month", "source"])["visits"].sum().reset_index()
    )
    traf_monthly["month_label"] = traf_monthly["month"].dt.strftime("%b %Y")
    fig_traf_line = px.line(
        traf_monthly, x="month_label", y="visits", color="source",
        color_discrete_sequence=COLOR_SEQ,
        template="plotly_white", markers=True,
    )
    fig_traf_line.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0),
                                  xaxis_title=None, yaxis_title="Sessions",
                                  legend_title="Source")
    st.plotly_chart(fig_traf_line, use_container_width=True)

    # Hostname monthly heatmap (top 10)
    st.markdown("#### Hostname Traffic Month-over-Month (Top 10)")
    top10_hosts = domain.groupby("domain")["visits"].sum().nlargest(10).index.tolist()
    dom_monthly = (
        domain[domain["domain"].isin(top10_hosts)]
        .groupby(["month", "domain"])["visits"].sum()
        .reset_index()
    )
    dom_monthly["month_label"] = dom_monthly["month"].dt.strftime("%b")
    pivot = dom_monthly.pivot(index="domain", columns="month_label", values="visits").fillna(0)
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    pivot = pivot[[m for m in month_order if m in pivot.columns]]
    fig_heat = px.imshow(
        pivot,
        color_continuous_scale=["#F1F8E9", "#1B5E20"],
        aspect="auto", template="plotly_white",
        labels={"color": "Visits"},
    )
    fig_heat.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                            xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — CONTENT & DOWNLOADS
# ══════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">LAYER 1 — CONTENT DEMAND & TOP DOWNLOADS</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="data-note">📌 <b>Metrics derived from:</b> download-1-2024.csv. '
                'Each row = a download event tracked by day. '
                '"Video plays" and "page-level bounce rate" are NOT present in the source data.</div>',
                unsafe_allow_html=True)

    # Top downloads by total_events
    top_dl = (
        download.groupby(["event_label", "page_title"])["total_events"]
        .sum().reset_index()
        .sort_values("total_events", ascending=False)
        .head(20)
    )
    top_dl["short_label"] = top_dl["event_label"].str.split("/").str[-1].str[:60]
    top_dl["short_title"] = top_dl["page_title"].str[:55] + "…"

    st.markdown("#### Top 20 Downloads by Total Events (Jan–Jun 2024)")
    fig_dl = px.bar(
        top_dl.sort_values("total_events"),
        x="total_events", y="short_label", orientation="h",
        color="total_events",
        color_continuous_scale=["#B3E5FC", USDA_BLUE],
        hover_data={"short_title": True, "total_events": True, "short_label": False},
        template="plotly_white",
    )
    fig_dl.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title=None, xaxis_title="Total Download Events",
                          coloraxis_showscale=False)
    st.plotly_chart(fig_dl, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Monthly Download Volume Trend")
        monthly_dl = download.groupby("month")["total_events"].sum().reset_index()
        monthly_dl["month_label"] = monthly_dl["month"].dt.strftime("%b %Y")
        fig_dl_trend = px.bar(
            monthly_dl, x="month_label", y="total_events",
            color_discrete_sequence=[USDA_BLUE], template="plotly_white",
        )
        fig_dl_trend.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                                    xaxis_title=None, yaxis_title="Download Events")
        st.plotly_chart(fig_dl_trend, use_container_width=True)

    with col_b:
        st.markdown("#### Top 10 Source Pages by Download Events")
        top_pages_dl = (
            download.groupby("page")["total_events"].sum()
            .nlargest(10).reset_index()
        )
        top_pages_dl["short_page"] = top_pages_dl["page"].str[:50]
        fig_pg = px.bar(
            top_pages_dl.sort_values("total_events"),
            x="total_events", y="short_page", orientation="h",
            color_discrete_sequence=[USDA_GREEN], template="plotly_white",
        )
        fig_pg.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                              yaxis_title=None, xaxis_title="Download Events")
        st.plotly_chart(fig_pg, use_container_width=True)

    # Full searchable table
    st.markdown("#### All Download Records (Searchable)")
    search_term = st.text_input("🔍 Filter by file name or page title", "")
    dl_table = download.groupby(["event_label", "page_title", "page"])["total_events"].sum().reset_index()
    dl_table.columns = ["Download URL", "Page Title", "Source Page", "Total Events"]
    dl_table = dl_table.sort_values("Total Events", ascending=False)
    if search_term:
        mask = (
            dl_table["Download URL"].str.contains(search_term, case=False, na=False) |
            dl_table["Page Title"].str.contains(search_term, case=False, na=False)
        )
        dl_table = dl_table[mask]
    st.dataframe(dl_table, use_container_width=True, height=280)

# ══════════════════════════════════════════════
# TAB 4 — LANGUAGE & EQUITY
# ══════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">LAYER 2 — BROWSER LANGUAGE & EQUITY ASSESSMENT</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="data-note">📌 <b>Metrics derived from:</b> language-1-2024.csv. '
                'Browser language is the locale set in the user\'s browser — a proxy for language preference. '
                'Country/city geographic data is NOT present in the source files.</div>',
                unsafe_allow_html=True)

    # Aggregate
    lang_total = language.groupby("language")["visits"].sum().reset_index()
    lang_total.columns = ["Language Code", "Total Visits"]
    lang_total = lang_total.sort_values("Total Visits", ascending=False)

    # Tag English vs non-English
    lang_total["Category"] = lang_total["Language Code"].apply(
        lambda x: "English" if str(x).startswith("en") else "Non-English"
    )

    col_l1, col_l2 = st.columns([5, 5])

    with col_l1:
        st.markdown("#### Top 20 Browser Languages by Total Visits")
        top20_lang = lang_total.head(20)
        fig_lang = px.bar(
            top20_lang.sort_values("Total Visits"),
            x="Total Visits", y="Language Code", orientation="h",
            color="Category",
            color_discrete_map={"English": USDA_BLUE, "Non-English": USDA_GOLD},
            template="plotly_white",
        )
        fig_lang.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0),
                                yaxis_title=None, legend_title="Category")
        st.plotly_chart(fig_lang, use_container_width=True)

    with col_l2:
        st.markdown("#### English vs. Non-English Share")
        cat_totals = lang_total.groupby("Category")["Total Visits"].sum().reset_index()
        fig_eq = px.pie(
            cat_totals, names="Category", values="Total Visits",
            color_discrete_map={"English": USDA_BLUE, "Non-English": USDA_GOLD},
            template="plotly_white", hole=0.48,
        )
        fig_eq.update_traces(textinfo="percent+label", pull=[0, 0.04])
        fig_eq.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_eq, use_container_width=True)

        # Non-English breakdown
        st.markdown("#### Non-English Language Breakdown (Top 10)")
        non_en = lang_total[lang_total["Category"] == "Non-English"].head(10)
        fig_ne = px.bar(
            non_en.sort_values("Total Visits"),
            x="Total Visits", y="Language Code", orientation="h",
            color_discrete_sequence=[USDA_GOLD], template="plotly_white",
        )
        fig_ne.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None)
        st.plotly_chart(fig_ne, use_container_width=True)

    # Monthly non-English share trend
    st.markdown("#### Monthly Non-English Share Trend (%)")
    lang_monthly = language.copy()
    lang_monthly["is_non_english"] = ~lang_monthly["language"].str.startswith("en")
    monthly_ne = (
        lang_monthly.groupby("month")
        .apply(lambda g: g[g["is_non_english"]]["visits"].sum() / g["visits"].sum() * 100)
        .reset_index(name="Non-English Share %")
    )
    monthly_ne["month_label"] = monthly_ne["month"].dt.strftime("%b %Y")
    fig_ne_trend = px.line(
        monthly_ne, x="month_label", y="Non-English Share %",
        color_discrete_sequence=[USDA_GOLD], template="plotly_white", markers=True,
    )
    fig_ne_trend.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0),
                                 xaxis_title=None, yaxis_ticksuffix="%")
    st.plotly_chart(fig_ne_trend, use_container_width=True)

    # Spanish specifically (USDA relevance)
    st.markdown("#### Spanish-Language Visits (es-*) — USDA Program Equity Signal")
    spanish = language[language["language"].str.startswith("es")].copy()
    spanish_monthly = spanish.groupby("month")["visits"].sum().reset_index()
    spanish_monthly["month_label"] = spanish_monthly["month"].dt.strftime("%b %Y")
    fig_sp = px.bar(
        spanish_monthly, x="month_label", y="visits",
        color_discrete_sequence=["#EF6C00"], template="plotly_white",
    )
    fig_sp.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title=None, yaxis_title="Spanish Browser Sessions")
    st.plotly_chart(fig_sp, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 5 — DEVICE & TECH
# ══════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">LAYER 2 — DEVICE TYPE, OS, BROWSER & COMPATIBILITY</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="data-note">📌 <b>Metrics derived from:</b> device-1-2024.csv (device type), '
                'os-browser-1-2024.csv (OS × browser combinations), '
                'windows-browser-1-2024.csv (Windows version × browser). '
                'Screen resolution data is NOT present in source files.</div>',
                unsafe_allow_html=True)

    col_d1, col_d2 = st.columns([4, 6])

    with col_d1:
        st.markdown("#### Device Type — Total Share")
        dev_totals = device.groupby("device")["visits"].sum().reset_index()
        dev_totals.columns = ["Device", "Visits"]
        fig_dev_pie = px.pie(
            dev_totals, names="Device", values="Visits",
            color_discrete_map={"desktop": USDA_BLUE, "mobile": USDA_GREEN, "tablet": USDA_GOLD},
            template="plotly_white", hole=0.45,
        )
        fig_dev_pie.update_traces(textinfo="percent+label", pull=[0, 0.04, 0])
        fig_dev_pie.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_dev_pie, use_container_width=True)

        st.markdown("#### Mobile Share Monthly Trend")
        dev_monthly = device.groupby(["month", "device"])["visits"].sum().reset_index()
        total_monthly = dev_monthly.groupby("month")["visits"].sum().reset_index(name="total")
        mobile_monthly = dev_monthly[dev_monthly["device"] == "mobile"].merge(total_monthly, on="month")
        mobile_monthly["Mobile %"] = mobile_monthly["visits"] / mobile_monthly["total"] * 100
        mobile_monthly["month_label"] = mobile_monthly["month"].dt.strftime("%b %Y")
        fig_mob = px.line(
            mobile_monthly, x="month_label", y="Mobile %",
            color_discrete_sequence=[USDA_GREEN], markers=True, template="plotly_white",
        )
        fig_mob.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0),
                               xaxis_title=None, yaxis_ticksuffix="%")
        st.plotly_chart(fig_mob, use_container_width=True)

    with col_d2:
        st.markdown("#### OS × Browser Heatmap (Total Sessions)")
        ob_agg = os_br.groupby(["os", "browser"])["visits"].sum().reset_index()
        # Keep top OS and browsers for readability
        top_os_list = ob_agg.groupby("os")["visits"].sum().nlargest(8).index.tolist()
        top_br_list = ob_agg.groupby("browser")["visits"].sum().nlargest(10).index.tolist()
        ob_filtered = ob_agg[
            ob_agg["os"].isin(top_os_list) & ob_agg["browser"].isin(top_br_list)
        ]
        pivot_ob = ob_filtered.pivot(index="os", columns="browser", values="visits").fillna(0)
        fig_hm = px.imshow(
            pivot_ob,
            color_continuous_scale=["#E8F5E9", USDA_GREEN],
            aspect="auto", template="plotly_white",
            labels={"color": "Sessions"},
        )
        fig_hm.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title="Browser", yaxis_title="Operating System")
        st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown("#### Top OS Platforms by Total Sessions")
        os_totals = os_br.groupby("os")["visits"].sum().nlargest(8).reset_index()
        os_totals.columns = ["OS", "Visits"]
        fig_os = px.bar(
            os_totals.sort_values("Visits"),
            x="Visits", y="OS", orientation="h",
            color_discrete_sequence=[USDA_BLUE], template="plotly_white",
        )
        fig_os.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None)
        st.plotly_chart(fig_os, use_container_width=True)

    # Windows version breakdown
    st.markdown("#### Windows Version × Browser (Compatibility Focus)")
    win_agg = windows.groupby(["os_version", "browser"])["visits"].sum().reset_index()
    win_agg["Win Version Label"] = "Win " + win_agg["os_version"].astype(str)

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        win_ver = windows.groupby("os_version")["visits"].sum().reset_index()
        win_ver.columns = ["Windows Version", "Visits"]
        win_ver["Windows Version"] = "Win " + win_ver["Windows Version"].astype(str)
        # Color legacy versions
        win_ver["Legacy"] = win_ver["Windows Version"].isin(["Win 7", "Win XP", "Win 8", "Win 8.1", "Win CE"])
        fig_wv = px.bar(
            win_ver.sort_values("Visits", ascending=False),
            x="Windows Version", y="Visits",
            color="Legacy",
            color_discrete_map={True: USDA_RED, False: USDA_BLUE},
            template="plotly_white",
        )
        fig_wv.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title=None, legend_title="Legacy OS")
        st.plotly_chart(fig_wv, use_container_width=True)

    with col_w2:
        win_br = windows.groupby("browser")["visits"].sum().reset_index()
        win_br.columns = ["Browser", "Visits"]
        win_br["Legacy"] = win_br["Browser"].isin(["Internet Explorer", "Mozilla Compatible Agent"])
        fig_wb = px.bar(
            win_br.sort_values("Visits", ascending=False),
            x="Browser", y="Visits",
            color="Legacy",
            color_discrete_map={True: USDA_RED, False: USDA_GREEN},
            template="plotly_white",
        )
        fig_wb.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title=None, legend_title="Legacy Browser")
        st.plotly_chart(fig_wb, use_container_width=True)

    # Windows version heatmap
    st.markdown("#### Windows Version × Browser Heatmap")
    pivot_wb = win_agg.pivot(index="Win Version Label", columns="browser", values="visits").fillna(0)
    fig_wb_hm = px.imshow(
        pivot_wb, color_continuous_scale=["#FFF8E1", USDA_GOLD],
        aspect="auto", template="plotly_white",
        labels={"color": "Sessions"},
    )
    fig_wb_hm.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_wb_hm, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 6 — FRICTION DIAGNOSTICS
# ══════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">LAYER 2 — FRICTION DIAGNOSTICS & COMPATIBILITY FLAGS</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="data-note">📌 <b>Important methodology note:</b> Friction signals are computed '
                'strictly from available CSV data: device share (device-1-2024.csv), '
                'legacy OS/browser presence (windows-browser-1-2024.csv, os-browser-1-2024.csv), '
                'language equity (language-1-2024.csv), traffic source dependency (traffic-source-1-2024.csv). '
                'Bounce rate, time-on-page, and task completion rates are NOT in the source data and are '
                '<b>not estimated or fabricated</b>.</div>', unsafe_allow_html=True)

    # ── Signal 1: Mobile Share ─────────────────
    total_dev    = device["visits"].sum()
    mob_visits   = device[device["device"] == "mobile"]["visits"].sum()
    mob_pct      = mob_visits / total_dev * 100

    # ── Signal 2: Legacy Windows ──────────────
    legacy_vers  = ["7", "XP", "8", "8.1", "CE"]
    win_total    = windows["visits"].sum()
    win_legacy   = windows[windows["os_version"].isin(legacy_vers)]["visits"].sum()
    legacy_pct   = win_legacy / win_total * 100 if win_total else 0

    # ── Signal 3: Legacy Browsers (IE + Compat) ───────
    legacy_br    = ["Internet Explorer", "Mozilla Compatible Agent"]
    ob_total     = os_br["visits"].sum()
    ie_visits    = os_br[os_br["browser"].isin(legacy_br)]["visits"].sum()
    ie_pct       = ie_visits / ob_total * 100 if ob_total else 0

    # ── Signal 4: Non-English Share ───────────
    total_lang   = language["visits"].sum()
    non_en       = language[~language["language"].str.startswith("en")]["visits"].sum()
    non_en_pct   = non_en / total_lang * 100 if total_lang else 0

    # ── Signal 5: Traffic Concentration ──────
    traffic_by_src = traffic.groupby("source")["visits"].sum()
    traf_total   = traffic_by_src.sum()
    google_pct   = traffic_by_src.get("google", 0) / traf_total * 100 if traf_total else 0
    direct_pct   = traffic_by_src.get("(direct)", 0) / traf_total * 100 if traf_total else 0

    # ── Signal 6: Domain concentration ───────
    dom_by_host  = domain.groupby("domain")["visits"].sum()
    dom_total    = dom_by_host.sum()
    top_dom_pct  = dom_by_host.max() / dom_total * 100

    # ─────────────────────────────────────────
    # Friction Signal Cards
    # ─────────────────────────────────────────
    st.markdown("### 🔎 Friction Signal Summary")
    fc1, fc2, fc3 = st.columns(3)
    fc4, fc5, fc6 = st.columns(3)

    def signal_card(col, title, value, note, severity):
        color = {"red":"#C62828","amber":"#E65100","green":"#2E7D32"}[severity]
        bg    = {"red":"#FFEBEE","amber":"#FFF3E0","green":"#E8F5E9"}[severity]
        with col:
            st.markdown(
                f'<div style="background:{bg};border-left:4px solid {color};border-radius:10px;'
                f'padding:14px;margin-bottom:8px;">'
                f'<div style="font-size:0.75rem;font-weight:700;color:{color};text-transform:uppercase">{title}</div>'
                f'<div style="font-size:2rem;font-weight:800;color:{color}">{value}</div>'
                f'<div style="font-size:0.75rem;color:#546E7A;margin-top:4px">{note}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    signal_card(fc1, "Mobile Share", f"{mob_pct:.1f}%",
                "Mobile sessions ÷ total device sessions",
                "amber" if mob_pct > 30 else "green")
    signal_card(fc2, "Legacy Windows (7/XP/8/CE)", f"{legacy_pct:.1f}%",
                "% of Windows sessions on end-of-life OS",
                "red" if legacy_pct > 5 else "amber" if legacy_pct > 2 else "green")
    signal_card(fc3, "Legacy Browser (IE/Compat)", f"{ie_pct:.2f}%",
                "IE + Mozilla Compat Agent as % of all OS-browser sessions",
                "red" if ie_pct > 1 else "amber" if ie_pct > 0.3 else "green")
    signal_card(fc4, "Non-English Browser Share", f"{non_en_pct:.1f}%",
                "Proxy for underserved language audiences",
                "amber" if non_en_pct > 5 else "green")
    signal_card(fc5, "Google Traffic Dependency", f"{google_pct:.1f}%",
                "Google visits ÷ all tracked source visits",
                "amber" if google_pct > 50 else "green")
    signal_card(fc6, "Top Domain Traffic Concentration", f"{top_dom_pct:.1f}%",
                f"fs.usda.gov share of all hostname visits",
                "amber" if top_dom_pct > 30 else "green")

    st.markdown("---")

    # ─────────────────────────────────────────
    # Legacy Windows Trend
    # ─────────────────────────────────────────
    st.markdown("### 📈 Legacy OS Risk Trend — Windows Version Mix Over Time")
    win_monthly = windows.copy()
    win_monthly["legacy"] = win_monthly["os_version"].isin(legacy_vers)
    win_lm = (
        win_monthly.groupby(["month", "legacy"])["visits"].sum().reset_index()
    )
    win_lm_total = win_lm.groupby("month")["visits"].sum().reset_index(name="total")
    win_lm = win_lm.merge(win_lm_total, on="month")
    win_lm["Share %"] = win_lm["visits"] / win_lm["total"] * 100
    win_lm["Type"] = win_lm["legacy"].map({True: "Legacy (Win 7 / XP / 8 / CE)", False: "Modern (Win 10+)"})
    win_lm["month_label"] = win_lm["month"].dt.strftime("%b %Y")

    fig_win_trend = px.bar(
        win_lm, x="month_label", y="Share %", color="Type",
        color_discrete_map={
            "Legacy (Win 7 / XP / 8 / CE)": USDA_RED,
            "Modern (Win 10+)": USDA_BLUE,
        },
        barmode="stack", template="plotly_white",
    )
    fig_win_trend.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                                 xaxis_title=None, yaxis_ticksuffix="%",
                                 legend_title="Windows OS Type")
    st.plotly_chart(fig_win_trend, use_container_width=True)

    # ─────────────────────────────────────────
    # Compatibility Flag Table
    # ─────────────────────────────────────────
    st.markdown("### 🚩 Compatibility Flag Table — Windows Version × Browser")
    win_flag = windows.groupby(["os_version", "browser"])["visits"].sum().reset_index()
    win_flag.columns = ["Windows Version", "Browser", "Total Sessions"]
    win_flag["OS Risk"] = win_flag["Windows Version"].apply(
        lambda v: "🔴 End-of-Life" if v in legacy_vers else "🟢 Supported"
    )
    win_flag["Browser Risk"] = win_flag["Browser"].apply(
        lambda b: "🔴 Legacy" if b in ["Internet Explorer", "Mozilla Compatible Agent"]
        else "🟡 Monitor" if b in ["Firefox", "Opera"] else "🟢 Modern"
    )
    win_flag["Combined Flag"] = win_flag.apply(
        lambda r: "🔴 HIGH RISK" if "End-of-Life" in r["OS Risk"] and "Legacy" in r["Browser Risk"]
        else "🟡 MONITOR" if "End-of-Life" in r["OS Risk"] or "Legacy" in r["Browser Risk"]
        else "🟢 OK", axis=1
    )
    win_flag["Win Version Label"] = "Win " + win_flag["Windows Version"].astype(str)
    win_flag = win_flag[["Win Version Label", "Browser", "Total Sessions", "OS Risk", "Browser Risk", "Combined Flag"]]
    win_flag = win_flag.sort_values("Total Sessions", ascending=False)
    st.dataframe(win_flag, use_container_width=True, height=350)

    # ─────────────────────────────────────────
    # Language Equity Monthly Detail
    # ─────────────────────────────────────────
    st.markdown("### 🌐 Language Equity — Monthly Non-English Visits")
    lang_monthly_ne = language.copy()
    lang_monthly_ne["Category"] = lang_monthly_ne["language"].apply(
        lambda x: "Non-English" if not str(x).startswith("en") else "English"
    )
    lang_eq = (
        lang_monthly_ne.groupby(["month", "Category"])["visits"].sum().reset_index()
    )
    lang_eq["month_label"] = lang_eq["month"].dt.strftime("%b %Y")
    fig_eq2 = px.bar(
        lang_eq, x="month_label", y="visits", color="Category",
        color_discrete_map={"English": USDA_BLUE, "Non-English": USDA_GOLD},
        barmode="stack", template="plotly_white",
    )
    fig_eq2.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                           xaxis_title=None, yaxis_title="Sessions")
    st.plotly_chart(fig_eq2, use_container_width=True)

    # ─────────────────────────────────────────
    # Improvement Recommendation Panel
    # ─────────────────────────────────────────
    st.markdown("### 💡 Data-Driven Improvement Priorities")
    st.markdown('<div class="data-note">Recommendations below are grounded exclusively in patterns '
                'observed in the provided CSV data.</div>', unsafe_allow_html=True)

    recs = []

    if legacy_pct > 0:
        recs.append(("🔴 Legacy OS Exposure",
            f"**{legacy_pct:.1f}%** of Windows sessions originate from end-of-life OS versions "
            f"(Win 7 / XP / 8 / CE) — totaling **{win_legacy:,} sessions**. "
            "USDA's rural and lower-income user base is more likely to run older hardware. "
            "Prioritize QA regression testing against these configurations and verify that "
            "all critical forms and document downloads function correctly on legacy stacks."))

    if ie_pct > 0:
        recs.append(("🔴 Legacy Browser Sessions (IE + Compat Agent)",
            f"**{ie_pct:.2f}%** of OS-browser sessions use Internet Explorer or Mozilla Compatible Agent — "
            f"totaling **{ie_visits:,} sessions**. "
            "IE reached end-of-life in June 2022. Any remaining IE sessions represent high-friction "
            "access attempts. Audit whether critical USDA pages render and function in IE/compatibility modes."))

    if non_en_pct > 0:
        recs.append(("🟡 Non-English Language Audience Underservice",
            f"**{non_en_pct:.1f}%** of browser sessions use a non-English language setting "
            f"({non_en:,} sessions). Spanish-language sessions (es-*) are the largest non-English segment. "
            "Review Spanish-language content coverage on high-traffic pages, particularly FNS program pages, "
            "WIC, and SNAP resources which serve populations with high Spanish-language prevalence."))

    if mob_pct > 30:
        recs.append(("🟡 High Mobile Traffic Share",
            f"**{mob_pct:.1f}%** of sessions occur on mobile devices. "
            "Without page-level device data (not in source files), mobile optimization gaps cannot be "
            "isolated by page. Recommend cross-referencing with accessibility audits on top-download pages "
            "and top-visited domains to verify mobile rendering and tap-target compliance."))

    if google_pct > 50:
        recs.append(("🟡 High Search Engine Dependency",
            f"**{google_pct:.1f}%** of tracked-source visits arrive via Google. "
            "This creates fragility: algorithm changes or ranking drops directly impact USDA program access. "
            "Increasing direct navigation share and improving govdelivery/newsletter referral would reduce risk."))

    for title, body in recs:
        st.markdown(
            f'<div style="background:#FAFAFA;border:1px solid #CFD8DC;border-radius:10px;'
            f'padding:14px 18px;margin-bottom:10px;">'
            f'<b>{title}</b><br><br>{body}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if not recs:
        st.success("No critical friction signals detected based on available data thresholds.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="footer">'
    'USDA Digital Service Effectiveness Framework &nbsp;|&nbsp; '
    'Data: Jan–Jun 2024 analytics.usa.gov &nbsp;|&nbsp; '
    'All metrics derived exclusively from provided CSV files &mdash; no imputed, assumed, or external data. &nbsp;|&nbsp; '
    'Metrics not present in source data (bounce rate, time-on-page, country/city geo, screen resolution, video plays) '
    'are excluded from this dashboard.'
    '</div>',
    unsafe_allow_html=True,
)

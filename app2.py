import os
import pathlib
import io
import zipfile
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ── Path resolution ────────────────────────────────────────────────────────────
try:
    DATA_DIR = pathlib.Path(__file__).parent.resolve()
except NameError:
    DATA_DIR = pathlib.Path.cwd()

def _find_file(name):
    bases = (DATA_DIR, pathlib.Path.cwd(), DATA_DIR / "data", pathlib.Path.cwd() / "data")
    for base in bases:
        p = base / name
        if p.exists():
            return str(p)
        p = base / (pathlib.Path(name).stem + ".zip")
        if p.exists():
            return str(p)
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

# ── Constants ──────────────────────────────────────────────────────────────────
AGENCY_MAP = {
    "fns.usda.gov": "Food and Nutrition Service",
    "ams.usda.gov": "Agricultural Marketing Service",
    "nrcs.usda.gov": "Natural Resources Conservation Service",
}

CLUSTER_INFO = {
    "Core Program": {
        "color": "#bbdefb",
        "dot": "🔵",
        "desc": "Low bounce, healthy session duration, reasonable stickiness. Users arrive, engage with content, and sometimes return. The site is working as intended for these pages.",
        "examples": "Housing loan pages, SF housing programs — mainstream service content.",
        "action": "Maintain content quality and page load speed. Use these pages as the benchmark for what good engagement looks like across the site.",
    },
    "Power User": {
        "color": "#c8e6c9",
        "dot": "🟢",
        "desc": "Highest stickiness of any cluster — people come back repeatedly. Duration is moderate. These are professional users: lenders, housing counselors, program administrators.",
        "examples": "Specialized reference content relied on by program administrators.",
        "action": "Prioritize stability and accuracy over redesign. Ensure content is updated on schedule. Consider adding bookmarking or reference-linking features.",
    },
    "Discovery": {
        "color": "#fff9c4",
        "dot": "🟡",
        "desc": "Moderate-to-high bounce and moderate duration. Users are exploring but many leave without finding what they need. Partially engaged but not converting to action.",
        "examples": "Program overview pages, general information sections.",
        "action": "Add clear calls-to-action. Improve internal navigation and cross-linking. Conduct user testing to find where journeys break down.",
    },
    "High Friction": {
        "color": "#ffcdd2",
        "dot": "🔴",
        "desc": "Very high bounce (83%+) and very short duration (under 15 seconds). Users arrive and almost immediately leave. Content is irrelevant, too hard to navigate, or inaccessible.",
        "examples": "Tribal Relations, Civil Rights Office, Reports pages.",
        "action": "Urgent content audit needed. Improve search-result alignment. Redesign information architecture. Conduct accessibility review.",
    },
}

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_system_data():
    def p(name):
        path = _find_file(name)
        if path is None:
            raise FileNotFoundError(
                f"Cannot find '{name}'. Commit all CSV files to the repo alongside app2.py."
            )
        return path

    def read(name):
        path = p(name)
        if path.endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
                with zf.open(csv_name) as f:
                    return pd.read_csv(f)
        return pd.read_csv(path)

    device    = read("device-1-2024.csv")
    domain    = read("domain-1-2024.csv")
    downloads = read("download-1-2024.csv")
    language  = read("language-1-2024.csv")
    traffic   = read("traffic-source-1-2024.csv")

    for df in (device, domain, downloads, language, traffic):
        df["date"]  = pd.to_datetime(df["date"], dayfirst=False)
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    return device, domain, downloads, language, traffic


@st.cache_data
def load_rd_data(file_source):
    """Accept a file path (str) to a csv or zip, or an UploadedFile object."""
    if isinstance(file_source, str):
        if file_source.endswith(".zip"):
            with zipfile.ZipFile(file_source) as zf:
                csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
                with zf.open(csv_name) as f:
                    raw = pd.read_csv(f, header=None, low_memory=False)
        else:
            raw = pd.read_csv(file_source, header=None, low_memory=False)
    else:
        if file_source.name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(file_source.read())) as zf:
                csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
                with zf.open(csv_name) as f:
                    raw = pd.read_csv(f, header=None, low_memory=False)
            file_source.seek(0)
        else:
            raw = pd.read_csv(file_source, header=None, low_memory=False)

    # ── Flatten two-row header (rows 6 & 7) ───────────────────────────────────
    h1, h2 = list(raw.iloc[6]), list(raw.iloc[7])
    cols, cur = [], ""
    for a, b in zip(h1, h2):
        if pd.notna(a) and str(a).strip():
            cur = str(a).strip()
        b_str = str(b).strip() if pd.notna(b) else ""
        if b_str:
            cols.append(cur + "_" + b_str if (cur and cur.split()[0].lower() not in b_str.lower()) else b_str)
        else:
            cols.append(cur or "unnamed")

    # Data starts at row 9 (row 8 is totals header)
    df = raw.iloc[9:].copy()
    df.columns = cols[: len(df.columns)]
    df = df.iloc[:, :-1]          # drop trailing empty column

    title_col   = cols[0]         # Page title
    section_col = cols[1]         # Section label
    path_col    = cols[5]         # Page path

    df = df[df[title_col].notna() & (df[title_col].astype(str).str.strip() != "")]
    df = df.rename(columns={
        title_col:   "Page title",
        section_col: "Section",
        cols[2]:     "Month",
        cols[3]:     "Day",
        path_col:    "Page path",
    })

    non_num = {"Page title", "Section", "Month", "Day", "Page path", cols[4]}
    for c in df.columns:
        if c not in non_num:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── Derive Section from page path where the CSV label is generic ───────────
    def _section(path):
        if pd.isna(path):
            return "Other"
        seg = str(path).strip("/").split("/")[0]
        return "Home" if not seg else seg.replace("-", " ").title()

    df["Section"] = df["Section"].where(
        df["Section"].notna()
        & (df["Section"].astype(str).str.strip() != "")
        & (df["Section"] != "Other"),
        df["Page path"].apply(_section),
    )

    # ── Column positions (verified against real data) ──────────────────────────
    # desktop: bounce=11, duration=10 | mobile: bounce=20, duration=19
    # tablet:  bounce=29, duration=28
    # totals:  active=42, sessions=44, views=45, duration=46, bounce=47,
    #          exits=48, returning=49, total_users=50
    def _c(i):
        return df.columns[i] if i < len(df.columns) else None

    df["Device Gap Score"]    = df[_c(20)].sub(df[_c(11)])
    df["Exit Pressure Index"] = df[_c(48)].div(df[_c(44)].replace(0, np.nan))
    df["Stickiness Ratio"]    = df[_c(49)].div(df[_c(50)].replace(0, np.nan))

    df = df.rename(columns={
        _c(47): "Bounce Rate",
        _c(46): "Session Duration",
        _c(45): "Views per Session",
        _c(42): "Active Users",
        _c(48): "Totals Exits",
        _c(44): "Totals Sessions",
        _c(49): "Returning Users",
        _c(50): "Total Users",
    })
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")

    # ── Aggregate to page level ────────────────────────────────────────────────
    RATE_COLS  = ["Bounce Rate", "Session Duration", "Views per Session",
                  "Exit Pressure Index", "Stickiness Ratio", "Device Gap Score"]
    COUNT_COLS = ["Active Users", "Returning Users", "Total Users"]
    agg = {c: "mean" for c in RATE_COLS if c in df.columns}
    agg.update({c: "sum" for c in COUNT_COLS if c in df.columns})
    agg["Section"] = "first"
    agg["Month"]   = "min"

    page = df.groupby("Page title").agg(agg).reset_index()
    page = page[page["Session Duration"] <= 1000]

    page["Underserved Score"] = (
        page["Bounce Rate"].fillna(0) * 0.4
        + page["Exit Pressure Index"].fillna(0) * 0.3
        + (1 - page["Stickiness Ratio"].fillna(0)) * 0.3
    )

    FEAT = ["Bounce Rate", "Session Duration", "Views per Session",
            "Exit Pressure Index", "Stickiness Ratio", "Device Gap Score"]
    scaled = StandardScaler().fit_transform(page[FEAT].fillna(0))

    return page, scaled, FEAT, df   # df = daily-level


@st.cache_data
def run_clustering(_page_df, _scaled):
    """Run k=4 KMeans and assign semantic labels based on cluster characteristics."""
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(_scaled)
    df = _page_df.copy()
    df["_cluster"] = labels

    stats = df.groupby("_cluster").agg(
        bounce    =("Bounce Rate",      "mean"),
        duration  =("Session Duration", "mean"),
        stickiness=("Stickiness Ratio", "mean"),
    )
    used = set()
    def pick(mask_fn, col, best):
        sub = stats[~stats.index.isin(used)]
        idx = sub[col].idxmax() if best == "max" else sub[col].idxmin()
        used.add(idx); return idx

    hf   = pick(None, "bounce",     "max")
    pu   = pick(None, "stickiness", "max")
    cp   = pick(None, "bounce",     "min")
    disc = [i for i in stats.index if i not in used][0]

    label_map = {hf: "High Friction", pu: "Power User", cp: "Core Program", disc: "Discovery"}
    df["Cluster Label"] = df["_cluster"].map(label_map)
    df = df.drop(columns=["_cluster"])
    return df


# ── OpenAI chat ────────────────────────────────────────────────────────────────
def build_system_prompt():
    return (
        "You are an expert digital analytics consultant embedded in a USDA website analytics dashboard. "
        "The dashboard covers January–June 2024 system-wide web traffic across all USDA agencies, plus "
        "detailed Rural Development engagement data. "
        "You help non-technical USDA decision-makers interpret charts and take action. "
        "Be concise, specific, and practical. Avoid jargon. "
        "When asked about a chart or metric, explain what it means for the agency's mission."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="USDA Digital Analytics Dashboard", layout="wide")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("USDA Analytics Dashboard")
st.sidebar.markdown("---")

RD_FILENAME   = "(Rural Development) Edited USDA data base.csv"
rd_default    = _find_file(RD_FILENAME)
rd_upload     = st.sidebar.file_uploader(
    "Override RD data (optional)",
    type=["csv", "zip"],
    help="Only needed if the bundled file is not in the repository.",
)

if rd_upload is not None:
    rd_source, rd_ok = rd_upload, True
    st.sidebar.success("Using uploaded file.")
elif rd_default:
    rd_source, rd_ok = rd_default, True
    st.sidebar.info("RD data loaded from repository.")
else:
    rd_source, rd_ok = None, False
    st.sidebar.error(f"'{RD_FILENAME}' not found. Upload it above.")

# ── ChatGPT Agent ──────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("💬 AI Analyst")

if not OPENAI_AVAILABLE:
    st.sidebar.warning("Install `openai` to enable AI chat.")
else:
    oai_key  = st.sidebar.text_input("OpenAI API Key", type="password", key="oai_key")
    asst_id  = st.sidebar.text_input(
        "Assistant ID (optional)", key="oai_asst",
        placeholder="asst_…  leave blank to use GPT-4o",
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    chat_box = st.sidebar.container()
    with chat_box:
        for m in st.session_state.messages:
            prefix = "**You:** " if m["role"] == "user" else "**AI:** "
            st.sidebar.markdown(prefix + m["content"])

    user_q = st.sidebar.text_input("Ask a question…", key="chat_q", label_visibility="collapsed",
                                    placeholder="Ask about the data…")
    send   = st.sidebar.button("Send", use_container_width=True)
    if st.sidebar.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if send and user_q.strip() and oai_key:
        st.session_state.messages.append({"role": "user", "content": user_q.strip()})
        try:
            client = OpenAI(api_key=oai_key)
            if asst_id.strip():
                thread = client.beta.threads.create()
                client.beta.threads.messages.create(
                    thread_id=thread.id, role="user", content=user_q.strip()
                )
                run = client.beta.threads.runs.create_and_poll(
                    thread_id=thread.id, assistant_id=asst_id.strip()
                )
                msgs = client.beta.threads.messages.list(thread_id=thread.id)
                reply = msgs.data[0].content[0].text.value
            else:
                api_msgs = [{"role": "system", "content": build_system_prompt()}]
                api_msgs += [{"role": m["role"], "content": m["content"]}
                             for m in st.session_state.messages]
                resp  = client.chat.completions.create(model="gpt-4o", messages=api_msgs)
                reply = resp.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "Layer 1 · System-Wide Analysis",
    "Layer 2 · Rural Development Baseline",
    "Layer 3 · Clustering & Underserved Analysis",
])

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — System-Wide Descriptive Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Layer 1: System-Wide Descriptive Analysis")
    st.caption("What is happening across USDA's web presence? Jan – Jun 2024.")
    st.markdown("---")

    try:
        dev_df, dom_df, dl_df, lang_df, src_df = load_system_data()

        # ── 1. Agency Traffic ─────────────────────────────────────────────────
        st.subheader("Agency Traffic: Top 15 USDA Hostnames by Total Visits")

        host_totals = dom_df.groupby("domain")["visits"].sum().reset_index()
        host_totals["label"] = host_totals["domain"].map(
            lambda d: f"{d}  —  {AGENCY_MAP[d]}" if d in AGENCY_MAP else d
        )
        top15 = host_totals.nlargest(15, "visits").sort_values("visits")

        fig_host = go.Figure(go.Bar(
            x=top15["visits"],
            y=top15["label"],
            orientation="h",
            marker_color="#1565c0",
            text=top15["visits"].apply(lambda v: f"{v/1e6:.1f}M"),
            textposition="outside",
        ))
        fig_host.update_layout(
            xaxis_title="Total Visits (Jan – Jun)",
            yaxis_title="",
            height=480,
            margin=dict(l=10, r=80, t=20, b=40),
            xaxis=dict(tickformat=","),
        )
        st.plotly_chart(fig_host, use_container_width=True)
        st.caption(
            "Agencies with the most visits represent USDA's highest-demand public services. "
            "Forest Service and Food & Nutrition Service dominate. Smaller agencies with growing "
            "traffic may need additional investment in content infrastructure."
        )

        st.markdown("---")

        # ── 2. Monthly Traffic by Source + Google/Social KPIs ─────────────────
        st.subheader("Monthly Visits by Traffic Source")

        def classify_src(row):
            s = str(row["source"]).lower()
            if str(row.get("has_social_referral", "No")) == "Yes":
                return "Social Referral"
            if s in ("google", "bing", "yahoo", "duckduckgo"):
                return "Organic Search"
            if s == "(direct)":
                return "Direct"
            return "Other"

        src_df = src_df.copy()
        src_df["category"] = src_df.apply(classify_src, axis=1)

        monthly_src = (
            src_df.groupby(["month", "category"])["visits"].sum().reset_index()
        )
        src_colors = {
            "Organic Search": "#1565c0",
            "Direct":         "#2e7d32",
            "Social Referral":"#f57c00",
            "Other":          "#9e9e9e",
        }
        fig_src = go.Figure()
        for cat in ["Organic Search", "Direct", "Social Referral", "Other"]:
            sub = monthly_src[monthly_src["category"] == cat]
            fig_src.add_trace(go.Bar(
                x=sub["month"].dt.strftime("%b %Y"),
                y=sub["visits"],
                name=cat,
                marker_color=src_colors[cat],
            ))
        fig_src.update_layout(
            barmode="stack",
            xaxis_title="Month",
            yaxis_title="Visits",
            height=370,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=50),
            yaxis=dict(tickformat=","),
        )
        st.plotly_chart(fig_src, use_container_width=True)
        st.caption(
            "Organic Search (Google, Bing, Yahoo) drives the majority of USDA traffic. "
            "A heavy reliance on search engines creates vulnerability — any algorithm change "
            "can significantly reduce public access to federal services."
        )

        # Google Dependency + Social Referral inline KPIs
        total_v  = src_df["visits"].sum()
        google_v = src_df[src_df["source"].str.lower() == "google"]["visits"].sum()
        social_v = src_df[src_df.get("has_social_referral", pd.Series(["No"]*len(src_df))) == "Yes"]["visits"].sum()
        g_pct    = google_v / total_v * 100 if total_v else 0
        s_pct    = social_v / total_v * 100 if total_v else 0
        g_label  = "Low" if g_pct < 40 else ("Moderate" if g_pct <= 60 else "High")
        g_color  = "#2e7d32" if g_pct < 40 else ("#f57c00" if g_pct <= 60 else "#c62828")

        kc1, kc2 = st.columns(2)
        kc1.markdown(
            f"**Google Dependency**  \n"
            f"<span style='font-size:1.8em;font-weight:700;color:{g_color}'>"
            f"{g_pct:.1f}% — {g_label}</span>  \n"
            f"<small>Below 40% = Low &nbsp;|&nbsp; 40–60% = Moderate &nbsp;|&nbsp; Above 60% = High</small>",
            unsafe_allow_html=True,
        )
        kc2.markdown(
            f"**Social Referral Share**  \n"
            f"<span style='font-size:1.8em;font-weight:700'>{s_pct:.2f}%</span>  \n"
            f"<small>of all traffic arrives via social platforms</small>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── 3. Top Downloads ──────────────────────────────────────────────────
        st.subheader("Top 20 Most-Downloaded Files")

        dl_df = dl_df.copy()
        dl_df["filename"] = dl_df["event_label"].apply(
            lambda x: str(x).split("/")[-1] if pd.notna(x) else x
        )
        dl_df["hostname"] = dl_df["page"].apply(
            lambda x: str(x).split("/")[0] if pd.notna(x) else x
        )
        top20_dl = (
            dl_df.groupby(["filename", "hostname"])["total_events"]
            .sum().reset_index()
            .nlargest(20, "total_events")
            .sort_values("total_events")
        )
        top20_dl["y_label"] = top20_dl["filename"] + "   (" + top20_dl["hostname"] + ")"

        fig_dl = go.Figure(go.Bar(
            x=top20_dl["total_events"],
            y=top20_dl["y_label"],
            orientation="h",
            marker_color="#4a148c",
            text=top20_dl["total_events"].apply(lambda v: f"{v:,.0f}"),
            textposition="outside",
        ))
        fig_dl.update_layout(
            xaxis_title="Total Download Events (Jan – Jun)",
            yaxis_title="",
            height=600,
            margin=dict(l=10, r=80, t=20, b=40),
            xaxis=dict(tickformat=","),
        )
        st.plotly_chart(fig_dl, use_container_width=True)
        st.caption(
            "These files represent the highest user demand for documents across all USDA agencies. "
            "Frequently downloaded files must remain current, accessible (508-compliant), and "
            "easy to locate from search engines."
        )

        st.markdown("---")

        # ── 4. Language Distribution ──────────────────────────────────────────
        st.subheader("Browser Language Distribution")
        st.info(
            "Browser language is a proxy for language preference, not geography. "
            "A user in the U.S. may use a non-English browser.",
            icon="ℹ️",
        )

        lang_totals = (
            lang_df.groupby("language")["visits"].sum().reset_index()
            .nlargest(15, "visits").sort_values("visits")
        )
        is_eng = lang_totals["language"].str.lower().str.startswith("en")

        fig_lang = go.Figure(go.Bar(
            x=lang_totals["visits"],
            y=lang_totals["language"],
            orientation="h",
            marker_color=["#1565c0" if e else "#e53935" for e in is_eng],
            text=lang_totals["visits"].apply(lambda v: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v:,.0f}"),
            textposition="outside",
        ))
        fig_lang.update_layout(
            xaxis_title="Total Visits (Jan – Jun)",
            yaxis_title="Browser Language Code",
            height=440,
            margin=dict(l=10, r=80, t=20, b=40),
            xaxis=dict(tickformat=","),
        )
        # Manual legend
        fig_lang.add_trace(go.Bar(x=[None], y=[None], marker_color="#1565c0", name="English variant", orientation="h"))
        fig_lang.add_trace(go.Bar(x=[None], y=[None], marker_color="#e53935", name="Non-English", orientation="h"))
        fig_lang.update_layout(
            barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_lang, use_container_width=True)

        total_lang  = lang_df["visits"].sum()
        non_eng_v   = lang_df[~lang_df["language"].str.lower().str.startswith("en")]["visits"].sum()
        non_eng_pct = non_eng_v / total_lang * 100 if total_lang else 0
        st.metric("Non-English Browser Share (Full Period)", f"{non_eng_pct:.1f}%")
        st.caption(
            "Spanish (es-*) and Chinese (zh-*) variants are the dominant non-English languages. "
            "These communities should be prioritized in multilingual content and service strategies."
        )

        st.markdown("---")

        # ── 5. Mobile Traffic Share Trend ─────────────────────────────────────
        st.subheader("Mobile Traffic Share Trend")

        monthly_dev = dev_df.groupby(["month", "device"])["visits"].sum().reset_index()
        monthly_tot = monthly_dev.groupby("month")["visits"].sum()
        mob_monthly = monthly_dev[monthly_dev["device"] == "mobile"].set_index("month")["visits"]
        mob_pct_ser = (mob_monthly / monthly_tot * 100).reset_index()
        mob_pct_ser.columns = ["month", "pct"]
        mob_pct_ser = mob_pct_ser.sort_values("month")

        fig_mob = go.Figure(go.Scatter(
            x=mob_pct_ser["month"],
            y=mob_pct_ser["pct"],
            mode="lines+markers+text",
            text=mob_pct_ser["pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="top center",
            line=dict(color="#f57c00", width=2),
            marker=dict(size=8),
        ))
        fig_mob.update_layout(
            xaxis_title="Month",
            yaxis_title="Mobile Share (%)",
            height=320,
            yaxis=dict(range=[0, 60]),
            margin=dict(t=20),
        )
        st.plotly_chart(fig_mob, use_container_width=True)

        pcts = mob_pct_ser["pct"].tolist()
        streak = max_streak = 0
        for i in range(1, len(pcts)):
            streak = streak + 1 if pcts[i] > pcts[i - 1] else 0
            max_streak = max(max_streak, streak)
        if max_streak >= 3:
            st.warning(
                f"📱 Trend flag: Mobile share rose for {max_streak} consecutive months. "
                "USDA services should prioritize mobile-responsive design and faster load times."
            )
        st.caption(
            "Rising mobile share means more citizens are accessing USDA services on phones. "
            "Pages that are not mobile-optimized create disproportionate barriers for these users."
        )

    except Exception as e:
        st.error(f"Error loading system-wide data: {e}")
        st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Rural Development Baseline
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Layer 2: Rural Development Descriptive Foundation")
    st.caption("How is the Rural Development site performing, and for whom is it performing worst?")
    st.markdown("---")

    if not rd_ok:
        st.warning("Upload the Rural Development CSV in the sidebar to view this analysis.")
        st.stop()

    try:
        page_df, scaled, feat_cols, daily_df = load_rd_data(rd_source)

        # ── KPI row ───────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total RD Users",       f"{page_df['Total Users'].sum():,.0f}")
        k2.metric("Mean Bounce Rate",     f"{page_df['Bounce Rate'].mean():.1%}")
        k3.metric("Mean Session Duration",f"{page_df['Session Duration'].mean():.0f}s")
        k4.metric("Mean Views / Session", f"{page_df['Views per Session'].mean():.2f}")

        st.markdown("---")

        # ── Heatmap with section toggle ────────────────────────────────────────
        st.subheader("Section-Level Performance Heatmap")

        HM_METRICS = ["Bounce Rate", "Session Duration", "Views per Session",
                      "Exit Pressure Index", "Stickiness Ratio"]

        hm_raw = (
            page_df.groupby("Section")[HM_METRICS].mean()
            .dropna(how="all")
            .reset_index()
        )
        all_sections = sorted(hm_raw["Section"].dropna().unique().tolist())

        with st.expander("Filter sections shown in heatmap", expanded=False):
            sel_sections = st.multiselect(
                "Sections to display",
                options=all_sections,
                default=all_sections,
                key="hm_sections",
            )

        if not sel_sections:
            st.info("Select at least one section above to display the heatmap.")
        else:
            hm_data = hm_raw[hm_raw["Section"].isin(sel_sections)].set_index("Section")

            # Normalize 0-1 per column; invert where lower = better
            norm = hm_data.copy()
            INVERT = {"Bounce Rate", "Exit Pressure Index"}
            for col in HM_METRICS:
                mn, mx = norm[col].min(), norm[col].max()
                norm[col] = (norm[col] - mn) / (mx - mn) if mx > mn else 0.5
                if col in INVERT:
                    norm[col] = 1 - norm[col]

            anns = []
            for i, row_lbl in enumerate(hm_data.index):
                for j, col in enumerate(HM_METRICS):
                    val = hm_data.loc[row_lbl, col]
                    txt = f"{val:.2f}" if abs(val) < 10 else f"{val:.0f}"
                    anns.append(dict(x=j, y=i, text=txt, showarrow=False,
                                     font=dict(size=9, color="black")))

            fig_hm = go.Figure(go.Heatmap(
                z=norm.values,
                x=HM_METRICS,
                y=list(hm_data.index),
                colorscale="RdYlGn",
                zmin=0, zmax=1,
                showscale=True,
                colorbar=dict(title="Performance", tickvals=[0, 0.5, 1],
                              ticktext=["Poor", "Average", "Good"]),
            ))
            fig_hm.update_layout(
                xaxis=dict(title="", side="top", tickangle=0),
                yaxis=dict(title="Site Section", autorange="reversed"),
                height=max(380, len(hm_data) * 30),
                margin=dict(t=80, l=10),
                annotations=anns,
            )
            st.plotly_chart(fig_hm, use_container_width=True)
            st.caption(
                "Green = strong performance; red = areas of concern. "
                "Sections with red across multiple metrics are highest-priority for content or UX intervention."
            )

        st.markdown("---")

        # ── Device engagement: two separate charts side by side ───────────────
        st.subheader("Engagement by Device Type")

        def _dev_stat(col_idx):
            col = daily_df.columns[col_idx] if col_idx < len(daily_df.columns) else None
            return pd.to_numeric(daily_df[col], errors="coerce").mean() if col else np.nan

        dev_bounce = {
            "Desktop": _dev_stat(11),
            "Mobile":  _dev_stat(20),
            "Tablet":  _dev_stat(29),
        }
        dev_dur = {
            "Desktop": _dev_stat(10),
            "Mobile":  _dev_stat(19),
            "Tablet":  _dev_stat(28),
        }

        dc1, dc2 = st.columns(2)
        with dc1:
            fig_br = go.Figure(go.Bar(
                x=list(dev_bounce.keys()),
                y=list(dev_bounce.values()),
                marker_color=["#1565c0", "#e53935", "#f57c00"],
                text=[f"{v:.1%}" for v in dev_bounce.values()],
                textposition="outside",
            ))
            fig_br.update_layout(
                title="Mean Bounce Rate by Device",
                xaxis_title="Device", yaxis_title="Bounce Rate",
                yaxis=dict(tickformat=".0%", range=[0, 1]),
                height=360, showlegend=False,
            )
            st.plotly_chart(fig_br, use_container_width=True)

        with dc2:
            fig_dur = go.Figure(go.Bar(
                x=list(dev_dur.keys()),
                y=list(dev_dur.values()),
                marker_color=["#1565c0", "#e53935", "#f57c00"],
                text=[f"{v:.0f}s" for v in dev_dur.values()],
                textposition="outside",
            ))
            fig_dur.update_layout(
                title="Mean Session Duration by Device",
                xaxis_title="Device", yaxis_title="Session Duration (seconds)",
                height=360, showlegend=False,
            )
            st.plotly_chart(fig_dur, use_container_width=True)

        st.caption(
            "Mobile users typically show higher bounce rates and shorter sessions, "
            "indicating friction in delivering content on smaller screens. "
            "Large gaps between mobile and desktop signal priority areas for mobile optimization."
        )

        st.markdown("---")

        # ── Monthly Engagement Trends (3 stacked subplots, shared x) ─────────
        st.subheader("Monthly Engagement Trends")

        if "Month" in daily_df.columns and "Bounce Rate" in daily_df.columns:
            MONTH_LABELS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                            7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            mt = (
                daily_df.groupby("Month")
                .agg(Bounce=("Bounce Rate","mean"),
                     Duration=("Session Duration","mean"),
                     Users=("Active Users","sum"))
                .reset_index()
                .dropna(subset=["Month"])
            )
            mt = mt[mt["Month"].between(1, 12)].sort_values("Month")
            mt["Month Label"] = mt["Month"].map(MONTH_LABELS)

            fig_mt = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                subplot_titles=("Bounce Rate", "Mean Session Duration (seconds)", "Active Users"),
                vertical_spacing=0.10,
            )
            fig_mt.add_trace(
                go.Scatter(x=mt["Month Label"], y=mt["Bounce"],
                           mode="lines+markers", line=dict(color="#e53935", width=2),
                           marker=dict(size=7), showlegend=False),
                row=1, col=1,
            )
            fig_mt.add_trace(
                go.Scatter(x=mt["Month Label"], y=mt["Duration"],
                           mode="lines+markers", line=dict(color="#1565c0", width=2),
                           marker=dict(size=7), showlegend=False),
                row=2, col=1,
            )
            fig_mt.add_trace(
                go.Bar(x=mt["Month Label"], y=mt["Users"],
                       marker_color="#2e7d32", showlegend=False),
                row=3, col=1,
            )
            fig_mt.update_yaxes(title_text="Rate",     tickformat=".0%", row=1, col=1)
            fig_mt.update_yaxes(title_text="Seconds",                     row=2, col=1)
            fig_mt.update_yaxes(title_text="Users",    tickformat=",",    row=3, col=1)
            fig_mt.update_xaxes(title_text="Month",                       row=3, col=1)
            fig_mt.update_layout(height=600, margin=dict(t=60))
            st.plotly_chart(fig_mt, use_container_width=True)
            st.caption(
                "Each panel shows a separate metric month-by-month. "
                "Rising bounce alongside declining session duration signals worsening content relevance or site friction. "
                "Active Users (green bars) shows overall traffic volume per month."
            )

    except Exception as e:
        st.error(f"Error in Rural Development analysis: {e}")
        st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Clustering & Underserved Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Layer 3: Page-Level Clustering & Underserved Analysis")
    st.markdown(
        "> **Clustering** asks: *Which pages behave similarly to each other?*  \n"
        "> **Underserved Score** asks: *How badly is this specific page failing its users?*"
    )
    st.markdown("---")

    if not rd_ok:
        st.warning("Upload the Rural Development CSV in the sidebar to view this analysis.")
        st.stop()

    try:
        page_df, scaled, feat_cols, _ = load_rd_data(rd_source)
        page_df = run_clustering(page_df, scaled)

        CLUSTER_COLORS = {
            "Core Program":  "#bbdefb",
            "Power User":    "#c8e6c9",
            "Discovery":     "#fff9c4",
            "High Friction": "#ffcdd2",
        }
        CLUSTER_LINE_COLORS = {
            "Core Program":  "#1565c0",
            "Power User":    "#2e7d32",
            "Discovery":     "#f9a825",
            "High Friction": "#c62828",
        }

        # ── Cluster definition cards ──────────────────────────────────────────
        st.subheader("What Each Cluster Represents")
        cols_cards = st.columns(4)
        for col_w, (name, info) in zip(cols_cards, CLUSTER_INFO.items()):
            with col_w:
                st.markdown(
                    f"<div style='background:{info['color']};border-radius:10px;"
                    f"padding:14px;height:100%'>"
                    f"<b>{info['dot']} {name}</b><br><br>"
                    f"<small>{info['desc']}</small><br><br>"
                    f"<i style='color:#555;font-size:0.8em'>e.g. {info['examples']}</i>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Cluster summary table ─────────────────────────────────────────────
        st.subheader("Cluster Summary: Metrics, Meaning & Recommended Actions")

        summary = (
            page_df.groupby("Cluster Label")
            .agg(
                Pages          =("Page title",         "count"),
                Bounce_Rate    =("Bounce Rate",         "mean"),
                Session_Dur    =("Session Duration",    "mean"),
                Views_Session  =("Views per Session",   "mean"),
                Exit_Pressure  =("Exit Pressure Index", "mean"),
                Stickiness     =("Stickiness Ratio",    "mean"),
                Device_Gap     =("Device Gap Score",    "mean"),
                Underserved_Avg=("Underserved Score",   "mean"),
            )
            .reset_index()
        )

        action_rows = []
        for _, row in summary.iterrows():
            info = CLUSTER_INFO.get(row["Cluster Label"], {})
            action_rows.append({
                "Cluster":              row["Cluster Label"],
                "Pages":                int(row["Pages"]),
                "Avg Bounce Rate":      f"{row['Bounce_Rate']:.0%}",
                "Avg Session (s)":      f"{row['Session_Dur']:.0f}",
                "Views / Session":      f"{row['Views_Session']:.2f}",
                "Exit Pressure":        f"{row['Exit_Pressure']:.3f}",
                "Stickiness":           f"{row['Stickiness']:.3f}",
                "Device Gap":           f"{row['Device_Gap']:.3f}",
                "Avg Underserved Score":f"{row['Underserved_Avg']:.3f}",
                "What It Means":        info.get("desc", ""),
                "Recommended Action":   info.get("action", ""),
            })

        action_df = pd.DataFrame(action_rows)

        def _color_cluster_row(row):
            bg = CLUSTER_COLORS.get(row["Cluster"], "#ffffff")
            return [f"background-color:{bg}"] * len(row)

        st.dataframe(
            action_df.style.apply(_color_cluster_row, axis=1),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Each row describes a distinct behavioral segment. "
            "The Avg Underserved Score combines bounce rate, exit pressure, and lack of return visits "
            "into a single failure indicator — higher = worse."
        )

        st.markdown("---")

        # ── 2D PCA ────────────────────────────────────────────────────────────
        st.subheader("Page Clusters Visualized in 2D (PCA)")

        pca2   = PCA(n_components=2, random_state=42)
        c2     = pca2.fit_transform(scaled)
        page_df["PC1"], page_df["PC2"] = c2[:, 0], c2[:, 1]

        fig_2d = px.scatter(
            page_df, x="PC1", y="PC2",
            color="Cluster Label",
            color_discrete_map=CLUSTER_LINE_COLORS,
            hover_data={
                "Page title": True, "Section": True,
                "Bounce Rate": ":.1%", "Session Duration": ":.0f",
                "PC1": False, "PC2": False,
            },
            title="",
            height=520,
        )
        fig_2d.update_layout(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            legend_title="Cluster",
        )
        st.plotly_chart(fig_2d, use_container_width=True)
        st.caption(
            "Each dot is a unique page, positioned by how similar its behavior is to other pages. "
            "Dots close together share similar engagement patterns. "
            "Note: PCA is used for visualization only — clustering was performed on all 6 features."
        )

        st.markdown("---")

        # ── 3D PCA ────────────────────────────────────────────────────────────
        st.subheader("Page Clusters Visualized in 3D (Interactive — drag to rotate)")

        pca3    = PCA(n_components=3, random_state=42)
        c3      = pca3.fit_transform(scaled)
        page_df["PC3"] = c3[:, 2]
        var3    = pca3.explained_variance_ratio_

        fig_3d = px.scatter_3d(
            page_df, x="PC1", y="PC2", z="PC3",
            color="Cluster Label",
            color_discrete_map=CLUSTER_LINE_COLORS,
            hover_data={
                "Page title": True, "Section": True,
                "Bounce Rate": ":.1%", "Session Duration": ":.0f",
            },
            height=600,
        )
        fig_3d.update_layout(
            scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
            legend_title="Cluster",
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        st.caption(
            f"Variance explained — PC1: {var3[0]:.1%} | PC2: {var3[1]:.1%} | PC3: {var3[2]:.1%}.  "
            "Rotate to explore which clusters are most distinct from each other."
        )

        st.markdown("---")

        # ── Radar chart ───────────────────────────────────────────────────────
        st.subheader("Cluster Behavioral Profiles — Radar Chart")

        radar_lbls = ["Bounce Rate", "Session Duration", "Views/Session",
                      "Exit Pressure", "Stickiness", "Device Gap"]
        cm = page_df.groupby("Cluster Label")[feat_cols].mean()
        cm_norm = cm.copy()
        for col in feat_cols:
            mn, mx = cm_norm[col].min(), cm_norm[col].max()
            cm_norm[col] = (cm_norm[col] - mn) / (mx - mn) if mx > mn else 0.5

        fig_radar = go.Figure()
        for lbl, row in cm_norm.iterrows():
            vals = list(row.values) + [row.values[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=radar_lbls + [radar_lbls[0]],
                fill="toself",
                name=lbl,
                line=dict(color=CLUSTER_LINE_COLORS.get(lbl, "#999"), width=2),
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=500,
            legend_title="Cluster",
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption(
            "Each line traces a cluster's normalized behavioral fingerprint across all six metrics. "
            "A spike on Bounce Rate and Exit Pressure with low Stickiness marks High Friction pages. "
            "Power User pages show a distinct spike on Stickiness."
        )

        st.markdown("---")

        # ── Underserved distribution chart ────────────────────────────────────
        st.subheader("Well-Served vs. Underserved Page Distribution")

        p25 = page_df["Underserved Score"].quantile(0.25)
        p75 = page_df["Underserved Score"].quantile(0.75)

        def _tier(s):
            if s <= p25:   return "Well-Served"
            if s <= p75:   return "Moderately Served"
            return "Underserved"

        page_df["Service Tier"] = page_df["Underserved Score"].apply(_tier)

        tier_order  = ["Well-Served", "Moderately Served", "Underserved"]
        tier_colors = {"Well-Served": "#2e7d32", "Moderately Served": "#f9a825", "Underserved": "#c62828"}
        tier_counts = page_df["Service Tier"].value_counts().reindex(tier_order, fill_value=0)
        tier_pct    = (tier_counts / tier_counts.sum() * 100).round(1)

        fig_tier = go.Figure()
        for tier in tier_order:
            fig_tier.add_trace(go.Bar(
                x=[tier],
                y=[tier_counts[tier]],
                name=tier,
                marker_color=tier_colors[tier],
                text=[f"{tier_counts[tier]} pages<br>({tier_pct[tier]}%)"],
                textposition="outside",
            ))
        fig_tier.update_layout(
            xaxis_title="Service Tier",
            yaxis_title="Number of Pages",
            showlegend=False,
            height=380,
            yaxis=dict(range=[0, tier_counts.max() * 1.25]),
            margin=dict(t=30),
        )
        st.plotly_chart(fig_tier, use_container_width=True)

        col_ws, col_ms, col_us = st.columns(3)
        col_ws.metric("Well-Served",       f"{tier_counts['Well-Served']} pages",       f"{tier_pct['Well-Served']}%")
        col_ms.metric("Moderately Served", f"{tier_counts['Moderately Served']} pages", f"{tier_pct['Moderately Served']}%")
        col_us.metric("Underserved",       f"{tier_counts['Underserved']} pages",       f"{tier_pct['Underserved']}%")
        st.caption(
            "Underserved Score = (Bounce Rate × 0.4) + (Exit Pressure × 0.3) + (1 − Stickiness) × 0.3.  \n"
            "Well-Served = bottom 25% of scores. Underserved = top 25%. "
            "These thresholds are relative to this dataset."
        )

        st.markdown("---")

        # ── Underserved Page Inventory ─────────────────────────────────────────
        st.subheader("Underserved Page Inventory")
        st.markdown(
            "Pages with an Underserved Score above the 75th percentile. "
            "Filter by section or cluster to prioritize specific areas."
        )

        underserved_df = (
            page_df[page_df["Service Tier"] == "Underserved"]
            .sort_values("Underserved Score", ascending=False)
        )

        f1, f2 = st.columns(2)
        with f1:
            section_opts = ["All Sections"] + sorted(underserved_df["Section"].dropna().unique().tolist())
            sel_sec = st.selectbox("Filter by Section", section_opts, key="us_section")
        with f2:
            cluster_opts = ["All Clusters"] + sorted(underserved_df["Cluster Label"].dropna().unique().tolist())
            sel_clus = st.selectbox("Filter by Cluster", cluster_opts, key="us_cluster")

        filtered = underserved_df.copy()
        if sel_sec  != "All Sections": filtered = filtered[filtered["Section"]       == sel_sec]
        if sel_clus != "All Clusters": filtered = filtered[filtered["Cluster Label"] == sel_clus]

        inv_cols = ["Page title", "Section", "Cluster Label", "Bounce Rate",
                    "Session Duration", "Exit Pressure Index", "Stickiness Ratio",
                    "Device Gap Score", "Underserved Score"]
        inv_cols = [c for c in inv_cols if c in filtered.columns]

        def _fmt_inv(df):
            fmt = {
                "Bounce Rate":        "{:.1%}",
                "Session Duration":   "{:.0f}",
                "Exit Pressure Index":"{:.3f}",
                "Stickiness Ratio":   "{:.3f}",
                "Device Gap Score":   "{:.3f}",
                "Underserved Score":  "{:.3f}",
            }

            def _score_color(val):
                """Red-yellow-green gradient without matplotlib."""
                try:
                    mn = filtered["Underserved Score"].min()
                    mx = filtered["Underserved Score"].max()
                    t  = (float(val) - mn) / (mx - mn) if mx > mn else 0.5
                    # t=0 → green, t=1 → red
                    r = int(255 * t)
                    g = int(255 * (1 - t))
                    text = "white" if t > 0.75 else "black"
                    return f"background-color: rgb({r},{g},30); color: {text}"
                except Exception:
                    return ""

            return (
                df[inv_cols]
                .style
                .format({k: v for k, v in fmt.items() if k in inv_cols})
                .map(_score_color, subset=["Underserved Score"])
            )

        st.dataframe(_fmt_inv(filtered), use_container_width=True, hide_index=True)
        st.caption(
            f"Showing {len(filtered)} pages (75th-percentile threshold: {p75:.3f}). "
            "Darker red = higher failure score. Sort or filter to prioritize remediation efforts."
        )

    except Exception as e:
        st.error(f"Error in clustering analysis: {e}")
        st.exception(e)

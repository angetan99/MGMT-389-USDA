# USDA Digital Analytics Dashboard

A Streamlit application for analyzing web traffic, engagement, and content performance across the USDA web ecosystem and the Rural Development site specifically.

---

## Overview

The dashboard is organized into three layers, each answering a distinct question:

| Layer | Page | Question |
|---|---|---|
| 1 | System-Wide Analysis | What is happening across USDA's web presence? |
| 2 | Rural Development | How is the Rural Development site performing, and for whom is it performing worst? |
| 3 | Clustering & Underserved | Which pages are failing users, and how severely? |

An **Ask the AI Analyst** link in the sidebar connects to a companion ChatGPT GPT for conversational exploration of the data.

---

## Requirements

**Python 3.9+**

Install dependencies:

```bash
pip install streamlit pandas numpy plotly scikit-learn
```

---

## Data Files

All data files must be placed in the same directory as `app2.py`, or in a `data/` subdirectory. Files may be raw `.csv` or zipped as `.zip` (the app will unzip automatically).

### Layer 1 — System-Wide (required)

| File | Contents |
|---|---|
| `device-1-2024.csv` | Daily visits broken down by device type (desktop, mobile, tablet) |
| `domain-1-2024.csv` | Daily visits by USDA hostname/subdomain |
| `download-1-2024.csv` | File download events with filename and source page |
| `language-1-2024.csv` | Daily visits by browser language code |
| `traffic-source-1-2024.csv` | Daily visits by traffic source (Google, direct, social, etc.) |

### Layer 2 & 3 — Rural Development (required)

| File | Contents |
|---|---|
| `(Rural Development) Edited USDA data base.csv` | Page-level engagement data exported from analytics, with a two-row header structure. Must match the filename exactly, or be zipped under that name. |

> If the Rural Development file is missing, Layers 2 and 3 will display an error and stop. Layer 1 will still function normally.

---

## Running the App

```bash
streamlit run app2.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Layer 1 — System-Wide Analysis

Covers January–June 2024 across all USDA hostnames.

**Charts included:**

- **Top 15 USDA Hostnames by Total Visits** — horizontal bar chart showing which agencies drive the most traffic. Known agencies (FNS, AMS, NRCS) are labeled by full name.
- **Monthly Visits by Traffic Source** — stacked bar chart breaking traffic into Organic Search, Direct, Social Referral, and Other. Includes live KPI callouts for Google dependency level and social referral share.
- **Top 20 Most-Downloaded Files** — horizontal bar chart of the highest-demand documents across all agencies, labeled by filename and source hostname.
- **Browser Language Distribution** — horizontal bar chart of the top 15 browser language codes. English variants are shown in blue; non-English in red. Displays total non-English browser share as a metric.
- **Mobile Traffic Share Trend** — line chart of mobile's share of total visits by month. Automatically flags if mobile share rose for 3 or more consecutive months.

---

## Layer 2 — Rural Development Baseline

Covers page-level performance data for the USDA Rural Development site.

**KPI row** at the top shows total users, mean bounce rate, mean session duration, and mean views per session across all pages.

**Charts included:**

- **Section-Level Performance Heatmap** — color-coded heatmap (red = poor, green = strong) showing mean bounce rate, session duration, views per session, exit pressure index, and stickiness ratio by site section. Horizontally scrollable. Sections can be filtered via an expander above the chart.
- **Engagement by Device Type** — side-by-side bar charts comparing mean bounce rate and mean session duration across desktop, mobile, and tablet.
- **Monthly Engagement Trends** — three stacked time series (shared x-axis) showing bounce rate, session duration, and active users month by month.

---

## Layer 3 — Clustering & Underserved Analysis

Uses the Rural Development data to cluster pages by behavioral similarity and score them for how well they serve users.

### Clustering

Pages are grouped using **K-Means clustering (k=4)** across six standardized engagement metrics. The value of k=4 is validated through WCSS (elbow method) and silhouette scores computed for k=2 through k=10.

**The six metrics used:**

| Metric | Definition |
|---|---|
| Bounce Rate | Share of sessions with no interaction |
| Session Duration | Average time spent on the page |
| Views per Session | Pages visited in a single session |
| Exit Pressure Index | Exits divided by total sessions |
| Stickiness Ratio | Returning users divided by total users |
| Device Gap Score | Difference in bounce rate between mobile and desktop |

**The four cluster labels:**

| Cluster | Behavior Pattern |
|---|---|
| Core Program | Low bounce, healthy duration, moderate return rate — the site is working |
| Power User | Highest stickiness; professional users who rely on specific pages repeatedly |
| Discovery | Moderate-to-high bounce; users exploring but not finding what they need |
| High Friction | Very high bounce (83%+), very short duration (<15s); users immediately leave |

**Charts included:**

- Cluster validation charts (WCSS elbow + silhouette scores)
- 2D PCA scatter plot of all pages colored by cluster
- 3D PCA scatter plot (interactive, drag to rotate)
- Radar chart comparing behavioral profiles across all four clusters

### Underserved Score

A separate composite score that measures how badly a page is failing its users, independent of its cluster label:

```
Underserved Score = (Bounce Rate × 0.4) + (Exit Pressure × 0.3) + (1 − Stickiness Ratio) × 0.3
```

Pages are tiered by score percentile:

| Tier | Threshold |
|---|---|
| Well-Served | Bottom 25% of scores |
| Moderately Served | Middle 50% |
| Underserved | Top 25% of scores |

**Charts included:**

- Bar chart of page counts by service tier
- Metric cards for each tier
- **Underserved Page Inventory** — filterable table of all pages above the 75th percentile threshold, showing all six metrics plus the composite score. Filter by section or cluster. Color-coded by score severity (green → red).

---

## File Structure

```
app2.py
device-1-2024.csv
domain-1-2024.csv
download-1-2024.csv
language-1-2024.csv
traffic-source-1-2024.csv
(Rural Development) Edited USDA data base.csv
```

All files can alternatively be placed in a `data/` subdirectory, or provided as `.zip` archives containing the corresponding `.csv`.

---

## Notes

- All data is cached using `@st.cache_data` for performance. Clear the cache by restarting the app or using the Streamlit menu.
- The Rural Development CSV uses a non-standard two-row header format. The loader parses rows 6–7 as headers and begins reading data at row 9. Do not alter the file structure.
- Browser language data reflects language preference settings, not geographic location. A user in the U.S. may use a non-English browser.
- The Underserved Score thresholds (25th and 75th percentile) are relative to the dataset loaded — they will shift if the data changes.

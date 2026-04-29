# USDA Digital Analytics Dashboard

A Streamlit application for analyzing web traffic, engagement, and content performance across the USDA web ecosystem and the Rural Development site. Deployed via GitHub on Streamlit Community Cloud.

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

## Layer 1 — System-Wide Analysis

Covers January–June 2024 across all USDA hostnames.

- **Top 15 USDA Hostnames by Total Visits** — which agencies drive the most traffic, with known agencies labeled by full name
- **Monthly Visits by Traffic Source** — traffic broken into Organic Search, Direct, Social Referral, and Other, with live callouts for Google dependency level and social referral share
- **Top 20 Most-Downloaded Files** — highest-demand documents across all agencies, labeled by filename and source hostname
- **Browser Language Distribution** — top 15 browser language codes; English variants vs. non-English, with total non-English share as a metric
- **Mobile Traffic Share Trend** — mobile's share of total visits by month; automatically flags if mobile share rose for 3 or more consecutive months

---

## Layer 2 — Rural Development Baseline

Page-level performance data for the USDA Rural Development site. KPIs at the top show total users, mean bounce rate, mean session duration, and mean views per session.

- **Section-Level Performance Heatmap** — color-coded heatmap (red = poor, green = strong) across five metrics by site section; horizontally scrollable and filterable by section
- **Engagement by Device Type** — mean bounce rate and session duration compared across desktop, mobile, and tablet
- **Monthly Engagement Trends** — bounce rate, session duration, and active users shown month by month on a shared time axis

---

## Layer 3 — Clustering & Underserved Analysis

### Clustering

Pages are grouped using **K-Means clustering (k=4)** across six engagement metrics. The choice of k=4 is validated through WCSS (elbow method) and silhouette scores.

**Metrics used:**

| Metric | Definition |
|---|---|
| Bounce Rate | Share of sessions with no interaction |
| Session Duration | Average time spent on the page |
| Views per Session | Pages visited in a single session |
| Exit Pressure Index | Exits divided by total sessions |
| Stickiness Ratio | Returning users divided by total users |
| Device Gap Score | Difference in bounce rate between mobile and desktop |

**Cluster labels:**

| Cluster | Behavior Pattern |
|---|---|
| Core Program | Low bounce, healthy duration, moderate return rate — the site is working |
| Power User | Highest stickiness; professional users who rely on specific pages repeatedly |
| Discovery | Moderate-to-high bounce; users exploring but not finding what they need |
| High Friction | Very high bounce (83%+), very short duration (<15s); users immediately leave |

Charts include cluster validation diagnostics, a 2D PCA scatter plot, an interactive 3D PCA scatter plot, and a radar chart comparing behavioral profiles across all four clusters.

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

The **Underserved Page Inventory** table lists all pages above the 75th percentile threshold, filterable by section or cluster, and color-coded by score severity.

---

## Notes

- Browser language data reflects language preference settings, not geographic location.
- Underserved Score thresholds are relative to the dataset — they shift if the underlying data changes.

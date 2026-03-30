"""
dashboard.py — Polymarket Signal Tracker
Streamlit app reading from Supabase PostgreSQL.

Run locally:    streamlit run dashboard.py
Deploy free:    https://streamlit.io/cloud (connect GitHub repo, set DATABASE_URL secret)

Environment variable required:
    DATABASE_URL — Supabase PostgreSQL connection string

Query architecture (two paths, kept separate for performance):
  HOT  — today's top-X table, re-runs on every UI interaction
  COLD — historical trend data, cached for 15 minutes via @st.cache_data
"""

import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import psycopg2.extras
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Polymarket Signal Tracker",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme — injected CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Dark slate background */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px 16px;
    }

    /* Table */
    .signal-table {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        width: 100%;
        border-collapse: collapse;
    }
    .signal-table th {
        background: #161b22;
        color: #8b949e;
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        padding: 8px 12px;
        border-bottom: 1px solid #30363d;
        text-align: left;
    }
    .signal-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #21262d;
        vertical-align: middle;
        color: #e6edf3;
    }
    .signal-table tr:hover td {
        background: #1c2128;
    }

    /* Direction badges */
    .dir-up   { color: #3fb950; font-size: 1.1em; }
    .dir-down { color: #f85149; font-size: 1.1em; }
    .dir-flat { color: #8b949e; font-size: 1.1em; }

    /* Score pill */
    .score-pill {
        display: inline-block;
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 4px;
        padding: 2px 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
    }

    /* Section headers */
    h2, h3 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        color: #e6edf3;
        border-bottom: 1px solid #30363d;
        padding-bottom: 6px;
    }

    /* Dividers */
    hr {
        border-color: #30363d;
    }

    /* Hide default Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------

@st.cache_resource
def get_connection():
    url = os.environ.get("DATABASE_URL")
    if not url:
        st.error("DATABASE_URL environment variable is not set.")
        st.stop()
    return psycopg2.connect(url)


def query(sql: str, params=None) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        # Reconnect on stale connection
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        return pd.read_sql_query(sql, conn, params=params)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900)   # 15-minute cache — cold path
def load_categories() -> list[str]:
    df = query("SELECT DISTINCT category FROM markets WHERE category IS NOT NULL ORDER BY category")
    return df["category"].tolist()


@st.cache_data(ttl=900)   # cold path
def load_config() -> dict:
    df = query("SELECT key, value FROM pipeline_config")
    return dict(zip(df["key"], df["value"]))


def load_topx(snapshot_date: date, top_n: int, categories: Optional[list[str]]) -> pd.DataFrame:
    """HOT path — top-X for a given date, optionally filtered by category."""
    if categories:
        return query(
            """
            SELECT
                t.rank_position,
                t.market_question,
                t.outcome_label,
                m.category,
                t.composite_score,
                t.volume_z_score,
                t.dp_dv_raw,
                t.dp_dv_direction
            FROM daily_topx t
            JOIN markets m ON m.market_id = t.market_id
            WHERE t.snapshot_date = %s
              AND t.top_n = %s
              AND m.category = ANY(%s)
            ORDER BY t.rank_position
            LIMIT %s
            """,
            (snapshot_date, top_n, categories, top_n),
        )
    else:
        return query(
            """
            SELECT
                t.rank_position,
                t.market_question,
                t.outcome_label,
                m.category,
                t.composite_score,
                t.volume_z_score,
                t.dp_dv_raw,
                t.dp_dv_direction
            FROM daily_topx t
            JOIN markets m ON m.market_id = t.market_id
            WHERE t.snapshot_date = %s
              AND t.top_n = %s
            ORDER BY t.rank_position
            LIMIT %s
            """,
            (snapshot_date, top_n, top_n),
        )


@st.cache_data(ttl=900)   # cold path
def load_historical_topx(start_date: date, end_date: date, top_n: int, categories: Optional[tuple]) -> pd.DataFrame:
    """
    Load daily composite scores for outcomes that appeared in the top-N
    at any point during the date range.
    Used for the rank trend chart and composite score history.
    """
    cat_filter = list(categories) if categories else None

    if cat_filter:
        return query(
            """
            SELECT
                t.snapshot_date,
                t.rank_position,
                t.market_question,
                t.outcome_label,
                m.category,
                t.composite_score,
                t.volume_z_score,
                t.dp_dv_direction,
                CONCAT(t.market_question, ' — ', t.outcome_label) AS display_label
            FROM daily_topx t
            JOIN markets m ON m.market_id = t.market_id
            WHERE t.snapshot_date BETWEEN %s AND %s
              AND t.top_n = %s
              AND m.category = ANY(%s)
            ORDER BY t.snapshot_date, t.rank_position
            """,
            (start_date, end_date, top_n, cat_filter),
        )
    else:
        return query(
            """
            SELECT
                t.snapshot_date,
                t.rank_position,
                t.market_question,
                t.outcome_label,
                m.category,
                t.composite_score,
                t.volume_z_score,
                t.dp_dv_direction,
                CONCAT(t.market_question, ' — ', t.outcome_label) AS display_label
            FROM daily_topx t
            JOIN markets m ON m.market_id = t.market_id
            WHERE t.snapshot_date BETWEEN %s AND %s
              AND t.top_n = %s
            ORDER BY t.snapshot_date, t.rank_position
            """,
            (start_date, end_date, top_n),
        )


@st.cache_data(ttl=900)
def load_new_markets(start_date: date, end_date: date, categories: Optional[tuple]) -> pd.DataFrame:
    """Markets first seen within the date range — topic emergence signal."""
    cat_filter = list(categories) if categories else None
    if cat_filter:
        return query(
            """
            SELECT question, category, first_seen_at::date AS first_seen
            FROM markets
            WHERE DATE(first_seen_at) BETWEEN %s AND %s
              AND category = ANY(%s)
            ORDER BY first_seen_at DESC
            """,
            (start_date, end_date, cat_filter),
        )
    else:
        return query(
            """
            SELECT question, category, first_seen_at::date AS first_seen
            FROM markets
            WHERE DATE(first_seen_at) BETWEEN %s AND %s
            ORDER BY first_seen_at DESC
            """,
            (start_date, end_date),
        )


@st.cache_data(ttl=900)
def load_volume_metrics(categories: Optional[tuple]) -> dict:
    """
    USDC volume metrics across tracked markets.
    Uses market_daily for yesterday aggregate + market_snapshots for intraday delta.
    Volume is market-level (use MAX per market to avoid double-counting binary outcomes).
    """
    cat_filter = list(categories) if categories else None
    yesterday = date.today() - timedelta(days=1)

    if cat_filter:
        df = query(
            """
            SELECT
                MAX(md.daily_volume_usdc)  AS market_volume,
                MAX(md.avg_liquidity_usdc) AS market_liquidity
            FROM market_daily md
            JOIN markets m ON m.market_id = md.market_id
            WHERE md.date = %s
              AND m.category = ANY(%s)
              AND md.volume_is_market_level = TRUE
            """,
            (yesterday, cat_filter),
        )
    else:
        df = query(
            """
            SELECT
                MAX(md.daily_volume_usdc)  AS market_volume,
                MAX(md.avg_liquidity_usdc) AS market_liquidity
            FROM market_daily md
            WHERE md.date = %s
              AND md.volume_is_market_level = TRUE
            """,
            (yesterday,),
        )

    total_volume    = float(df["market_volume"].sum())   if not df.empty else 0
    total_liquidity = float(df["market_liquidity"].sum()) if not df.empty else 0
    return {"total_volume": total_volume, "total_liquidity": total_liquidity}


@st.cache_data(ttl=900)
def load_category_volume(start_date: date, end_date: date, categories: Optional[tuple]) -> pd.DataFrame:
    """
    Daily volume per category over the date range for the bar chart.
    Uses MAX per market per day to avoid double-counting binary outcomes.
    """
    cat_filter = list(categories) if categories else None

    if cat_filter:
        return query(
            """
            SELECT
                md.date,
                m.category,
                SUM(subq.market_volume) AS total_volume
            FROM (
                SELECT market_id, date, MAX(daily_volume_usdc) AS market_volume
                FROM market_daily
                WHERE date BETWEEN %s AND %s
                  AND volume_is_market_level = TRUE
                GROUP BY market_id, date
            ) subq
            JOIN market_daily md ON md.market_id = subq.market_id AND md.date = subq.date
            JOIN markets m ON m.market_id = subq.market_id
            WHERE m.category = ANY(%s)
            GROUP BY md.date, m.category
            ORDER BY md.date, total_volume DESC
            """,
            (start_date, end_date, cat_filter),
        )
    else:
        return query(
            """
            SELECT
                subq.date,
                m.category,
                SUM(subq.market_volume) AS total_volume
            FROM (
                SELECT market_id, date, MAX(daily_volume_usdc) AS market_volume
                FROM market_daily
                WHERE date BETWEEN %s AND %s
                  AND volume_is_market_level = TRUE
                GROUP BY market_id, date
            ) subq
            JOIN markets m ON m.market_id = subq.market_id
            GROUP BY subq.date, m.category
            ORDER BY subq.date, total_volume DESC
            """,
            (start_date, end_date),
        )


@st.cache_data(ttl=900)
def load_topx_with_volume(snapshot_date: date, top_n: int, categories: Optional[tuple]) -> pd.DataFrame:
    """
    Top-X enriched with latest volume and liquidity from market_snapshots.
    Joins to most recent snapshot per market for intraday volume data.
    """
    cat_filter = list(categories) if categories else None
    if cat_filter:
        return query(
            """
            SELECT
                t.rank_position,
                t.market_question,
                t.outcome_label,
                m.category,
                t.composite_score,
                t.volume_z_score,
                t.dp_dv_raw,
                t.dp_dv_direction,
                snap.cumulative_volume_usdc,
                snap.period_volume_usdc,
                snap.liquidity_usdc
            FROM daily_topx t
            JOIN markets m ON m.market_id = t.market_id
            LEFT JOIN LATERAL (
                SELECT cumulative_volume_usdc, period_volume_usdc, liquidity_usdc
                FROM market_snapshots ms
                WHERE ms.outcome_id = t.outcome_id
                ORDER BY snapshot_at DESC
                LIMIT 1
            ) snap ON TRUE
            WHERE t.snapshot_date = %s
              AND t.top_n = %s
              AND m.category = ANY(%s)
            ORDER BY t.rank_position
            LIMIT %s
            """,
            (snapshot_date, top_n, cat_filter, top_n),
        )
    else:
        return query(
            """
            SELECT
                t.rank_position,
                t.market_question,
                t.outcome_label,
                m.category,
                t.composite_score,
                t.volume_z_score,
                t.dp_dv_raw,
                t.dp_dv_direction,
                snap.cumulative_volume_usdc,
                snap.period_volume_usdc,
                snap.liquidity_usdc
            FROM daily_topx t
            JOIN markets m ON m.market_id = t.market_id
            LEFT JOIN LATERAL (
                SELECT cumulative_volume_usdc, period_volume_usdc, liquidity_usdc
                FROM market_snapshots ms
                WHERE ms.outcome_id = t.outcome_id
                ORDER BY snapshot_at DESC
                LIMIT 1
            ) snap ON TRUE
            WHERE t.snapshot_date = %s
              AND t.top_n = %s
            ORDER BY t.rank_position
            LIMIT %s
            """,
            (snapshot_date, top_n, top_n),
        )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

DIRECTION_HTML = {
    1:  '<span class="dir-up">↑</span>',
    -1: '<span class="dir-down">↓</span>',
    0:  '<span class="dir-flat">→</span>',
}


def direction_symbol(val) -> str:
    try:
        return DIRECTION_HTML.get(int(val), '<span class="dir-flat">—</span>')
    except (TypeError, ValueError):
        return '<span class="dir-flat">—</span>'


def score_fmt(val) -> str:
    try:
        return f'<span class="score-pill">{float(val):.4f}</span>'
    except (TypeError, ValueError):
        return "—"


def fmt_usdc(val) -> str:
    """Format USDC value with K/M suffix."""
    try:
        v = float(val)
        if v >= 1_000_000:
            return f"${v/1_000_000:.1f}M"
        elif v >= 1_000:
            return f"${v/1_000:.1f}K"
        else:
            return f"${v:.0f}"
    except (TypeError, ValueError):
        return "—"


def render_topx_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No data for the selected filters.")
        return

    has_volume = "cumulative_volume_usdc" in df.columns

    rows_html = ""
    for _, row in df.iterrows():
        direction_cell = direction_symbol(row.get("dp_dv_direction"))
        composite_cell = score_fmt(row.get("composite_score"))
        z_score_cell   = score_fmt(row.get("volume_z_score"))

        question = str(row.get("market_question", ""))
        question_display = question if len(question) <= 60 else question[:57] + "…"

        vol_cell    = fmt_usdc(row.get("cumulative_volume_usdc")) if has_volume else "—"
        delta_cell  = fmt_usdc(row.get("period_volume_usdc"))     if has_volume else "—"
        liq_cell    = fmt_usdc(row.get("liquidity_usdc"))         if has_volume else "—"

        # Color the period volume delta green if positive
        try:
            period_v = float(row.get("period_volume_usdc") or 0)
            delta_color = "#3fb950" if period_v > 0 else "#8b949e"
        except (TypeError, ValueError):
            delta_color = "#8b949e"

        rows_html += f"""
        <tr>
            <td style="color:#8b949e">{int(row['rank_position'])}</td>
            <td title="{question}">{question_display}</td>
            <td style="color:#8b949e">{row.get('outcome_label', '')}</td>
            <td><span style="background:#1c2128;padding:2px 6px;border-radius:3px;font-size:0.75rem">{row.get('category', '—')}</span></td>
            <td>{direction_cell} {composite_cell}</td>
            <td>{z_score_cell}</td>
            <td style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem">{vol_cell}</td>
            <td style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;color:{delta_color}">{delta_cell}</td>
            <td style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;color:#8b949e">{liq_cell}</td>
        </tr>
        """

    if has_volume:
        extra_headers = "<th>Total Vol</th><th>Δ Vol</th><th>Liquidity</th>"
    else:
        extra_headers = ""

    table_html = (
        '<table class="signal-table"><thead><tr>'
        "<th>#</th><th>Market</th><th>Outcome</th><th>Category</th>"
        "<th>Composite ↑↓</th><th>Vol Z-Score</th>"
        + extra_headers
        + "</tr></thead><tbody>"
        + rows_html
        + "</tbody></table>"
    )
    components.html(
        f"<style>.signal-table{{width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:0.82rem}}.signal-table th{{background:#161b22;color:#8b949e;text-transform:uppercase;font-size:0.72rem;padding:8px 12px;border-bottom:1px solid #30363d;text-align:left}}.signal-table td{{padding:10px 12px;border-bottom:1px solid #21262d;color:#e6edf3}}.score-pill{{background:#1c2128;border:1px solid #30363d;border-radius:4px;padding:2px 8px}}.dir-up{{color:#3fb950}}.dir-down{{color:#f85149}}.dir-flat{{color:#8b949e}}</style>{table_html}",
        height=600,
        scrolling=True,
    )


def render_rank_trend(df: pd.DataFrame) -> None:
    """
    Bump chart: rank position over time per outcome.
    Lower rank = better = plotted lower on an inverted Y axis.
    Only shows outcomes that appeared in top-N on at least 2 days
    to avoid chart noise from single-day spikes.
    """
    if df.empty:
        return

    appearance_counts = df.groupby("display_label")["snapshot_date"].nunique()
    qualified = appearance_counts[appearance_counts >= 2].index.tolist()

    if not qualified:
        st.caption("No outcomes appeared in the top-N on multiple days yet — chart will populate after more history accumulates.")
        return

    plot_df = df[df["display_label"].isin(qualified)].copy()
    plot_df["snapshot_date"] = pd.to_datetime(plot_df["snapshot_date"])

    fig = px.line(
        plot_df,
        x="snapshot_date",
        y="rank_position",
        color="display_label",
        markers=True,
        labels={
            "snapshot_date": "",
            "rank_position": "Rank",
            "display_label": "",
        },
    )

    fig.update_yaxes(
        autorange="reversed",        # rank 1 at top
        dtick=1,
        gridcolor="#21262d",
        title_text="Rank Position",
    )
    fig.update_xaxes(gridcolor="#21262d")
    fig.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(family="IBM Plex Mono", color="#8b949e", size=11),
        legend=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
        hovermode="x unified",
    )
    fig.update_traces(line=dict(width=2), marker=dict(size=6))

    st.plotly_chart(fig, use_container_width=True)


def render_composite_history(df: pd.DataFrame) -> None:
    """
    Line chart of composite score over time.
    Same qualification filter as rank trend — 2+ day appearances.
    """
    if df.empty:
        return

    appearance_counts = df.groupby("display_label")["snapshot_date"].nunique()
    qualified = appearance_counts[appearance_counts >= 2].index.tolist()

    if not qualified:
        return

    plot_df = df[df["display_label"].isin(qualified)].copy()
    plot_df["snapshot_date"] = pd.to_datetime(plot_df["snapshot_date"])

    fig = px.line(
        plot_df,
        x="snapshot_date",
        y="composite_score",
        color="display_label",
        markers=True,
        labels={
            "snapshot_date": "",
            "composite_score": "Composite Score",
            "display_label": "",
        },
    )

    fig.update_yaxes(gridcolor="#21262d", range=[0, 1])
    fig.update_xaxes(gridcolor="#21262d")
    fig.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(family="IBM Plex Mono", color="#8b949e", size=11),
        legend=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=340,
        hovermode="x unified",
    )
    fig.update_traces(line=dict(width=2), marker=dict(size=5))

    st.plotly_chart(fig, use_container_width=True)


def render_category_volume(df: pd.DataFrame) -> None:
    """Stacked bar chart of daily USDC volume by category."""
    if df.empty:
        st.caption("No volume data available for this period.")
        return

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["total_volume"] = df["total_volume"].astype(float)

    # Keep top 8 categories by total volume, group rest as "Other"
    top_cats = (
        df.groupby("category")["total_volume"]
        .sum()
        .nlargest(8)
        .index.tolist()
    )
    df["cat_display"] = df["category"].apply(lambda c: c if c in top_cats else "Other")

    plot_df = df.groupby(["date", "cat_display"])["total_volume"].sum().reset_index()

    fig = px.bar(
        plot_df,
        x="date",
        y="total_volume",
        color="cat_display",
        labels={"date": "", "total_volume": "USDC Volume", "cat_display": ""},
        barmode="stack",
    )
    fig.update_yaxes(
        gridcolor="#21262d",
        tickprefix="$",
        tickformat=",.0f",
    )
    fig.update_xaxes(gridcolor="#21262d")
    fig.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(family="IBM Plex Mono", color="#8b949e", size=11),
        legend=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=340,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_new_markets(df: pd.DataFrame) -> None:
    if df.empty:
        st.caption("No new markets in the selected window.")
        return

    rows_html = ""
    for _, row in df.iterrows():
        q = str(row.get("question", ""))
        q_display = q if len(q) <= 80 else q[:77] + "…"
        rows_html += f"""
        <tr>
            <td style="color:#8b949e;font-family:'IBM Plex Mono',monospace;font-size:0.78rem">{row.get('first_seen', '')}</td>
            <td title="{q}">{q_display}</td>
            <td><span style="background:#1c2128;padding:2px 6px;border-radius:3px;font-size:0.75rem">{row.get('category', '—')}</span></td>
        </tr>
        """

    st.markdown(
        f"""
        <table class="signal-table">
            <thead>
                <tr>
                    <th>First Seen</th>
                    <th>Market</th>
                    <th>Category</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📡 Signal Tracker")
    st.markdown("---")

    config = load_config()
    default_top_n = int(config.get("top_n", 10))
    categories_available = load_categories()

    st.markdown("### Filters")

    top_n = st.slider(
        "Top N markets",
        min_value=5,
        max_value=50,
        value=default_top_n,
        step=5,
        help="Number of markets shown in the ranked table and charts.",
    )

    selected_categories = st.multiselect(
        "Categories",
        options=categories_available,
        default=[],
        placeholder="All categories",
    )
    # None = no filter; empty list also means no filter
    active_categories = selected_categories if selected_categories else None

    st.markdown("### Date Range")

    today = date.today()
    preset = st.radio(
        "Quick select",
        options=["7 days", "14 days", "30 days", "Custom"],
        index=0,
        horizontal=True,
    )

    if preset == "7 days":
        start_date = today - timedelta(days=7)
        end_date   = today
    elif preset == "14 days":
        start_date = today - timedelta(days=14)
        end_date   = today
    elif preset == "30 days":
        start_date = today - timedelta(days=30)
        end_date   = today
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From", value=today - timedelta(days=7))
        with col2:
            end_date = st.date_input("To", value=today)

    st.markdown("---")
    st.caption(
        f"Pipeline config:  \n"
        f"Window: {config.get('window_days', '5')}d  \n"
        f"Vol floor: ${float(config.get('volume_floor_usdc', 1000)):,.0f}  \n"
        f"Polling: {config.get('polling_interval_minutes', '15')} min"
    )

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.markdown("## Today's Top Signals")
st.caption(f"Ranked by composite score (Vol Z-Score + |ΔP/ΔV|) · {today.strftime('%B %d, %Y')}")

# Summary metrics row
col1, col2, col3, col4 = st.columns(4)

topx_today = load_topx_with_volume(
    today, top_n,
    tuple(active_categories) if active_categories else None
)

with col1:
    st.metric("Outcomes Ranked", len(topx_today))
with col2:
    if not topx_today.empty and "composite_score" in topx_today.columns:
        top_score = topx_today["composite_score"].max()
        st.metric("Top Score", f"{top_score:.4f}" if pd.notna(top_score) else "—")
    else:
        st.metric("Top Score", "—")
with col3:
    if not topx_today.empty and "dp_dv_direction" in topx_today.columns:
        bullish = (topx_today["dp_dv_direction"] == 1).sum()
        bearish = (topx_today["dp_dv_direction"] == -1).sum()
        st.metric("↑ Bullish / ↓ Bearish", f"{bullish} / {bearish}")
    else:
        st.metric("↑ Bullish / ↓ Bearish", "— / —")
with col4:
    if not topx_today.empty and "category" in topx_today.columns:
        top_cat = topx_today["category"].value_counts().idxmax() if not topx_today["category"].isna().all() else "—"
        st.metric("Dominant Category", top_cat)
    else:
        st.metric("Dominant Category", "—")

# Volume metrics panel
vol_metrics = load_volume_metrics(
    tuple(active_categories) if active_categories else None
)
vcol1, vcol2, vcol3 = st.columns(3)
with vcol1:
    st.metric(
        "Total USDC Volume (yesterday)",
        fmt_usdc(vol_metrics["total_volume"]) if vol_metrics["total_volume"] else "—",
    )
with vcol2:
    st.metric(
        "Total Liquidity",
        fmt_usdc(vol_metrics["total_liquidity"]) if vol_metrics["total_liquidity"] else "—",
    )
with vcol3:
    active_count = query("SELECT COUNT(*) as c FROM markets WHERE is_active = TRUE").iloc[0]["c"]
    st.metric("Active Markets", f"{int(active_count):,}")

st.markdown("")
render_topx_table(topx_today)

# ---------------------------------------------------------------------------
# Historical charts
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## Historical Trends")
st.caption(f"{start_date.strftime('%b %d')} → {end_date.strftime('%b %d, %Y')} · outcomes appearing in top-{top_n} on 2+ days shown")

hist_df = load_historical_topx(
    start_date,
    end_date,
    top_n,
    tuple(active_categories) if active_categories else None,
)

chart_col1, chart_col2 = st.columns([1, 1])

with chart_col1:
    st.markdown("### Rank Movement")
    st.caption("Position over time — lower = stronger signal")
    render_rank_trend(hist_df)

with chart_col2:
    st.markdown("### Composite Score")
    st.caption("Signal strength over time — higher = stronger")
    render_composite_history(hist_df)

# ---------------------------------------------------------------------------
# Category volume
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## Volume by Category")
st.caption(f"Daily USDC volume across top categories · {start_date.strftime('%b %d')} → {end_date.strftime('%b %d, %Y')}")

cat_vol_df = load_category_volume(
    start_date,
    end_date,
    tuple(active_categories) if active_categories else None,
)
render_category_volume(cat_vol_df)

# ---------------------------------------------------------------------------
# New markets (topic emergence)
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## New Markets")
st.caption(f"Markets first seen between {start_date.strftime('%b %d')} and {end_date.strftime('%b %d')} — topic emergence signal")

new_markets_df = load_new_markets(
    start_date,
    end_date,
    tuple(active_categories) if active_categories else None,
)

st.markdown(f"**{len(new_markets_df)} new markets** in this window")
render_new_markets(new_markets_df)
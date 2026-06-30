from __future__ import annotations

from dataclasses import dataclass
from html import escape
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_prediction.config import (  # noqa: E402
    DEFAULT_ODDS_BOOKMAKERS,
    DEFAULT_PUBLIC_ODDS_DATE_RANGE,
    DEFAULT_ODDS_SPORT_KEY,
    FIFA_HUB_URL,
    GOALSCORERS_URL,
    HISTORICAL_RESULTS_URL,
    ODDS_API_KEY_ENV,
    STATSBOMB_COMPETITIONS_URL,
)
from worldcup_prediction.evaluation import (  # noqa: E402
    prediction_performance_summary,
    rolling_match_predictions,
)
from worldcup_prediction.model import (  # noqa: E402
    RECENT_MATCH_HALFLIFE_DAYS,
    predict_matchup,
    recent_results_with_weights,
    team_match_frame,
)
from worldcup_prediction.odds import (  # noqa: E402
    attach_model_edges,
    best_odds_by_outcome,
    event_summary_frame,
    format_american_odds,
    load_cached_odds,
    matching_event_odds,
    odds_for_event,
    refresh_bookmaker_odds,
    refresh_public_bookmaker_odds,
)
from worldcup_prediction.pipeline import DataBundle, load_cached_data, refresh_data  # noqa: E402
from worldcup_prediction.schedule import fetch_upcoming_fixtures  # noqa: E402
from worldcup_prediction.source_catalog import source_catalog_frame  # noqa: E402


st.set_page_config(
    page_title="World Cup Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "World Cup Prediction Dashboard: transparent model forecasts, scorer estimates, and sportsbook-market comparisons.",
    },
)

DATA_CACHE_VERSION = 4
ODDS_CACHE_VERSION = 2
SCHEDULE_CACHE_VERSION = 1
EVALUATION_CACHE_VERSION = 1
BOOKMAKER_LABELS = {
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "betmgm": "BetMGM",
    "betrivers": "BetRivers",
    "williamhill_us": "Caesars",
    "espnbet": "ESPN BET",
    "fanatics": "Fanatics",
    "hardrockbet": "Hard Rock Bet",
    "betonlineag": "BetOnline.ag",
    "bovada": "Bovada",
    "betus": "BetUS",
    "mybookieag": "MyBookie.ag",
    "lowvig": "LowVig.ag",
}


def ensure_bundle_schema(bundle: DataBundle) -> DataBundle:
    for attribute in DataBundle.__dataclass_fields__:
        if not hasattr(bundle, attribute):
            setattr(bundle, attribute, {} if attribute == "metadata" else pd.DataFrame())
    return bundle


@st.cache_data(show_spinner=False)
def load_data(cache_version: int) -> DataBundle:
    _ = cache_version
    return ensure_bundle_schema(load_cached_data())


@st.cache_data(show_spinner=False)
def load_odds_data(cache_version: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    _ = cache_version
    return load_cached_odds()


@st.cache_data(show_spinner=False)
def load_upcoming_matchups(cache_version: int) -> pd.DataFrame:
    _ = cache_version
    return fetch_upcoming_fixtures()


@st.cache_data(show_spinner=False)
def load_prediction_audit(
    cache_version: int,
    combined_results: pd.DataFrame,
    report_team_stats: pd.DataFrame,
    statsbomb_team_stats: pd.DataFrame,
    tournament: str | None,
    max_matches: int,
    training_start: str | None,
    min_training_matches: int,
) -> pd.DataFrame:
    _ = cache_version
    return rolling_match_predictions(
        combined_results,
        report_team_stats=report_team_stats,
        statsbomb_team_stats=statsbomb_team_stats,
        tournament=tournament,
        max_matches=max_matches,
        training_start=training_start,
        min_training_matches=min_training_matches,
    )


def bundle_frame_len(bundle: DataBundle, attribute: str) -> int:
    frame = getattr(bundle, attribute, None)
    return len(frame) if frame is not None else 0


@dataclass(frozen=True)
class VisualTheme:
    name: str
    app_bg: str
    app_bg_2: str
    surface: str
    surface_muted: str
    border: str
    text: str
    muted: str
    accent: str
    accent_soft: str
    warning: str
    plot_bg: str
    grid: str
    heat_low: str
    heat_mid: str
    heat_high: str
    button_text: str


LIGHT_THEME = VisualTheme(
    name="Light",
    app_bg="#fbfcfb",
    app_bg_2="#f5f8f6",
    surface="#ffffff",
    surface_muted="#f6f8f7",
    border="#dce5e0",
    text="#14211c",
    muted="#5c6f66",
    accent="#147a64",
    accent_soft="#e2f3ee",
    warning="#b7791f",
    plot_bg="#ffffff",
    grid="#e4ece8",
    heat_low="#f7fbf9",
    heat_mid="#8ccfc0",
    heat_high="#126b58",
    button_text="#ffffff",
)

DARK_THEME = VisualTheme(
    name="Dark",
    app_bg="#0f1513",
    app_bg_2="#111c18",
    surface="#17211f",
    surface_muted="#202d29",
    border="#2d403a",
    text="#edf5f1",
    muted="#a6b8b0",
    accent="#5ee0ba",
    accent_soft="#17382f",
    warning="#f0b45d",
    plot_bg="#121a17",
    grid="#2d403a",
    heat_low="#13201c",
    heat_mid="#247d67",
    heat_high="#9cf5d3",
    button_text="#10201a",
)


def get_visual_theme(theme_name: str) -> VisualTheme:
    return DARK_THEME if theme_name == "Dark" else LIGHT_THEME


def apply_theme(theme: VisualTheme, auto_mode: bool = False) -> None:
    dark_override = ""
    if auto_mode:
        dark_override = """
        @media (prefers-color-scheme: dark) {{
            :root {{
                --app-bg: __DARK_APP_BG__;
                --app-bg-2: __DARK_APP_BG_2__;
                --surface: __DARK_SURFACE__;
                --surface-muted: __DARK_SURFACE_MUTED__;
                --border: __DARK_BORDER__;
                --text: __DARK_TEXT__;
                --muted: __DARK_MUTED__;
                --accent: __DARK_ACCENT__;
                --accent-soft: __DARK_ACCENT_SOFT__;
                --warning: __DARK_WARNING__;
            }}
        }}
        """.replace("__DARK_APP_BG__", DARK_THEME.app_bg).replace(
            "__DARK_APP_BG_2__", DARK_THEME.app_bg_2
        ).replace("__DARK_SURFACE__", DARK_THEME.surface).replace(
            "__DARK_SURFACE_MUTED__", DARK_THEME.surface_muted
        ).replace("__DARK_BORDER__", DARK_THEME.border).replace(
            "__DARK_TEXT__", DARK_THEME.text
        ).replace("__DARK_MUTED__", DARK_THEME.muted).replace(
            "__DARK_ACCENT__", DARK_THEME.accent
        ).replace("__DARK_ACCENT_SOFT__", DARK_THEME.accent_soft).replace(
            "__DARK_WARNING__", DARK_THEME.warning
        ).replace("{{", "{").replace("}}", "}")

    css = """
        <style>
        :root {{
            --app-bg: __APP_BG__;
            --app-bg-2: __APP_BG_2__;
            --surface: __SURFACE__;
            --surface-muted: __SURFACE_MUTED__;
            --border: __BORDER__;
            --text: __TEXT__;
            --muted: __MUTED__;
            --accent: __ACCENT__;
            --accent-soft: __ACCENT_SOFT__;
            --warning: __WARNING__;
            --button-text: __BUTTON_TEXT__;
        }}
        __DARK_OVERRIDE__
        #MainMenu,
        footer {{
            visibility: hidden;
        }}
        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stAppDeployButton"],
        [data-testid="stDecoration"] {{
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }}
        .block-container,
        [data-testid="stMainBlockContainer"],
        [data-testid="stAppViewContainer"] .main .block-container {{
            max-width: 1380px;
            padding: 0.9rem 2rem 3rem !important;
        }}
        .stApp {{
            background: linear-gradient(180deg, var(--app-bg) 0%, var(--app-bg-2) 100%);
            color: var(--text);
        }}
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background: var(--app-bg) !important;
            color: var(--text) !important;
        }}
        h1, h2, h3 {{
            letter-spacing: 0;
            color: var(--text);
        }}
        h2 {{
            margin-top: 1.2rem;
        }}
        .stApp, .stMarkdown, .stText, p, li, label, span {{
            color: var(--text);
        }}
        div[data-testid="column"] {{
            min-width: 0;
        }}
        section[data-testid="stSidebar"] {{
            background: var(--surface);
            border-right: 1px solid var(--border);
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"],
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {{
            gap: 0.65rem;
        }}
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div {{
            color: var(--text) !important;
        }}
        section[data-testid="stSidebar"] [data-baseweb="radio"] label,
        section[data-testid="stSidebar"] [data-baseweb="checkbox"] label,
        div[data-testid="stCheckbox"] label,
        div[data-testid="stRadio"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stMultiSelect"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stTextInput"] label {{
            color: var(--text) !important;
        }}
        div[data-baseweb="radio"] label,
        div[data-baseweb="checkbox"] label {{
            color: var(--text) !important;
        }}
        input,
        textarea,
        div[data-baseweb="input"] input,
        div[data-baseweb="select"] > div,
        div[data-baseweb="popover"] div,
        ul[role="listbox"],
        li[role="option"] {{
            background-color: var(--surface) !important;
            border-color: var(--border) !important;
            color: var(--text) !important;
        }}
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="popover"] span,
        li[role="option"] span {{
            color: var(--text) !important;
        }}
        .stButton > button,
        button[kind="primary"] {{
            background: var(--accent) !important;
            border: 1px solid var(--accent) !important;
            border-radius: 8px !important;
            color: var(--button-text) !important;
            font-weight: 700 !important;
            min-height: 2.55rem;
        }}
        .stButton > button *,
        button[kind="primary"] * {{
            color: var(--button-text) !important;
        }}
        button,
        button span {{
            color: inherit;
        }}
        div[data-testid="stMetric"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.85rem 0.9rem;
            box-shadow: 0 8px 22px rgba(18, 42, 33, 0.045);
            min-height: 5.4rem;
        }}
        div[data-testid="stMetricLabel"] p {{
            color: var(--muted);
            font-size: 0.86rem;
        }}
        div[data-testid="stMetricValue"] {{
            color: var(--text);
        }}
        .app-header {{
            padding: 0.9rem 1rem;
            background: linear-gradient(135deg, var(--surface) 0%, var(--surface-muted) 100%);
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: 0 10px 26px rgba(18, 42, 33, 0.06);
            margin-bottom: 0.85rem;
        }}
        .header-top {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
        }}
        .header-copy {{
            min-width: 0;
        }}
        .header-kicker {{
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }}
        .header-title {{
            margin: 0 0 0.35rem 0;
            font-size: 1.72rem;
            line-height: 1.1;
            letter-spacing: 0;
            color: var(--text);
            font-weight: 800;
        }}
        .app-header p {{
            margin: 0;
            color: var(--muted);
            max-width: 58rem;
            line-height: 1.45;
        }}
        .pill-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.7rem;
        }}
        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            color: var(--accent);
            background: var(--accent-soft);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.32rem 0.65rem;
            font-size: 0.82rem;
            font-weight: 600;
            max-width: 100%;
        }}
        .section-note {{
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: -0.35rem;
            margin-bottom: 0.8rem;
            line-height: 1.45;
        }}
        .toolbar-note {{
            color: var(--muted);
            font-size: 0.88rem;
            margin: 0.15rem 0 0.6rem;
        }}
        .forecast-summary {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.85rem 0 1rem;
            box-shadow: 0 8px 22px rgba(18, 42, 33, 0.045);
        }}
        .forecast-topline {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 0.85rem;
        }}
        .forecast-title {{
            font-size: 1rem;
            font-weight: 800;
            color: var(--text);
            margin-bottom: 0.25rem;
        }}
        .forecast-subtitle {{
            color: var(--muted);
            font-size: 0.9rem;
        }}
        .confidence-badge {{
            flex: 0 0 auto;
            background: var(--accent-soft);
            border: 1px solid var(--border);
            border-radius: 999px;
            color: var(--accent);
            font-weight: 800;
            padding: 0.35rem 0.7rem;
            font-size: 0.82rem;
        }}
        .probability-strip {{
            display: grid;
            grid-template-columns: var(--home-width) var(--draw-width) var(--away-width);
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 8px;
            min-height: 2.4rem;
            background: var(--surface-muted);
        }}
        .probability-segment {{
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 0;
            padding: 0.45rem 0.25rem;
            color: var(--button-text);
            font-size: 0.84rem;
            font-weight: 800;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .probability-segment.home {{
            background: var(--accent);
        }}
        .probability-segment.draw {{
            background: var(--muted);
        }}
        .probability-segment.away {{
            background: var(--warning);
        }}
        .insight-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
            margin-top: 0.8rem;
        }}
        .insight-item {{
            background: var(--surface-muted);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem;
        }}
        .insight-label {{
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }}
        .insight-value {{
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 800;
        }}
        .source-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            min-height: 6.2rem;
        }}
        .source-card strong {{
            color: var(--text);
        }}
        .source-card span {{
            color: var(--muted);
            font-size: 0.86rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.4rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.35rem;
            flex-wrap: wrap;
            overflow: visible;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.55rem 0.85rem;
            min-height: 2.5rem;
            flex: 0 1 auto;
        }}
        .stTabs [aria-selected="true"] {{
            border-color: var(--accent);
            color: var(--accent);
            background: var(--accent-soft);
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }}
        div[data-testid="stAlert"] {{
            border-radius: 8px;
            border-color: var(--border);
        }}
        a {{
            color: var(--accent) !important;
        }}
        @media (max-width: 760px) {{
            header[data-testid="stHeader"] {{
                display: block !important;
                visibility: visible !important;
                height: 3rem !important;
                background: var(--app-bg) !important;
                border-bottom: 1px solid var(--border);
                color: var(--text) !important;
            }}
            [data-testid="stToolbar"] {{
                display: flex !important;
                visibility: visible !important;
                height: 3rem !important;
                background: transparent !important;
            }}
            header[data-testid="stHeader"] button,
            button[data-testid="stExpandSidebarButton"],
            [data-testid="collapsedControl"],
            [data-testid="stSidebarCollapsedControl"] {{
                display: inline-flex !important;
                visibility: visible !important;
                color: var(--text) !important;
            }}
            .block-container,
            [data-testid="stMainBlockContainer"],
            [data-testid="stAppViewContainer"] .main .block-container {{
                padding: 3.75rem 0.85rem 2rem !important;
            }}
            .header-top,
            .forecast-topline {{
                flex-direction: column;
            }}
            .header-title {{
                font-size: 1.45rem;
            }}
            .probability-strip {{
                grid-template-columns: 1fr;
            }}
            .probability-segment {{
                justify-content: flex-start;
                padding-left: 0.75rem;
            }}
            .insight-grid {{
                grid-template-columns: 1fr 1fr;
            }}
            .stTabs [data-baseweb="tab"] {{
                flex: 1 1 calc(50% - 0.4rem);
                justify-content: center;
            }}
        }}
        </style>
    """
    css = (
        css.replace("__APP_BG__", theme.app_bg)
        .replace("__APP_BG_2__", theme.app_bg_2)
        .replace("__SURFACE__", theme.surface)
        .replace("__SURFACE_MUTED__", theme.surface_muted)
        .replace("__BORDER__", theme.border)
        .replace("__TEXT__", theme.text)
        .replace("__MUTED__", theme.muted)
        .replace("__ACCENT__", theme.accent)
        .replace("__ACCENT_SOFT__", theme.accent_soft)
        .replace("__WARNING__", theme.warning)
        .replace("__BUTTON_TEXT__", theme.button_text)
        .replace("__DARK_OVERRIDE__", dark_override)
    )
    css = css.replace("{{", "{").replace("}}", "}")
    st.html(css)


def style_plotly(fig: go.Figure, theme: VisualTheme) -> go.Figure:
    fig.update_layout(
        plot_bgcolor=theme.plot_bg,
        paper_bgcolor=theme.surface,
        font_color=theme.text,
        title_font_color=theme.text,
        legend_font_color=theme.text,
    )
    fig.update_xaxes(gridcolor=theme.grid, zerolinecolor=theme.grid, color=theme.muted, title_font_color=theme.muted)
    fig.update_yaxes(gridcolor=theme.grid, zerolinecolor=theme.grid, color=theme.muted, title_font_color=theme.muted)
    return fig


def pct(value: float) -> str:
    return f"{value:.1%}"


def pct_or_blank(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return ""


def signed_pct_or_blank(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):+.1%}"
    except (TypeError, ValueError):
        return ""


def rounded_frame(frame: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    display = frame.copy()
    numeric_columns = display.select_dtypes(include="number").columns
    display[numeric_columns] = display[numeric_columns].round(decimals)
    return display


def default_index(teams: list[str], candidates: list[str], fallback: int = 0) -> int:
    for candidate in candidates:
        if candidate in teams:
            return teams.index(candidate)
    return min(fallback, max(0, len(teams) - 1))


def safe_html(value: Any) -> str:
    return escape(str(value if value is not None else ""))


def compact_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return safe_html(value)
    if pd.isna(number):
        return "n/a"
    if number >= 1000:
        return f"{number:,.0f}"
    if number.is_integer():
        return f"{number:.0f}"
    return f"{number:.1f}"


def compact_timestamp(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return safe_html(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%b %d, %Y %H:%M UTC")


def render_hero(
    bundle: DataBundle,
    title: str = "World Cup Prediction Dashboard",
    subtitle: str = (
        "Transparent Elo, current World Cup report stats, historical results, and scorer profiles "
        "combined into matchup probabilities with recency-weighted team form."
    ),
) -> None:
    metadata = bundle.metadata or {}
    refreshed_at = compact_timestamp(metadata.get("refreshed_at", "not yet"))
    chips = [
        f"{compact_number(metadata.get('report_matches_parsed', len(bundle.report_matches)))} reports parsed",
        f"{compact_number(metadata.get('historical_results_rows', len(bundle.combined_results)))} result rows",
        f"{compact_number(metadata.get('player_profiles_rows', len(bundle.player_profiles)))} player profiles",
        f"Updated {refreshed_at}",
    ]
    chip_html = "".join(f'<span class="pill">{chip}</span>' for chip in chips)
    st.markdown(
        f"""
        <div class="app-header">
            <div class="header-top">
                <div class="header-copy">
                    <div class="header-kicker">World Cup Forecasting</div>
                    <div class="header-title" role="heading" aria-level="1">{safe_html(title)}</div>
                    <p>{safe_html(subtitle)}</p>
                </div>
            </div>
            <div class="pill-row">
                {chip_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def refresh_controls() -> str:
    st.sidebar.header("Display")
    theme_mode = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=0)

    st.sidebar.header("Data Refresh")
    download_behavior = st.sidebar.selectbox("Source files", ["Use cache when available", "Force redownload"])
    force = download_behavior == "Force redownload"
    limit_behavior = st.sidebar.selectbox("Report scope", ["All reports", "Quick check"])
    limit_enabled = limit_behavior == "Quick check"
    max_reports = None
    if limit_enabled:
        max_reports = st.sidebar.number_input("Max reports", min_value=1, value=3, step=1)

    if st.sidebar.button("Refresh model data", type="primary", width="stretch"):
        with st.spinner("Downloading sources, parsing PDFs, and rebuilding model inputs..."):
            try:
                refresh_data(force=force, max_reports=int(max_reports) if max_reports else None)
            except Exception as exc:  # noqa: BLE001 - surface refresh failures in the dashboard.
                st.sidebar.error(f"Refresh failed: {exc}")
            else:
                load_data.clear()
                st.sidebar.success("Data refreshed.")
    return theme_mode


def show_empty_state(bundle: DataBundle) -> bool:
    if not bundle.team_profiles.empty:
        return False

    st.warning("No processed data is available yet.")
    st.write("Use the sidebar refresh button, or run:")
    st.code(".venv312/bin/python3 scripts/update_data.py", language="bash")
    return True


def scoreline_heatmap(team_a: str, team_b: str, matrix: pd.DataFrame, theme: VisualTheme) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values * 100,
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale=[
                [0, theme.heat_low],
                [0.5, theme.heat_mid],
                [1, theme.heat_high],
            ],
            hovertemplate=f"{team_a} goals: %{{y}}<br>{team_b} goals: %{{x}}<br>Probability: %{{z:.2f}}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Scoreline Probability Heatmap",
        xaxis_title=f"{team_b} goals",
        yaxis_title=f"{team_a} goals",
        height=520,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return style_plotly(fig, theme)


def goal_total_heatmap(goal_totals: pd.DataFrame, theme: VisualTheme) -> go.Figure:
    display = goal_totals.copy()
    fig = go.Figure(
        data=go.Heatmap(
            z=[display["probability"].to_numpy() * 100],
            x=display["total_goals"].astype(str),
            y=["Probability"],
            colorscale=[
                [0, theme.heat_low],
                [0.5, theme.heat_mid],
                [1, theme.heat_high],
            ],
            hovertemplate="Total goals: %{x}<br>Probability: %{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Total Goals Probability",
        xaxis_title="Total goals",
        yaxis_title="",
        height=230,
        margin=dict(l=20, r=20, t=60, b=35),
    )
    return style_plotly(fig, theme)


def factor_heatmap(factors: pd.DataFrame, team_a: str, team_b: str, theme: VisualTheme) -> go.Figure:
    factor_rows = factors.copy()
    z_values = []
    text_values = []
    for _, row in factor_rows.iterrows():
        values = np.array([float(row[team_a]), float(row[team_b])])
        if "against" in str(row["factor"]).lower():
            values = -values
        if values.max() == values.min():
            normalized = np.array([0.5, 0.5])
        else:
            normalized = (values - values.min()) / (values.max() - values.min())
        z_values.append(normalized)
        text_values.append([f"{row[team_a]:.2f}", f"{row[team_b]:.2f}"])

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=[team_a, team_b],
            y=factor_rows["factor"],
            text=text_values,
            texttemplate="%{text}",
            colorscale=[
                [0, theme.surface_muted],
                [0.5, theme.plot_bg],
                [1, theme.accent],
            ],
            hovertemplate="Team: %{x}<br>Factor: %{y}<br>Value: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Model Factor Comparison",
        height=420,
        margin=dict(l=20, r=20, t=60, b=35),
    )
    return style_plotly(fig, theme)


def betting_odds_display(edge_rows: pd.DataFrame) -> pd.DataFrame:
    display = edge_rows.copy()
    if display.empty:
        return display
    for column in (
        "model_probability",
        "push_probability",
        "implied_probability",
        "no_vig_probability",
        "edge_vs_no_vig",
        "expected_value",
        "kelly_fraction",
    ):
        if column not in display:
            display[column] = pd.NA
    if "point" not in display:
        display["point"] = pd.NA
    display["odds"] = display["price_american"].map(format_american_odds)
    display["line"] = display.apply(format_market_line, axis=1)
    display["market"] = display["market"].map(format_market_name)
    display["model_probability"] = display["model_probability"].map(pct_or_blank)
    display["push_probability"] = display["push_probability"].map(pct_or_blank)
    display["implied_probability"] = display["implied_probability"].map(pct_or_blank)
    display["no_vig_probability"] = display["no_vig_probability"].map(pct_or_blank)
    display["edge_vs_no_vig"] = display["edge_vs_no_vig"].map(signed_pct_or_blank)
    display["expected_value"] = display["expected_value"].map(signed_pct_or_blank)
    display["kelly_fraction"] = display["kelly_fraction"].map(pct_or_blank)
    return display[
        [
            "commence_time",
            "home_team",
            "away_team",
            "market",
            "line",
            "outcome_name",
            "bookmaker_title",
            "odds",
            "model_probability",
            "push_probability",
            "no_vig_probability",
            "edge_vs_no_vig",
            "expected_value",
            "kelly_fraction",
            "link",
        ]
    ]


def format_market_name(value: Any) -> str:
    labels = {
        "h2h": "Moneyline",
        "spreads": "Spread",
        "totals": "Total",
    }
    return labels.get(str(value), str(value or ""))


def format_market_line(row: pd.Series) -> str:
    try:
        value = row.get("point")
        if pd.isna(value):
            return ""
        point = float(value)
    except (TypeError, ValueError):
        return ""
    if str(row.get("market") or "") == "spreads":
        return f"{point:+g}"
    return f"{point:+g}" if point < 0 else f"{point:g}"


def sportsbook_options() -> list[str]:
    options = list(dict.fromkeys([*DEFAULT_ODDS_BOOKMAKERS, *BOOKMAKER_LABELS.keys()]))
    return options


def confidence_label(probabilities: list[float]) -> str:
    ordered = sorted(probabilities, reverse=True)
    if not ordered:
        return "Low confidence"
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
    if margin >= 0.18:
        return "High confidence"
    if margin >= 0.08:
        return "Medium confidence"
    return "Tight matchup"


def render_matchup_summary(team_a: str, team_b: str, prediction: dict[str, Any], venue_label: str) -> None:
    outcome_rows = [
        (team_a, float(prediction["team_a_win_probability"])),
        ("Draw", float(prediction["draw_probability"])),
        (team_b, float(prediction["team_b_win_probability"])),
    ]
    favorite_label, favorite_probability = max(outcome_rows, key=lambda row: row[1])
    total_expected_goals = float(prediction["team_a_expected_goals"]) + float(prediction["team_b_expected_goals"])
    advantage = float(prediction["breakdown"]["advantage"])
    st.markdown(
        f"""
        <div class="forecast-summary">
            <div class="forecast-topline">
                <div>
                    <div class="forecast-title">Lean: {safe_html(favorite_label)} at {pct(favorite_probability)}</div>
                    <div class="forecast-subtitle">
                        {safe_html(team_a)} vs {safe_html(team_b)} · {safe_html(venue_label)}
                    </div>
                </div>
                <div class="confidence-badge">{safe_html(confidence_label([row[1] for row in outcome_rows]))}</div>
            </div>
            <div
                class="probability-strip"
                style="--home-width: {max(outcome_rows[0][1], 0.02):.4f}fr; --draw-width: {max(outcome_rows[1][1], 0.02):.4f}fr; --away-width: {max(outcome_rows[2][1], 0.02):.4f}fr;"
            >
                <div class="probability-segment home">{safe_html(team_a)} {pct(outcome_rows[0][1])}</div>
                <div class="probability-segment draw">Draw {pct(outcome_rows[1][1])}</div>
                <div class="probability-segment away">{safe_html(team_b)} {pct(outcome_rows[2][1])}</div>
            </div>
            <div class="insight-grid">
                <div class="insight-item">
                    <div class="insight-label">{safe_html(team_a)} xG</div>
                    <div class="insight-value">{float(prediction["team_a_expected_goals"]):.2f}</div>
                </div>
                <div class="insight-item">
                    <div class="insight-label">{safe_html(team_b)} xG</div>
                    <div class="insight-value">{float(prediction["team_b_expected_goals"]):.2f}</div>
                </div>
                <div class="insight-item">
                    <div class="insight-label">Expected total</div>
                    <div class="insight-value">{total_expected_goals:.2f}</div>
                </div>
                <div class="insight-item">
                    <div class="insight-label">Model edge</div>
                    <div class="insight-value">{advantage:+.2f}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_betting_odds_tab(
    team_a: str,
    team_b: str,
    prediction: dict[str, Any],
    venue_label: str,
    theme: VisualTheme,
) -> None:
    st.subheader("Betting Odds")
    st.markdown(
        '<p class="section-note">Compares cached sportsbook moneyline odds with the model probabilities for this selected matchup. Informational only; verify legality, availability, and final lines in your sportsbook.</p>',
        unsafe_allow_html=True,
    )

    odds_rows, odds_metadata = load_odds_data(ODDS_CACHE_VERSION)
    st.markdown("#### No-key public odds scrape")
    st.caption(
        "Refreshes the local odds cache from ESPN's public World Cup scoreboard feed. "
        "That feed currently exposes DraftKings moneyline, spread, and total prices when available."
    )
    public_cols = st.columns([1.0, 0.5], vertical_alignment="bottom")
    with public_cols[0]:
        public_date_range = st.text_input(
            "ESPN date range",
            value=odds_metadata.get("date_range", DEFAULT_PUBLIC_ODDS_DATE_RANGE),
            help="Use YYYYMMDD or YYYYMMDD-YYYYMMDD. The default covers the full 2026 World Cup.",
        )
    with public_cols[1]:
        scrape_clicked = st.button("Scrape public odds", type="primary", width="stretch")

    if scrape_clicked:
        with st.spinner("Refreshing public sportsbook odds only..."):
            try:
                odds_rows, odds_metadata = refresh_public_bookmaker_odds(date_range=public_date_range)
            except Exception as exc:  # noqa: BLE001 - show network/source failures in the dashboard.
                st.error(f"Public odds scrape failed: {exc}")
            else:
                load_odds_data.clear()
                if odds_rows.empty:
                    st.warning("Public scrape completed, but no moneyline odds were returned.")
                else:
                    st.success("Public sportsbook odds refreshed.")

    with st.expander("Optional API refresh for more sportsbooks"):
        controls = st.columns([1.0, 1.0, 1.4, 0.75], vertical_alignment="bottom")
        with controls[0]:
            sport_key = st.text_input("Odds sport key", value=odds_metadata.get("sport_key", DEFAULT_ODDS_SPORT_KEY))
        with controls[1]:
            selected_markets = st.text_input("Markets", value=odds_metadata.get("markets", "h2h,spreads,totals"))
        with controls[2]:
            options = sportsbook_options()
            default_bookmakers = [bookmaker for bookmaker in DEFAULT_ODDS_BOOKMAKERS if bookmaker in options]
            metadata_bookmakers = odds_metadata.get("bookmakers") if odds_metadata.get("source") == "The Odds API" else None
            selected_bookmakers = st.multiselect(
                "Sportsbooks",
                options,
                default=metadata_bookmakers or default_bookmakers,
                format_func=lambda key: BOOKMAKER_LABELS.get(key, key),
            )
        with controls[3]:
            refresh_clicked = st.button("Refresh API odds", width="stretch")

        if refresh_clicked:
            with st.spinner("Refreshing sportsbook odds only..."):
                try:
                    odds_rows, odds_metadata = refresh_bookmaker_odds(
                        sport_key=sport_key,
                        bookmakers=selected_bookmakers,
                        markets=selected_markets,
                    )
                except Exception as exc:  # noqa: BLE001 - show API/config failures in the dashboard.
                    st.error(f"Odds refresh failed: {exc}")
                    st.caption(f"Set `{ODDS_API_KEY_ENV}` in the environment before refreshing API odds.")
                else:
                    load_odds_data.clear()
                    st.success("Bookmaker odds refreshed.")

    metrics = st.columns(5)
    metrics[0].metric("Cached odds rows", len(odds_rows))
    metrics[1].metric("Events returned", odds_metadata.get("events_returned", 0))
    metrics[2].metric("Odds source", odds_metadata.get("source", "n/a"))
    metrics[3].metric("Odds refreshed", odds_metadata.get("refreshed_at", "n/a"))
    metrics[4].metric("Model context", venue_label)

    matching_odds = matching_event_odds(odds_rows, team_a, team_b)
    if matching_odds.empty:
        st.info("No cached sportsbook odds match this selected matchup yet.")
        if not odds_rows.empty:
            event_preview = (
                odds_rows[["commence_time", "home_team", "away_team"]]
                .drop_duplicates()
                .sort_values("commence_time")
                .head(20)
            )
            st.dataframe(event_preview, width="stretch", hide_index=True)
        return

    event_rows = event_summary_frame(matching_odds)
    if event_rows.empty:
        st.info("No usable sportsbook events match this selected matchup yet.")
        return
    selected_event_key = event_rows.iloc[0]["event_key"]
    if len(event_rows) > 1:
        labels = {
            row.event_key: f"{row.commence_time} | {row.home_team} vs {row.away_team} | {row.bookmakers} books"
            for row in event_rows.itertuples(index=False)
        }
        selected_event_key = st.selectbox(
            "Odds fixture",
            event_rows["event_key"].tolist(),
            format_func=lambda key: labels.get(key, key),
        )
    else:
        event = event_rows.iloc[0]
        st.caption(f"Odds fixture: {event['commence_time']} | {event['home_team']} vs {event['away_team']}")

    selected_odds = odds_for_event(matching_odds, selected_event_key)
    edge_rows = attach_model_edges(selected_odds, prediction)
    h2h_edges = edge_rows[edge_rows["market"].astype(str).eq("h2h")].copy()
    best_rows = best_odds_by_outcome(h2h_edges)
    if not best_rows.empty:
        st.subheader("Best Available Moneyline Prices")
        st.dataframe(betting_odds_display(best_rows), width="stretch", hide_index=True)

    market_edges = edge_rows[edge_rows["market"].astype(str).isin(["spreads", "totals"])].copy()
    if not market_edges.empty:
        st.subheader("Spread and Total Markets")
        st.dataframe(betting_odds_display(market_edges), width="stretch", hide_index=True)

    goalscorer_mask = edge_rows["market"].astype(str).str.contains("goalscorer|goal_scorer|player", case=False, na=False)
    goalscorer_edges = edge_rows[goalscorer_mask].copy()
    st.subheader("Goalscorer Odds")
    if goalscorer_edges.empty:
        st.info(
            "No goalscorer prices were returned by the no-key public feed. "
            "The app still shows model-estimated scorer probabilities in the Player Scoring tab."
        )
    else:
        st.dataframe(betting_odds_display(goalscorer_edges), width="stretch", hide_index=True)

    positive_edges = edge_rows[
        (pd.to_numeric(edge_rows["expected_value"], errors="coerce") > 0)
        & pd.to_numeric(edge_rows["model_probability"], errors="coerce").notna()
    ]
    if not positive_edges.empty:
        fig = px.bar(
            positive_edges.head(20),
            x="expected_value",
            y="bookmaker_title",
            color="outcome_name",
            orientation="h",
            hover_data={
                "outcome_name": True,
                "price_american": True,
                "model_probability": ":.1%",
                "no_vig_probability": ":.1%",
                "edge_vs_no_vig": ":.1%",
            },
            title="Positive Expected Value Rows",
        )
        fig.update_layout(height=max(320, 30 * len(positive_edges.head(20)) + 140), xaxis_tickformat=".0%")
        st.plotly_chart(style_plotly(fig, theme), width="stretch")

    st.subheader("All Matching Odds Rows")
    st.dataframe(betting_odds_display(edge_rows), width="stretch", hide_index=True)


def player_scoring_chart(player_scoring: pd.DataFrame, team: str, theme: VisualTheme) -> go.Figure:
    rows = player_scoring[player_scoring["team"].eq(team)].copy()
    rows = rows.sort_values("score_probability", ascending=True)
    fig = px.bar(
        rows,
        x="score_probability",
        y="player",
        orientation="h",
        text=rows["score_probability"].map(pct),
        color="score_probability",
        color_continuous_scale=[theme.surface_muted, theme.accent],
        hover_data={
            "expected_goals": ":.3f",
            "recent_goals_24m": True,
            "goals": True,
            "score_probability": ":.1%",
            "player": False,
        },
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        title=f"{team} player scoring chances",
        xaxis_title="Chance to score at least once",
        yaxis_title="",
        height=max(320, 38 * len(rows) + 130),
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=20, r=70, t=60, b=35),
    )
    fig.update_xaxes(tickformat=".0%")
    return style_plotly(fig, theme)


def ensure_recent_profile_columns(profiles: pd.DataFrame) -> pd.DataFrame:
    display = profiles.copy()
    if "weighted_recent_points_per_match" not in display and "recent_points_per_match" in display:
        display["weighted_recent_points_per_match"] = display["recent_points_per_match"]
    if "weighted_recent_goals_for" not in display and "recent_goals_for" in display:
        display["weighted_recent_goals_for"] = display["recent_goals_for"]
    if "weighted_recent_goals_against" not in display and "recent_goals_against" in display:
        display["weighted_recent_goals_against"] = display["recent_goals_against"]
    if "weighted_recent_goal_diff" not in display:
        goals_for = pd.to_numeric(display.get("weighted_recent_goals_for", 0), errors="coerce").fillna(0)
        goals_against = pd.to_numeric(display.get("weighted_recent_goals_against", 0), errors="coerce").fillna(0)
        display["weighted_recent_goal_diff"] = goals_for - goals_against
    if "recent_result_weight" not in display:
        display["recent_result_weight"] = 0.0

    recent_columns = [
        "weighted_recent_points_per_match",
        "weighted_recent_goals_for",
        "weighted_recent_goals_against",
        "weighted_recent_goal_diff",
        "recent_result_weight",
    ]
    for column in recent_columns:
        display[column] = pd.to_numeric(display[column], errors="coerce").fillna(0)
    return display


def recent_form_ranking_chart(profiles: pd.DataFrame, theme: VisualTheme) -> go.Figure:
    rows = ensure_recent_profile_columns(profiles).sort_values("weighted_recent_points_per_match", ascending=False).head(25)
    rows = rows.sort_values("weighted_recent_points_per_match", ascending=True)
    fig = px.bar(
        rows,
        x="weighted_recent_points_per_match",
        y="team",
        orientation="h",
        color="weighted_recent_goal_diff",
        color_continuous_scale=[theme.warning, theme.surface_muted, theme.accent],
        color_continuous_midpoint=0,
        hover_data={
            "rating": ":.0f" if "rating" in rows else False,
            "recent_matches": True if "recent_matches" in rows else False,
            "recent_result_weight": ":.2f",
            "weighted_recent_goal_diff": ":.2f",
        },
        title="Best Recency-Weighted Team Form",
    )
    fig.update_layout(
        height=660,
        xaxis_title="Weighted recent points per match",
        yaxis_title="",
        coloraxis_colorbar_title="Weighted GD",
    )
    return style_plotly(fig, theme)


def recent_form_scatter_chart(profiles: pd.DataFrame, theme: VisualTheme) -> go.Figure:
    rows = ensure_recent_profile_columns(profiles)
    if "rating" not in rows:
        rows["rating"] = 0
    rows["scatter_size"] = rows["recent_result_weight"].clip(lower=0.2)
    fig = px.scatter(
        rows,
        x="rating",
        y="weighted_recent_points_per_match",
        size="scatter_size",
        size_max=24,
        color="weighted_recent_goal_diff",
        color_continuous_scale=[theme.warning, theme.surface_muted, theme.accent],
        color_continuous_midpoint=0,
        hover_name="team",
        hover_data={
            "scatter_size": False,
            "recent_result_weight": ":.2f",
            "weighted_recent_goals_for": ":.2f",
            "weighted_recent_goals_against": ":.2f",
            "weighted_recent_goal_diff": ":.2f",
        },
        title="Rating vs Recency-Weighted Form",
    )
    fig.update_layout(
        height=520,
        xaxis_title="Elo-style rating",
        yaxis_title="Weighted recent points per match",
        coloraxis_colorbar_title="Weighted GD",
    )
    return style_plotly(fig, theme)


def recent_inputs_heatmap(profiles: pd.DataFrame, selected_team: str, theme: VisualTheme) -> go.Figure:
    rows = ensure_recent_profile_columns(profiles)
    if selected_team != "All teams" and selected_team in set(rows["team"]):
        selected_row = rows[rows["team"].eq(selected_team)]
        comparison_rows = rows.sort_values("weighted_recent_points_per_match", ascending=False).head(19)
        rows = pd.concat([selected_row, comparison_rows], ignore_index=True).drop_duplicates("team")
    else:
        rows = rows.sort_values("weighted_recent_points_per_match", ascending=False).head(20)

    metrics = [
        ("weighted_recent_points_per_match", "Weighted PPM", False),
        ("weighted_recent_goal_diff", "Weighted GD", False),
        ("weighted_recent_goals_for", "Weighted GF", False),
        ("weighted_recent_goals_against", "Weighted GA", True),
        ("recent_result_weight", "Effective matches", False),
    ]
    z_values = []
    text_values = []
    for column, _, invert in metrics:
        values = pd.to_numeric(rows[column], errors="coerce").fillna(0).to_numpy(dtype=float)
        comparable_values = -values if invert else values
        if comparable_values.max() == comparable_values.min():
            normalized = np.full_like(comparable_values, 0.5, dtype=float)
        else:
            normalized = (comparable_values - comparable_values.min()) / (comparable_values.max() - comparable_values.min())
        z_values.append(normalized)
        text_values.append([f"{value:.2f}" for value in values])

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=rows["team"],
            y=[label for _, label, _ in metrics],
            text=text_values,
            texttemplate="%{text}",
            colorscale=[
                [0, theme.surface_muted],
                [0.5, theme.plot_bg],
                [1, theme.accent],
            ],
            hovertemplate="Team: %{x}<br>Metric: %{y}<br>Value: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Recent Form Inputs Heatmap",
        height=430,
        margin=dict(l=20, r=20, t=60, b=80),
    )
    fig.update_xaxes(tickangle=-35)
    return style_plotly(fig, theme)


def recent_team_timeline(team_rows: pd.DataFrame, selected_team: str, theme: VisualTheme) -> go.Figure:
    rows = team_rows[team_rows["team"].eq(selected_team)].sort_values("date").tail(30).copy()
    color_map = {"W": theme.accent, "D": theme.muted, "L": theme.warning}
    marker_colors = rows["result"].map(color_map).fillna(theme.accent)
    marker_sizes = 8 + (pd.to_numeric(rows["recency_weight"], errors="coerce").fillna(0) * 18)
    fig = go.Figure(
        data=go.Scatter(
            x=rows["date"],
            y=rows["points"],
            mode="lines+markers",
            marker=dict(size=marker_sizes, color=marker_colors, line=dict(color=theme.border, width=1)),
            line=dict(color=theme.muted, width=2),
            customdata=rows[["opponent", "goals_for", "goals_against", "result", "recency_weight"]],
            hovertemplate=(
                "Opponent: %{customdata[0]}<br>"
                "Score: %{customdata[1]}-%{customdata[2]}<br>"
                "Result: %{customdata[3]}<br>"
                "Recency weight: %{customdata[4]:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"{selected_team} Result Timeline",
        height=410,
        yaxis_title="Points",
        xaxis_title="Match date",
        yaxis=dict(tickmode="array", tickvals=[0, 1, 3], range=[-0.25, 3.25]),
    )
    return style_plotly(fig, theme)


def render_matchup_analysis(
    team_a: str,
    team_b: str,
    venue_label: str,
    bundle: DataBundle,
    theme: VisualTheme,
    home_team: str | None = None,
) -> None:
    profiles = bundle.team_profiles
    neutral = home_team is None
    prediction = predict_matchup(
        team_a,
        team_b,
        profiles,
        bundle.player_profiles,
        neutral=neutral,
        home_team=home_team,
    )
    render_matchup_summary(team_a, team_b, prediction, venue_label)
    probabilities = st.columns(5)
    probabilities[0].metric(f"{team_a} win", pct(prediction["team_a_win_probability"]))
    probabilities[1].metric("Draw", pct(prediction["draw_probability"]))
    probabilities[2].metric(f"{team_b} win", pct(prediction["team_b_win_probability"]))
    probabilities[3].metric(f"{team_a} xG", f"{prediction['team_a_expected_goals']:.2f}")
    probabilities[4].metric(f"{team_b} xG", f"{prediction['team_b_expected_goals']:.2f}")

    tabs = st.tabs(["Prediction", "Heatmaps", "Player Scoring", "Betting Odds", "Model Inputs"])
    with tabs[0]:
        st.subheader("Match Forecast")
        st.markdown(
            '<p class="section-note">Expected goals are converted to an ensemble Poisson scoreline distribution.</p>',
            unsafe_allow_html=True,
        )
        left_chart, right_chart = st.columns([1.25, 0.75])
        with left_chart:
            outcome_frame = pd.DataFrame(
                {
                    "Outcome": [f"{team_a} win", "Draw", f"{team_b} win"],
                    "Probability": [
                        prediction["team_a_win_probability"],
                        prediction["draw_probability"],
                        prediction["team_b_win_probability"],
                    ],
                }
            )
            fig = px.bar(
                outcome_frame,
                x="Outcome",
                y="Probability",
                text=outcome_frame["Probability"].map(pct),
                color="Outcome",
                color_discrete_sequence=[theme.accent, theme.muted, theme.warning],
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(height=430, showlegend=False, yaxis_tickformat=".0%", yaxis_range=[0, 1])
            st.plotly_chart(style_plotly(fig, theme), width="stretch")
        with right_chart:
            st.metric("Most likely score", prediction["most_likely_score"])
            st.metric("Expected total goals", f"{prediction['team_a_expected_goals'] + prediction['team_b_expected_goals']:.2f}")
            st.metric("Model edge", f"{prediction['breakdown']['advantage']:+.2f}")

    with tabs[1]:
        st.subheader("Heatmaps")
        st.plotly_chart(scoreline_heatmap(team_a, team_b, prediction["score_matrix"], theme), width="stretch")
        lower_left, lower_right = st.columns([0.9, 1.1])
        with lower_left:
            st.plotly_chart(goal_total_heatmap(prediction["goal_total_probabilities"], theme), width="stretch")
        with lower_right:
            st.plotly_chart(factor_heatmap(prediction["factor_frame"], team_a, team_b, theme), width="stretch")

    with tabs[2]:
        st.subheader("Estimated Player Scoring Chances")
        st.markdown(
            '<p class="section-note">These are estimated from historical international goalscorer records and the team expected-goal forecast, not confirmed lineups.</p>',
            unsafe_allow_html=True,
        )
        player_scoring = prediction["player_scoring"]
        if player_scoring.empty:
            st.info("No player scorer profiles are available for this matchup yet.")
        else:
            scorer_left, scorer_right = st.columns(2)
            with scorer_left:
                st.plotly_chart(player_scoring_chart(player_scoring, team_a, theme), width="stretch")
            with scorer_right:
                st.plotly_chart(player_scoring_chart(player_scoring, team_b, theme), width="stretch")
            display = player_scoring.copy()
            display["score_probability"] = display["score_probability"].map(pct)
            display["first_team_goal_probability"] = display["first_team_goal_probability"].map(pct)
            st.dataframe(rounded_frame(display), width="stretch", hide_index=True)

    with tabs[3]:
        render_betting_odds_tab(team_a, team_b, prediction, venue_label, theme)

    with tabs[4]:
        st.subheader("Model Inputs")
        breakdown = prediction["breakdown"]
        score_model = pd.DataFrame(
            [
                {
                    "model": "Structural form + rating",
                    team_a: breakdown["structural_team_a_expected_goals"],
                    team_b: breakdown["structural_team_b_expected_goals"],
                    "weight": 1 - breakdown["ensemble_attack_defense_weight"],
                },
                {
                    "model": "Attack/defense scoring rates",
                    team_a: breakdown["attack_defense_team_a_expected_goals"],
                    team_b: breakdown["attack_defense_team_b_expected_goals"],
                    "weight": breakdown["ensemble_attack_defense_weight"],
                },
                {
                    "model": "Final ensemble",
                    team_a: prediction["team_a_expected_goals"],
                    team_b: prediction["team_b_expected_goals"],
                    "weight": 1.0,
                },
            ]
        )
        st.markdown(
            '<p class="section-note">Score probabilities blend the existing rating/form model with a smaller attack-defense scoring-rate model. A rolling recent-match check favored a low-weight blend for score log likelihood.</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(rounded_frame(score_model), width="stretch", hide_index=True)

        comparison_columns = [
            "team",
            "rating",
            "matches",
            "recent_matches",
            "recent_points_per_match",
            "recent_goals_for",
            "recent_goals_against",
            "weighted_recent_points_per_match",
            "weighted_recent_goals_for",
            "weighted_recent_goals_against",
            "weighted_recent_goal_diff",
            "recent_result_weight",
            "report_matches",
            "xg_for_2026",
            "xg_for_2026_shrunk",
            "xg_against_2026",
            "xg_against_2026_shrunk",
            "attempts_for_2026",
            "attempts_against_2026",
            "possession_pct_2026",
            "statsbomb_matches",
            "statsbomb_xg_for",
            "statsbomb_xg_for_shrunk",
            "statsbomb_xg_against",
            "statsbomb_xg_against_shrunk",
            "statsbomb_shots_for",
            "statsbomb_shots_against",
        ]
        available_columns = [column for column in comparison_columns if column in profiles.columns]
        comparison = profiles.loc[profiles["team"].isin([team_a, team_b]), available_columns]
        st.dataframe(rounded_frame(comparison), width="stretch", hide_index=True)


def matchup_page(bundle: DataBundle, theme: VisualTheme) -> None:
    render_hero(
        bundle,
        "Matchup Predictor",
        "Set two teams and venue context, then review win probability, xG, scoreline, scorer, and betting-market views.",
    )
    if show_empty_state(bundle):
        return

    profiles = bundle.team_profiles
    teams = sorted(profiles["team"].dropna().astype(str).unique())
    if len(teams) < 2:
        st.warning("At least two teams are required for a matchup.")
        return

    st.subheader("Configure Matchup")
    st.markdown(
        '<p class="toolbar-note">Venue context adjusts expected goals only when one selected team is treated as home.</p>',
        unsafe_allow_html=True,
    )
    left, right, action = st.columns([1.1, 1.1, 0.8], vertical_alignment="bottom")
    with left:
        team_a = st.selectbox("Team A", teams, index=default_index(teams, ["USA", "Mexico", "Argentina"]))
    with right:
        team_b = st.selectbox("Team B", teams, index=default_index(teams, ["Brazil", "Germany", "France"], fallback=1))
    with action:
        venue_options = list(dict.fromkeys(["Neutral site", f"{team_a} home", f"{team_b} home"]))
        venue_label = st.selectbox("Venue", venue_options)

    if team_a == team_b:
        st.info("Select two different teams.")
        return

    home_team = None
    if venue_label == f"{team_a} home":
        home_team = team_a
    elif venue_label == f"{team_b} home":
        home_team = team_b

    render_matchup_analysis(team_a, team_b, venue_label, bundle, theme, home_team=home_team)


def upcoming_matchups_page(bundle: DataBundle, theme: VisualTheme) -> None:
    render_hero(
        bundle,
        "Upcoming Matchups",
        "Pick a scheduled World Cup fixture and run the same forecast, scorer, and betting-market workflow.",
    )
    if show_empty_state(bundle):
        return

    profiles = bundle.team_profiles
    teams = sorted(profiles["team"].dropna().astype(str).unique())
    if len(teams) < 2:
        st.warning("At least two teams are required for a matchup.")
        return

    header_cols = st.columns([1.4, 0.9, 0.7], vertical_alignment="bottom")
    with header_cols[0]:
        st.subheader("Scheduled Fixtures")
        st.markdown(
            '<p class="toolbar-note">Fixtures are loaded from ESPN public schedule data; refresh here without rebuilding the model.</p>',
            unsafe_allow_html=True,
        )
    with header_cols[2]:
        if st.button("Refresh schedule", type="primary", width="stretch"):
            load_upcoming_matchups.clear()

    try:
        fixtures = load_upcoming_matchups(SCHEDULE_CACHE_VERSION)
    except Exception as exc:  # noqa: BLE001 - keep the app usable when the public schedule endpoint is unavailable.
        st.error(f"Upcoming fixtures failed to load: {exc}")
        return

    if fixtures.empty:
        st.info("No upcoming fixtures were returned by the public schedule feed.")
        return

    total_fixture_count = len(fixtures)
    available_teams = set(teams)
    fixtures = fixtures[
        fixtures["home_team"].astype(str).isin(available_teams)
        & fixtures["away_team"].astype(str).isin(available_teams)
    ].copy()
    if fixtures.empty:
        st.warning("Upcoming fixtures loaded, but none match teams currently available in the model profiles.")
        return
    if len(fixtures) < total_fixture_count:
        hidden_count = total_fixture_count - len(fixtures)
        st.caption(
            f"Showing {len(fixtures)} fixtures with confirmed model teams; "
            f"{hidden_count} bracket-placeholder fixtures are hidden until teams are known."
        )

    fixtures["kickoff_ct"] = pd.to_datetime(fixtures["commence_time"], utc=True, errors="coerce").dt.tz_convert(
        "America/Chicago"
    )
    fixtures["fixture_label"] = fixtures.apply(
        lambda row: (
            f"{row['kickoff_ct'].strftime('%b %d, %-I:%M %p CT')} | "
            f"{row['home_team']} vs {row['away_team']}"
        ),
        axis=1,
    )

    team_options = ["All teams"] + sorted(
        set(fixtures["home_team"].dropna().astype(str)).union(fixtures["away_team"].dropna().astype(str))
    )
    filters = st.columns([0.8, 1.5, 0.9], vertical_alignment="bottom")
    with filters[0]:
        team_filter = st.selectbox("Team filter", team_options)

    visible_fixtures = fixtures
    if team_filter != "All teams":
        visible_fixtures = fixtures[
            fixtures["home_team"].eq(team_filter) | fixtures["away_team"].eq(team_filter)
        ].copy()
    if visible_fixtures.empty:
        st.info("No upcoming fixtures match that team filter.")
        return

    label_by_id = dict(zip(visible_fixtures["event_id"], visible_fixtures["fixture_label"], strict=False))
    with filters[1]:
        selected_event_id = st.selectbox(
            "Upcoming fixture",
            visible_fixtures["event_id"].astype(str).tolist(),
            format_func=lambda event_id: label_by_id.get(str(event_id), str(event_id)),
        )
    selected = visible_fixtures[visible_fixtures["event_id"].astype(str).eq(str(selected_event_id))].iloc[0]
    team_a = str(selected["home_team"])
    team_b = str(selected["away_team"])
    with filters[2]:
        venue_options = list(dict.fromkeys(["Neutral site", f"{team_a} home", f"{team_b} home"]))
        venue_label = st.selectbox("Model venue context", venue_options)

    home_team = None
    if venue_label == f"{team_a} home":
        home_team = team_a
    elif venue_label == f"{team_b} home":
        home_team = team_b

    fixture_metrics = st.columns(4)
    fixture_metrics[0].metric("Kickoff", selected["kickoff_ct"].strftime("%b %d, %-I:%M %p CT"))
    fixture_metrics[1].metric("Status", str(selected.get("status") or "Scheduled"))
    fixture_metrics[2].metric("Venue", str(selected.get("venue") or "TBD"))
    fixture_metrics[3].metric("Source", str(selected.get("source") or "Public feed"))

    table = visible_fixtures[["kickoff_ct", "home_team", "away_team", "venue", "status", "source"]].copy()
    table["kickoff_ct"] = table["kickoff_ct"].dt.strftime("%b %d, %-I:%M %p CT")
    st.dataframe(table.head(30), width="stretch", hide_index=True)

    render_matchup_analysis(team_a, team_b, venue_label, bundle, theme, home_team=home_team)


def model_performance_page(bundle: DataBundle, theme: VisualTheme) -> None:
    render_hero(
        bundle,
        "Model Performance",
        "Compare rolling pre-match predictions with completed match results to audit how the model is performing.",
    )
    if bundle.combined_results.empty:
        st.info("No completed-match results are available yet.")
        return

    results = bundle.combined_results.copy()
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results = results.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    if results.empty:
        st.info("No completed-match results are available yet.")
        return

    tournaments = sorted(results["tournament"].dropna().astype(str).unique())
    tournament_options = ["All tournaments"] + tournaments
    default_scope = "FIFA World Cup 2026" if "FIFA World Cup 2026" in tournament_options else "All tournaments"

    st.subheader("Prediction Audit")
    st.markdown(
        '<p class="toolbar-note">Each row is predicted with only matches and report stats dated before that match. Lower Brier and log-loss scores are better.</p>',
        unsafe_allow_html=True,
    )
    controls = st.columns([1.2, 0.8, 0.8, 0.8], vertical_alignment="bottom")
    with controls[0]:
        selected_tournament = st.selectbox(
            "Result scope",
            tournament_options,
            index=default_index(tournament_options, [default_scope]),
        )
    with controls[1]:
        match_count = st.slider("Completed matches", min_value=10, max_value=200, value=60, step=10)
    with controls[2]:
        training_label = st.selectbox("Training history", ["Since 2018", "Since 2020", "Since 2022", "All prior"])
    with controls[3]:
        min_training_matches = st.slider("Min prior matches", min_value=10, max_value=200, value=50, step=10)

    training_start = {
        "Since 2018": "2018-01-01",
        "Since 2020": "2020-01-01",
        "Since 2022": "2022-01-01",
        "All prior": None,
    }[training_label]
    tournament_filter = None if selected_tournament == "All tournaments" else selected_tournament

    with st.spinner("Building rolling pre-match prediction audit..."):
        audit = load_prediction_audit(
            EVALUATION_CACHE_VERSION,
            bundle.combined_results,
            bundle.report_team_stats,
            bundle.statsbomb_team_stats,
            tournament_filter,
            match_count,
            training_start,
            min_training_matches,
        )

    if audit.empty:
        st.warning("No auditable matches were available for the selected scope and training-history settings.")
        return

    summary = prediction_performance_summary(audit)
    metrics = st.columns(6)
    metrics[0].metric("Matches audited", f"{int(summary['matches'])}")
    metrics[1].metric("Outcome accuracy", pct(summary["outcome_accuracy"]))
    metrics[2].metric("Exact score hit", pct(summary["exact_score_accuracy"]))
    metrics[3].metric("Avg actual-result probability", pct(summary["average_actual_outcome_probability"]))
    metrics[4].metric("Brier score", f"{summary['average_brier_score']:.3f}")
    metrics[5].metric("Score log loss", f"{summary['average_score_log_loss']:.3f}")

    chart_rows = audit.sort_values("date").copy()
    chart_rows["match"] = chart_rows["home_team"] + " vs " + chart_rows["away_team"]
    chart_rows["correct_label"] = np.where(chart_rows["prediction_correct"], "Correct outcome", "Missed outcome")
    left_chart, right_chart = st.columns([1.15, 0.85])
    with left_chart:
        fig = px.scatter(
            chart_rows,
            x="date",
            y="actual_outcome_probability",
            color="correct_label",
            size="exact_score_probability",
            hover_data={
                "match": True,
                "actual_score": True,
                "predicted_score": True,
                "actual_outcome_probability": ":.1%",
                "exact_score_probability": ":.2%",
                "correct_label": False,
            },
            color_discrete_map={"Correct outcome": theme.accent, "Missed outcome": theme.warning},
        )
        fig.update_layout(
            title="Probability Assigned to Actual Outcome",
            height=420,
            yaxis_tickformat=".0%",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(style_plotly(fig, theme), width="stretch")
    with right_chart:
        confusion = (
            audit.groupby(["actual_outcome", "predicted_outcome"], as_index=False)
            .size()
            .rename(columns={"size": "matches"})
        )
        fig = px.bar(
            confusion,
            x="actual_outcome",
            y="matches",
            color="predicted_outcome",
            barmode="group",
            color_discrete_map={"Home": theme.accent, "Draw": theme.muted, "Away": theme.warning},
        )
        fig.update_layout(title="Outcome Prediction Mix", height=420)
        st.plotly_chart(style_plotly(fig, theme), width="stretch")

    st.subheader("Predictions vs Actual Results")
    display = audit.copy()
    display["date"] = pd.to_datetime(display["date"], errors="coerce").dt.date.astype(str)
    display["match"] = display["home_team"] + " vs " + display["away_team"]
    display["prediction_correct"] = np.where(display["prediction_correct"], "Yes", "No")
    display["exact_score_hit"] = np.where(display["exact_score_hit"], "Yes", "No")
    for column in [
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
        "actual_outcome_probability",
        "exact_score_probability",
    ]:
        display[column] = display[column].map(pct)
    display = display[
        [
            "date",
            "tournament",
            "match",
            "actual_score",
            "predicted_score",
            "actual_outcome",
            "predicted_outcome",
            "prediction_correct",
            "exact_score_hit",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "actual_outcome_probability",
            "exact_score_probability",
            "home_expected_goals",
            "away_expected_goals",
            "brier_score",
            "outcome_log_loss",
            "score_log_loss",
            "training_matches",
        ]
    ]
    st.dataframe(rounded_frame(display), width="stretch", hide_index=True)


def recent_results_page(bundle: DataBundle, theme: VisualTheme) -> None:
    render_hero(
        bundle,
        "Recent Results",
        "Audit recency-weighted form inputs, latest matches, and team-level trends used by the forecast.",
    )
    st.subheader("Recent Results and Model Form Inputs")
    st.markdown(
        '<p class="section-note">The model now applies stronger weight to fresh results through Elo update size and recency-weighted team form features.</p>',
        unsafe_allow_html=True,
    )
    if bundle.combined_results.empty:
        st.info("No completed-match results are available yet.")
        return

    results = bundle.combined_results.copy()
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results = results.dropna(subset=["date", "home_team", "away_team"]).sort_values("date", ascending=False)
    teams = sorted(set(results["home_team"].dropna().astype(str)).union(set(results["away_team"].dropna().astype(str))))
    profiles = ensure_recent_profile_columns(bundle.team_profiles)
    team_rows = recent_results_with_weights(team_match_frame(results)).sort_values("date", ascending=False)

    controls = st.columns([1.1, 0.7, 0.7], vertical_alignment="bottom")
    with controls[0]:
        selected_team = st.selectbox("Team focus", ["All teams"] + teams)
    with controls[1]:
        row_limit = st.slider("Rows shown", min_value=10, max_value=150, value=50, step=10)
    with controls[2]:
        st.metric("Form half-life", f"{RECENT_MATCH_HALFLIFE_DAYS} days")

    if selected_team != "All teams":
        team_profile = profiles[profiles["team"].eq(selected_team)]
        selected_rows = team_rows[team_rows["team"].eq(selected_team)]
        metric_cols = st.columns(4)
        if not team_profile.empty:
            profile_row = team_profile.iloc[0]
            metric_cols[0].metric("Weighted points / match", f"{profile_row['weighted_recent_points_per_match']:.2f}")
            metric_cols[1].metric("Weighted goal diff", f"{profile_row['weighted_recent_goal_diff']:+.2f}")
            metric_cols[2].metric("Effective recent matches", f"{profile_row['recent_result_weight']:.1f}")
        else:
            metric_cols[0].metric("Weighted points / match", "0.00")
            metric_cols[1].metric("Weighted goal diff", "+0.00")
            metric_cols[2].metric("Effective recent matches", "0.0")
        latest_team_date = selected_rows["date"].max() if not selected_rows.empty else pd.NaT
        metric_cols[3].metric("Latest match", latest_team_date.date().isoformat() if pd.notna(latest_team_date) else "n/a")
    else:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Latest match date", results["date"].max().date().isoformat())
        metric_cols[1].metric("Completed matches", len(results))
        metric_cols[2].metric("Teams in results", len(teams))
        metric_cols[3].metric("Team result rows", len(team_rows))

    chart_left, chart_right = st.columns([1.05, 0.95])
    with chart_left:
        st.plotly_chart(recent_form_ranking_chart(profiles, theme), width="stretch")
    with chart_right:
        st.plotly_chart(recent_form_scatter_chart(profiles, theme), width="stretch")

    st.plotly_chart(recent_inputs_heatmap(profiles, selected_team, theme), width="stretch")
    if selected_team != "All teams":
        st.plotly_chart(recent_team_timeline(team_rows, selected_team, theme), width="stretch")

    st.subheader("Latest Completed Matches")
    match_rows = results.copy()
    if selected_team != "All teams":
        match_rows = match_rows[match_rows["home_team"].eq(selected_team) | match_rows["away_team"].eq(selected_team)]
    match_rows = match_rows.head(row_limit).copy()
    match_rows["score"] = match_rows["home_score"].astype(str) + "-" + match_rows["away_score"].astype(str)
    match_columns = [
        "date",
        "home_team",
        "score",
        "away_team",
        "tournament",
        "city",
        "country",
        "neutral",
    ]
    st.dataframe(
        rounded_frame(match_rows[[column for column in match_columns if column in match_rows.columns]]),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Weighted Team-Perspective Result Rows")
    detail_rows = team_rows.copy()
    if selected_team != "All teams":
        detail_rows = detail_rows[detail_rows["team"].eq(selected_team)]
    detail_columns = [
        "date",
        "team",
        "opponent",
        "result",
        "goals_for",
        "goals_against",
        "goal_diff",
        "points",
        "recency_weight",
        "tournament",
    ]
    st.dataframe(
        rounded_frame(detail_rows.head(row_limit)[[column for column in detail_columns if column in detail_rows.columns]], 3),
        width="stretch",
        hide_index=True,
    )


def reports_page(bundle: DataBundle, theme: VisualTheme) -> None:
    render_hero(
        bundle,
        "2026 Report Review",
        "Inspect parsed FIFA match-report data and current-tournament team statistics feeding the model.",
    )
    st.subheader("2026 FIFA Report Summaries")
    if bundle.report_matches.empty:
        st.info("No FIFA report summaries have been parsed yet.")
        return

    matches = bundle.report_matches.sort_values(["date", "match_number"], na_position="last")
    stat_cols = [
        "match_number",
        "group",
        "date",
        "home_team",
        "home_score",
        "away_score",
        "away_team",
        "home_xg",
        "away_xg",
        "home_attempts",
        "away_attempts",
        "home_possession_pct",
        "away_possession_pct",
        "venue",
    ]
    st.dataframe(rounded_frame(matches[[column for column in stat_cols if column in matches.columns]]), width="stretch", hide_index=True)

    if not bundle.report_team_stats.empty and {"xg", "goals_for", "attempts", "team"}.issubset(bundle.report_team_stats.columns):
        team_stats = bundle.report_team_stats.copy()
        fig = px.scatter(
            team_stats,
            x="xg",
            y="goals_for",
            size="attempts",
            color="result",
            hover_name="team",
            hover_data=["opponent", "attempts_on_target"] if "attempts_on_target" in team_stats else ["opponent"],
            title="Report xG vs Goals",
            color_discrete_sequence=[theme.accent, theme.muted, theme.warning],
        )
        fig.update_layout(height=460)
        st.plotly_chart(style_plotly(fig, theme), width="stretch")

    st.subheader("Team Report Rows")
    st.dataframe(rounded_frame(bundle.report_team_stats), width="stretch", hide_index=True)


def ratings_page(bundle: DataBundle, theme: VisualTheme) -> None:
    render_hero(
        bundle,
        "Team Ratings",
        "Review Elo-style ratings, recent form, xG shrinkage inputs, and other team profile features.",
    )
    st.subheader("Team Ratings")
    if show_empty_state(bundle):
        return

    profiles = bundle.team_profiles.copy()
    query = st.text_input("Filter teams", "")
    if query:
        profiles = profiles[profiles["team"].str.contains(query, case=False, na=False)]

    top_profiles = profiles.sort_values("rating", ascending=False).head(25)
    fig = px.bar(
        top_profiles.sort_values("rating"),
        x="rating",
        y="team",
        orientation="h",
        title="Top Team Ratings",
        color="rating",
        color_continuous_scale=[theme.surface_muted, theme.accent],
    )
    fig.update_layout(height=650, yaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(style_plotly(fig, theme), width="stretch")
    st.dataframe(rounded_frame(profiles), width="stretch", hide_index=True)


def players_page(bundle: DataBundle, theme: VisualTheme) -> None:
    render_hero(
        bundle,
        "Player Scorer Profiles",
        "Explore historical scorer weights used to allocate team expected goals to player scoring estimates.",
    )
    st.subheader("Player Scorer Profiles")
    if bundle.player_profiles.empty:
        st.info("No player profiles are available yet. Refresh data to download goalscorers.csv.")
        return

    profiles = bundle.player_profiles.copy()
    teams = ["All teams"] + sorted(profiles["team"].dropna().astype(str).unique())
    selected_team = st.selectbox("Team", teams)
    if selected_team != "All teams":
        profiles = profiles[profiles["team"].eq(selected_team)]

    top_profiles = profiles.sort_values(["scoring_weight", "recent_goals_24m", "goals"], ascending=False).head(40)
    fig = px.scatter(
        top_profiles,
        x="recent_goals_24m",
        y="scoring_weight",
        size="goals",
        color="team",
        hover_name="player",
        hover_data=["last_goal_date", "penalty_goals", "team_goal_share"],
        title="Recent Goals vs Recency-Weighted Scoring Profile",
    )
    fig.update_layout(height=560)
    st.plotly_chart(style_plotly(fig, theme), width="stretch")
    st.dataframe(rounded_frame(top_profiles), width="stretch", hide_index=True)


def data_sources_page(bundle: DataBundle, theme: VisualTheme) -> None:
    render_hero(
        bundle,
        "Data Sources",
        "Review candidate and integrated public football data sources for automation fit and model value.",
    )
    st.subheader("Football Data Source Review")
    st.markdown(
        '<p class="section-note">Sources from the linked Sports Data Campus article are classified by automation fit and usefulness for this national-team model.</p>',
        unsafe_allow_html=True,
    )
    catalog = source_catalog_frame()
    status_order = ["Integrated", "Candidate", "Needs credentials", "Manual review", "Reference", "Low priority"]
    selected_statuses = st.multiselect("Status", status_order, default=status_order)
    if selected_statuses:
        catalog = catalog[catalog["status"].isin(selected_statuses)]

    status_counts = catalog.groupby("status", as_index=False).size()
    if not status_counts.empty:
        fig = px.bar(
            status_counts,
            x="status",
            y="size",
            color="status",
            color_discrete_sequence=[theme.accent, theme.warning, theme.muted, "#6b8f85", "#8aa097", "#4d5d58"],
            title="Source Automation Fit",
        )
        fig.update_layout(height=340, showlegend=False, yaxis_title="Sources", xaxis_title="")
        st.plotly_chart(style_plotly(fig, theme), width="stretch")

    st.dataframe(catalog, width="stretch", hide_index=True)


def data_health_page(bundle: DataBundle) -> None:
    render_hero(
        bundle,
        "Data Health",
        "Check source freshness, parsed-row counts, and local cache status before trusting model outputs.",
    )
    st.subheader("Data Health")

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f'<div class="source-card"><strong>FIFA reports</strong><br><span><a href="{FIFA_HUB_URL}">Training Centre PDF hub</a></span></div>',
        unsafe_allow_html=True,
    )
    col2.markdown(
        f'<div class="source-card"><strong>Historical results</strong><br><span><a href="{HISTORICAL_RESULTS_URL}">international_results results.csv</a></span></div>',
        unsafe_allow_html=True,
    )
    col3.markdown(
        f'<div class="source-card"><strong>Goalscorers</strong><br><span><a href="{GOALSCORERS_URL}">international_results goalscorers.csv</a></span></div>',
        unsafe_allow_html=True,
    )
    col4.markdown(
        f'<div class="source-card"><strong>StatsBomb Open Data</strong><br><span><a href="{STATSBOMB_COMPETITIONS_URL}">competitions and event JSON</a></span></div>',
        unsafe_allow_html=True,
    )

    metadata = bundle.metadata or {}
    if metadata:
        cols = st.columns(6)
        cols[0].metric("Report links", metadata.get("report_links_found", 0))
        cols[1].metric("Reports parsed", metadata.get("report_matches_parsed", 0))
        cols[2].metric("Historical rows", metadata.get("historical_results_rows", 0))
        cols[3].metric("Player profiles", metadata.get("player_profiles_rows", bundle_frame_len(bundle, "player_profiles")))
        cols[4].metric("StatsBomb matches", metadata.get("statsbomb_matches_loaded", bundle_frame_len(bundle, "statsbomb_team_stats") // 2))
        cols[5].metric("Team profiles", metadata.get("team_profiles_rows", bundle_frame_len(bundle, "team_profiles")))
        st.json(metadata)
    else:
        st.info("No metadata is available yet.")


def main() -> None:
    theme_mode = refresh_controls()
    theme = get_visual_theme(theme_mode)
    apply_theme(theme)
    bundle = load_data(DATA_CACHE_VERSION)

    st.sidebar.header("View")
    page = st.sidebar.selectbox(
        "Dashboard page",
        [
            "Matchup Predictor",
            "Upcoming Matchups",
            "Model Performance",
            "Recent Results",
            "2026 Reports",
            "Team Ratings",
            "Player Profiles",
            "Data Sources",
            "Data Health",
        ],
    )

    if page == "Matchup Predictor":
        matchup_page(bundle, theme)
    elif page == "Upcoming Matchups":
        upcoming_matchups_page(bundle, theme)
    elif page == "Model Performance":
        model_performance_page(bundle, theme)
    elif page == "Recent Results":
        recent_results_page(bundle, theme)
    elif page == "2026 Reports":
        reports_page(bundle, theme)
    elif page == "Team Ratings":
        ratings_page(bundle, theme)
    elif page == "Player Profiles":
        players_page(bundle, theme)
    elif page == "Data Sources":
        data_sources_page(bundle, theme)
    else:
        data_health_page(bundle)


if __name__ == "__main__":
    main()

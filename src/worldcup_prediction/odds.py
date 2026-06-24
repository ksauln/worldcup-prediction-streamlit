from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from worldcup_prediction.config import (
    BOOKMAKER_ODDS_METADATA_PATH,
    BOOKMAKER_ODDS_PATH,
    DEFAULT_ODDS_BOOKMAKERS,
    DEFAULT_PUBLIC_ODDS_DATE_RANGE,
    DEFAULT_ODDS_REGIONS,
    DEFAULT_ODDS_SPORT_KEY,
    ESPN_WORLD_CUP_SCOREBOARD_URL,
    HTTP_TIMEOUT_SECONDS,
    ODDS_API_BASE_URL,
    ODDS_API_KEY_ENV,
    USER_AGENT,
)
from worldcup_prediction.teams import canonical_team_name


ODDS_COLUMNS = [
    "fetched_at",
    "event_id",
    "sport_key",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker_key",
    "bookmaker_title",
    "bookmaker_last_update",
    "market",
    "outcome_name",
    "outcome_type",
    "point",
    "price_american",
    "price_decimal",
    "implied_probability",
    "bookmaker_overround",
    "no_vig_probability",
    "link",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def empty_odds_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ODDS_COLUMNS)


def american_to_decimal(price: Any) -> float:
    price_value = float(price)
    if price_value > 0:
        return 1.0 + price_value / 100.0
    if price_value < 0:
        return 1.0 + 100.0 / abs(price_value)
    return 1.0


def american_to_implied_probability(price: Any) -> float:
    return 1.0 / american_to_decimal(price)


def format_american_odds(price: Any) -> str:
    price_value = int(round(float(price)))
    return f"+{price_value}" if price_value > 0 else str(price_value)


def _manual_bookmaker_key(bookmaker_title: str) -> str:
    slug = _slug_key(bookmaker_title)
    return f"manual_{slug or 'bookmaker'}"


def _slug_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _parse_american_price(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    if not re.fullmatch(r"[+-]?\d+(?:\.0+)?", text):
        raise ValueError(f"Invalid American odds value: {value!r}")
    price = float(text)
    if price == 0:
        raise ValueError("American odds cannot be 0.")
    return price


def _parse_point(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    text = str(value).strip().lower()
    if not text:
        return None
    match = re.search(r"[+-]?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _outcome_type(home_team: str, away_team: str, outcome_name: str, market: str = "h2h") -> str:
    market_key = str(market or "").lower()
    outcome_text = str(outcome_name or "").strip().lower()
    if market_key == "totals":
        if outcome_text.startswith("over"):
            return "over"
        if outcome_text.startswith("under"):
            return "under"
    outcome = canonical_team_name(outcome_name)
    if market_key == "spreads":
        if outcome == canonical_team_name(home_team):
            return "home_spread"
        if outcome == canonical_team_name(away_team):
            return "away_spread"
    if outcome == canonical_team_name(home_team):
        return "home_win"
    if outcome == canonical_team_name(away_team):
        return "away_win"
    if outcome_text == "draw":
        return "draw"
    return "other"


def flatten_odds_events(events: list[dict[str, Any]], fetched_at: str | None = None) -> pd.DataFrame:
    fetched_at = fetched_at or _utc_now_iso()
    rows: list[dict[str, Any]] = []

    for event in events:
        home_team = canonical_team_name(event.get("home_team") or "")
        away_team = canonical_team_name(event.get("away_team") or "")
        for bookmaker in event.get("bookmakers") or []:
            for market in bookmaker.get("markets") or []:
                market_key = market.get("key", "")
                for outcome in market.get("outcomes") or []:
                    price = outcome.get("price")
                    if price is None:
                        continue
                    outcome_name = canonical_team_name(outcome.get("name") or "")
                    try:
                        decimal_price = american_to_decimal(price)
                        implied_probability = american_to_implied_probability(price)
                    except (TypeError, ValueError, ZeroDivisionError):
                        continue
                    rows.append(
                        {
                            "fetched_at": fetched_at,
                            "event_id": event.get("id", ""),
                            "sport_key": event.get("sport_key", ""),
                            "commence_time": event.get("commence_time", ""),
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker_key": bookmaker.get("key", ""),
                            "bookmaker_title": bookmaker.get("title", bookmaker.get("key", "")),
                            "bookmaker_last_update": bookmaker.get("last_update", ""),
                            "market": market_key,
                            "outcome_name": outcome_name,
                            "outcome_type": _outcome_type(home_team, away_team, outcome_name, market_key),
                            "point": _parse_point(outcome.get("point")),
                            "price_american": float(price),
                            "price_decimal": decimal_price,
                            "implied_probability": implied_probability,
                            "link": outcome.get("link") or market.get("link") or bookmaker.get("link") or "",
                        }
                    )

    if not rows:
        return empty_odds_frame()

    odds = pd.DataFrame(rows)
    group_columns = ["event_id", "bookmaker_key", "market"]
    overround = odds.groupby(group_columns)["implied_probability"].transform("sum")
    odds["bookmaker_overround"] = overround
    odds["no_vig_probability"] = odds["implied_probability"] / overround.replace(0, pd.NA)
    return odds[ODDS_COLUMNS].reset_index(drop=True)


def _link_href(link: Any) -> str:
    if isinstance(link, dict):
        return str(link.get("href") or "")
    return ""


def _espn_team_name(competition: dict[str, Any], home_away: str) -> str:
    for competitor in competition.get("competitors") or []:
        if not isinstance(competitor, dict):
            continue
        if str(competitor.get("homeAway") or "").lower() != home_away:
            continue
        team = competitor.get("team") or {}
        if not isinstance(team, dict):
            team = {}
        return canonical_team_name(
            team.get("displayName")
            or team.get("shortDisplayName")
            or team.get("name")
            or team.get("location")
            or ""
        )
    return ""


def _espn_moneyline_price(moneyline: dict[str, Any], side: str) -> tuple[float | None, str]:
    if not isinstance(moneyline, dict):
        return None, ""
    side_rows = moneyline.get(side) or {}
    if not isinstance(side_rows, dict):
        return None, ""
    close = side_rows.get("close") or {}
    if not isinstance(close, dict):
        close = {}
    open_price = side_rows.get("open") or {}
    if not isinstance(open_price, dict):
        open_price = {}
    try:
        price = _parse_american_price(close.get("odds") if close.get("odds") is not None else open_price.get("odds"))
    except (TypeError, ValueError):
        price = None
    link = _link_href(close.get("link")) or _link_href(open_price.get("link"))
    return price, link


def _espn_market_price(market: dict[str, Any], side: str) -> tuple[float | None, float | None, str]:
    if not isinstance(market, dict):
        return None, None, ""
    side_rows = market.get(side) or {}
    if not isinstance(side_rows, dict):
        return None, None, ""
    close = side_rows.get("close") or {}
    if not isinstance(close, dict):
        close = {}
    open_price = side_rows.get("open") or {}
    if not isinstance(open_price, dict):
        open_price = {}
    try:
        price = _parse_american_price(close.get("odds") if close.get("odds") is not None else open_price.get("odds"))
    except (TypeError, ValueError):
        price = None
    point = _parse_point(close.get("line") if close.get("line") is not None else open_price.get("line"))
    link = _link_href(close.get("link")) or _link_href(open_price.get("link"))
    return price, point, link


def _append_espn_row(
    rows: list[dict[str, Any]],
    *,
    fetched_at: str,
    event_id: str,
    sport_key: str,
    commence_time: str,
    home_team: str,
    away_team: str,
    bookmaker_key: str,
    bookmaker_title: str,
    market: str,
    outcome_name: str,
    outcome_type: str,
    price: float,
    point: float | None = None,
    link: str = "",
) -> None:
    rows.append(
        {
            "fetched_at": fetched_at,
            "event_id": event_id,
            "sport_key": sport_key,
            "commence_time": commence_time,
            "home_team": home_team,
            "away_team": away_team,
            "bookmaker_key": bookmaker_key,
            "bookmaker_title": bookmaker_title,
            "bookmaker_last_update": fetched_at,
            "market": market,
            "outcome_name": outcome_name,
            "outcome_type": outcome_type,
            "point": point,
            "price_american": price,
            "price_decimal": american_to_decimal(price),
            "implied_probability": american_to_implied_probability(price),
            "link": link,
        }
    )


def flatten_espn_scoreboard_odds(payload: dict[str, Any], fetched_at: str | None = None) -> pd.DataFrame:
    fetched_at = fetched_at or _utc_now_iso()
    league_slug = ""
    leagues = payload.get("leagues") if isinstance(payload, dict) else []
    if leagues:
        league_slug = str((leagues[0] or {}).get("slug") or "")
    sport_key = f"espn:{league_slug or 'fifa.world'}"
    rows: list[dict[str, Any]] = []

    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        for competition in event.get("competitions") or []:
            if not isinstance(competition, dict):
                continue
            home_team = _espn_team_name(competition, "home")
            away_team = _espn_team_name(competition, "away")
            if not home_team or not away_team:
                continue
            event_id = str(competition.get("id") or event.get("id") or "")
            commence_time = competition.get("date") or event.get("date") or ""

            for odds_entry in competition.get("odds") or []:
                if not isinstance(odds_entry, dict):
                    continue
                provider = odds_entry.get("provider") or {}
                bookmaker_title = str(provider.get("displayName") or provider.get("name") or "ESPN Public Odds")
                bookmaker_key = _slug_key(bookmaker_title) or str(provider.get("id") or "espn_public")
                moneyline = odds_entry.get("moneyline") or {}
                if not isinstance(moneyline, dict):
                    moneyline = {}
                default_link = _link_href(odds_entry.get("link"))
                outcomes = [
                    ("home", home_team, "home_win"),
                    ("draw", "Draw", "draw"),
                    ("away", away_team, "away_win"),
                ]

                for side, outcome_name, outcome_type in outcomes:
                    price, link = _espn_moneyline_price(moneyline, side)
                    if price is None and side == "draw":
                        draw_odds = odds_entry.get("drawOdds") or {}
                        if not isinstance(draw_odds, dict):
                            draw_odds = {}
                        try:
                            price = _parse_american_price(draw_odds.get("moneyLine"))
                        except (TypeError, ValueError):
                            price = None
                        link = _link_href(draw_odds.get("link"))
                    if price is None:
                        continue
                    _append_espn_row(
                        rows,
                        fetched_at=fetched_at,
                        event_id=event_id,
                        sport_key=sport_key,
                        commence_time=commence_time,
                        home_team=home_team,
                        away_team=away_team,
                        bookmaker_key=bookmaker_key,
                        bookmaker_title=bookmaker_title,
                        market="h2h",
                        outcome_name=outcome_name,
                        outcome_type=outcome_type,
                        price=price,
                        link=link or default_link,
                    )

                point_spread = odds_entry.get("pointSpread") or {}
                for side, outcome_name, outcome_type in [
                    ("home", home_team, "home_spread"),
                    ("away", away_team, "away_spread"),
                ]:
                    price, point, link = _espn_market_price(point_spread, side)
                    if price is None:
                        continue
                    _append_espn_row(
                        rows,
                        fetched_at=fetched_at,
                        event_id=event_id,
                        sport_key=sport_key,
                        commence_time=commence_time,
                        home_team=home_team,
                        away_team=away_team,
                        bookmaker_key=bookmaker_key,
                        bookmaker_title=bookmaker_title,
                        market="spreads",
                        outcome_name=outcome_name,
                        outcome_type=outcome_type,
                        point=point,
                        price=price,
                        link=link or default_link,
                    )

                total = odds_entry.get("total") or {}
                fallback_total = _parse_point(odds_entry.get("overUnder"))
                for side, outcome_name, outcome_type in [
                    ("over", "Over", "over"),
                    ("under", "Under", "under"),
                ]:
                    price, point, link = _espn_market_price(total, side)
                    if price is None:
                        continue
                    _append_espn_row(
                        rows,
                        fetched_at=fetched_at,
                        event_id=event_id,
                        sport_key=sport_key,
                        commence_time=commence_time,
                        home_team=home_team,
                        away_team=away_team,
                        bookmaker_key=bookmaker_key,
                        bookmaker_title=bookmaker_title,
                        market="totals",
                        outcome_name=outcome_name,
                        outcome_type=outcome_type,
                        point=point if point is not None else fallback_total,
                        price=price,
                        link=link or default_link,
                    )

    if not rows:
        return empty_odds_frame()

    odds = pd.DataFrame(rows)
    group_columns = ["event_id", "bookmaker_key", "market"]
    overround = odds.groupby(group_columns)["implied_probability"].transform("sum")
    odds["bookmaker_overround"] = overround
    odds["no_vig_probability"] = odds["implied_probability"] / overround.replace(0, pd.NA)
    return odds[ODDS_COLUMNS].reset_index(drop=True)


def manual_moneyline_odds(
    team_a: str,
    team_b: str,
    team_a_price: Any,
    draw_price: Any = None,
    team_b_price: Any = None,
    bookmaker_title: str = "Manual",
    commence_time: str = "",
    fetched_at: str | None = None,
) -> pd.DataFrame:
    fetched_at = fetched_at or _utc_now_iso()
    home_team = canonical_team_name(team_a)
    away_team = canonical_team_name(team_b)
    bookmaker_key = _manual_bookmaker_key(bookmaker_title)
    event_id = "manual:" + "|".join([commence_time, home_team, away_team, bookmaker_key])
    outcomes = [
        (home_team, "home_win", _parse_american_price(team_a_price)),
        ("Draw", "draw", _parse_american_price(draw_price)),
        (away_team, "away_win", _parse_american_price(team_b_price)),
    ]
    rows: list[dict[str, Any]] = []
    for outcome_name, outcome_type, price in outcomes:
        if price is None:
            continue
        rows.append(
            {
                "fetched_at": fetched_at,
                "event_id": event_id,
                "sport_key": "manual",
                "commence_time": commence_time,
                "home_team": home_team,
                "away_team": away_team,
                "bookmaker_key": bookmaker_key,
                "bookmaker_title": bookmaker_title.strip() or "Manual",
                "bookmaker_last_update": fetched_at,
                "market": "h2h",
                "outcome_name": outcome_name,
                "outcome_type": outcome_type,
                "point": None,
                "price_american": price,
                "price_decimal": american_to_decimal(price),
                "implied_probability": american_to_implied_probability(price),
                "link": "",
            }
        )

    if len(rows) < 2:
        raise ValueError("Enter odds for at least two outcomes.")
    odds = pd.DataFrame(rows)
    overround = odds["implied_probability"].sum()
    odds["bookmaker_overround"] = overround
    odds["no_vig_probability"] = odds["implied_probability"] / overround
    return odds[ODDS_COLUMNS].reset_index(drop=True)


def save_manual_odds(
    manual_rows: pd.DataFrame,
    odds_path: Path = BOOKMAKER_ODDS_PATH,
    metadata_path: Path = BOOKMAKER_ODDS_METADATA_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if manual_rows is None or manual_rows.empty:
        raise ValueError("No manual odds rows to save.")
    existing_rows, existing_metadata = load_cached_odds(odds_path=odds_path, metadata_path=metadata_path)
    manual_key_columns = ["event_id", "bookmaker_key", "market"]
    replacement_keys = set(map(tuple, manual_rows[manual_key_columns].astype(str).to_numpy()))

    if existing_rows.empty:
        combined = manual_rows.copy()
    else:
        existing_keys = existing_rows[manual_key_columns].astype(str).apply(tuple, axis=1)
        combined = pd.concat([existing_rows[~existing_keys.isin(replacement_keys)], manual_rows], ignore_index=True)

    combined = combined[ODDS_COLUMNS].reset_index(drop=True)
    odds_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(odds_path, index=False)
    metadata = {
        **existing_metadata,
        "refreshed_at": _utc_now_iso(),
        "source": "Manual odds entry",
        "odds_rows": len(combined),
        "manual_odds_rows": len(manual_rows),
    }
    _write_json(metadata_path, metadata)
    return combined, metadata


def refresh_bookmaker_odds(
    api_key: str | None = None,
    sport_key: str = DEFAULT_ODDS_SPORT_KEY,
    bookmakers: list[str] | None = None,
    regions: str = DEFAULT_ODDS_REGIONS,
    markets: str = "h2h,spreads,totals",
    output_path: Path = BOOKMAKER_ODDS_PATH,
    metadata_path: Path = BOOKMAKER_ODDS_METADATA_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    api_key = api_key or os.getenv(ODDS_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"Set {ODDS_API_KEY_ENV} before refreshing bookmaker odds.")

    endpoint = f"{ODDS_API_BASE_URL}/v4/sports/{sport_key}/odds/"
    selected_bookmakers = bookmakers if bookmakers is not None else DEFAULT_ODDS_BOOKMAKERS
    params: dict[str, str] = {
        "apiKey": api_key,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
        "includeLinks": "true",
    }
    if selected_bookmakers:
        params["bookmakers"] = ",".join(selected_bookmakers)
    else:
        params["regions"] = regions

    response = requests.get(
        endpoint,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:300].strip()
        raise RuntimeError(f"Odds refresh failed with HTTP {response.status_code}: {body}") from exc

    events = response.json()
    if not isinstance(events, list):
        raise ValueError("Unexpected odds payload: expected a list of events.")

    fetched_at = _utc_now_iso()
    odds = flatten_odds_events(events, fetched_at=fetched_at)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    odds.to_csv(output_path, index=False)

    metadata = {
        "refreshed_at": fetched_at,
        "source": "The Odds API",
        "sport_key": sport_key,
        "markets": markets,
        "regions": regions if not selected_bookmakers else "",
        "bookmakers": selected_bookmakers,
        "events_returned": len(events),
        "odds_rows": len(odds),
        "requests_remaining": response.headers.get("x-requests-remaining", ""),
        "requests_used": response.headers.get("x-requests-used", ""),
        "requests_last": response.headers.get("x-requests-last", ""),
    }
    _write_json(metadata_path, metadata)
    return odds, metadata


def refresh_public_bookmaker_odds(
    date_range: str = DEFAULT_PUBLIC_ODDS_DATE_RANGE,
    output_path: Path = BOOKMAKER_ODDS_PATH,
    metadata_path: Path = BOOKMAKER_ODDS_METADATA_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    date_range = str(date_range or DEFAULT_PUBLIC_ODDS_DATE_RANGE).strip()
    params = {"dates": date_range, "limit": "500"}
    response = requests.get(
        ESPN_WORLD_CUP_SCOREBOARD_URL,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:300].strip()
        raise RuntimeError(f"Public odds scrape failed with HTTP {response.status_code}: {body}") from exc

    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Unexpected ESPN odds payload: expected a JSON object.")

    fetched_at = _utc_now_iso()
    odds = flatten_espn_scoreboard_odds(payload, fetched_at=fetched_at)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    odds.to_csv(output_path, index=False)

    bookmakers = sorted(odds["bookmaker_key"].dropna().astype(str).unique().tolist()) if not odds.empty else []
    metadata = {
        "refreshed_at": fetched_at,
        "source": "ESPN public scoreboard",
        "source_url": ESPN_WORLD_CUP_SCOREBOARD_URL,
        "date_range": date_range,
        "markets": "h2h,spreads,totals",
        "bookmakers": bookmakers,
        "events_returned": len(payload.get("events") or []),
        "odds_rows": len(odds),
    }
    _write_json(metadata_path, metadata)
    return odds, metadata


def load_cached_odds(
    odds_path: Path = BOOKMAKER_ODDS_PATH,
    metadata_path: Path = BOOKMAKER_ODDS_METADATA_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not odds_path.exists():
        return empty_odds_frame(), _read_json(metadata_path)
    try:
        odds = pd.read_csv(odds_path)
    except pd.errors.EmptyDataError:
        odds = empty_odds_frame()
    for column in ODDS_COLUMNS:
        if column not in odds:
            odds[column] = pd.NA
    return odds[ODDS_COLUMNS], _read_json(metadata_path)


def _event_key(row: pd.Series) -> str:
    event_id = str(row.get("event_id") or "").strip()
    if event_id:
        return event_id
    return "|".join(
        [
            str(row.get("commence_time") or ""),
            canonical_team_name(str(row.get("home_team") or "")),
            canonical_team_name(str(row.get("away_team") or "")),
        ]
    )


def event_summary_frame(odds_rows: pd.DataFrame) -> pd.DataFrame:
    if odds_rows is None or odds_rows.empty:
        return pd.DataFrame(
            columns=[
                "event_key",
                "event_id",
                "commence_time",
                "home_team",
                "away_team",
                "bookmakers",
                "odds_rows",
            ]
        )

    rows = odds_rows.copy()
    rows["event_key"] = rows.apply(_event_key, axis=1)
    summary = (
        rows.groupby("event_key", as_index=False)
        .agg(
            event_id=("event_id", "first"),
            commence_time=("commence_time", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            bookmakers=("bookmaker_key", "nunique"),
            odds_rows=("event_key", "size"),
        )
        .sort_values(["commence_time", "home_team", "away_team"])
        .reset_index(drop=True)
    )
    return summary


def odds_for_event(odds_rows: pd.DataFrame, event_key: str) -> pd.DataFrame:
    if odds_rows is None or odds_rows.empty:
        return empty_odds_frame()
    rows = odds_rows.copy()
    rows["event_key"] = rows.apply(_event_key, axis=1)
    selected = rows[rows["event_key"].eq(event_key)].drop(columns=["event_key"])
    return selected.reset_index(drop=True)


def matching_event_odds(odds_rows: pd.DataFrame, team_a: str, team_b: str) -> pd.DataFrame:
    if odds_rows is None or odds_rows.empty:
        return empty_odds_frame()

    rows = odds_rows.copy()
    team_a_name = canonical_team_name(team_a)
    team_b_name = canonical_team_name(team_b)
    home = rows["home_team"].map(canonical_team_name)
    away = rows["away_team"].map(canonical_team_name)
    mask = ((home.eq(team_a_name) & away.eq(team_b_name)) | (home.eq(team_b_name) & away.eq(team_a_name)))
    sort_columns = [column for column in ("commence_time", "bookmaker_title", "outcome_type") if column in rows]
    matched = rows[mask]
    if sort_columns:
        matched = matched.sort_values(sort_columns)
    return matched.reset_index(drop=True)


def attach_model_edges(odds_rows: pd.DataFrame, prediction: dict[str, Any]) -> pd.DataFrame:
    if odds_rows is None or odds_rows.empty:
        output = empty_odds_frame()
        for column in (
            "model_probability",
            "push_probability",
            "edge_vs_implied",
            "edge_vs_no_vig",
            "expected_value",
            "kelly_fraction",
        ):
            output[column] = pd.Series(dtype=float)
        return output

    rows = odds_rows.copy()
    if "point" not in rows:
        rows["point"] = pd.NA
    team_probabilities = {
        canonical_team_name(prediction["team_a"]): float(prediction["team_a_win_probability"]),
        canonical_team_name(prediction["team_b"]): float(prediction["team_b_win_probability"]),
    }
    draw_probability = float(prediction["draw_probability"])
    score_matrix = prediction.get("score_matrix")

    def scoreline_market_probability(row: pd.Series) -> tuple[float, float]:
        if not isinstance(score_matrix, pd.DataFrame):
            return float("nan"), 0.0
        point = _parse_point(row.get("point"))
        if point is None:
            return float("nan"), 0.0

        values = score_matrix.to_numpy()
        home_team = canonical_team_name(str(row.get("home_team", "")))
        away_team = canonical_team_name(str(row.get("away_team", "")))
        team_a = canonical_team_name(prediction["team_a"])
        team_b = canonical_team_name(prediction["team_b"])
        if home_team == team_a and away_team == team_b:
            home_index = 0
        elif home_team == team_b and away_team == team_a:
            home_index = 1
        else:
            return float("nan"), 0.0

        outcome_type = str(row.get("outcome_type") or "")
        win_probability = 0.0
        push_probability = 0.0
        epsilon = 1e-12
        for team_a_goals in range(values.shape[0]):
            for team_b_goals in range(values.shape[1]):
                probability = float(values[team_a_goals, team_b_goals])
                if home_index == 0:
                    home_goals, away_goals = team_a_goals, team_b_goals
                else:
                    home_goals, away_goals = team_b_goals, team_a_goals

                if outcome_type == "over":
                    margin = team_a_goals + team_b_goals - point
                elif outcome_type == "under":
                    margin = point - (team_a_goals + team_b_goals)
                elif outcome_type == "home_spread":
                    margin = home_goals + point - away_goals
                elif outcome_type == "away_spread":
                    margin = away_goals + point - home_goals
                else:
                    return float("nan"), 0.0

                if margin > epsilon:
                    win_probability += probability
                elif abs(margin) <= epsilon:
                    push_probability += probability
        return win_probability, push_probability

    def model_probability(row: pd.Series) -> float:
        if row.get("outcome_type") == "draw":
            return draw_probability
        if row.get("market") == "h2h":
            return team_probabilities.get(canonical_team_name(str(row.get("outcome_name", ""))), float("nan"))
        return scoreline_market_probability(row)[0]

    rows["model_probability"] = rows.apply(model_probability, axis=1)
    rows["push_probability"] = rows.apply(lambda row: scoreline_market_probability(row)[1], axis=1)
    rows["edge_vs_implied"] = rows["model_probability"] - pd.to_numeric(rows["implied_probability"], errors="coerce")
    rows["edge_vs_no_vig"] = rows["model_probability"] - pd.to_numeric(rows["no_vig_probability"], errors="coerce")
    decimal_price = pd.to_numeric(rows["price_decimal"], errors="coerce")
    push_probability = pd.to_numeric(rows["push_probability"], errors="coerce").fillna(0.0)
    rows["expected_value"] = rows["model_probability"] * decimal_price + push_probability - 1.0
    rows["kelly_fraction"] = (rows["expected_value"] / (decimal_price - 1.0)).clip(lower=0.0)
    return rows.sort_values("expected_value", ascending=False).reset_index(drop=True)


def best_odds_by_outcome(edge_rows: pd.DataFrame) -> pd.DataFrame:
    if edge_rows is None or edge_rows.empty:
        return pd.DataFrame()
    rows = edge_rows.copy()
    rows["price_decimal"] = pd.to_numeric(rows["price_decimal"], errors="coerce")
    return (
        rows.sort_values(["market", "point", "outcome_name", "price_decimal"], ascending=[True, True, True, False])
        .groupby(["market", "point", "outcome_name"], as_index=False, dropna=False)
        .head(1)
        .sort_values("expected_value", ascending=False)
        .reset_index(drop=True)
    )

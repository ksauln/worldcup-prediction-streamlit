from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from worldcup_prediction.config import (
    DEFAULT_PUBLIC_ODDS_DATE_RANGE,
    ESPN_WORLD_CUP_SCOREBOARD_URL,
    HTTP_TIMEOUT_SECONDS,
    USER_AGENT,
)
from worldcup_prediction.teams import canonical_team_name

FIXTURE_COLUMNS = [
    "event_id",
    "commence_time",
    "home_team",
    "away_team",
    "venue",
    "status",
    "completed",
    "source",
]


def empty_fixture_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=FIXTURE_COLUMNS)


def _team_name_from_competitor(competitor: dict[str, Any]) -> str:
    team = competitor.get("team") or {}
    return canonical_team_name(
        team.get("displayName")
        or team.get("shortDisplayName")
        or team.get("name")
        or competitor.get("displayName")
        or competitor.get("name")
        or ""
    )


def _team_by_home_away(competition: dict[str, Any], home_away: str) -> str:
    for competitor in competition.get("competitors", []) or []:
        if str(competitor.get("homeAway", "")).lower() == home_away:
            return _team_name_from_competitor(competitor)
    return ""


def _event_status(event: dict[str, Any], competition: dict[str, Any]) -> tuple[str, bool]:
    status = competition.get("status") or event.get("status") or {}
    status_type = status.get("type") or {}
    label = status_type.get("description") or status_type.get("detail") or status_type.get("name") or ""
    completed = bool(status_type.get("completed", False))
    return str(label), completed


def _event_venue(event: dict[str, Any], competition: dict[str, Any]) -> str:
    venue = competition.get("venue") or event.get("venue") or {}
    return str(venue.get("fullName") or venue.get("displayName") or venue.get("name") or "")


def flatten_espn_scoreboard_fixtures(payload: dict[str, Any], now: str | pd.Timestamp | None = None) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise ValueError("Unexpected ESPN schedule payload: expected a JSON object.")

    rows: list[dict[str, Any]] = []
    for event in payload.get("events", []) or []:
        competitions = event.get("competitions", []) or [{}]
        for competition in competitions:
            home_team = _team_by_home_away(competition, "home")
            away_team = _team_by_home_away(competition, "away")
            commence_time = competition.get("date") or event.get("date")
            if not home_team or not away_team or not commence_time:
                continue
            status, completed = _event_status(event, competition)
            event_id = event.get("id") or competition.get("id") or f"{commence_time}:{home_team}:{away_team}"
            rows.append(
                {
                    "event_id": str(event_id),
                    "commence_time": commence_time,
                    "home_team": home_team,
                    "away_team": away_team,
                    "venue": _event_venue(event, competition),
                    "status": status,
                    "completed": completed,
                    "source": "ESPN",
                }
            )

    fixtures = pd.DataFrame(rows, columns=FIXTURE_COLUMNS)
    if fixtures.empty:
        return empty_fixture_frame()

    fixtures["commence_time"] = pd.to_datetime(fixtures["commence_time"], utc=True, errors="coerce")
    fixtures = fixtures.dropna(subset=["commence_time", "home_team", "away_team"])
    if now is None:
        reference_time = pd.Timestamp.now(tz="UTC")
    else:
        reference_time = pd.to_datetime(now, utc=True, errors="coerce")
        if pd.isna(reference_time):
            reference_time = pd.Timestamp.now(tz="UTC")

    fixtures["completed"] = fixtures["completed"].astype(bool)
    fixtures = fixtures[(fixtures["commence_time"].ge(reference_time)) & ~fixtures["completed"]]
    fixtures = fixtures.sort_values(["commence_time", "home_team", "away_team"]).drop_duplicates("event_id")
    return fixtures[FIXTURE_COLUMNS].reset_index(drop=True)


def fetch_upcoming_fixtures(
    date_range: str = DEFAULT_PUBLIC_ODDS_DATE_RANGE,
    now: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    response = requests.get(
        ESPN_WORLD_CUP_SCOREBOARD_URL,
        params={"dates": date_range, "limit": "500"},
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    return flatten_espn_scoreboard_fixtures(payload, now=now)

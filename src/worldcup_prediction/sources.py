from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from worldcup_prediction.config import (
    FIFA_HUB_URL,
    GOALSCORERS_PATH,
    GOALSCORERS_URL,
    HISTORICAL_RESULTS_PATH,
    HISTORICAL_RESULTS_URL,
    HTTP_TIMEOUT_SECONDS,
    REPORT_PDF_DIR,
    STATSBOMB_COMPETITIONS_PATH,
    STATSBOMB_COMPETITIONS_URL,
    STATSBOMB_EVENTS_BASE_URL,
    STATSBOMB_EVENTS_DIR,
    STATSBOMB_MATCHES_BASE_URL,
    STATSBOMB_MATCHES_DIR,
    USER_AGENT,
)
from worldcup_prediction.teams import canonical_team_name


DEFAULT_SOURCE_CACHE_MAX_AGE_HOURS = 6.0


@dataclass(frozen=True)
class ReportLink:
    label: str
    url: str
    match_number: int | None


@dataclass(frozen=True)
class DownloadedReport:
    link: ReportLink
    path: Path
    downloaded: bool


def clean_space(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def extract_match_number(value: str) -> int | None:
    decoded = unquote(value)
    match = re.search(r"\bPMSR[-_\s]*M(?P<number>\d{1,3})\b", decoded, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\bMatch\s+(?P<number>\d{1,3})\b", decoded, flags=re.IGNORECASE)
    return int(match.group("number")) if match else None


def extract_report_links(html: str, base_url: str = FIFA_HUB_URL) -> list[ReportLink]:
    """Extract completed-match PDF links from the FIFA report hub HTML."""
    soup = BeautifulSoup(html, "html.parser")
    links: list[ReportLink] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if ".pdf" not in href.lower():
            continue

        url = urljoin(base_url, href)
        if url in seen:
            continue

        seen.add(url)
        filename = Path(unquote(urlparse(url).path)).name
        label = clean_space(anchor.get_text(" ", strip=True)) or filename
        links.append(ReportLink(label=label, url=url, match_number=extract_match_number(url)))

    return sorted(links, key=lambda item: (item.match_number is None, item.match_number or 9999, item.url))


def fetch_text(url: str) -> str:
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.text


def safe_filename_from_url(url: str) -> str:
    filename = Path(unquote(urlparse(url).path)).name or "report.pdf"
    filename = re.sub(r"[^A-Za-z0-9._ -]+", "_", filename)
    filename = re.sub(r"\s+", "_", filename).strip("._ ")
    return filename or "report.pdf"


def download_url(url: str, destination: Path, force: bool = False) -> bool:
    """Download a URL to disk. Returns True when a network download happened."""
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return False

    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return True


def load_json_url(url: str, destination: Path, force: bool = False) -> object:
    download_url(url, destination, force=force)
    return json.loads(destination.read_text(encoding="utf-8"))


def download_report_pdfs(
    links: list[ReportLink],
    output_dir: Path = REPORT_PDF_DIR,
    force: bool = False,
    max_reports: int | None = None,
    pause_seconds: float = 0.15,
) -> list[DownloadedReport]:
    selected_links = links[:max_reports] if max_reports is not None else links
    downloaded: list[DownloadedReport] = []

    for link in selected_links:
        prefix = f"M{link.match_number:03d}_" if link.match_number is not None else ""
        destination = output_dir / f"{prefix}{safe_filename_from_url(link.url)}"
        was_downloaded = download_url(link.url, destination, force=force)
        downloaded.append(DownloadedReport(link=link, path=destination, downloaded=was_downloaded))
        if was_downloaded and pause_seconds > 0:
            time.sleep(pause_seconds)

    return downloaded


def cache_file_is_stale(
    cache_path: Path,
    max_age_hours: float | None = DEFAULT_SOURCE_CACHE_MAX_AGE_HOURS,
    now: datetime | None = None,
) -> bool:
    if not cache_path.exists():
        return True
    if max_age_hours is None:
        return False
    reference = now or datetime.now(timezone.utc)
    age_seconds = max(reference.timestamp() - cache_path.stat().st_mtime, 0.0)
    return age_seconds >= max(float(max_age_hours), 0.0) * 60 * 60


def load_historical_results(
    force: bool = False,
    source_url: str = HISTORICAL_RESULTS_URL,
    cache_path: Path = HISTORICAL_RESULTS_PATH,
    max_cache_age_hours: float | None = DEFAULT_SOURCE_CACHE_MAX_AGE_HOURS,
) -> pd.DataFrame:
    if force or cache_file_is_stale(cache_path, max_cache_age_hours):
        download_url(source_url, cache_path, force=True)

    results = pd.read_csv(cache_path)
    required_columns = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    }
    missing = sorted(required_columns - set(results.columns))
    if missing:
        raise ValueError(f"Historical results file is missing required columns: {missing}")

    results = results.copy()
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results["home_score"] = pd.to_numeric(results["home_score"], errors="coerce")
    results["away_score"] = pd.to_numeric(results["away_score"], errors="coerce")
    results["neutral"] = results["neutral"].astype(str).str.upper().eq("TRUE")
    results = results.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    results["home_score"] = results["home_score"].astype(int)
    results["away_score"] = results["away_score"].astype(int)
    return results


def load_goalscorers(
    force: bool = False,
    source_url: str = GOALSCORERS_URL,
    cache_path: Path = GOALSCORERS_PATH,
    max_cache_age_hours: float | None = DEFAULT_SOURCE_CACHE_MAX_AGE_HOURS,
) -> pd.DataFrame:
    if force or cache_file_is_stale(cache_path, max_cache_age_hours):
        download_url(source_url, cache_path, force=True)

    goalscorers = pd.read_csv(cache_path)
    required_columns = {
        "date",
        "home_team",
        "away_team",
        "team",
        "scorer",
        "minute",
        "own_goal",
        "penalty",
    }
    missing = sorted(required_columns - set(goalscorers.columns))
    if missing:
        raise ValueError(f"Goalscorers file is missing required columns: {missing}")

    goalscorers = goalscorers.copy()
    goalscorers["date"] = pd.to_datetime(goalscorers["date"], errors="coerce")
    for column in ("home_team", "away_team", "team"):
        goalscorers[column] = goalscorers[column].map(canonical_team_name)
    goalscorers["scorer"] = goalscorers["scorer"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    goalscorers["minute"] = pd.to_numeric(goalscorers["minute"], errors="coerce")
    goalscorers["own_goal"] = goalscorers["own_goal"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    goalscorers["penalty"] = goalscorers["penalty"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    goalscorers = goalscorers.dropna(subset=["date", "team", "scorer"])
    goalscorers = goalscorers[goalscorers["scorer"].ne("")]
    return goalscorers.sort_values("date").reset_index(drop=True)


def load_statsbomb_competitions(force: bool = False) -> pd.DataFrame:
    competitions = load_json_url(STATSBOMB_COMPETITIONS_URL, STATSBOMB_COMPETITIONS_PATH, force=force)
    frame = pd.DataFrame(competitions)
    if frame.empty:
        return frame
    frame["competition_international"] = frame["competition_international"].astype(bool)
    frame["competition_youth"] = frame["competition_youth"].astype(bool)
    return frame


def statsbomb_competition_candidates(competitions: pd.DataFrame) -> pd.DataFrame:
    if competitions.empty:
        return competitions
    candidates = competitions[
        competitions["competition_international"].eq(True)
        & competitions["competition_youth"].eq(False)
        & competitions["competition_gender"].astype(str).str.lower().eq("male")
    ].copy()
    candidates["match_available"] = pd.to_datetime(candidates["match_available"], errors="coerce")
    return candidates.sort_values(["match_available", "competition_name", "season_name"], ascending=[False, True, False])


def load_statsbomb_matches_for_competition(
    competition_id: int,
    season_id: int,
    force: bool = False,
) -> list[dict]:
    path = STATSBOMB_MATCHES_DIR / str(competition_id) / f"{season_id}.json"
    url = f"{STATSBOMB_MATCHES_BASE_URL}/{competition_id}/{season_id}.json"
    matches = load_json_url(url, path, force=force)
    if not isinstance(matches, list):
        raise ValueError(f"Unexpected StatsBomb matches payload for {competition_id}/{season_id}")
    return matches


def load_statsbomb_events(match_id: int, force: bool = False) -> list[dict]:
    path = STATSBOMB_EVENTS_DIR / f"{match_id}.json"
    url = f"{STATSBOMB_EVENTS_BASE_URL}/{match_id}.json"
    events = load_json_url(url, path, force=force)
    if not isinstance(events, list):
        raise ValueError(f"Unexpected StatsBomb events payload for match {match_id}")
    return events


def _statsbomb_team_name(team_payload: dict | None) -> str:
    payload = team_payload or {}
    return canonical_team_name(
        payload.get("name")
        or payload.get("home_team_name")
        or payload.get("away_team_name")
        or payload.get("team_name")
        or ""
    )


def statsbomb_events_to_team_stats(match: dict, events: list[dict]) -> list[dict]:
    home_team = _statsbomb_team_name(match.get("home_team"))
    away_team = _statsbomb_team_name(match.get("away_team"))
    match_date = match.get("match_date")
    competition = (match.get("competition") or {}).get("competition_name", "")
    season = (match.get("season") or {}).get("season_name", "")
    match_id = match.get("match_id")
    stats = {
        home_team: {
            "team": home_team,
            "opponent": away_team,
            "is_home": True,
            "statsbomb_shots": 0,
            "statsbomb_xg": 0.0,
            "statsbomb_goals": 0,
            "statsbomb_on_target": 0,
        },
        away_team: {
            "team": away_team,
            "opponent": home_team,
            "is_home": False,
            "statsbomb_shots": 0,
            "statsbomb_xg": 0.0,
            "statsbomb_goals": 0,
            "statsbomb_on_target": 0,
        },
    }

    on_target_outcomes = {"Goal", "Saved", "Saved to Post"}
    for event in events:
        if (event.get("type") or {}).get("name") != "Shot":
            continue
        team = _statsbomb_team_name(event.get("team"))
        if team not in stats:
            continue
        shot = event.get("shot") or {}
        outcome = (shot.get("outcome") or {}).get("name", "")
        stats[team]["statsbomb_shots"] += 1
        stats[team]["statsbomb_xg"] += float(shot.get("statsbomb_xg") or 0.0)
        stats[team]["statsbomb_goals"] += 1 if outcome == "Goal" else 0
        stats[team]["statsbomb_on_target"] += 1 if outcome in on_target_outcomes else 0

    rows: list[dict] = []
    for row in stats.values():
        opponent_row = stats.get(row["opponent"], {})
        rows.append(
            {
                "source": "StatsBomb Open Data",
                "match_id": match_id,
                "date": match_date,
                "competition": competition,
                "season": season,
                **row,
                "statsbomb_xg_against": opponent_row.get("statsbomb_xg", 0.0),
                "statsbomb_shots_against": opponent_row.get("statsbomb_shots", 0),
            }
        )
    return rows


def load_statsbomb_team_stats(
    force: bool = False,
    max_matches: int = 80,
) -> tuple[pd.DataFrame, dict[str, int]]:
    competitions = load_statsbomb_competitions(force=force)
    candidates = statsbomb_competition_candidates(competitions)
    match_rows: list[dict] = []
    event_errors = 0

    for competition in candidates.itertuples(index=False):
        try:
            matches = load_statsbomb_matches_for_competition(
                int(competition.competition_id),
                int(competition.season_id),
                force=force,
            )
        except Exception:
            event_errors += 1
            continue
        for match in matches:
            enriched = dict(match)
            enriched.setdefault(
                "competition",
                {
                    "competition_id": int(competition.competition_id),
                    "competition_name": competition.competition_name,
                },
            )
            enriched.setdefault(
                "season",
                {
                    "season_id": int(competition.season_id),
                    "season_name": competition.season_name,
                },
            )
            match_rows.append(enriched)

    match_rows = sorted(match_rows, key=lambda item: item.get("match_date") or "", reverse=True)
    selected_matches = match_rows[:max_matches] if max_matches > 0 else []
    stat_rows: list[dict] = []
    for match in selected_matches:
        try:
            events = load_statsbomb_events(int(match["match_id"]), force=force)
            stat_rows.extend(statsbomb_events_to_team_stats(match, events))
        except Exception:
            event_errors += 1

    stats = pd.DataFrame(stat_rows)
    if not stats.empty:
        stats["date"] = pd.to_datetime(stats["date"], errors="coerce")
        stats = stats.dropna(subset=["date", "team", "opponent"])
    metadata = {
        "statsbomb_competitions_found": len(competitions),
        "statsbomb_international_competitions": len(candidates),
        "statsbomb_matches_available": len(match_rows),
        "statsbomb_matches_loaded": len(selected_matches),
        "statsbomb_event_errors": event_errors,
    }
    return stats, metadata

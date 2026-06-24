from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from worldcup_prediction.config import (
    COMBINED_RESULTS_PATH,
    FIFA_HUB_URL,
    GOALSCORERS_PATH,
    GOALSCORERS_URL,
    HISTORICAL_RESULTS_URL,
    HUB_HTML_PATH,
    METADATA_PATH,
    PLAYER_PROFILES_PATH,
    PROCESSED_DATA_DIR,
    REPORT_MATCHES_PATH,
    REPORT_TEAM_STATS_PATH,
    STATSBOMB_COMPETITIONS_URL,
    STATSBOMB_TEAM_STATS_PATH,
    TEAM_PROFILES_PATH,
    ensure_data_dirs,
)
from worldcup_prediction.model import build_player_profiles, build_team_profiles, combine_results
from worldcup_prediction.pdf_reports import parse_report_pdf
from worldcup_prediction.sources import (
    download_report_pdfs,
    extract_report_links,
    fetch_text,
    load_goalscorers,
    load_historical_results,
    load_statsbomb_team_stats,
)


REQUIRED_TEAM_PROFILE_COLUMNS = {
    "weighted_recent_points_per_match",
    "weighted_recent_goals_for",
    "weighted_recent_goals_against",
    "weighted_recent_goal_diff",
    "recent_result_weight",
    "xg_for_2026_shrunk",
    "xg_against_2026_shrunk",
    "statsbomb_xg_for_shrunk",
    "statsbomb_xg_against_shrunk",
}

REQUIRED_PLAYER_PROFILE_COLUMNS = {
    "team",
    "player",
    "goals",
    "recent_goals_24m",
    "penalty_goals",
    "scoring_weight",
    "team_goal_share",
}


@dataclass
class DataBundle:
    historical_results: pd.DataFrame
    goalscorers: pd.DataFrame
    report_matches: pd.DataFrame
    report_team_stats: pd.DataFrame
    statsbomb_team_stats: pd.DataFrame
    combined_results: pd.DataFrame
    team_profiles: pd.DataFrame
    player_profiles: pd.DataFrame
    metadata: dict[str, Any]


def _read_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except ValueError:
        frame = pd.read_csv(path)
        for column in parse_dates or []:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        return frame


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def refresh_data(
    force: bool = False,
    max_reports: int | None = None,
    include_statsbomb: bool = True,
    max_statsbomb_matches: int = 80,
) -> DataBundle:
    ensure_data_dirs()

    hub_html = fetch_text(FIFA_HUB_URL)
    HUB_HTML_PATH.write_text(hub_html, encoding="utf-8")
    report_links = extract_report_links(hub_html)
    downloaded_reports = download_report_pdfs(report_links, force=force, max_reports=max_reports)

    match_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    for report in downloaded_reports:
        try:
            match_row, parsed_team_rows = parse_report_pdf(report.path, source_url=report.link.url)
            if report.link.match_number is not None and not match_row.get("match_number"):
                match_row["match_number"] = report.link.match_number
            match_rows.append(match_row)
            team_rows.extend(parsed_team_rows)
        except Exception as exc:  # noqa: BLE001 - keep refresh resilient to a single bad PDF.
            parse_errors.append({"url": report.link.url, "file": str(report.path), "error": str(exc)})

    report_matches = pd.DataFrame(match_rows)
    report_team_stats = pd.DataFrame(team_rows)
    historical_results = load_historical_results(force=force)
    goalscorers = load_goalscorers(force=force)
    statsbomb_metadata: dict[str, Any] = {
        "statsbomb_competitions_found": 0,
        "statsbomb_international_competitions": 0,
        "statsbomb_matches_available": 0,
        "statsbomb_matches_loaded": 0,
        "statsbomb_event_errors": 0,
        "statsbomb_refresh_error": "",
    }
    statsbomb_team_stats = pd.DataFrame()
    if include_statsbomb:
        try:
            statsbomb_team_stats, statsbomb_metadata = load_statsbomb_team_stats(
                force=force,
                max_matches=max_statsbomb_matches,
            )
        except Exception as exc:  # noqa: BLE001 - do not block the core model if an optional open source fails.
            statsbomb_metadata["statsbomb_refresh_error"] = str(exc)
    combined_results = combine_results(historical_results, report_matches)
    team_profiles = build_team_profiles(combined_results, report_team_stats, statsbomb_team_stats)
    player_profiles = build_player_profiles(goalscorers, combined_results)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    report_matches.to_csv(REPORT_MATCHES_PATH, index=False)
    report_team_stats.to_csv(REPORT_TEAM_STATS_PATH, index=False)
    statsbomb_team_stats.to_csv(STATSBOMB_TEAM_STATS_PATH, index=False)
    combined_results.to_csv(COMBINED_RESULTS_PATH, index=False)
    team_profiles.to_csv(TEAM_PROFILES_PATH, index=False)
    player_profiles.to_csv(PLAYER_PROFILES_PATH, index=False)

    metadata = {
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "fifa_hub_url": FIFA_HUB_URL,
        "historical_results_url": HISTORICAL_RESULTS_URL,
        "goalscorers_url": GOALSCORERS_URL,
        "statsbomb_competitions_url": STATSBOMB_COMPETITIONS_URL,
        "report_links_found": len(report_links),
        "reports_downloaded_or_cached": len(downloaded_reports),
        "reports_downloaded_this_run": sum(1 for report in downloaded_reports if report.downloaded),
        "report_matches_parsed": len(report_matches),
        "report_team_rows_parsed": len(report_team_stats),
        "historical_results_rows": len(historical_results),
        "goalscorer_rows": len(goalscorers),
        "combined_results_rows": len(combined_results),
        "team_profiles_rows": len(team_profiles),
        "player_profiles_rows": len(player_profiles),
        "parse_errors": parse_errors,
        **statsbomb_metadata,
    }
    _write_json(METADATA_PATH, metadata)

    return DataBundle(
        historical_results=historical_results,
        goalscorers=goalscorers,
        report_matches=report_matches,
        report_team_stats=report_team_stats,
        statsbomb_team_stats=statsbomb_team_stats,
        combined_results=combined_results,
        team_profiles=team_profiles,
        player_profiles=player_profiles,
        metadata=metadata,
    )


def load_cached_data() -> DataBundle:
    metadata = _read_json(METADATA_PATH)
    report_matches = _read_csv(REPORT_MATCHES_PATH, parse_dates=["date"])
    report_team_stats = _read_csv(REPORT_TEAM_STATS_PATH, parse_dates=["date"])
    statsbomb_team_stats = _read_csv(STATSBOMB_TEAM_STATS_PATH, parse_dates=["date"])
    combined_results = _read_csv(COMBINED_RESULTS_PATH, parse_dates=["date"])
    team_profiles = _read_csv(TEAM_PROFILES_PATH)
    player_profiles = _read_csv(PLAYER_PROFILES_PATH, parse_dates=["last_goal_date"])

    has_current_profile_schema = REQUIRED_TEAM_PROFILE_COLUMNS.issubset(set(team_profiles.columns))
    if (team_profiles.empty or not has_current_profile_schema) and not combined_results.empty:
        team_profiles = build_team_profiles(combined_results, report_team_stats, statsbomb_team_stats)
        TEAM_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
        team_profiles.to_csv(TEAM_PROFILES_PATH, index=False)

    has_current_player_schema = REQUIRED_PLAYER_PROFILE_COLUMNS.issubset(set(player_profiles.columns))
    if player_profiles.empty or not has_current_player_schema:
        cached_goalscorers = _read_csv(GOALSCORERS_PATH, parse_dates=["date"])
        if not cached_goalscorers.empty:
            result_context = combined_results if not combined_results.empty else None
            player_profiles = build_player_profiles(cached_goalscorers, result_context)
            PLAYER_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
            player_profiles.to_csv(PLAYER_PROFILES_PATH, index=False)
            metadata["player_profiles_rows"] = len(player_profiles)
            _write_json(METADATA_PATH, metadata)

    return DataBundle(
        historical_results=pd.DataFrame(),
        goalscorers=pd.DataFrame(),
        report_matches=report_matches,
        report_team_stats=report_team_stats,
        statsbomb_team_stats=statsbomb_team_stats,
        combined_results=combined_results,
        team_profiles=team_profiles,
        player_profiles=player_profiles,
        metadata=metadata,
    )

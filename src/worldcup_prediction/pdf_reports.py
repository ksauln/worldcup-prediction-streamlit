from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from worldcup_prediction.teams import canonical_team_name


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def _parse_scoreline(lines: list[str]) -> dict[str, Any]:
    score_re = re.compile(r"^(?P<home>.+?)\s+(?P<home_score>\d+)\s*-\s*(?P<away_score>\d+)\s+(?P<away>.+?)$")
    compressed_score_re = re.compile(r"^(?P<home>.+?)(?P<home_score>\d+)\s*-\s*(?P<away_score>\d+)$")
    for line in lines[:30]:
        match = score_re.match(clean_text(line))
        if match:
            return {
                "home_team": canonical_team_name(match.group("home").strip()),
                "away_team": canonical_team_name(match.group("away").strip()),
                "home_score": int(match.group("home_score")),
                "away_score": int(match.group("away_score")),
            }
    for index, line in enumerate(lines[:30]):
        match = compressed_score_re.match(clean_text(line))
        if match and index + 1 < len(lines):
            return {
                "home_team": canonical_team_name(match.group("home").strip()),
                "away_team": canonical_team_name(lines[index + 1].strip()),
                "home_score": int(match.group("home_score")),
                "away_score": int(match.group("away_score")),
            }
    raise ValueError("Could not find report scoreline in PDF text.")


def _parse_group_and_match(text: str) -> tuple[str | None, int | None]:
    match = re.search(r"\b(?P<group>Group\s+[A-L])\s+-\s+Match\s+(?P<number>\d{1,3})\b", text, flags=re.IGNORECASE)
    if not match:
        return None, None
    return match.group("group").title(), int(match.group("number"))


def _parse_date(text: str) -> str | None:
    match = re.search(r"\b(?P<date>\d{1,2}\s+[A-Za-z]+\s+2026)\b", text)
    if not match:
        return None
    try:
        return datetime.strptime(match.group("date"), "%d %B %Y").date().isoformat()
    except ValueError:
        return match.group("date")


def _parse_kickoff(text: str) -> str | None:
    match = re.search(r"\b(?P<kickoff>\d{1,2}:\d{2})\s+Kick\b", text, flags=re.IGNORECASE)
    return match.group("kickoff") if match else None


def _parse_venue(lines: list[str]) -> str | None:
    for index, line in enumerate(lines[:12]):
        if re.search(r"\bKick\b", line, flags=re.IGNORECASE) and index + 1 < len(lines):
            return clean_text(lines[index + 1])
    return None


def _match_id(date: str | None, match_number: int | None, home_team: str, away_team: str) -> str:
    if match_number is not None:
        return f"2026_M{match_number:03d}"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{home_team}_{away_team}").strip("_")
    return f"{date or 'unknown_date'}_{slug}"


def _result(goals_for: int, goals_against: int) -> str:
    if goals_for > goals_against:
        return "W"
    if goals_for < goals_against:
        return "L"
    return "D"


def _points(result: str) -> int:
    return {"W": 3, "D": 1, "L": 0}[result]


def _set_pair(rows: list[dict[str, Any]], key: str, values: tuple[Any, Any]) -> None:
    rows[0][key] = values[0]
    rows[1][key] = values[1]


def _number(value: str) -> int | float:
    return float(value) if "." in value else int(value)


def _parse_key_statistics(text: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    compact = clean_text(text)
    match_updates: dict[str, Any] = {}

    patterns: list[tuple[str, str, str, str]] = [
        ("xg", r"(?P<home>\d+(?:\.\d+)?)\s+xG\s*\(Expected Goals\)\s+(?P<away>\d+(?:\.\d+)?)", "home_xg", "away_xg"),
        ("pass_completion_pct", r"(?P<home>\d+(?:\.\d+)?)\s*%\s+Pass Completion\s*%\s+(?P<away>\d+(?:\.\d+)?)\s*%", "home_pass_completion_pct", "away_pass_completion_pct"),
        ("completed_line_breaks", r"(?P<home>\d+)\s+Completed Line Breaks\s+(?P<away>\d+)", "home_completed_line_breaks", "away_completed_line_breaks"),
        ("defensive_line_breaks", r"(?P<home>\d+)\s+Defensive Line Breaks\s+(?P<away>\d+)", "home_defensive_line_breaks", "away_defensive_line_breaks"),
        ("final_third_receptions", r"(?P<home>\d+)\s+Receptions in the Final Third\s+(?P<away>\d+)", "home_final_third_receptions", "away_final_third_receptions"),
        ("crosses", r"(?P<home>\d+)\s+Crosses\s+(?P<away>\d+)", "home_crosses", "away_crosses"),
        ("ball_progressions", r"(?P<home>\d+)\s+Ball Progressions\s+(?P<away>\d+)", "home_ball_progressions", "away_ball_progressions"),
        ("forced_turnovers", r"(?P<home>\d+)\s+Forced Turnovers\s+(?P<away>\d+)", "home_forced_turnovers", "away_forced_turnovers"),
        ("second_balls", r"(?P<home>\d+)\s+Second Balls\s+(?P<away>\d+)", "home_second_balls", "away_second_balls"),
        ("distance_km", r"(?P<home>\d+(?:\.\d+)?)\s*km\s+Total Distance Covered\s+(?P<away>\d+(?:\.\d+)?)\s*km", "home_distance_km", "away_distance_km"),
        ("zone4_sprint_km", r"(?P<home>\d+(?:\.\d+)?)\s*km\s+Zone 4.*?20-25 km/h\s+(?P<away>\d+(?:\.\d+)?)\s*km", "home_zone4_sprint_km", "away_zone4_sprint_km"),
    ]

    for row_key, pattern, home_key, away_key in patterns:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if not match:
            continue
        home_value = _number(match.group("home"))
        away_value = _number(match.group("away"))
        _set_pair(rows, row_key, (home_value, away_value))
        match_updates[home_key] = home_value
        match_updates[away_key] = away_value

    attempts = re.search(
        r"(?P<home_attempts>\d+)\s*\((?P<home_on>\d+)\)\s+Attempts at Goal\s*\(On Target\)\s+"
        r"(?P<away_attempts>\d+)\s*\((?P<away_on>\d+)\)",
        compact,
        flags=re.IGNORECASE,
    )
    if attempts:
        _set_pair(rows, "attempts", (int(attempts.group("home_attempts")), int(attempts.group("away_attempts"))))
        _set_pair(rows, "attempts_on_target", (int(attempts.group("home_on")), int(attempts.group("away_on"))))
        match_updates.update(
            {
                "home_attempts": int(attempts.group("home_attempts")),
                "away_attempts": int(attempts.group("away_attempts")),
                "home_attempts_on_target": int(attempts.group("home_on")),
                "away_attempts_on_target": int(attempts.group("away_on")),
            }
        )

    passes = re.search(
        r"(?P<home_passes>\d+)\s*\((?P<home_complete>\d+)\)\s+Total Passes\s*\(Complete\)\s+"
        r"(?P<away_passes>\d+)\s*\((?P<away_complete>\d+)\)",
        compact,
        flags=re.IGNORECASE,
    )
    if passes:
        _set_pair(rows, "total_passes", (int(passes.group("home_passes")), int(passes.group("away_passes"))))
        _set_pair(rows, "completed_passes", (int(passes.group("home_complete")), int(passes.group("away_complete"))))
        match_updates.update(
            {
                "home_total_passes": int(passes.group("home_passes")),
                "away_total_passes": int(passes.group("away_passes")),
                "home_completed_passes": int(passes.group("home_complete")),
                "away_completed_passes": int(passes.group("away_complete")),
            }
        )

    pressures = re.search(
        r"(?P<home_pressures>\d+)\s*\((?P<home_direct>\d+)\)\s+Defensive Pressures Applied\s*"
        r"\(Direct Pressures\)\s+(?P<away_pressures>\d+)\s*\((?P<away_direct>\d+)\)",
        compact,
        flags=re.IGNORECASE,
    )
    if pressures:
        _set_pair(rows, "defensive_pressures", (int(pressures.group("home_pressures")), int(pressures.group("away_pressures"))))
        _set_pair(rows, "direct_pressures", (int(pressures.group("home_direct")), int(pressures.group("away_direct"))))
        match_updates.update(
            {
                "home_defensive_pressures": int(pressures.group("home_pressures")),
                "away_defensive_pressures": int(pressures.group("away_pressures")),
                "home_direct_pressures": int(pressures.group("home_direct")),
                "away_direct_pressures": int(pressures.group("away_direct")),
            }
        )

    possession = re.search(
        r"Possession\s+Total\s+(?P<home>\d+(?:\.\d+)?)%\s+(?P<contest>\d+(?:\.\d+)?)%\s+"
        r"(?P<away>\d+(?:\.\d+)?)%\s+Total",
        compact,
        flags=re.IGNORECASE,
    )
    if possession:
        home_possession = float(possession.group("home"))
        contest_possession = float(possession.group("contest"))
        away_possession = float(possession.group("away"))
        _set_pair(rows, "possession_pct", (home_possession, away_possession))
        _set_pair(rows, "possession_contest_pct", (contest_possession, contest_possession))
        match_updates.update(
            {
                "home_possession_pct": home_possession,
                "away_possession_pct": away_possession,
                "possession_contest_pct": contest_possession,
            }
        )

    return match_updates


def parse_report_text(
    pages: list[str],
    source_url: str = "",
    pdf_file: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    full_text = "\n".join(pages)
    lines = [line.strip() for page in pages for line in page.splitlines() if line.strip()]
    scoreline = _parse_scoreline(lines)
    group, match_number = _parse_group_and_match(full_text)
    date = _parse_date(full_text)
    kickoff = _parse_kickoff(full_text)
    venue = _parse_venue(lines)
    match_id = _match_id(date, match_number, scoreline["home_team"], scoreline["away_team"])

    home_result = _result(scoreline["home_score"], scoreline["away_score"])
    away_result = _result(scoreline["away_score"], scoreline["home_score"])
    rows = [
        {
            "match_id": match_id,
            "match_number": match_number,
            "date": date,
            "team": scoreline["home_team"],
            "opponent": scoreline["away_team"],
            "is_home": True,
            "goals_for": scoreline["home_score"],
            "goals_against": scoreline["away_score"],
            "result": home_result,
            "points": _points(home_result),
        },
        {
            "match_id": match_id,
            "match_number": match_number,
            "date": date,
            "team": scoreline["away_team"],
            "opponent": scoreline["home_team"],
            "is_home": False,
            "goals_for": scoreline["away_score"],
            "goals_against": scoreline["home_score"],
            "result": away_result,
            "points": _points(away_result),
        },
    ]

    match_row: dict[str, Any] = {
        "match_id": match_id,
        "match_number": match_number,
        "group": group,
        "date": date,
        "kickoff": kickoff,
        "venue": venue,
        "home_team": scoreline["home_team"],
        "away_team": scoreline["away_team"],
        "home_score": scoreline["home_score"],
        "away_score": scoreline["away_score"],
        "tournament": "FIFA World Cup 2026",
        "source_url": source_url,
        "pdf_file": pdf_file,
    }

    match_row.update(_parse_key_statistics(full_text, rows))
    for row in rows:
        if "xg" in row:
            row["xg_against"] = rows[1]["xg"] if row["is_home"] else rows[0]["xg"]
        if "attempts" in row:
            row["attempts_against"] = rows[1]["attempts"] if row["is_home"] else rows[0]["attempts"]
        if "attempts_on_target" in row:
            row["attempts_on_target_against"] = rows[1]["attempts_on_target"] if row["is_home"] else rows[0]["attempts_on_target"]
        row["source_url"] = source_url
        row["pdf_file"] = pdf_file

    return match_row, rows


def parse_report_pdf(pdf_path: Path, source_url: str = "") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pages = extract_pdf_pages(pdf_path)
    return parse_report_text(pages, source_url=source_url, pdf_file=str(pdf_path))

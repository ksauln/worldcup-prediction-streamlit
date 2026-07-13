from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from worldcup_prediction.teams import canonical_team_name

DEFAULT_RATING = 1500.0
HOME_ADVANTAGE_POINTS = 55.0
RECENT_MATCH_HALFLIFE_DAYS = 365
RECENT_ELO_HALFLIFE_DAYS = 365 * 3
RECENT_PLAYER_WINDOW_DAYS = 730
PLAYER_RECENCY_HALFLIFE_DAYS = 900
XG_PRIOR = 1.28
REPORT_XG_SHRINKAGE_MATCHES = 4.0
STATSBOMB_XG_SHRINKAGE_MATCHES = 8.0
HOME_GOAL_ADVANTAGE_COMPONENT = 0.08
ATTACK_DEFENSE_ENSEMBLE_WEIGHT = 0.20
ATTACK_DEFENSE_PRIOR_MATCHES = 6.0
ATTACK_DEFENSE_ATTACK_POWER = 0.70
ATTACK_DEFENSE_DEFENSE_POWER = 0.65
REPORT_XG_ATTACK_DEFENSE_WEIGHT = 0.28
STATSBOMB_XG_ATTACK_DEFENSE_WEIGHT = 0.14
POISSON_TAIL_PROBABILITY = 0.0025
MIN_SCORELINE_MAX_GOALS = 7
MAX_SCORELINE_MAX_GOALS = 14


def tournament_weight(name: Any) -> float:
    value = str(name or "").lower()
    if "fifa world cup" in value and "qualification" not in value:
        return 1.35
    if "qualification" in value:
        return 1.1
    if any(token in value for token in ("uefa euro", "copa am", "africa cup", "asian cup", "gold cup", "nations league")):
        return 1.15
    if "friendly" in value:
        return 0.75
    return 1.0


def _coerce_results(results: pd.DataFrame) -> pd.DataFrame:
    if results is None or results.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "tournament",
                "city",
                "country",
                "neutral",
            ]
        )

    coerced = results.copy()
    for column in ("city", "country", "tournament"):
        if column not in coerced:
            coerced[column] = ""
    if "neutral" not in coerced:
        coerced["neutral"] = True

    coerced["date"] = pd.to_datetime(coerced["date"], errors="coerce")
    coerced["home_team"] = coerced["home_team"].map(canonical_team_name)
    coerced["away_team"] = coerced["away_team"].map(canonical_team_name)
    coerced["home_score"] = pd.to_numeric(coerced["home_score"], errors="coerce")
    coerced["away_score"] = pd.to_numeric(coerced["away_score"], errors="coerce")
    coerced["neutral"] = coerced["neutral"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    coerced = coerced.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    coerced["home_score"] = coerced["home_score"].astype(int)
    coerced["away_score"] = coerced["away_score"].astype(int)
    return coerced.sort_values("date").reset_index(drop=True)


def report_matches_to_results(report_matches: pd.DataFrame) -> pd.DataFrame:
    if report_matches is None or report_matches.empty:
        return _coerce_results(pd.DataFrame())

    rows = report_matches.copy()
    rows["city"] = rows.get("venue", "")
    rows["country"] = "FIFA World Cup 2026"
    rows["neutral"] = True
    rows["tournament"] = rows.get("tournament", "FIFA World Cup 2026")
    return _coerce_results(
        rows[
            [
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "tournament",
                "city",
                "country",
                "neutral",
            ]
        ]
    )


def combine_results(historical_results: pd.DataFrame, report_matches: pd.DataFrame | None = None) -> pd.DataFrame:
    historical = _coerce_results(historical_results)
    reports = report_matches_to_results(report_matches if report_matches is not None else pd.DataFrame())
    combined = pd.concat([historical, reports], ignore_index=True)
    if combined.empty:
        return combined

    combined = combined.drop_duplicates(
        subset=["date", "home_team", "away_team", "home_score", "away_score"],
        keep="last",
    )
    return combined.sort_values("date").reset_index(drop=True)


def active_team_names(
    results: pd.DataFrame | None,
    profiles: pd.DataFrame | None = None,
    reference_date: str | pd.Timestamp | None = None,
    window_days: int = 1460,
) -> list[str]:
    profile_teams = set()
    if profiles is not None and not profiles.empty and "team" in profiles:
        profile_teams = set(profiles["team"].dropna().astype(str))

    if results is None or results.empty or "date" not in results:
        return sorted(profile_teams)

    dated = results.copy()
    dated["date"] = pd.to_datetime(dated["date"], errors="coerce")
    dated = dated.dropna(subset=["date"])
    if dated.empty:
        return sorted(profile_teams)

    reference = pd.to_datetime(reference_date, errors="coerce") if reference_date is not None else dated["date"].max()
    if pd.isna(reference):
        reference = dated["date"].max()
    cutoff = reference - pd.Timedelta(days=max(int(window_days), 1))
    active_rows = dated[dated["date"].ge(cutoff)]
    active = set()
    for column in ("home_team", "away_team"):
        if column in active_rows:
            active.update(active_rows[column].dropna().map(canonical_team_name).astype(str))
    if profile_teams:
        active &= profile_teams
    return sorted(active or profile_teams)


def compute_elo_ratings(results: pd.DataFrame) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    results = _coerce_results(results)
    max_date = results["date"].max() if not results.empty else pd.NaT

    for row in results.itertuples(index=False):
        home = str(row.home_team)
        away = str(row.away_team)
        home_rating = ratings.get(home, DEFAULT_RATING)
        away_rating = ratings.get(away, DEFAULT_RATING)
        adjusted_home_rating = home_rating + (0.0 if row.neutral else HOME_ADVANTAGE_POINTS)
        expected_home = 1 / (1 + 10 ** ((away_rating - adjusted_home_rating) / 400))

        if row.home_score > row.away_score:
            actual_home = 1.0
        elif row.home_score < row.away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        margin = abs(row.home_score - row.away_score)
        margin_multiplier = 1.0 if margin <= 1 else math.log(margin + 1) * 1.15
        age_days = max((max_date - row.date).days, 0) if pd.notna(max_date) else 0
        recency_multiplier = 0.75 + 0.75 * math.exp(-math.log(2) * age_days / RECENT_ELO_HALFLIFE_DAYS)
        k_factor = 22.0 * tournament_weight(row.tournament) * margin_multiplier * recency_multiplier
        change = k_factor * (actual_home - expected_home)
        ratings[home] = home_rating + change
        ratings[away] = away_rating - change

    return (
        pd.DataFrame([{"team": team, "rating": rating} for team, rating in ratings.items()])
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )


def team_match_frame(results: pd.DataFrame) -> pd.DataFrame:
    results = _coerce_results(results)
    rows: list[dict[str, Any]] = []
    for row in results.itertuples(index=False):
        home_result = "W" if row.home_score > row.away_score else "L" if row.home_score < row.away_score else "D"
        away_result = "W" if row.away_score > row.home_score else "L" if row.away_score < row.home_score else "D"
        rows.append(
            {
                "date": row.date,
                "team": row.home_team,
                "opponent": row.away_team,
                "goals_for": row.home_score,
                "goals_against": row.away_score,
                "points": {"W": 3, "D": 1, "L": 0}[home_result],
                "result": home_result,
                "tournament": row.tournament,
            }
        )
        rows.append(
            {
                "date": row.date,
                "team": row.away_team,
                "opponent": row.home_team,
                "goals_for": row.away_score,
                "goals_against": row.home_score,
                "points": {"W": 3, "D": 1, "L": 0}[away_result],
                "result": away_result,
                "tournament": row.tournament,
            }
        )
    return pd.DataFrame(rows)


def recent_results_with_weights(
    team_matches: pd.DataFrame,
    reference_date: str | pd.Timestamp | None = None,
    half_life_days: int = RECENT_MATCH_HALFLIFE_DAYS,
) -> pd.DataFrame:
    output_columns = [
        "date",
        "team",
        "opponent",
        "goals_for",
        "goals_against",
        "points",
        "result",
        "tournament",
        "goal_diff",
        "recency_weight",
    ]
    if team_matches is None or team_matches.empty:
        return pd.DataFrame(columns=output_columns)

    weighted = team_matches.copy()
    weighted["date"] = pd.to_datetime(weighted["date"], errors="coerce")
    weighted = weighted.dropna(subset=["date", "team"])
    if weighted.empty:
        return pd.DataFrame(columns=output_columns)

    for column in ("goals_for", "goals_against", "points"):
        if column not in weighted:
            weighted[column] = 0
        weighted[column] = pd.to_numeric(weighted[column], errors="coerce").fillna(0)

    reference = pd.to_datetime(reference_date, errors="coerce") if reference_date is not None else weighted["date"].max()
    if pd.isna(reference):
        reference = weighted["date"].max()
    age_days = (reference - weighted["date"]).dt.days.clip(lower=0)
    weighted["recency_weight"] = np.exp(-np.log(2) * age_days / max(float(half_life_days), 1.0))
    weighted["goal_diff"] = weighted["goals_for"] - weighted["goals_against"]
    return weighted[[column for column in output_columns if column in weighted.columns]].sort_values("date").reset_index(drop=True)


def recency_weighted_team_form(team_matches: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "team",
        "recent_result_weight",
        "weighted_recent_points_per_match",
        "weighted_recent_goals_for",
        "weighted_recent_goals_against",
        "weighted_recent_goal_diff",
    ]
    weighted = recent_results_with_weights(team_matches)
    if weighted.empty:
        return pd.DataFrame(columns=output_columns)

    weighted["_weighted_points"] = weighted["points"] * weighted["recency_weight"]
    weighted["_weighted_goals_for"] = weighted["goals_for"] * weighted["recency_weight"]
    weighted["_weighted_goals_against"] = weighted["goals_against"] * weighted["recency_weight"]
    weighted["_weighted_goal_diff"] = weighted["goal_diff"] * weighted["recency_weight"]
    aggregate = (
        weighted.groupby("team")
        .agg(
            recent_result_weight=("recency_weight", "sum"),
            _weighted_points=("_weighted_points", "sum"),
            _weighted_goals_for=("_weighted_goals_for", "sum"),
            _weighted_goals_against=("_weighted_goals_against", "sum"),
            _weighted_goal_diff=("_weighted_goal_diff", "sum"),
        )
        .reset_index()
    )
    denominator = aggregate["recent_result_weight"].replace(0, np.nan)
    aggregate["weighted_recent_points_per_match"] = aggregate["_weighted_points"] / denominator
    aggregate["weighted_recent_goals_for"] = aggregate["_weighted_goals_for"] / denominator
    aggregate["weighted_recent_goals_against"] = aggregate["_weighted_goals_against"] / denominator
    aggregate["weighted_recent_goal_diff"] = aggregate["_weighted_goal_diff"] / denominator
    aggregate = aggregate[output_columns]
    numeric_columns = [column for column in aggregate.columns if column != "team"]
    aggregate[numeric_columns] = aggregate[numeric_columns].fillna(0)
    return aggregate


def _mean_if_present(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    return frame.groupby("team")[column].mean()


def _report_aggregates(report_team_stats: pd.DataFrame | None) -> pd.DataFrame:
    if report_team_stats is None or report_team_stats.empty:
        return pd.DataFrame(columns=["team", "report_matches"])

    reports = report_team_stats.copy()
    column_map = {
        "xg": "xg_for_2026",
        "xg_against": "xg_against_2026",
        "attempts": "attempts_for_2026",
        "attempts_against": "attempts_against_2026",
        "attempts_on_target": "on_target_for_2026",
        "attempts_on_target_against": "on_target_against_2026",
        "possession_pct": "possession_pct_2026",
    }
    for source_column in column_map:
        if source_column in reports:
            reports[source_column] = pd.to_numeric(reports[source_column], errors="coerce")

    grouped = reports.groupby("team")
    aggregate = grouped.size().rename("report_matches").to_frame()
    for source_column, output_column in column_map.items():
        if source_column in reports:
            aggregate[output_column] = grouped[source_column].mean()

    return aggregate.reset_index()


def _statsbomb_aggregates(statsbomb_team_stats: pd.DataFrame | None) -> pd.DataFrame:
    if statsbomb_team_stats is None or statsbomb_team_stats.empty:
        return pd.DataFrame(columns=["team", "statsbomb_matches"])

    stats = statsbomb_team_stats.copy()
    column_map = {
        "statsbomb_xg": "statsbomb_xg_for",
        "statsbomb_xg_against": "statsbomb_xg_against",
        "statsbomb_shots": "statsbomb_shots_for",
        "statsbomb_shots_against": "statsbomb_shots_against",
        "statsbomb_on_target": "statsbomb_on_target_for",
    }
    for source_column in column_map:
        if source_column in stats:
            stats[source_column] = pd.to_numeric(stats[source_column], errors="coerce")

    grouped = stats.groupby("team")
    aggregate = grouped.size().rename("statsbomb_matches").to_frame()
    for source_column, output_column in column_map.items():
        if source_column in stats:
            aggregate[output_column] = grouped[source_column].mean()
    return aggregate.reset_index()


def _add_shrunk_rate_feature(
    profiles: pd.DataFrame,
    value_column: str,
    count_column: str,
    output_column: str,
    prior: float = XG_PRIOR,
    prior_matches: float = REPORT_XG_SHRINKAGE_MATCHES,
) -> None:
    if value_column not in profiles:
        profiles[value_column] = 0.0
    if count_column not in profiles:
        profiles[count_column] = 0.0

    values = pd.to_numeric(profiles[value_column], errors="coerce").fillna(0.0)
    counts = pd.to_numeric(profiles[count_column], errors="coerce").fillna(0.0).clip(lower=0.0)
    weights = counts / (counts + max(float(prior_matches), 1.0))
    profiles[output_column] = prior + weights * (values - prior)


def _add_shrunk_xg_features(profiles: pd.DataFrame) -> pd.DataFrame:
    output = profiles.copy()
    _add_shrunk_rate_feature(
        output,
        "xg_for_2026",
        "report_matches",
        "xg_for_2026_shrunk",
        prior_matches=REPORT_XG_SHRINKAGE_MATCHES,
    )
    _add_shrunk_rate_feature(
        output,
        "xg_against_2026",
        "report_matches",
        "xg_against_2026_shrunk",
        prior_matches=REPORT_XG_SHRINKAGE_MATCHES,
    )
    _add_shrunk_rate_feature(
        output,
        "statsbomb_xg_for",
        "statsbomb_matches",
        "statsbomb_xg_for_shrunk",
        prior_matches=STATSBOMB_XG_SHRINKAGE_MATCHES,
    )
    _add_shrunk_rate_feature(
        output,
        "statsbomb_xg_against",
        "statsbomb_matches",
        "statsbomb_xg_against_shrunk",
        prior_matches=STATSBOMB_XG_SHRINKAGE_MATCHES,
    )
    return output


def _coerce_goalscorers(goalscorers: pd.DataFrame | None) -> pd.DataFrame:
    if goalscorers is None or goalscorers.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "home_team",
                "away_team",
                "team",
                "scorer",
                "minute",
                "own_goal",
                "penalty",
            ]
        )

    scorers = goalscorers.copy()
    for column in ("home_team", "away_team", "team"):
        if column not in scorers:
            scorers[column] = ""
        scorers[column] = scorers[column].map(canonical_team_name)
    if "minute" not in scorers:
        scorers["minute"] = np.nan
    if "own_goal" not in scorers:
        scorers["own_goal"] = False
    if "penalty" not in scorers:
        scorers["penalty"] = False

    scorers["date"] = pd.to_datetime(scorers["date"], errors="coerce")
    scorers["scorer"] = scorers["scorer"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    scorers["minute"] = pd.to_numeric(scorers["minute"], errors="coerce")
    scorers["own_goal"] = scorers["own_goal"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    scorers["penalty"] = scorers["penalty"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    scorers = scorers.dropna(subset=["date", "team", "scorer"])
    scorers = scorers[scorers["scorer"].ne("")]
    return scorers.sort_values("date").reset_index(drop=True)


def build_player_profiles(goalscorers: pd.DataFrame | None, results: pd.DataFrame | None = None) -> pd.DataFrame:
    scorers = _coerce_goalscorers(goalscorers)
    scorers = scorers[~scorers["own_goal"]].copy()
    if scorers.empty:
        return pd.DataFrame(
            columns=[
                "team",
                "player",
                "goals",
                "recent_goals_24m",
                "penalty_goals",
                "first_goal_count",
                "avg_goal_minute",
                "last_goal_date",
                "scoring_weight",
                "team_goal_share",
            ]
        )

    max_date = scorers["date"].max()
    age_days = (max_date - scorers["date"]).dt.days.clip(lower=0)
    scorers["recency_weight"] = np.exp(-age_days / PLAYER_RECENCY_HALFLIFE_DAYS)
    scorers["adjusted_goal_weight"] = scorers["recency_weight"] * np.where(scorers["penalty"], 0.72, 1.0)
    scorers["recent_goal"] = age_days <= RECENT_PLAYER_WINDOW_DAYS

    first_goal_rows = (
        scorers.sort_values(["date", "minute"])
        .groupby(["date", "home_team", "away_team"], as_index=False)
        .first()[["date", "home_team", "away_team", "team", "scorer"]]
    )
    first_goal_rows["first_goal"] = True
    scorers = scorers.merge(
        first_goal_rows,
        on=["date", "home_team", "away_team", "team", "scorer"],
        how="left",
    )
    scorers["first_goal"] = scorers["first_goal"].fillna(False)

    grouped = scorers.groupby(["team", "scorer"], dropna=False)
    profiles = grouped.agg(
        goals=("scorer", "size"),
        recent_goals_24m=("recent_goal", "sum"),
        penalty_goals=("penalty", "sum"),
        first_goal_count=("first_goal", "sum"),
        avg_goal_minute=("minute", "mean"),
        last_goal_date=("date", "max"),
        scoring_weight=("adjusted_goal_weight", "sum"),
    ).reset_index()
    profiles = profiles.rename(columns={"scorer": "player"})

    if results is not None and not results.empty:
        team_goals = team_match_frame(results).groupby("team")["goals_for"].sum().rename("team_total_goals")
    else:
        team_goals = scorers.groupby("team").size().rename("team_total_goals")
    profiles = profiles.merge(team_goals.reset_index(), on="team", how="left")
    profiles["team_total_goals"] = profiles["team_total_goals"].replace(0, np.nan)
    profiles["team_goal_share"] = profiles["goals"] / profiles["team_total_goals"]
    profiles["team_goal_share"] = profiles["team_goal_share"].fillna(0)
    profiles["last_goal_date"] = pd.to_datetime(profiles["last_goal_date"], errors="coerce").dt.date.astype(str)
    numeric_columns = [
        "goals",
        "recent_goals_24m",
        "penalty_goals",
        "first_goal_count",
        "avg_goal_minute",
        "scoring_weight",
        "team_total_goals",
        "team_goal_share",
    ]
    profiles[numeric_columns] = profiles[numeric_columns].fillna(0)
    return profiles.sort_values(["team", "scoring_weight", "goals"], ascending=[True, False, False]).reset_index(drop=True)


def build_team_profiles(
    results: pd.DataFrame,
    report_team_stats: pd.DataFrame | None = None,
    statsbomb_team_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    results = _coerce_results(results)
    if results.empty:
        return pd.DataFrame(columns=["team", "rating"])

    ratings = compute_elo_ratings(results)
    team_matches = team_match_frame(results)
    career = team_matches.groupby("team").agg(matches=("team", "size")).reset_index()
    recent = (
        team_matches.sort_values(["team", "date"])
        .groupby("team", group_keys=False)
        .tail(12)
        .groupby("team")
        .agg(
            recent_matches=("team", "size"),
            recent_points_per_match=("points", "mean"),
            recent_goals_for=("goals_for", "mean"),
            recent_goals_against=("goals_against", "mean"),
        )
        .reset_index()
    )
    report_features = _report_aggregates(report_team_stats)
    statsbomb_features = _statsbomb_aggregates(statsbomb_team_stats)
    weighted_recent = recency_weighted_team_form(team_matches)

    profiles = ratings.merge(career, on="team", how="left").merge(recent, on="team", how="left")
    profiles = profiles.merge(weighted_recent, on="team", how="left")
    profiles = profiles.merge(report_features, on="team", how="left")
    profiles = profiles.merge(statsbomb_features, on="team", how="left")
    numeric_columns = [column for column in profiles.columns if column != "team"]
    profiles[numeric_columns] = profiles[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    profiles = _add_shrunk_xg_features(profiles)
    return profiles.sort_values("rating", ascending=False).reset_index(drop=True)


def _safe_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(numeric_value):
        return float(default)
    return numeric_value


def _profile_for_team(profiles: pd.DataFrame, team: str) -> dict[str, float]:
    if profiles is not None and not profiles.empty and team in set(profiles["team"]):
        row = profiles.loc[profiles["team"] == team].iloc[0].to_dict()
    else:
        row = {"team": team, "rating": DEFAULT_RATING}

    recent_points = _safe_float(row.get("recent_points_per_match"), 1.3)
    recent_goals_for = _safe_float(row.get("recent_goals_for"), 1.2)
    recent_goals_against = _safe_float(row.get("recent_goals_against"), 1.2)
    defaults = {
        "rating": DEFAULT_RATING,
        "recent_points_per_match": recent_points,
        "recent_goals_for": recent_goals_for,
        "recent_goals_against": recent_goals_against,
        "weighted_recent_points_per_match": recent_points,
        "weighted_recent_goals_for": recent_goals_for,
        "weighted_recent_goals_against": recent_goals_against,
        "weighted_recent_goal_diff": recent_goals_for - recent_goals_against,
        "recent_result_weight": 0.0,
        "xg_for_2026": 0.0,
        "xg_against_2026": 0.0,
        "report_matches": 0.0,
        "statsbomb_xg_for": 0.0,
        "statsbomb_xg_against": 0.0,
        "statsbomb_matches": 0.0,
    }
    for key, default in defaults.items():
        row[key] = _safe_float(row.get(key, default), default)
    row["xg_for_2026_shrunk"] = _safe_float(
        row.get("xg_for_2026_shrunk"),
        row["xg_for_2026"] if row["report_matches"] > 0 else XG_PRIOR,
    )
    row["xg_against_2026_shrunk"] = _safe_float(
        row.get("xg_against_2026_shrunk"),
        row["xg_against_2026"] if row["report_matches"] > 0 else XG_PRIOR,
    )
    row["statsbomb_xg_for_shrunk"] = _safe_float(
        row.get("statsbomb_xg_for_shrunk"),
        row["statsbomb_xg_for"] if row["statsbomb_matches"] > 0 else XG_PRIOR,
    )
    row["statsbomb_xg_against_shrunk"] = _safe_float(
        row.get("statsbomb_xg_against_shrunk"),
        row["statsbomb_xg_against"] if row["statsbomb_matches"] > 0 else XG_PRIOR,
    )
    return row


def model_factor_frame(team_a: str, team_b: str, profiles: pd.DataFrame) -> pd.DataFrame:
    profile_a = _profile_for_team(profiles, team_a)
    profile_b = _profile_for_team(profiles, team_b)
    rows = [
        {
            "factor": "Rating",
            team_a: profile_a["rating"],
            team_b: profile_b["rating"],
            "edge": profile_a["rating"] - profile_b["rating"],
        },
        {
            "factor": "Weighted recent points / match",
            team_a: profile_a["weighted_recent_points_per_match"],
            team_b: profile_b["weighted_recent_points_per_match"],
            "edge": profile_a["weighted_recent_points_per_match"] - profile_b["weighted_recent_points_per_match"],
        },
        {
            "factor": "Weighted recent goal diff",
            team_a: profile_a["weighted_recent_goal_diff"],
            team_b: profile_b["weighted_recent_goal_diff"],
            "edge": profile_a["weighted_recent_goal_diff"] - profile_b["weighted_recent_goal_diff"],
        },
        {
            "factor": "Recent points / match",
            team_a: profile_a["recent_points_per_match"],
            team_b: profile_b["recent_points_per_match"],
            "edge": profile_a["recent_points_per_match"] - profile_b["recent_points_per_match"],
        },
        {
            "factor": "Recent goals for",
            team_a: profile_a["recent_goals_for"],
            team_b: profile_b["recent_goals_for"],
            "edge": profile_a["recent_goals_for"] - profile_b["recent_goals_for"],
        },
        {
            "factor": "Recent goals against",
            team_a: profile_a["recent_goals_against"],
            team_b: profile_b["recent_goals_against"],
            "edge": profile_b["recent_goals_against"] - profile_a["recent_goals_against"],
        },
        {
            "factor": "2026 xG for (shrunk)",
            team_a: profile_a["xg_for_2026_shrunk"],
            team_b: profile_b["xg_for_2026_shrunk"],
            "edge": profile_a["xg_for_2026_shrunk"] - profile_b["xg_for_2026_shrunk"],
        },
        {
            "factor": "2026 xG against (shrunk)",
            team_a: profile_a["xg_against_2026_shrunk"],
            team_b: profile_b["xg_against_2026_shrunk"],
            "edge": profile_b["xg_against_2026_shrunk"] - profile_a["xg_against_2026_shrunk"],
        },
        {
            "factor": "StatsBomb open xG for (shrunk)",
            team_a: profile_a["statsbomb_xg_for_shrunk"],
            team_b: profile_b["statsbomb_xg_for_shrunk"],
            "edge": profile_a["statsbomb_xg_for_shrunk"] - profile_b["statsbomb_xg_for_shrunk"],
        },
        {
            "factor": "StatsBomb open xG against (shrunk)",
            team_a: profile_a["statsbomb_xg_against_shrunk"],
            team_b: profile_b["statsbomb_xg_against_shrunk"],
            "edge": profile_b["statsbomb_xg_against_shrunk"] - profile_a["statsbomb_xg_against_shrunk"],
        },
    ]
    return pd.DataFrame(rows)


def _home_component(team_a: str, team_b: str, neutral: bool, home_team: str | None = None) -> float:
    if neutral:
        return 0.0
    canonical_home = canonical_team_name(home_team or team_a)
    if canonical_home == canonical_team_name(team_a):
        return HOME_GOAL_ADVANTAGE_COMPONENT
    if canonical_home == canonical_team_name(team_b):
        return -HOME_GOAL_ADVANTAGE_COMPONENT
    return 0.0


def _shrunk_observed_rate(
    value: Any,
    evidence_weight: Any,
    prior: float = XG_PRIOR,
    prior_matches: float = ATTACK_DEFENSE_PRIOR_MATCHES,
) -> float:
    numeric_value = _safe_float(value, prior)
    numeric_weight = max(_safe_float(evidence_weight, 0.0), 0.0)
    weight = numeric_weight / (numeric_weight + max(float(prior_matches), 1.0))
    return prior + weight * (numeric_value - prior)


def _attack_defense_rates(profile: dict[str, float]) -> tuple[float, float]:
    evidence_weight = profile.get("recent_result_weight", 0.0)
    attack_rate = _shrunk_observed_rate(
        profile.get("weighted_recent_goals_for", profile.get("recent_goals_for", XG_PRIOR)),
        evidence_weight,
    )
    defense_allowed_rate = _shrunk_observed_rate(
        profile.get("weighted_recent_goals_against", profile.get("recent_goals_against", XG_PRIOR)),
        evidence_weight,
    )

    report_matches = max(_safe_float(profile.get("report_matches"), 0.0), 0.0)
    if report_matches > 0:
        report_weight = min(report_matches / REPORT_XG_SHRINKAGE_MATCHES, 1.0) * REPORT_XG_ATTACK_DEFENSE_WEIGHT
        attack_rate = (1 - report_weight) * attack_rate + report_weight * profile["xg_for_2026_shrunk"]
        defense_allowed_rate = (
            (1 - report_weight) * defense_allowed_rate + report_weight * profile["xg_against_2026_shrunk"]
        )

    statsbomb_matches = max(_safe_float(profile.get("statsbomb_matches"), 0.0), 0.0)
    if statsbomb_matches > 0:
        statsbomb_weight = (
            min(statsbomb_matches / STATSBOMB_XG_SHRINKAGE_MATCHES, 1.0) * STATSBOMB_XG_ATTACK_DEFENSE_WEIGHT
        )
        attack_rate = (1 - statsbomb_weight) * attack_rate + statsbomb_weight * profile["statsbomb_xg_for_shrunk"]
        defense_allowed_rate = (
            (1 - statsbomb_weight) * defense_allowed_rate + statsbomb_weight * profile["statsbomb_xg_against_shrunk"]
        )

    return _clamp(attack_rate, 0.2, 4.8), _clamp(defense_allowed_rate, 0.2, 4.8)


def attack_defense_expected_goals_breakdown(
    team_a: str,
    team_b: str,
    profiles: pd.DataFrame,
    neutral: bool = True,
    home_team: str | None = None,
) -> dict[str, float]:
    profile_a = _profile_for_team(profiles, team_a)
    profile_b = _profile_for_team(profiles, team_b)
    team_a_attack_rate, team_a_defense_allowed_rate = _attack_defense_rates(profile_a)
    team_b_attack_rate, team_b_defense_allowed_rate = _attack_defense_rates(profile_b)

    team_a_goals = XG_PRIOR * (team_a_attack_rate / XG_PRIOR) ** ATTACK_DEFENSE_ATTACK_POWER
    team_a_goals *= (team_b_defense_allowed_rate / XG_PRIOR) ** ATTACK_DEFENSE_DEFENSE_POWER
    team_b_goals = XG_PRIOR * (team_b_attack_rate / XG_PRIOR) ** ATTACK_DEFENSE_ATTACK_POWER
    team_b_goals *= (team_a_defense_allowed_rate / XG_PRIOR) ** ATTACK_DEFENSE_DEFENSE_POWER

    home_component = _home_component(team_a, team_b, neutral=neutral, home_team=home_team)
    if home_component > 0:
        team_a_goals *= math.exp(home_component)
        team_b_goals *= math.exp(-home_component / 2)
    elif home_component < 0:
        team_a_goals *= math.exp(home_component / 2)
        team_b_goals *= math.exp(-home_component)

    team_a_goals = _clamp(team_a_goals, 0.2, 4.8)
    team_b_goals = _clamp(team_b_goals, 0.2, 4.8)
    return {
        "team_a_attack_rate": team_a_attack_rate,
        "team_a_defense_allowed_rate": team_a_defense_allowed_rate,
        "team_b_attack_rate": team_b_attack_rate,
        "team_b_defense_allowed_rate": team_b_defense_allowed_rate,
        "team_a_expected_goals": team_a_goals,
        "team_b_expected_goals": team_b_goals,
    }


def _structural_expected_goals_breakdown(
    team_a: str,
    team_b: str,
    profiles: pd.DataFrame,
    neutral: bool = True,
    home_team: str | None = None,
) -> dict[str, float]:
    profile_a = _profile_for_team(profiles, team_a)
    profile_b = _profile_for_team(profiles, team_b)

    rating_component = (profile_a["rating"] - profile_b["rating"]) / 750.0
    recent_component = 0.24 * (
        profile_a["weighted_recent_points_per_match"] - profile_b["weighted_recent_points_per_match"]
    )
    goal_form_component = 0.12 * (profile_a["weighted_recent_goal_diff"] - profile_b["weighted_recent_goal_diff"])

    xg_component = 0.0
    if profile_a["report_matches"] > 0 or profile_b["report_matches"] > 0:
        xg_component = 0.14 * (
            (profile_a["xg_for_2026_shrunk"] - profile_a["xg_against_2026_shrunk"])
            - (profile_b["xg_for_2026_shrunk"] - profile_b["xg_against_2026_shrunk"])
        )

    open_data_component = 0.0
    if profile_a["statsbomb_matches"] > 0 or profile_b["statsbomb_matches"] > 0:
        open_data_component = 0.06 * (
            (profile_a["statsbomb_xg_for_shrunk"] - profile_a["statsbomb_xg_against_shrunk"])
            - (profile_b["statsbomb_xg_for_shrunk"] - profile_b["statsbomb_xg_against_shrunk"])
        )

    home_component = _home_component(team_a, team_b, neutral=neutral, home_team=home_team)
    advantage = _clamp(
        rating_component + recent_component + goal_form_component + xg_component + open_data_component + home_component,
        -1.15,
        1.15,
    )
    base_goals = 1.28
    team_a_goals = _clamp(base_goals * math.exp(advantage), 0.2, 4.8)
    team_b_goals = _clamp(base_goals * math.exp(-advantage), 0.2, 4.8)
    return {
        "rating_component": rating_component,
        "recent_component": recent_component,
        "goal_form_component": goal_form_component,
        "xg_component": xg_component,
        "open_data_component": open_data_component,
        "home_component": home_component,
        "advantage": advantage,
        "team_a_expected_goals": team_a_goals,
        "team_b_expected_goals": team_b_goals,
    }


def expected_goals_breakdown(
    team_a: str,
    team_b: str,
    profiles: pd.DataFrame,
    neutral: bool = True,
    home_team: str | None = None,
) -> dict[str, float]:
    structural = _structural_expected_goals_breakdown(
        team_a,
        team_b,
        profiles,
        neutral=neutral,
        home_team=home_team,
    )
    attack_defense = attack_defense_expected_goals_breakdown(
        team_a,
        team_b,
        profiles,
        neutral=neutral,
        home_team=home_team,
    )
    weight = _clamp(ATTACK_DEFENSE_ENSEMBLE_WEIGHT, 0.0, 1.0)
    team_a_goals = (
        (1 - weight) * structural["team_a_expected_goals"] + weight * attack_defense["team_a_expected_goals"]
    )
    team_b_goals = (
        (1 - weight) * structural["team_b_expected_goals"] + weight * attack_defense["team_b_expected_goals"]
    )

    breakdown = structural.copy()
    breakdown.update(
        {
            "structural_advantage": structural["advantage"],
            "structural_team_a_expected_goals": structural["team_a_expected_goals"],
            "structural_team_b_expected_goals": structural["team_b_expected_goals"],
            "attack_defense_team_a_expected_goals": attack_defense["team_a_expected_goals"],
            "attack_defense_team_b_expected_goals": attack_defense["team_b_expected_goals"],
            "team_a_attack_rate": attack_defense["team_a_attack_rate"],
            "team_a_defense_allowed_rate": attack_defense["team_a_defense_allowed_rate"],
            "team_b_attack_rate": attack_defense["team_b_attack_rate"],
            "team_b_defense_allowed_rate": attack_defense["team_b_defense_allowed_rate"],
            "ensemble_attack_defense_weight": weight,
            "team_a_expected_goals": team_a_goals,
            "team_b_expected_goals": team_b_goals,
            "advantage": 0.5 * math.log(team_a_goals / team_b_goals),
        }
    )
    return breakdown


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def expected_goals(
    team_a: str,
    team_b: str,
    profiles: pd.DataFrame,
    neutral: bool = True,
    home_team: str | None = None,
) -> tuple[float, float]:
    breakdown = expected_goals_breakdown(team_a, team_b, profiles, neutral=neutral, home_team=home_team)
    return breakdown["team_a_expected_goals"], breakdown["team_b_expected_goals"]


def poisson_probabilities(lam: float, max_goals: int = 7) -> np.ndarray:
    lam = max(float(lam), 0.0)
    probabilities = np.array([math.exp(-lam) * lam**goals / math.factorial(goals) for goals in range(max_goals + 1)])
    return probabilities / probabilities.sum()


def poisson_tail_probability(lam: float, max_goals: int) -> float:
    lam = max(float(lam), 0.0)
    cumulative = sum(math.exp(-lam) * lam**goals / math.factorial(goals) for goals in range(max_goals + 1))
    return max(0.0, 1.0 - cumulative)


def scoreline_max_goals(*expected_goals_values: float) -> int:
    max_lam = max([max(float(value), 0.0) for value in expected_goals_values] or [0.0])
    max_goals = MIN_SCORELINE_MAX_GOALS
    while (
        max_goals < MAX_SCORELINE_MAX_GOALS
        and poisson_tail_probability(max_lam, max_goals) > POISSON_TAIL_PROBABILITY
    ):
        max_goals += 1
    return max_goals


def scoreline_matrix(team_a_goals: float, team_b_goals: float, max_goals: int | None = None) -> pd.DataFrame:
    if max_goals is None:
        max_goals = scoreline_max_goals(team_a_goals, team_b_goals)
    team_a_distribution = poisson_probabilities(team_a_goals, max_goals=max_goals)
    team_b_distribution = poisson_probabilities(team_b_goals, max_goals=max_goals)
    matrix = np.outer(team_a_distribution, team_b_distribution)
    matrix = matrix / matrix.sum()
    return pd.DataFrame(matrix, index=range(max_goals + 1), columns=range(max_goals + 1))


def goal_total_probabilities(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    values = matrix.to_numpy()
    for team_a_goals in range(values.shape[0]):
        for team_b_goals in range(values.shape[1]):
            rows.append(
                {
                    "team_a_goals": team_a_goals,
                    "team_b_goals": team_b_goals,
                    "total_goals": team_a_goals + team_b_goals,
                    "probability": float(values[team_a_goals, team_b_goals]),
                }
            )
    return pd.DataFrame(rows).groupby("total_goals", as_index=False)["probability"].sum()


def predict_player_scorers(
    team: str,
    team_expected_goals: float,
    player_profiles: pd.DataFrame | None,
    max_players: int = 10,
    scorer_capture_rate: float = 0.88,
) -> pd.DataFrame:
    if player_profiles is None or player_profiles.empty:
        return pd.DataFrame(
            columns=[
                "team",
                "player",
                "expected_goals",
                "score_probability",
                "first_team_goal_probability",
                "goals",
                "recent_goals_24m",
                "penalty_goals",
                "scoring_weight",
                "last_goal_date",
            ]
        )

    canonical_team = canonical_team_name(team)
    candidates = player_profiles[player_profiles["team"].map(canonical_team_name).eq(canonical_team)].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["team", "player", "expected_goals", "score_probability", "last_goal_date"])

    if "last_goal_date" in candidates and "last_goal_date" in player_profiles:
        candidate_dates = pd.to_datetime(candidates["last_goal_date"], errors="coerce")
        all_dates = pd.to_datetime(player_profiles["last_goal_date"], errors="coerce")
        max_date = all_dates.max()
        if pd.notna(max_date):
            active_cutoff = max_date - pd.Timedelta(days=1460)
            active_candidates = candidates[candidate_dates.ge(active_cutoff)]
            if len(active_candidates) >= 2:
                candidates = active_candidates

    for column in ("scoring_weight", "recent_goals_24m", "goals", "penalty_goals"):
        if column not in candidates:
            candidates[column] = 0
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce").fillna(0)

    candidates["candidate_weight"] = (
        candidates["scoring_weight"]
        + 0.35 * candidates["recent_goals_24m"]
        + 0.08 * candidates["goals"]
        + 0.12 * candidates["penalty_goals"]
    )
    candidates = candidates[candidates["candidate_weight"] > 0]
    if candidates.empty:
        return pd.DataFrame(columns=["team", "player", "expected_goals", "score_probability", "last_goal_date"])

    candidates = candidates.sort_values(["candidate_weight", "recent_goals_24m", "goals"], ascending=False).head(max_players)
    total_weight = candidates["candidate_weight"].sum()
    allocated_goals = max(0.0, team_expected_goals * scorer_capture_rate)
    candidates["expected_goals"] = allocated_goals * candidates["candidate_weight"] / total_weight
    candidates["score_probability"] = 1 - np.exp(-candidates["expected_goals"])
    candidates["first_team_goal_probability"] = candidates["candidate_weight"] / total_weight
    if "last_goal_date" not in candidates:
        candidates["last_goal_date"] = ""
    output_columns = [
        "team",
        "player",
        "expected_goals",
        "score_probability",
        "first_team_goal_probability",
        "goals",
        "recent_goals_24m",
        "penalty_goals",
        "scoring_weight",
        "last_goal_date",
    ]
    return candidates[output_columns].reset_index(drop=True)


def predict_matchup(
    team_a: str,
    team_b: str,
    profiles: pd.DataFrame,
    player_profiles: pd.DataFrame | None = None,
    max_goals: int | None = None,
    neutral: bool = True,
    home_team: str | None = None,
) -> dict[str, Any]:
    if team_a == team_b:
        raise ValueError("Select two different teams.")

    breakdown = expected_goals_breakdown(team_a, team_b, profiles, neutral=neutral, home_team=home_team)
    team_a_goals = breakdown["team_a_expected_goals"]
    team_b_goals = breakdown["team_b_expected_goals"]
    matrix = scoreline_matrix(team_a_goals, team_b_goals, max_goals=max_goals)
    values = matrix.to_numpy()
    team_a_win = float(np.tril(values, k=-1).sum())
    draw = float(np.trace(values))
    team_b_win = float(np.triu(values, k=1).sum())
    best_index = np.unravel_index(values.argmax(), values.shape)
    team_a_scorers = predict_player_scorers(team_a, team_a_goals, player_profiles)
    team_b_scorers = predict_player_scorers(team_b, team_b_goals, player_profiles)
    player_scoring = pd.concat([team_a_scorers, team_b_scorers], ignore_index=True)

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_expected_goals": team_a_goals,
        "team_b_expected_goals": team_b_goals,
        "team_a_win_probability": team_a_win,
        "draw_probability": draw,
        "team_b_win_probability": team_b_win,
        "most_likely_score": f"{best_index[0]}-{best_index[1]}",
        "score_matrix": matrix,
        "goal_total_probabilities": goal_total_probabilities(matrix),
        "factor_frame": model_factor_frame(team_a, team_b, profiles),
        "player_scoring": player_scoring,
        "breakdown": breakdown,
    }

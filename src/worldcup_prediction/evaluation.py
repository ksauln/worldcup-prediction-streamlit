from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from worldcup_prediction.model import build_team_profiles, predict_matchup
from worldcup_prediction.teams import canonical_team_name

EVALUATION_COLUMNS = [
    "date",
    "tournament",
    "home_team",
    "away_team",
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
    "outcome_log_loss",
    "score_log_loss",
    "brier_score",
    "home_expected_goals",
    "away_expected_goals",
    "training_matches",
]


def prediction_audit_metric_guide() -> list[dict[str, str]]:
    """Return plain-language guidance for interpreting the prediction audit."""
    return [
        {
            "metric": "Matches audited",
            "meaning": "Completed matches included in the selected rolling evaluation.",
            "direction": "More is stronger evidence",
            "reference": "Small samples can move sharply after one match.",
            "how_to_read": "Use this to judge how much confidence to place in every other summary score.",
        },
        {
            "metric": "Outcome accuracy",
            "meaning": "Share of matches where the highest-probability home, draw, or away outcome occurred.",
            "direction": "Higher is better",
            "reference": "Compare with simple favorite and rating baselines on the same matches.",
            "how_to_read": "Useful for picks, but it ignores whether a correct forecast was 36% or 90% confident.",
        },
        {
            "metric": "Exact score hit",
            "meaning": "Share of matches where the single most likely scoreline exactly matched the result.",
            "direction": "Higher is better",
            "reference": "Exact-score hit rates are naturally much lower than outcome accuracy.",
            "how_to_read": "Do not treat the predicted score as the only plausible result; inspect the score matrix too.",
        },
        {
            "metric": "Avg actual-result probability",
            "meaning": "Average probability the model assigned to the home, draw, or away outcome that occurred.",
            "direction": "Higher is better",
            "reference": "A uniform three-outcome forecast assigns 33.3% to every actual result.",
            "how_to_read": "Rewards forecasts that consistently place substantial probability on what happens.",
        },
        {
            "metric": "Home / draw / away probabilities",
            "meaning": "The model's three mutually exclusive regulation-result probabilities for an individual match.",
            "direction": "Must total 100%",
            "reference": "The largest probability is the predicted outcome, not a guarantee.",
            "how_to_read": "A 45% favorite is still expected not to win 55% of the time.",
        },
        {
            "metric": "Actual-outcome probability",
            "meaning": "Probability assigned to the outcome that actually occurred in one audited match.",
            "direction": "Higher is better",
            "reference": "Read across many matches rather than judging the model from one upset.",
            "how_to_read": "Low values identify surprises or confident misses; repeated low values indicate poor calibration.",
        },
        {
            "metric": "Brier score",
            "meaning": "Sum of squared errors across the home, draw, and away probabilities.",
            "direction": "Lower is better",
            "reference": "0 is perfect; 2 is worst; a uniform 1X2 forecast scores 0.667.",
            "how_to_read": "Measures probability calibration and accuracy together; confident wrong forecasts are penalized more.",
        },
        {
            "metric": "Outcome log loss",
            "meaning": "Negative log of the probability assigned to the actual home, draw, or away outcome.",
            "direction": "Lower is better",
            "reference": "0 is perfect; there is no upper limit; a uniform 1X2 forecast scores 1.099.",
            "how_to_read": "Especially punishes assigning tiny probability to an outcome that occurs.",
        },
        {
            "metric": "Score log loss",
            "meaning": "Negative log of the probability assigned to the exact final scoreline.",
            "direction": "Lower is better",
            "reference": "0 is perfect and there is no upper limit; compare models on the same matches.",
            "how_to_read": "Evaluates the full score distribution, so it is more informative than exact-score hit rate alone.",
        },
        {
            "metric": "Predicted score",
            "meaning": "The single scoreline with the highest probability in the model's score matrix.",
            "direction": "Descriptive, not a quality score",
            "reference": "Several neighboring scorelines may have nearly identical probabilities.",
            "how_to_read": "Use it as a summary of the distribution, not as a claim that this score is likely by itself.",
        },
        {
            "metric": "Expected goals (xG)",
            "meaning": "Model-implied average goals for each team across many repetitions of the matchup.",
            "direction": "Descriptive, not a quality score",
            "reference": "An expectation of 1.6 goals does not mean the team will score exactly 1.6 or even exactly 2.",
            "how_to_read": "Compare the two teams and their total to understand strength and expected match tempo.",
        },
        {
            "metric": "Training matches",
            "meaning": "Number of earlier result rows available when that audited match was predicted.",
            "direction": "Context only",
            "reference": "More history is not automatically better if it is stale or less relevant.",
            "how_to_read": "Very small histories make team ratings and form estimates less stable.",
        },
    ]


def empty_evaluation_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EVALUATION_COLUMNS)


def _coerce_match_results(results: pd.DataFrame | None) -> pd.DataFrame:
    if results is None or results.empty:
        return pd.DataFrame()

    rows = results.copy()
    for column in ("home_team", "away_team"):
        rows[column] = rows[column].map(canonical_team_name)
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows["home_score"] = pd.to_numeric(rows["home_score"], errors="coerce")
    rows["away_score"] = pd.to_numeric(rows["away_score"], errors="coerce")
    if "neutral" not in rows:
        rows["neutral"] = True
    rows["neutral"] = rows["neutral"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    if "tournament" not in rows:
        rows["tournament"] = ""
    rows = rows.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    rows["home_score"] = rows["home_score"].astype(int)
    rows["away_score"] = rows["away_score"].astype(int)
    return rows.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def _filter_dated_frame(
    frame: pd.DataFrame | None,
    before_date: pd.Timestamp,
    on_or_after_date: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    if frame is None or frame.empty or "date" not in frame:
        return frame
    rows = frame.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    if on_or_after_date is not None and pd.notna(on_or_after_date):
        rows = rows[rows["date"].ge(on_or_after_date)]
    return rows[rows["date"].lt(before_date)].copy()


def _actual_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "Home"
    if away_score > home_score:
        return "Away"
    return "Draw"


def _predicted_outcome(home_probability: float, draw_probability: float, away_probability: float) -> str:
    values = {
        "Home": float(home_probability),
        "Draw": float(draw_probability),
        "Away": float(away_probability),
    }
    return max(values, key=values.get)


def _score_probability(prediction: dict[str, Any], home_score: int, away_score: int) -> float:
    matrix = prediction.get("score_matrix")
    if not isinstance(matrix, pd.DataFrame) or matrix.empty:
        return 1e-12
    if home_score in matrix.index and away_score in matrix.columns:
        return max(float(matrix.loc[home_score, away_score]), 1e-12)
    return 1e-12


def rolling_match_predictions(
    results: pd.DataFrame,
    report_team_stats: pd.DataFrame | None = None,
    statsbomb_team_stats: pd.DataFrame | None = None,
    tournament: str | None = None,
    max_matches: int = 60,
    training_start: str | pd.Timestamp | None = "2018-01-01",
    min_training_matches: int = 50,
) -> pd.DataFrame:
    matches = _coerce_match_results(results)
    if matches.empty:
        return empty_evaluation_frame()

    evaluation_pool = matches
    if tournament:
        evaluation_pool = evaluation_pool[evaluation_pool["tournament"].astype(str).eq(tournament)]
    evaluation_pool = evaluation_pool.tail(max(int(max_matches), 1))
    if evaluation_pool.empty:
        return empty_evaluation_frame()

    training_start_date = pd.to_datetime(training_start, errors="coerce") if training_start is not None else pd.NaT
    rows: list[dict[str, Any]] = []
    for match in evaluation_pool.itertuples(index=False):
        prior_matches = matches[matches["date"].lt(match.date)]
        if pd.notna(training_start_date):
            prior_matches = prior_matches[prior_matches["date"].ge(training_start_date)]
        if len(prior_matches) < min_training_matches:
            continue

        prior_report_stats = _filter_dated_frame(report_team_stats, match.date, training_start_date)
        prior_statsbomb_stats = _filter_dated_frame(statsbomb_team_stats, match.date, training_start_date)
        profiles = build_team_profiles(
            prior_matches,
            report_team_stats=prior_report_stats,
            statsbomb_team_stats=prior_statsbomb_stats,
        )
        if profiles.empty:
            continue

        neutral = bool(match.neutral)
        home_team = None if neutral else str(match.home_team)
        prediction = predict_matchup(
            str(match.home_team),
            str(match.away_team),
            profiles,
            player_profiles=None,
            neutral=neutral,
            home_team=home_team,
        )

        home_probability = float(prediction["team_a_win_probability"])
        draw_probability = float(prediction["draw_probability"])
        away_probability = float(prediction["team_b_win_probability"])
        actual_outcome = _actual_outcome(int(match.home_score), int(match.away_score))
        predicted_outcome = _predicted_outcome(home_probability, draw_probability, away_probability)
        actual_probability = {
            "Home": home_probability,
            "Draw": draw_probability,
            "Away": away_probability,
        }[actual_outcome]
        outcome_targets = np.array(
            [
                1.0 if actual_outcome == "Home" else 0.0,
                1.0 if actual_outcome == "Draw" else 0.0,
                1.0 if actual_outcome == "Away" else 0.0,
            ]
        )
        outcome_probabilities = np.array([home_probability, draw_probability, away_probability])
        exact_score_probability = _score_probability(prediction, int(match.home_score), int(match.away_score))

        rows.append(
            {
                "date": match.date,
                "tournament": str(match.tournament),
                "home_team": str(match.home_team),
                "away_team": str(match.away_team),
                "actual_score": f"{int(match.home_score)}-{int(match.away_score)}",
                "predicted_score": prediction["most_likely_score"],
                "actual_outcome": actual_outcome,
                "predicted_outcome": predicted_outcome,
                "prediction_correct": predicted_outcome == actual_outcome,
                "exact_score_hit": prediction["most_likely_score"] == f"{int(match.home_score)}-{int(match.away_score)}",
                "home_win_probability": home_probability,
                "draw_probability": draw_probability,
                "away_win_probability": away_probability,
                "actual_outcome_probability": actual_probability,
                "exact_score_probability": exact_score_probability,
                "outcome_log_loss": -math.log(max(actual_probability, 1e-12)),
                "score_log_loss": -math.log(exact_score_probability),
                "brier_score": float(np.sum((outcome_probabilities - outcome_targets) ** 2)),
                "home_expected_goals": float(prediction["team_a_expected_goals"]),
                "away_expected_goals": float(prediction["team_b_expected_goals"]),
                "training_matches": len(prior_matches),
            }
        )

    if not rows:
        return empty_evaluation_frame()
    return pd.DataFrame(rows, columns=EVALUATION_COLUMNS).sort_values("date", ascending=False).reset_index(drop=True)


def prediction_performance_summary(evaluation: pd.DataFrame) -> dict[str, float]:
    if evaluation is None or evaluation.empty:
        return {
            "matches": 0,
            "outcome_accuracy": 0.0,
            "exact_score_accuracy": 0.0,
            "average_actual_outcome_probability": 0.0,
            "average_brier_score": 0.0,
            "average_outcome_log_loss": 0.0,
            "average_score_log_loss": 0.0,
        }

    return {
        "matches": float(len(evaluation)),
        "outcome_accuracy": float(evaluation["prediction_correct"].mean()),
        "exact_score_accuracy": float(evaluation["exact_score_hit"].mean()),
        "average_actual_outcome_probability": float(evaluation["actual_outcome_probability"].mean()),
        "average_brier_score": float(evaluation["brier_score"].mean()),
        "average_outcome_log_loss": float(evaluation["outcome_log_loss"].mean()),
        "average_score_log_loss": float(evaluation["score_log_loss"].mean()),
    }

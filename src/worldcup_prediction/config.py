from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORT_PDF_DIR = RAW_DATA_DIR / "fifa_reports"

FIFA_HUB_URL = "https://www.fifatrainingcentre.com/en/fifa-world-cup-2026/match-report-hub.php"
HISTORICAL_RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
GOALSCORERS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
STATSBOMB_COMPETITIONS_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"
STATSBOMB_MATCHES_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches"
STATSBOMB_EVENTS_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events"

HUB_HTML_PATH = RAW_DATA_DIR / "fifa_match_report_hub.html"
HISTORICAL_RESULTS_PATH = RAW_DATA_DIR / "historical_results.csv"
GOALSCORERS_PATH = RAW_DATA_DIR / "goalscorers.csv"
STATSBOMB_DIR = RAW_DATA_DIR / "statsbomb"
STATSBOMB_COMPETITIONS_PATH = STATSBOMB_DIR / "competitions.json"
STATSBOMB_MATCHES_DIR = STATSBOMB_DIR / "matches"
STATSBOMB_EVENTS_DIR = STATSBOMB_DIR / "events"
REPORT_MATCHES_PATH = PROCESSED_DATA_DIR / "fifa_2026_report_matches.csv"
REPORT_TEAM_STATS_PATH = PROCESSED_DATA_DIR / "fifa_2026_report_team_stats.csv"
STATSBOMB_TEAM_STATS_PATH = PROCESSED_DATA_DIR / "statsbomb_team_stats.csv"
COMBINED_RESULTS_PATH = PROCESSED_DATA_DIR / "combined_results.csv"
TEAM_PROFILES_PATH = PROCESSED_DATA_DIR / "team_profiles.csv"
PLAYER_PROFILES_PATH = PROCESSED_DATA_DIR / "player_profiles.csv"
BOOKMAKER_ODDS_PATH = PROCESSED_DATA_DIR / "bookmaker_odds.csv"
BOOKMAKER_ODDS_METADATA_PATH = PROCESSED_DATA_DIR / "bookmaker_odds_metadata.json"
METADATA_PATH = PROCESSED_DATA_DIR / "metadata.json"

HTTP_TIMEOUT_SECONDS = 45
USER_AGENT = "WorldCupPrediction/0.1 (+https://www.fifatrainingcentre.com)"
ODDS_API_BASE_URL = "https://api.the-odds-api.com"
ODDS_API_KEY_ENV = "ODDS_API_KEY"
DEFAULT_ODDS_SPORT_KEY = "soccer_fifa_world_cup"
DEFAULT_ODDS_REGIONS = "us,us2"
ESPN_WORLD_CUP_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
DEFAULT_PUBLIC_ODDS_DATE_RANGE = "20260611-20260719"
DEFAULT_ODDS_BOOKMAKERS = [
    "draftkings",
    "fanduel",
    "betmgm",
    "betrivers",
    "williamhill_us",
    "espnbet",
    "fanatics",
    "hardrockbet",
]


def ensure_data_dirs() -> None:
    """Create data folders used by the refresh pipeline."""
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORT_PDF_DIR, STATSBOMB_DIR, STATSBOMB_MATCHES_DIR, STATSBOMB_EVENTS_DIR):
        path.mkdir(parents=True, exist_ok=True)

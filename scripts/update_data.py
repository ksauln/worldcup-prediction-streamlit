from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup_prediction.pipeline import refresh_data  # noqa: E402
from worldcup_prediction.sources import load_goalscorers, load_historical_results  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh FIFA World Cup report data and prediction inputs.")
    parser.add_argument("--force", action="store_true", help="Redownload cached source files.")
    parser.add_argument(
        "--refresh-source-csvs",
        action="store_true",
        help="Redownload historical results and goalscorers without forcing PDF or StatsBomb downloads.",
    )
    parser.add_argument("--max-reports", type=int, default=None, help="Limit the number of report PDFs for quick checks.")
    parser.add_argument("--skip-statsbomb", action="store_true", help="Skip optional StatsBomb Open Data refresh.")
    parser.add_argument(
        "--max-statsbomb-matches",
        type=int,
        default=80,
        help="Limit optional StatsBomb event matches to download and parse.",
    )
    args = parser.parse_args()

    if args.refresh_source_csvs and not args.force:
        load_historical_results(force=True)
        load_goalscorers(force=True)

    bundle = refresh_data(
        force=args.force,
        max_reports=args.max_reports,
        include_statsbomb=not args.skip_statsbomb,
        max_statsbomb_matches=args.max_statsbomb_matches,
    )
    metadata = bundle.metadata
    print("Data refresh complete")
    print(f"FIFA report links found: {metadata['report_links_found']}")
    print(f"Report matches parsed: {metadata['report_matches_parsed']}")
    print(f"Historical result rows: {metadata['historical_results_rows']}")
    print(f"Goalscorer rows: {metadata['goalscorer_rows']}")
    print(f"StatsBomb matches loaded: {metadata['statsbomb_matches_loaded']}")
    print(f"Combined result rows: {metadata['combined_results_rows']}")
    print(f"Team profiles: {metadata['team_profiles_rows']}")
    print(f"Player profiles: {metadata['player_profiles_rows']}")
    if metadata["parse_errors"]:
        print("Parse errors:")
        for error in metadata["parse_errors"]:
            print(f"- {error['file']}: {error['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

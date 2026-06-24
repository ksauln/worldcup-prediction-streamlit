from __future__ import annotations

import pandas as pd


SOURCE_CATALOG = [
    {
        "source": "StatsBomb Open Data",
        "url": "https://github.com/statsbomb/open-data",
        "automation": "Direct JSON",
        "status": "Integrated",
        "model_use": "Open event data xG and shot profiles for senior men's international matches.",
        "notes": "Best fit from the list: public GitHub data, documented JSON structure, no credentials.",
    },
    {
        "source": "Football-Data.co.uk",
        "url": "https://www.football-data.co.uk/data.php",
        "automation": "Direct CSV",
        "status": "Candidate",
        "model_use": "Could calibrate club-level market and result priors, but it is not a national-team feed.",
        "notes": "Useful for European club models; weaker fit for World Cup national-team prediction without squad mapping.",
    },
    {
        "source": "FBref",
        "url": "https://fbref.com",
        "automation": "Manual/export-oriented",
        "status": "Manual review",
        "model_use": "Could add player and club form if exports are downloaded manually.",
        "notes": "Rich data, but automated scraping needs terms/robots review and rate-limit care.",
    },
    {
        "source": "Understat",
        "url": "https://understat.com",
        "automation": "Unofficial scrape",
        "status": "Manual review",
        "model_use": "Could add club xG form for players if mapped to squads.",
        "notes": "Useful xG source for major clubs, but not direct national-team data.",
    },
    {
        "source": "API-Football",
        "url": "https://www.api-football.com",
        "automation": "API key",
        "status": "Needs credentials",
        "model_use": "Could add fixtures, lineups, injuries, live stats, and team/player data.",
        "notes": "Good structured API, but the free tier still requires an account and key.",
    },
    {
        "source": "Sofascore",
        "url": "https://www.sofascore.com",
        "automation": "No stable public data contract",
        "status": "Manual review",
        "model_use": "Could inform ratings, heatmaps, and lineups if an approved API/export is available.",
        "notes": "Excellent public UI, but not suitable for unattended scraping by default.",
    },
    {
        "source": "FotMob",
        "url": "https://www.fotmob.com",
        "automation": "No stable public data contract",
        "status": "Manual review",
        "model_use": "Could add player ratings and live match stats if an approved feed is available.",
        "notes": "Strong interface, but app endpoints are not a dependable public contract.",
    },
    {
        "source": "FootyStats.org",
        "url": "https://footystats.org",
        "automation": "Limited export/API",
        "status": "Candidate",
        "model_use": "Could add league-level probability priors.",
        "notes": "Potentially useful, but needs endpoint/license review before automating.",
    },
    {
        "source": "Futbolme",
        "url": "https://www.futbolme.com",
        "automation": "Website tables",
        "status": "Low priority",
        "model_use": "Mostly Spanish domestic/youth context; not a direct World Cup model input.",
        "notes": "Useful for local scouting, weaker fit for current national-team forecasts.",
    },
    {
        "source": "PlayeRank/Wyscout",
        "url": "https://github.com/mesosbrodleto/playerank",
        "automation": "Research/methodology",
        "status": "Reference",
        "model_use": "Can inspire player contribution modeling.",
        "notes": "Not a general current-data feed without Wyscout access.",
    },
    {
        "source": "WhoScored",
        "url": "https://www.whoscored.com",
        "automation": "No stable public data contract",
        "status": "Manual review",
        "model_use": "Could add ratings and player style data if an approved export/feed is available.",
        "notes": "Good public analysis site; automated scraping should be avoided without permission.",
    },
    {
        "source": "Transfermarkt",
        "url": "https://www.transfermarkt.com",
        "automation": "Website tables",
        "status": "Manual review",
        "model_use": "Could add squad value, age, injury, and market priors.",
        "notes": "Useful contextual data, but scraping needs terms review and careful throttling.",
    },
    {
        "source": "Opta Analyst",
        "url": "https://theanalyst.com",
        "automation": "Editorial/visual analysis",
        "status": "Reference",
        "model_use": "Useful as validation context, not raw data ingestion.",
        "notes": "Insight source rather than a bulk public dataset.",
    },
]


def source_catalog_frame() -> pd.DataFrame:
    return pd.DataFrame(SOURCE_CATALOG)

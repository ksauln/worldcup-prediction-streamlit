from __future__ import annotations

import re
import unicodedata


TEAM_ALIASES = {
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "cape verde": "Cabo Verde",
    "cabo verde": "Cabo Verde",
    "curacao": "Curaçao",
    "curaçao": "Curaçao",
    "czech republic": "Czechia",
    "czechia": "Czechia",
    "democratic republic of congo": "Congo DR",
    "dr congo": "Congo DR",
    "congo dr": "Congo DR",
    "iran": "IR Iran",
    "ir iran": "IR Iran",
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "côte d'ivoire": "Côte d'Ivoire",
    "south korea": "Korea Republic",
    "korea republic": "Korea Republic",
    "turkey": "Türkiye",
    "turkiye": "Türkiye",
    "türkiye": "Türkiye",
    "united states": "USA",
    "united states of america": "USA",
    "usa": "USA",
}


def _alias_key(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name or "")
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def canonical_team_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(name or "")).strip()
    return TEAM_ALIASES.get(_alias_key(cleaned), cleaned)

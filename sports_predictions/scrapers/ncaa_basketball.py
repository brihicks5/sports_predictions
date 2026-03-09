"""NCAA Basketball data importers.

Supports:
1. Kaggle March Machine Learning Mania CSV files (historical bulk import)
2. KenPom API for advanced metrics (adjusted efficiency, tempo, etc.)
3. ESPN API for current-season game results
"""

import csv
import os
from datetime import datetime
from pathlib import Path

import requests

from sports_predictions.db import (
    get_db, get_or_create_team, upsert_game, upsert_team_stat
)

SPORT = "ncaa_basketball"


def import_kaggle_games(csv_path: str):
    """Import game results from Kaggle's March Machine Learning Mania dataset.

    Expects the 'MRegularSeasonDetailedResults.csv' or
    'MRegularSeasonCompactResults.csv' format with columns:
    Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT
    """
    conn = get_db(SPORT)
    teams_csv = Path(csv_path).parent / "MTeams.csv"

    # Load team name mapping if available
    team_names = {}
    if teams_csv.exists():
        with open(teams_csv) as f:
            for row in csv.DictReader(f):
                team_names[row["TeamID"]] = row["TeamName"]

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            season = int(row["Season"])
            w_team_raw = row["WTeamID"]
            l_team_raw = row["LTeamID"]

            w_name = team_names.get(w_team_raw, f"Team_{w_team_raw}")
            l_name = team_names.get(l_team_raw, f"Team_{l_team_raw}")

            w_id = get_or_create_team(conn, w_name, source="kaggle")
            l_id = get_or_create_team(conn, l_name, source="kaggle")

            # WLoc: H = winner was home, A = winner was away, N = neutral
            wloc = row.get("WLoc", "N")
            if wloc == "H":
                home_id, away_id = w_id, l_id
                home_score, away_score = int(row["WScore"]), int(row["LScore"])
            elif wloc == "A":
                home_id, away_id = l_id, w_id
                home_score, away_score = int(row["LScore"]), int(row["WScore"])
            else:
                home_id, away_id = w_id, l_id
                home_score, away_score = int(row["WScore"]), int(row["LScore"])

            neutral = wloc == "N"

            # DayNum is days since a reference date; use it as a proxy date
            day_num = row.get("DayNum", "0")
            date_str = f"{season}-{int(day_num):03d}"

            upsert_game(conn, season, date_str, home_id, away_id,
                        home_score, away_score, neutral_site=neutral)
            count += 1
            if count % 1000 == 0:
                conn.commit()

    conn.commit()
    conn.close()
    print(f"Imported {count} games from Kaggle CSV")


def import_kaggle_tourney(csv_path: str):
    """Import NCAA tournament results from Kaggle dataset.

    Same format as regular season but marks games as postseason.
    """
    conn = get_db(SPORT)
    teams_csv = Path(csv_path).parent / "MTeams.csv"

    team_names = {}
    if teams_csv.exists():
        with open(teams_csv) as f:
            for row in csv.DictReader(f):
                team_names[row["TeamID"]] = row["TeamName"]

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            season = int(row["Season"])
            w_name = team_names.get(row["WTeamID"], f"Team_{row['WTeamID']}")
            l_name = team_names.get(row["LTeamID"], f"Team_{row['LTeamID']}")

            w_id = get_or_create_team(conn, w_name, source="kaggle")
            l_id = get_or_create_team(conn, l_name, source="kaggle")

            wloc = row.get("WLoc", "N")
            if wloc == "H":
                home_id, away_id = w_id, l_id
                home_score, away_score = int(row["WScore"]), int(row["LScore"])
            elif wloc == "A":
                home_id, away_id = l_id, w_id
                home_score, away_score = int(row["LScore"]), int(row["WScore"])
            else:
                home_id, away_id = w_id, l_id
                home_score, away_score = int(row["WScore"]), int(row["LScore"])

            day_num = row.get("DayNum", "0")
            date_str = f"{season}-{int(day_num):03d}"

            upsert_game(conn, season, date_str, home_id, away_id,
                        home_score, away_score, neutral_site=(wloc == "N"),
                        postseason=True)
            count += 1

    conn.commit()
    conn.close()
    print(f"Imported {count} tournament games from Kaggle CSV")


def _get_kenpom_api_token():
    """Load KenPom API token from .env file."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    token = os.environ.get("KENPOM_API_TOKEN")
    if not token:
        raise ValueError(
            "KENPOM_API_TOKEN not found. Add it to .env file. "
            "See .env.example for format."
        )
    return token


def _kenpom_api_request(endpoint: str, params: dict) -> list:
    """Make an authenticated request to the KenPom API."""
    token = _get_kenpom_api_token()
    params["endpoint"] = endpoint
    resp = requests.get(
        "https://kenpom.com/api.php",
        params=params,
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"KenPom API error {resp.status_code}: {resp.text}"
        )
    return resp.json()


# Mapping from KenPom API field names to our stat names.
# Only float fields that are useful as model features.
KENPOM_RATINGS_FIELDS = {
    "AdjEM": "adj_efficiency_margin",
    "AdjOE": "adj_offensive_efficiency",
    "AdjDE": "adj_defensive_efficiency",
    "AdjTempo": "adj_tempo",
    "Pythag": "pythag",
    "Luck": "luck",
    "SOS": "sos",
    "SOSO": "sos_offense",
    "SOSD": "sos_defense",
    "NCSOS": "nc_sos",
    "OE": "offensive_efficiency",
    "DE": "defensive_efficiency",
    "Tempo": "tempo",
}


def fetch_kenpom_ratings(season: int) -> int:
    """Fetch team ratings from the KenPom API and store in the database.

    Uses the ratings endpoint to get adjusted efficiency, tempo,
    strength of schedule, and other advanced metrics.

    Returns the number of stats that were inserted or changed.
    """
    data = _kenpom_api_request("ratings", {"y": season})

    conn = get_db(SPORT)
    team_count = 0
    changed = 0

    for entry in data:
        team_name = entry.get("TeamName")
        if not team_name:
            continue

        conf = entry.get("ConfShort")
        team_id = get_or_create_team(conn, team_name, conference=conf,
                                         source="kenpom")

        for api_field, stat_name in KENPOM_RATINGS_FIELDS.items():
            value = entry.get(api_field)
            if value is not None:
                if upsert_team_stat(conn, team_id, season, stat_name, float(value)):
                    changed += 1

        team_count += 1

    conn.commit()
    conn.close()
    print(f"Imported KenPom ratings for {team_count} teams (season {season}), {changed} stats changed")
    return changed


# Four-factors: the fundamental drivers of scoring efficiency.
# Only new fields not already covered by ratings endpoint.
KENPOM_FOUR_FACTORS_FIELDS = {
    "eFG_Pct": "efg_pct",
    "TO_Pct": "to_pct",
    "OR_Pct": "or_pct",
    "FT_Rate": "ft_rate",
    "DeFG_Pct": "def_efg_pct",
    "DTO_Pct": "def_to_pct",
    "DOR_Pct": "def_or_pct",
    "DFT_Rate": "def_ft_rate",
}


def fetch_kenpom_four_factors(season: int) -> int:
    """Fetch four-factors stats from the KenPom API and store in the database.

    The four factors of basketball: effective FG%, turnover rate,
    offensive rebound rate, and free throw rate — on both offense and defense.

    Returns the number of stats that were inserted or changed.
    """
    data = _kenpom_api_request("four-factors", {"y": season})

    conn = get_db(SPORT)
    team_count = 0
    changed = 0

    for entry in data:
        team_name = entry.get("TeamName")
        if not team_name:
            continue

        conf = entry.get("ConfShort")
        team_id = get_or_create_team(conn, team_name, conference=conf,
                                         source="kenpom")

        for api_field, stat_name in KENPOM_FOUR_FACTORS_FIELDS.items():
            value = entry.get(api_field)
            if value is not None:
                if upsert_team_stat(conn, team_id, season, stat_name, float(value)):
                    changed += 1

        team_count += 1

    conn.commit()
    conn.close()
    print(f"Imported KenPom four-factors for {team_count} teams (season {season}), {changed} stats changed")
    return changed


ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/scoreboard"
)


def _date_to_season(date_str: str) -> int:
    """Convert a date string (YYYY-MM-DD) to NCAA season year.

    NCAA season spans Nov-Apr. Games from Nov-Dec belong to the next
    calendar year's season (e.g., Nov 2025 = season 2026).
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if dt.month >= 11:
        return dt.year + 1
    return dt.year


def fetch_espn_games(date_str: str):
    """Fetch completed game results from ESPN for a given date.

    Args:
        date_str: Date in YYYY-MM-DD format.

    Returns the number of games that were new or changed.
    """
    espn_date = date_str.replace("-", "")
    resp = requests.get(ESPN_SCOREBOARD_URL,
                        params={"dates": espn_date, "limit": 200,
                                "groups": 50})
    if resp.status_code != 200:
        raise RuntimeError(
            f"ESPN API error {resp.status_code}: {resp.text}"
        )

    data = resp.json()
    events = data.get("events", [])
    season = _date_to_season(date_str)

    conn = get_db(SPORT)
    total = 0
    changed = 0
    unmatched = []

    for event in events:
        comp = event["competitions"][0]

        # Skip incomplete games
        if not comp["status"]["type"].get("completed", False):
            continue

        competitors = comp["competitors"]
        home = next(c for c in competitors if c["homeAway"] == "home")
        away = next(c for c in competitors if c["homeAway"] == "away")

        home_name = home["team"]["shortDisplayName"]
        away_name = away["team"]["shortDisplayName"]

        # Check if teams exist before creating — track truly new ones
        home_is_new = (conn.execute(
            "SELECT team_id FROM team_aliases WHERE alias = ?",
            (home_name,)
        ).fetchone() is None and conn.execute(
            "SELECT id FROM teams WHERE name = ?", (home_name,)
        ).fetchone() is None)

        away_is_new = (conn.execute(
            "SELECT team_id FROM team_aliases WHERE alias = ?",
            (away_name,)
        ).fetchone() is None and conn.execute(
            "SELECT id FROM teams WHERE name = ?", (away_name,)
        ).fetchone() is None)

        home_id = get_or_create_team(conn, home_name, source="espn")
        away_id = get_or_create_team(conn, away_name, source="espn")

        if home_is_new:
            unmatched.append(home_name)
        if away_is_new:
            unmatched.append(away_name)

        home_score = int(home["score"])
        away_score = int(away["score"])
        neutral = comp.get("neutralSite", False)

        if upsert_game(conn, season, date_str, home_id, away_id,
                        home_score, away_score, neutral_site=neutral):
            changed += 1
        total += 1

    conn.commit()
    conn.close()

    if unmatched:
        unique = sorted(set(unmatched))
        print(f"WARNING: {len(unique)} unmatched ESPN teams "
              f"(may need alias mapping): {unique}")

    print(f"ESPN {date_str}: {changed} changed, {total} total games")
    return changed


def compute_season_stats(season: int):
    """Compute derived stats from game results for a season.

    Calculates per team: win_pct, avg_margin, avg_points_for,
    avg_points_against, games_played.
    """
    conn = get_db(SPORT)

    teams = conn.execute("SELECT id, name FROM teams").fetchall()

    for team in teams:
        tid = team["id"]

        games = conn.execute("""
            SELECT home_score, away_score,
                   CASE WHEN home_team_id = ? THEN 1 ELSE 0 END as is_home
            FROM games
            WHERE season = ? AND (home_team_id = ? OR away_team_id = ?)
              AND home_score IS NOT NULL
        """, (tid, season, tid, tid)).fetchall()

        if not games:
            continue

        wins = 0
        total_margin = 0
        total_pts_for = 0
        total_pts_against = 0

        for g in games:
            if g["is_home"]:
                pts_for = g["home_score"]
                pts_against = g["away_score"]
            else:
                pts_for = g["away_score"]
                pts_against = g["home_score"]

            total_pts_for += pts_for
            total_pts_against += pts_against
            margin = pts_for - pts_against
            total_margin += margin
            if margin > 0:
                wins += 1

        n = len(games)
        upsert_team_stat(conn, tid, season, "win_pct", wins / n)
        upsert_team_stat(conn, tid, season, "avg_margin", total_margin / n)
        upsert_team_stat(conn, tid, season, "avg_points_for", total_pts_for / n)
        upsert_team_stat(conn, tid, season, "avg_points_against",
                         total_pts_against / n)
        upsert_team_stat(conn, tid, season, "games_played", n)

    conn.commit()
    conn.close()
    print(f"Computed season stats for {season}")

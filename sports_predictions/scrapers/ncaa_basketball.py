"""NCAA Basketball data importers.

Supports:
1. Kaggle March Machine Learning Mania CSV files (historical bulk import)
2. KenPom API for advanced metrics (adjusted efficiency, tempo, etc.)
3. ESPN API for current-season game results
"""

import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

import requests

from sports_predictions.db import (
    get_db, get_or_create_team, upsert_game, upsert_team_stat,
    upsert_team_rating_by_date
)

SPORT = "ncaa_basketball"


def _first_monday_in_november(year: int) -> datetime:
    """Return the first Monday in November for a given year."""
    # Nov 1, then advance to Monday
    nov1 = datetime(year, 11, 1)
    days_until_monday = (7 - nov1.weekday()) % 7
    return nov1 + timedelta(days=days_until_monday)


def kaggle_day_to_date(season: int, day_num: int) -> str:
    """Convert a Kaggle season + DayNum to an ISO date string.

    Kaggle DayNum counts from the first Monday in November of the year
    before the season label (e.g. season 2026 day 0 = Mon Nov 3 2025).
    """
    day0 = _first_monday_in_november(season - 1)
    return (day0 + timedelta(days=day_num)).strftime("%Y-%m-%d")


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

            day_num = int(row.get("DayNum", "0"))
            date_str = kaggle_day_to_date(season, day_num)

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

            day_num = int(row.get("DayNum", "0"))
            date_str = kaggle_day_to_date(season, day_num)

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


# Archive endpoint fields — point-in-time adjusted ratings
KENPOM_ARCHIVE_FIELDS = {
    "AdjEM": "adj_efficiency_margin",
    "AdjOE": "adj_offensive_efficiency",
    "AdjDE": "adj_defensive_efficiency",
    "AdjTempo": "adj_tempo",
}


def fetch_kenpom_archive_date(date_str: str) -> int:
    """Fetch KenPom ratings for a specific date and store in team_ratings_by_date.

    Args:
        date_str: Date in YYYY-MM-DD format.

    Returns the number of stats that were inserted or changed.
    Returns -1 if the date is out of range for the archive endpoint.
    """
    token = _get_kenpom_api_token()
    resp = requests.get(
        "https://kenpom.com/api.php",
        params={"endpoint": "archive", "d": date_str},
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status_code in (400, 404):
        return -1
    if resp.status_code != 200:
        raise RuntimeError(
            f"KenPom API error {resp.status_code}: {resp.text}"
        )
    data = resp.json()

    conn = get_db(SPORT)
    changed = 0

    for entry in data:
        team_name = entry.get("TeamName")
        if not team_name:
            continue

        team_id = get_or_create_team(conn, team_name, source="kenpom")

        for api_field, stat_name in KENPOM_ARCHIVE_FIELDS.items():
            value = entry.get(api_field)
            if value is not None:
                if upsert_team_rating_by_date(
                    conn, team_id, date_str, stat_name, float(value)
                ):
                    changed += 1

    conn.commit()
    conn.close()
    return changed


def fetch_kenpom_archive_season(season: int) -> int:
    """Fetch KenPom archive ratings for all game dates in a season.

    Finds all unique game dates for the season in the database,
    then fetches archive ratings for each date.

    Returns the total number of stats that were inserted or changed.
    """
    conn = get_db(SPORT)
    dates = conn.execute(
        "SELECT DISTINCT date FROM games WHERE season = ? ORDER BY date",
        (season,)
    ).fetchall()
    conn.close()

    unique_dates = [r["date"] for r in dates]
    print(f"Fetching KenPom archive for {len(unique_dates)} dates "
          f"(season {season})")

    total_changed = 0
    skipped = 0
    for i, date_str in enumerate(unique_dates):
        changed = fetch_kenpom_archive_date(date_str)
        if changed == -1:
            skipped += 1
        else:
            total_changed += changed
        if (i + 1) % 25 == 0:
            print(f"  {i + 1} / {len(unique_dates)} dates fetched...")

    print(f"KenPom archive for season {season}: {total_changed} stats "
          f"across {len(unique_dates) - skipped} dates "
          f"({skipped} dates out of range)")
    return total_changed


ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/scoreboard"
)
ESPN_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/summary"
)

# ESPN shortDisplayName -> canonical team name mappings.
# ESPN uses different abbreviations/names than Kaggle/KenPom.
ESPN_NAME_OVERRIDES = {
    "AR-Pine Bluff": "Arkansas Pine Bluff",
    "Abilene Chrstn": "Abilene Christian",
    "App State": "Appalachian St.",
    "Bakersfield": "Cal St. Bakersfield",
    "Ball State": "Ball St.",
    "Bethune": "Bethune Cookman",
    "Boston U": "Boston University",
    "C Arkansas": "Central Arkansas",
    "C Connecticut": "Central Connecticut",
    "CA Baptist": "Cal Baptist",
    "CSU Northridge": "CSUN",
    "Coastal": "Coastal Carolina",
    "Detroit Mercy": "Detroit",
    "E Texas A&M": "East Texas A&M",
    "FAU": "Florida Atlantic",
    "FDU": "Fairleigh Dickinson",
    "Fullerton": "Cal St. Fullerton",
    "GA Southern": "Georgia Southern",
    "Gardner-Webb": "Gardner Webb",
    "Hawai\u2019i": "Hawaii",
    "Hou Christian": "Houston Christian",
    "IU Indy": "IUPUI",
    "Iowa State": "Iowa St.",
    "Jax State": "Jacksonville St.",
    "Kennesaw St": "Kennesaw St.",
    "Kent State": "Kent St.",
    "LMU": "Loyola Marymount",
    "Long Island": "LIU Brooklyn",
    "MD Eastern": "Maryland Eastern Shore",
    "McNeese": "McNeese St.",
    "Miami": "Miami FL",
    "Miss Valley St": "Mississippi Valley St.",
    "Mount St Marys": "Mount St. Mary's",
    "N Arizona": "Northern Arizona",
    "N\u2019Western St": "Northwestern St.",
    "Nicholls": "Nicholls St.",
    "Ohio State": "Ohio St.",
    "Ole Miss": "Mississippi",
    "Omaha": "Nebraska Omaha",
    "Penn State": "Penn St.",
    "Pitt": "Pittsburgh",
    "Purdue FW": "Purdue Fort Wayne",
    "SC State": "South Carolina St.",
    "SE Missouri": "Southeast Missouri",
    "Sacramento St": "Sacramento St.",
    "Saint Francis": "St. Francis PA",
    "Sam Houston": "Sam Houston St.",
    "San Jos\u00e9 St": "San Jose St.",
    "Santa Barbara": "UC Santa Barbara",
    "Seattle U": "Seattle",
    "So Indiana": "Southern Indiana",
    "St Thomas (MN)": "St. Thomas",
    "Texas A&M Commerce": "East Texas A&M",
    "Texas A&M-CC": "Texas A&M Corpus Chris",
    "UAlbany": "Albany",
    "UConn": "Connecticut",
    "UIC": "Illinois Chicago",
    "UL Monroe": "Louisiana Monroe",
    "UMass": "Massachusetts",
    "UT Martin": "Tennessee Martin",
    "UT Rio Grande": "UT Rio Grande Valley",
    "Utah State": "Utah St.",
    "Western KY": "Western Kentucky",
}


def _date_to_season(date_str: str) -> int:
    """Convert a date string (YYYY-MM-DD) to NCAA season year.

    NCAA season spans Nov-Apr. Games from Nov-Dec belong to the next
    calendar year's season (e.g., Nov 2025 = season 2026).
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if dt.month >= 11:
        return dt.year + 1
    return dt.year


def _extract_espn_odds(pick: dict) -> dict:
    """Extract odds from ESPN pickcenter data for storage.

    Returns kwargs suitable for upsert_game(). Spread is from the home team
    perspective (positive = home favored), matching our model convention.
    ESPN's spread is negative when home is favored, so we negate it.
    """
    result = {}

    # Spread
    spread_data = pick.get("pointSpread", {})
    if spread_data:
        home_line = spread_data.get("home", {}).get("close", {}).get("line")
        if home_line is not None:
            result["vegas_spread"] = -float(home_line)
    if "vegas_spread" not in result and "spread" in pick:
        result["vegas_spread"] = -float(pick["spread"])

    # Total
    total_data = pick.get("total", {})
    if total_data:
        over_line = total_data.get("over", {}).get("close", {}).get("line", "")
        if over_line:
            result["vegas_total"] = float(str(over_line).lstrip("oO"))
    if "vegas_total" not in result and "overUnder" in pick:
        result["vegas_total"] = float(pick["overUnder"])

    # Moneyline
    ml_data = pick.get("moneyline", {})
    if ml_data:
        home_ml = ml_data.get("home", {}).get("close", {}).get("odds")
        away_ml = ml_data.get("away", {}).get("close", {}).get("odds")
        try:
            if home_ml:
                result["vegas_home_ml"] = int(home_ml)
        except (ValueError, TypeError):
            pass
        try:
            if away_ml:
                result["vegas_away_ml"] = int(away_ml)
        except (ValueError, TypeError):
            pass

    if result:
        result["odds_provider"] = pick.get("provider", {}).get("name", "unknown")

    return result


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

        # Apply name overrides for known ESPN variants
        home_name = ESPN_NAME_OVERRIDES.get(home_name, home_name)
        away_name = ESPN_NAME_OVERRIDES.get(away_name, away_name)

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

        # Fetch odds from ESPN summary/pickcenter
        odds_kwargs = {}
        game_id = event.get("id")
        if game_id:
            try:
                summary = requests.get(ESPN_SUMMARY_URL,
                                       params={"event": game_id}, timeout=10)
                summary.raise_for_status()
                pickcenter = summary.json().get("pickcenter", [])
                if pickcenter:
                    odds_kwargs = _extract_espn_odds(pickcenter[0])
            except requests.RequestException:
                pass

        if upsert_game(conn, season, date_str, home_id, away_id,
                        home_score, away_score, neutral_site=neutral,
                        **odds_kwargs):
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


def import_kaggle_rankings(csv_path: str, min_season: int = 2010):
    """Import weekly consensus rankings from Kaggle's MMasseyOrdinals.csv.

    For each team+date, computes the median ordinal rank across all
    ranking systems. Stores as 'consensus_rank' in team_ratings_by_date
    so it can be looked up at game time.
    """
    from collections import defaultdict
    from statistics import median

    teams_csv = Path(csv_path).parent / "MTeams.csv"

    # Load Kaggle TeamID -> name mapping
    team_names = {}
    if teams_csv.exists():
        with open(teams_csv) as f:
            for row in csv.DictReader(f):
                team_names[row["TeamID"]] = row["TeamName"]

    # Read rankings: group by (season, day_num, kaggle_team_id) -> [ranks]
    rankings = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            season = int(row["Season"])
            if season < min_season:
                continue
            key = (season, int(row["RankingDayNum"]), row["TeamID"])
            rankings[key].append(int(row["OrdinalRank"]))

    conn = get_db(SPORT)

    # Resolve Kaggle team names to our DB team IDs (cache the mapping)
    kaggle_to_db = {}
    for kid, name in team_names.items():
        from sports_predictions.db import resolve_team
        tid = resolve_team(conn, name)
        if tid:
            kaggle_to_db[kid] = tid

    changed = 0
    total = 0
    for (season, day_num, kaggle_tid), ranks in rankings.items():
        db_tid = kaggle_to_db.get(kaggle_tid)
        if not db_tid:
            continue

        date_str = kaggle_day_to_date(season, day_num)
        consensus = median(ranks)

        if upsert_team_rating_by_date(
            conn, db_tid, date_str, "consensus_rank", consensus
        ):
            changed += 1
        total += 1

        if total % 10000 == 0:
            conn.commit()

    conn.commit()
    conn.close()
    print(f"Imported weekly consensus rankings: {changed} changed, "
          f"{total} total team-date entries (seasons {min_season}+)")


# Massey team name -> our DB name overrides (where names don't match)
MASSEY_NAME_OVERRIDES = {
    "Connecticut": "UConn",
    "Miami FL": "Miami",
    "St John's": "St John's",
    "St Mary's CA": "St. Mary's",
    "N Dakota St": "North Dakota St.",
    "SF Austin": "SF Austin",
    "TAM C. Christi": "Texas A&M Corpus Chris",
    "CS Northridge": "Cal St. Northridge",
    "CS Fullerton": "Cal St. Fullerton",
    "CS Bakersfield": "Cal St. Bakersfield",
    "CS Sacramento": "Sacramento St.",
    "LIU Brooklyn": "LIU",
    "UC Irvine": "UC Irvine",
    "UC San Diego": "UC San Diego",
    "UC Santa Barbara": "UC Santa Barbara",
    "UC Davis": "UC Davis",
    "UC Riverside": "UC Riverside",
    "IL Chicago": "Ill. Chicago",
    "St Louis": "Saint Louis",
    "WI Green Bay": "Green Bay",
    "WI Milwaukee": "Milwaukee",
    "St Thomas MN": "St. Thomas (MN)",
    "E Washington": "Eastern Washington",
    "E Michigan": "Eastern Michigan",
    "E Kentucky": "Eastern Kentucky",
    "E Illinois": "Eastern Illinois",
    "W Carolina": "Western Carolina",
    "W Michigan": "Western Michigan",
    "W Illinois": "Western Illinois",
    "N Colorado": "Northern Colorado",
    "N Illinois": "Northern Illinois",
    "N Kentucky": "Northern Kentucky",
    "S Illinois": "Southern Illinois",
    "S Dakota St": "South Dakota St.",
    "SE Missouri St": "Southeast Missouri St.",
    "NE Omaha": "Nebraska Omaha",
    "PFW": "Purdue Fort Wayne",
    "SIUE": "SIU Edwardsville",
    "MTSU": "Middle Tennessee",
    "WKU": "Western Kentucky",
    "UTRGV": "UT Rio Grande Valley",
    "UTEP": "UT El Paso",
    "FGCU": "Florida Gulf Coast",
    "Col Charleston": "Charleston",
    "MA Lowell": "UMass Lowell",
    "UNC Wilmington": "UNC Wilmington",
    "UNC Asheville": "UNC Asheville",
    "UNC Greensboro": "UNC Greensboro",
    "Cal Poly": "Cal Poly",
    "UMBC": "UMBC",
    "MD E Shore": "Maryland Eastern Shore",
    "Ark Little Rock": "Little Rock",
    "Ark Pine Bluff": "Arkansas Pine Bluff",
    "MS Valley St": "Mississippi Valley St.",
    "IUPUI": "IU Indianapolis",
    "Missouri KC": "Kansas City",
    "Loy Marymount": "Loyola Marymount",
    "Northwestern LA": "Northwestern St.",
    "St Peter's": "Saint Peter's",
    "St Bonaventure": "St. Bonaventure",
    "St Joseph's PA": "Saint Joseph's",
    "East Texas A&M": "East Texas A&M",
    "FL Atlantic": "Florida Atlantic",
    "Florida Intl": "FIU",
    "Ga Southern": "Georgia Southern",
    "Charleston So": "Charleston Southern",
    "Cent Arkansas": "Central Arkansas",
    "Central Conn": "Central Connecticut",
    "Mt St Mary's": "Mount St. Mary's",
    "SC Upstate": "South Carolina Upstate",
    "SE Louisiana": "Southeastern Louisiana",
    "Southern Miss": "Southern Mississippi",
}


def import_massey_composite(csv_path: str, date_str: str, season: int):
    """Import Massey Ratings composite rankings from a CSV file.

    The CSV should have columns from masseyratings.com/ranks?s=cb including
    Team and CMP (composite rank).

    Args:
        csv_path: Path to the CSV file with rankings.
        date_str: Date for these rankings (YYYY-MM-DD).
        season: Season year (e.g. 2026).
    """
    from sports_predictions.db import resolve_team

    conn = get_db(SPORT)

    changed = 0
    total = 0
    unmatched = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        # Strip whitespace from header names (some exports have " CMP" etc.)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            team_name = row["Team"].strip()
            try:
                composite_rank = int(row["CMP"].strip())
            except (ValueError, KeyError):
                continue

            # Try name override first, then raw name
            db_name = MASSEY_NAME_OVERRIDES.get(team_name, team_name)
            team_id = resolve_team(conn, db_name)

            # If override didn't work, try raw name
            if not team_id and db_name != team_name:
                team_id = resolve_team(conn, team_name)

            if not team_id:
                unmatched.append(team_name)
                continue

            if upsert_team_rating_by_date(
                conn, team_id, date_str, "consensus_rank", float(composite_rank)
            ):
                changed += 1
            total += 1

    conn.commit()
    conn.close()

    if unmatched:
        print(f"Warning: {len(unmatched)} unmatched teams:")
        for name in sorted(set(unmatched)):
            print(f"  {name}")

    print(f"Imported Massey composite for {date_str}: {changed} changed, "
          f"{total} total teams")

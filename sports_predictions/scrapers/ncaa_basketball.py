"""NCAA Basketball data importers.

Supports:
1. Kaggle March Machine Learning Mania CSV files (historical bulk import)
2. KenPom scraping for advanced metrics (adjusted efficiency, tempo, etc.)
3. sportsipy for in-season game results
"""

import csv
import re
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

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

            w_id = get_or_create_team(conn, w_name)
            l_id = get_or_create_team(conn, l_name)

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

            w_id = get_or_create_team(conn, w_name)
            l_id = get_or_create_team(conn, l_name)

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


def scrape_kenpom(season: int, username: str = None, password: str = None):
    """Scrape KenPom ratings for a given season.

    KenPom requires a subscription for most data. If credentials are provided,
    logs in first. Otherwise attempts to scrape the free summary page.

    Stores these stats per team:
    - adj_efficiency_margin: adjusted efficiency margin
    - adj_offensive_efficiency: adjusted offensive efficiency
    - adj_defensive_efficiency: adjusted defensive efficiency
    - adj_tempo: adjusted tempo
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    })

    if username and password:
        login_url = "https://kenpom.com/handlers/login_handler.php"
        session.post(login_url, data={
            "email": username,
            "password": password,
        })

    url = f"https://kenpom.com/index.php?y={season}"
    resp = session.get(url)
    if resp.status_code != 200:
        print(f"Failed to fetch KenPom data: {resp.status_code}")
        return

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"id": "ratings-table"})
    if not table:
        print("Could not find ratings table — may need KenPom subscription")
        return

    conn = get_db(SPORT)
    count = 0

    tbody = table.find("tbody")
    if not tbody:
        print("No table body found")
        conn.close()
        return

    for row in tbody.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 8:
            continue

        # Column layout: Rank, Team, Conf, W-L, AdjEM, AdjO, AdjD, AdjT
        team_cell = cells[1]
        team_name = team_cell.get_text(strip=True)
        # Remove seed numbers that sometimes appear
        team_name = re.sub(r'\s*\d+$', '', team_name).strip()

        conf = cells[2].get_text(strip=True)

        try:
            adj_em = float(cells[4].get_text(strip=True))
            adj_o = float(cells[5].get_text(strip=True))
            adj_d = float(cells[6].get_text(strip=True))
            adj_t = float(cells[7].get_text(strip=True))
        except (ValueError, IndexError):
            continue

        team_id = get_or_create_team(conn, team_name, conference=conf)
        upsert_team_stat(conn, team_id, season, "adj_efficiency_margin", adj_em)
        upsert_team_stat(conn, team_id, season, "adj_offensive_efficiency", adj_o)
        upsert_team_stat(conn, team_id, season, "adj_defensive_efficiency", adj_d)
        upsert_team_stat(conn, team_id, season, "adj_tempo", adj_t)
        count += 1

    conn.commit()
    conn.close()
    print(f"Imported KenPom stats for {count} teams (season {season})")


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

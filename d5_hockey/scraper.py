"""Scraper for District Five Hockey website.

Pulls standings, player stats, rosters, and game results
from districtfivehockey.com for all available seasons.
"""

import re
import time

import requests
from bs4 import BeautifulSoup

from d5_hockey.db import (
    get_db, get_or_create_season, get_or_create_player,
    get_or_create_team, add_alias, resolve_player
)

BASE_URL = "https://districtfivehockey.com"

# Map of season URL slugs to (year, period) metadata.
# Period ordering: spring=1, summer=2, fall=3
SEASONS = [
    ("summer-19", 2019, "summer"),
    ("fall-19", 2019, "fall"),
    ("spring-21", 2021, "spring"),
    ("summer-fall-21", 2021, "fall"),
    ("spring-22", 2022, "spring"),
    ("summer-22", 2022, "summer"),
    ("fall-22", 2022, "fall"),
    ("spring-23", 2023, "spring"),
    ("summer-23", 2023, "summer"),
    ("fall-23", 2023, "fall"),
    ("spring-24", 2024, "spring"),
    ("summer-24", 2024, "summer"),
    ("fall-24", 2024, "fall"),
    ("spring-25", 2025, "spring"),
    ("summer-25", 2025, "summer"),
    ("fall-25", 2025, "fall"),
]

# Team abbreviation to full name mapping (updated as we discover them)
ABBREV_MAP = {
    "SEX": "Great Sexpectations",
    "GLAT": "Hockey at Glatt's",
    "LEG": "Legally Blonde",
    "PIE": "Return Of The Pie Pie",
    "RED": "Rediohead",
    "DUCK": "The Mighty Ducks",
    "CLAP": "Return Of The Clap",
    "MILK": "The MilkMob",
    "PMD": "Purple Monkey Dishwasher",
    "SCHV": "I'm Schvitzing",
    "CUP": "My Cup Size Is Stanley",
    "A$$": "A$$ Fault Green",
}

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
})


def _fetch(path: str) -> BeautifulSoup:
    """Fetch a page and return parsed HTML."""
    url = f"{BASE_URL}/{path}" if not path.startswith("http") else path
    time.sleep(0.5)  # be polite
    resp = SESSION.get(url)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def scrape_standings(season_slug: str, season_id: int, conn):
    """Scrape standings page for a season."""
    try:
        soup = _fetch(f"standings-{season_slug}/")
    except requests.HTTPError:
        print(f"  No standings page for {season_slug}")
        return

    tables = soup.find_all("table")
    if not tables:
        print(f"  No standings table found for {season_slug}")
        return

    for table in tables:
        rows = table.find_all("tr")
        for row in rows[1:]:  # skip header
            cells = row.find_all("td")
            if len(cells) < 6:
                continue

            # Try to find team name - usually in the 2nd column
            team_name = cells[1].get_text(strip=True)
            if not team_name or team_name.isdigit():
                team_name = cells[0].get_text(strip=True)

            if not team_name:
                continue

            get_or_create_team(conn, team_name, season_id)

    print(f"  Scraped standings for {season_slug}")


def scrape_stats(season_slug: str, season_id: int, conn):
    """Scrape player stats page for a season."""
    try:
        soup = _fetch(f"stats-{season_slug}/")
    except requests.HTTPError:
        print(f"  No stats page for {season_slug}")
        return

    tables = soup.find_all("table")
    skater_count = 0
    goalie_count = 0

    for table in tables:
        headers = [th.get_text(strip=True).upper()
                   for th in table.find_all("th")]

        # If no <th> headers, check the first <tr>'s <td> cells
        rows = table.find_all("tr")
        if not headers and rows:
            first_cells = rows[0].find_all("td")
            first_vals = [c.get_text(strip=True).upper() for c in first_cells]
            if any(v in ("NAME", "PLAYER", "RANK", "GP") for v in first_vals):
                headers = first_vals
                rows = rows[1:]  # skip header row

        # Detect if this is a goalie table
        is_goalie = "GAA" in headers or "WINS" in headers

        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue

            # Parse based on detected columns
            try:
                if is_goalie:
                    stats = _parse_goalie_row(cells, headers)
                    if stats and "/" not in stats["name"]:
                        _store_goalie_stats(stats, season_id, conn)
                        goalie_count += 1
                else:
                    stats = _parse_skater_row(cells, headers)
                    # Skip tandem entries (e.g. "Hicks/James D") —
                    # each player has their own individual row
                    if stats and "/" not in stats["name"]:
                        _store_skater_stats(stats, season_id, conn)
                        skater_count += 1
            except (ValueError, IndexError):
                continue

    print(f"  Scraped stats for {season_slug}: "
          f"{skater_count} skaters, {goalie_count} goalies")


def _parse_skater_row(cells, headers):
    """Parse a skater stats row, adapting to column layout."""
    values = [c.get_text(strip=True) for c in cells]

    # Find column indices
    def col(name, default=None):
        try:
            return headers.index(name)
        except ValueError:
            return default

    # Common layouts: [Rank, Name, Team, GP, G, A, Pts, PIM, ...]
    # or without rank: [Name, Team, GP, G, A, Pts, PIM, ...]
    name_idx = col("NAME", col("PLAYER", 1))
    team_idx = col("TEAM", col("TM", 2))
    gp_idx = col("GP", 3)
    g_idx = col("G", 4)
    a_idx = col("A", 5)
    pts_idx = col("PTS", col("POINTS", 6))
    pim_idx = col("PIM", 7)
    ppg_idx = col("PPG", None)
    shg_idx = col("SHG", None)
    gwg_idx = col("GWG", None)

    name = values[name_idx] if name_idx < len(values) else None
    if not name or name.isdigit():
        return None

    team = values[team_idx] if team_idx < len(values) else ""

    def safe_int(idx):
        if idx is None or idx >= len(values):
            return 0
        try:
            return int(values[idx])
        except ValueError:
            return 0

    return {
        "name": name,
        "team": team,
        "gp": safe_int(gp_idx),
        "goals": safe_int(g_idx),
        "assists": safe_int(a_idx),
        "points": safe_int(pts_idx),
        "pim": safe_int(pim_idx),
        "ppg": safe_int(ppg_idx),
        "shg": safe_int(shg_idx),
        "gwg": safe_int(gwg_idx),
    }


def _parse_goalie_row(cells, headers):
    """Parse a goalie stats row."""
    values = [c.get_text(strip=True) for c in cells]

    def col(name, default=None):
        try:
            return headers.index(name)
        except ValueError:
            return default

    name_idx = col("NAME", col("PLAYER", 1))
    team_idx = col("TEAM", col("TM", 2))
    gp_idx = col("GP", 3)
    wins_idx = col("WINS", col("W", 4))
    gaa_idx = col("GAA", 5)
    so_idx = col("SO", col("SHUTOUTS", 6))

    name = values[name_idx] if name_idx < len(values) else None
    if not name or name.isdigit():
        return None

    team = values[team_idx] if team_idx < len(values) else ""

    def safe_int(idx):
        if idx is None or idx >= len(values):
            return 0
        try:
            return int(values[idx])
        except ValueError:
            return 0

    def safe_float(idx):
        if idx is None or idx >= len(values):
            return None
        try:
            return float(values[idx])
        except ValueError:
            return None

    return {
        "name": name,
        "team": team,
        "gp": safe_int(gp_idx),
        "wins": safe_int(wins_idx),
        "gaa": safe_float(gaa_idx),
        "shutouts": safe_int(so_idx),
    }


def _store_skater_stats(stats, season_id, conn):
    """Store skater stats, creating player if needed."""
    player_id = resolve_player(conn, stats["name"])
    if player_id is None:
        player_id = get_or_create_player(conn, stats["name"])

    conn.execute("""
        INSERT INTO player_season_stats
            (player_id, season_id, team_abbrev, gp, goals, assists,
             points, pim, ppg, shg, gwg, is_goalie)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        ON CONFLICT(player_id, season_id)
        DO UPDATE SET team_abbrev=excluded.team_abbrev,
            gp=excluded.gp, goals=excluded.goals, assists=excluded.assists,
            points=excluded.points, pim=excluded.pim, ppg=excluded.ppg,
            shg=excluded.shg, gwg=excluded.gwg
    """, (player_id, season_id, stats["team"], stats["gp"],
          stats["goals"], stats["assists"], stats["points"],
          stats["pim"], stats["ppg"], stats["shg"], stats["gwg"]))


def _store_goalie_stats(stats, season_id, conn):
    """Store goalie stats."""
    player_id = resolve_player(conn, stats["name"])
    if player_id is None:
        player_id = get_or_create_player(conn, stats["name"])

    conn.execute("""
        INSERT INTO player_season_stats
            (player_id, season_id, team_abbrev, gp, is_goalie,
             wins, gaa, shutouts)
        VALUES (?, ?, ?, ?, 1, ?, ?, ?)
        ON CONFLICT(player_id, season_id)
        DO UPDATE SET team_abbrev=excluded.team_abbrev,
            gp=excluded.gp, is_goalie=1,
            wins=excluded.wins, gaa=excluded.gaa,
            shutouts=excluded.shutouts
    """, (player_id, season_id, stats["team"], stats["gp"],
          stats["wins"], stats["gaa"], stats["shutouts"]))


def scrape_schedule(season_slug: str, season_id: int, conn):
    """Scrape game results from schedule page.

    The D5 site formats each game as HTML with <a> tags for teams and score:
        <a>Team1</a> at <a>Team2</a> (<a>4-0 ABBREV</a>)
    separated by <br/> tags within date-grouped <p> blocks.
    We extract team names from the anchor tags to avoid issues with
    team names containing " at " (e.g. "Hockey at Glatt's").
    """
    try:
        soup = _fetch(f"schedule-{season_slug}/")
    except requests.HTTPError:
        print(f"  No schedule page for {season_slug}")
        return

    content = soup.find("div", class_="entry-content")
    if not content:
        print(f"  No content found for {season_slug}")
        return

    game_count = 0
    is_playoff = False

    score_pattern = re.compile(
        r'(\d+)-(\d+)\s*(?:\(OT\)\s*)?([A-Z\$]+)'
    )

    for p_tag in content.find_all("p"):
        p_text = p_tag.get_text()

        # Detect playoff markers
        if "playoff" in p_text.lower() or "championship" in p_text.lower():
            is_playoff = True

        # Extract date from the block
        current_date = None
        date_match = re.search(
            r'(January|February|March|April|May|June|July|August|'
            r'September|October|November|December)\s+\d+', p_text
        )
        if date_match:
            current_date = date_match.group(0)

        # Split the <p> HTML by <br/> to get individual game segments
        segments = str(p_tag).split('<br/>')
        for seg in segments:
            seg_soup = BeautifulSoup(seg, "lxml")
            links = seg_soup.find_all("a")

            # Each game has 2-3 links: team1, team2, and optionally score
            if len(links) < 2:
                continue

            team1_name = links[0].get_text(strip=True)
            team2_name = links[1].get_text(strip=True)

            if not team1_name or not team2_name:
                continue

            # Score is either in the 3rd link or in the segment text
            score_text = ""
            if len(links) >= 3:
                score_text = links[2].get_text(strip=True)
            else:
                # Fall back to searching segment text for score pattern
                score_text = seg_soup.get_text()

            match = score_pattern.search(score_text)
            if not match:
                continue

            score1 = int(match.group(1))
            score2 = int(match.group(2))
            winner_abbrev = match.group(3).strip()

            team1_id = get_or_create_team(conn, team1_name, season_id)
            team2_id = get_or_create_team(conn, team2_name, season_id)

            # Figure out which team won based on abbreviation
            winner_id = None
            for tid, tname in [(team1_id, team1_name),
                                (team2_id, team2_name)]:
                team_row = conn.execute(
                    "SELECT abbrev FROM teams WHERE id = ?", (tid,)
                ).fetchone()
                if team_row and team_row["abbrev"] == winner_abbrev:
                    winner_id = tid
                    break
                full_name = ABBREV_MAP.get(winner_abbrev)
                if full_name and full_name.lower() in tname.lower():
                    winner_id = tid
                    break
                if winner_abbrev.lower() in tname.lower():
                    winner_id = tid
                    break

            # Assign scores: winner gets the higher score
            if winner_id == team1_id:
                t1_score = max(score1, score2)
                t2_score = min(score1, score2)
            elif winner_id == team2_id:
                t1_score = min(score1, score2)
                t2_score = max(score1, score2)
            else:
                t1_score, t2_score = score1, score2

            try:
                conn.execute("""
                    INSERT INTO games (season_id, date, team1_id, team2_id,
                                      team1_score, team2_score, winner_id,
                                      is_playoff)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(season_id, date, team1_id, team2_id)
                    DO UPDATE SET team1_score=excluded.team1_score,
                        team2_score=excluded.team2_score,
                        winner_id=excluded.winner_id,
                        is_playoff=excluded.is_playoff
                """, (season_id, current_date, team1_id, team2_id,
                      t1_score, t2_score, winner_id, int(is_playoff)))
                game_count += 1
            except Exception:
                pass

    print(f"  Scraped schedule for {season_slug}: {game_count} games")


def scrape_rosters(season_slug: str, season_id: int, conn):
    """Scrape team roster pages for a season."""
    # Get teams for this season
    teams = conn.execute(
        "SELECT id, name FROM teams WHERE season_id = ?", (season_id,)
    ).fetchall()

    if not teams:
        print(f"  No teams found for {season_slug} to scrape rosters")
        return

    roster_count = 0
    for team in teams:
        # Build URL slug from team name
        slug = team["name"].lower()
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'\s+', '-', slug).strip('-')
        url = f"{slug}-{season_slug}/"

        try:
            soup = _fetch(url)
        except requests.HTTPError:
            # Try alternate URL patterns
            continue

        # Look for player names in gallery captions or text
        # The site uses image galleries with player names as filenames
        player_names = set()

        # Check image filenames for player names
        for img in soup.find_all("img"):
            src = img.get("src", "")
            alt = img.get("alt", "")
            if alt and "logo" not in alt.lower():
                # Clean up alt text
                name = alt.strip()
                if name and len(name) > 2:
                    player_names.add(name)

        # Check figcaptions
        for caption in soup.find_all("figcaption"):
            name = caption.get_text(strip=True)
            if name and len(name) > 2:
                player_names.add(name)

        # Check for text content that looks like a roster list
        content = soup.find("div", class_="entry-content")
        if content:
            for li in content.find_all("li"):
                name = li.get_text(strip=True)
                if name and len(name) > 2 and not name.startswith("http"):
                    player_names.add(name)

        for name in player_names:
            if "wildcard" in name.lower() or "goalie" in name.lower():
                continue
            player_id = resolve_player(conn, name)
            if player_id is None:
                player_id = get_or_create_player(conn, name)
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO roster_entries
                        (player_id, team_id, season_id)
                    VALUES (?, ?, ?)
                """, (player_id, team["id"], season_id))
                roster_count += 1
            except Exception:
                pass

    print(f"  Scraped rosters for {season_slug}: {roster_count} entries")


def scrape_all():
    """Scrape all available seasons."""
    conn = get_db()

    for slug, year, period in SEASONS:
        season_name = f"{period.title()} {year}"
        print(f"\n=== {season_name} ({slug}) ===")

        season_id = get_or_create_season(conn, season_name, year, period)

        scrape_standings(slug, season_id, conn)
        conn.commit()

        scrape_stats(slug, season_id, conn)
        conn.commit()

        scrape_schedule(slug, season_id, conn)
        conn.commit()

        scrape_rosters(slug, season_id, conn)
        conn.commit()

    conn.close()
    print("\n=== Scraping complete ===")


if __name__ == "__main__":
    scrape_all()

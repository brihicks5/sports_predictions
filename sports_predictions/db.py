"""Generic SQLite database helpers for sports predictions.

Each sport gets its own .db file in data/, but they all share the same schema.
This makes it easy to add new sports without changing the database code.
"""

import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def get_db(sport: str) -> sqlite3.Connection:
    """Get a connection to the database for a given sport."""
    DATA_DIR.mkdir(exist_ok=True)
    db_path = DATA_DIR / f"{sport}.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection):
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            conference TEXT
        );

        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            date TEXT,
            home_team_id INTEGER NOT NULL REFERENCES teams(id),
            away_team_id INTEGER NOT NULL REFERENCES teams(id),
            home_score INTEGER,
            away_score INTEGER,
            neutral_site INTEGER DEFAULT 0,
            postseason INTEGER DEFAULT 0,
            UNIQUE(season, date, home_team_id, away_team_id)
        );

        CREATE TABLE IF NOT EXISTS team_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id INTEGER NOT NULL REFERENCES teams(id),
            season INTEGER NOT NULL,
            stat_name TEXT NOT NULL,
            stat_value REAL,
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(team_id, season, stat_name)
        );

        CREATE TABLE IF NOT EXISTS team_aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id INTEGER NOT NULL REFERENCES teams(id),
            alias TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS team_ratings_by_date (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id INTEGER NOT NULL REFERENCES teams(id),
            date TEXT NOT NULL,
            stat_name TEXT NOT NULL,
            stat_value REAL,
            UNIQUE(team_id, date, stat_name)
        );

        CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);
        CREATE INDEX IF NOT EXISTS idx_games_date ON games(date);
        CREATE INDEX IF NOT EXISTS idx_team_stats_lookup
            ON team_stats(team_id, season, stat_name);
        CREATE INDEX IF NOT EXISTS idx_team_aliases_alias
            ON team_aliases(alias);
        CREATE INDEX IF NOT EXISTS idx_team_aliases_team
            ON team_aliases(team_id);
        CREATE INDEX IF NOT EXISTS idx_team_ratings_by_date_lookup
            ON team_ratings_by_date(team_id, date, stat_name);
    """)


def resolve_team(conn: sqlite3.Connection, name: str) -> int | None:
    """Look up a team by alias or canonical name. Returns team id or None."""
    row = conn.execute(
        "SELECT team_id FROM team_aliases WHERE alias = ?", (name,)
    ).fetchone()
    if row:
        return row["team_id"]
    row = conn.execute(
        "SELECT id FROM teams WHERE name = ?", (name,)
    ).fetchone()
    if row:
        return row["id"]
    return None


def add_team_alias(conn: sqlite3.Connection, team_id: int,
                   alias: str, source: str):
    """Add an alias for a team. Ignores if alias already exists."""
    conn.execute(
        "INSERT OR IGNORE INTO team_aliases (team_id, alias, source) "
        "VALUES (?, ?, ?)",
        (team_id, alias, source)
    )


def get_or_create_team(conn: sqlite3.Connection, name: str,
                       conference: str = None, source: str = None) -> int:
    """Get existing team id (by alias or name) or create a new team.

    Returns the team id.
    """
    # Check aliases first
    row = conn.execute(
        "SELECT team_id FROM team_aliases WHERE alias = ?", (name,)
    ).fetchone()
    if row:
        team_id = row["team_id"]
        if conference:
            conn.execute(
                "UPDATE teams SET conference = ? WHERE id = ?",
                (conference, team_id)
            )
        return team_id

    # Fall back to canonical name lookup
    row = conn.execute(
        "SELECT id FROM teams WHERE name = ?", (name,)
    ).fetchone()
    if row:
        team_id = row["id"]
        if conference:
            conn.execute(
                "UPDATE teams SET conference = ? WHERE id = ?",
                (conference, team_id)
            )
        # Backfill alias
        add_team_alias(conn, team_id, name, source or "canonical")
        return team_id

    # Create new team + aliases
    cursor = conn.execute(
        "INSERT INTO teams (name, conference) VALUES (?, ?)",
        (name, conference)
    )
    team_id = cursor.lastrowid
    add_team_alias(conn, team_id, name, "canonical")
    if source and source != "canonical":
        add_team_alias(conn, team_id, name, source)
    return team_id


def upsert_game(conn: sqlite3.Connection, season: int, date: str,
                 home_team_id: int, away_team_id: int,
                 home_score: int, away_score: int,
                 neutral_site: bool = False, postseason: bool = False) -> bool:
    """Insert or update a game result.

    Returns True if the row was inserted or changed, False if unchanged.
    """
    row = conn.execute(
        "SELECT home_score, away_score, neutral_site, postseason FROM games "
        "WHERE season=? AND date=? AND home_team_id=? AND away_team_id=?",
        (season, date, home_team_id, away_team_id)
    ).fetchone()

    if (row is not None
            and row["home_score"] == home_score
            and row["away_score"] == away_score
            and row["neutral_site"] == int(neutral_site)
            and row["postseason"] == int(postseason)):
        return False

    conn.execute("""
        INSERT INTO games (season, date, home_team_id, away_team_id,
                          home_score, away_score, neutral_site, postseason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(season, date, home_team_id, away_team_id)
        DO UPDATE SET home_score=excluded.home_score,
                      away_score=excluded.away_score,
                      neutral_site=excluded.neutral_site,
                      postseason=excluded.postseason
    """, (season, date, home_team_id, away_team_id,
          home_score, away_score, int(neutral_site), int(postseason)))
    return True


def upsert_team_stat(conn: sqlite3.Connection, team_id: int, season: int,
                     stat_name: str, stat_value: float) -> bool:
    """Insert or update a team stat for a season.

    Returns True if the value was inserted or changed, False if unchanged.
    """
    row = conn.execute(
        "SELECT stat_value FROM team_stats WHERE team_id=? AND season=? AND stat_name=?",
        (team_id, season, stat_name)
    ).fetchone()

    if row is not None and row["stat_value"] == stat_value:
        return False

    conn.execute("""
        INSERT INTO team_stats (team_id, season, stat_name, stat_value,
                               updated_at)
        VALUES (?, ?, ?, ?, datetime('now'))
        ON CONFLICT(team_id, season, stat_name)
        DO UPDATE SET stat_value=excluded.stat_value,
                      updated_at=datetime('now')
    """, (team_id, season, stat_name, stat_value))
    return True


def upsert_team_rating_by_date(conn: sqlite3.Connection, team_id: int,
                                date: str, stat_name: str,
                                stat_value: float) -> bool:
    """Insert or update a team's rating for a specific date.

    Returns True if the value was inserted or changed, False if unchanged.
    """
    row = conn.execute(
        "SELECT stat_value FROM team_ratings_by_date "
        "WHERE team_id=? AND date=? AND stat_name=?",
        (team_id, date, stat_name)
    ).fetchone()

    if row is not None and row["stat_value"] == stat_value:
        return False

    conn.execute("""
        INSERT INTO team_ratings_by_date (team_id, date, stat_name, stat_value)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(team_id, date, stat_name)
        DO UPDATE SET stat_value=excluded.stat_value
    """, (team_id, date, stat_name, stat_value))
    return True

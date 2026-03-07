"""D5 Hockey database schema.

Separate from the generic sports_predictions DB because D5 needs
player-level tracking and roster-based team strength calculation.
"""

import sqlite3
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def get_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    db_path = DATA_DIR / "d5_hockey.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS seasons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            year INTEGER NOT NULL,
            period TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            season_id INTEGER NOT NULL REFERENCES seasons(id),
            abbrev TEXT,
            UNIQUE(name, season_id)
        );

        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS player_aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL REFERENCES players(id),
            alias TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS roster_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL REFERENCES players(id),
            team_id INTEGER NOT NULL REFERENCES teams(id),
            season_id INTEGER NOT NULL REFERENCES seasons(id),
            UNIQUE(player_id, team_id, season_id)
        );

        CREATE TABLE IF NOT EXISTS player_season_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL REFERENCES players(id),
            season_id INTEGER NOT NULL REFERENCES seasons(id),
            team_abbrev TEXT,
            gp INTEGER DEFAULT 0,
            goals INTEGER DEFAULT 0,
            assists INTEGER DEFAULT 0,
            points INTEGER DEFAULT 0,
            pim INTEGER DEFAULT 0,
            ppg INTEGER DEFAULT 0,
            shg INTEGER DEFAULT 0,
            gwg INTEGER DEFAULT 0,
            is_goalie INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            gaa REAL,
            shutouts INTEGER DEFAULT 0,
            UNIQUE(player_id, season_id)
        );

        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season_id INTEGER NOT NULL REFERENCES seasons(id),
            date TEXT,
            team1_id INTEGER NOT NULL REFERENCES teams(id),
            team2_id INTEGER NOT NULL REFERENCES teams(id),
            team1_score INTEGER,
            team2_score INTEGER,
            winner_id INTEGER REFERENCES teams(id),
            is_playoff INTEGER DEFAULT 0,
            UNIQUE(season_id, date, team1_id, team2_id)
        );

        CREATE INDEX IF NOT EXISTS idx_roster_season
            ON roster_entries(season_id);
        CREATE INDEX IF NOT EXISTS idx_player_stats_season
            ON player_season_stats(season_id);
        CREATE INDEX IF NOT EXISTS idx_games_season
            ON games(season_id);
    """)


def get_or_create_season(conn, name: str, year: int, period: str) -> int:
    row = conn.execute(
        "SELECT id FROM seasons WHERE name = ?", (name,)
    ).fetchone()
    if row:
        return row["id"]
    cursor = conn.execute(
        "INSERT INTO seasons (name, year, period) VALUES (?, ?, ?)",
        (name, year, period)
    )
    return cursor.lastrowid


def get_or_create_player(conn, canonical_name: str) -> int:
    row = conn.execute(
        "SELECT id FROM players WHERE canonical_name = ?", (canonical_name,)
    ).fetchone()
    if row:
        return row["id"]
    cursor = conn.execute(
        "INSERT INTO players (canonical_name) VALUES (?)", (canonical_name,)
    )
    return cursor.lastrowid


def add_alias(conn, player_id: int, alias: str):
    try:
        conn.execute(
            "INSERT OR IGNORE INTO player_aliases (player_id, alias) "
            "VALUES (?, ?)",
            (player_id, alias)
        )
    except sqlite3.IntegrityError:
        pass


def resolve_player(conn, name: str) -> Optional[int]:
    """Look up a player by canonical name or alias."""
    row = conn.execute(
        "SELECT id FROM players WHERE canonical_name = ?", (name,)
    ).fetchone()
    if row:
        return row["id"]
    row = conn.execute(
        "SELECT player_id FROM player_aliases WHERE alias = ?", (name,)
    ).fetchone()
    if row:
        return row["player_id"]
    return None


def get_or_create_team(conn, name: str, season_id: int,
                       abbrev: str = None) -> int:
    row = conn.execute(
        "SELECT id FROM teams WHERE name = ? AND season_id = ?",
        (name, season_id)
    ).fetchone()
    if row:
        if abbrev:
            conn.execute(
                "UPDATE teams SET abbrev = ? WHERE id = ?",
                (abbrev, row["id"])
            )
        return row["id"]
    cursor = conn.execute(
        "INSERT INTO teams (name, season_id, abbrev) VALUES (?, ?, ?)",
        (name, season_id, abbrev)
    )
    return cursor.lastrowid

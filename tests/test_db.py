"""Tests for database helpers."""

import sqlite3

import pytest

from sports_predictions.db import (
    get_db, upsert_team_stat, upsert_game, get_or_create_team, resolve_team,
    _ensure_schema,
)


@pytest.fixture
def conn():
    """In-memory SQLite database with schema applied."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    _ensure_schema(c)
    return c


class TestUpsertTeamStat:
    """Tests for upsert_team_stat's change detection (drives skip-training)."""

    def test_insert_returns_true(self, conn):
        conn.execute("INSERT INTO teams (name) VALUES ('Duke')")
        assert upsert_team_stat(conn, 1, 2026, "win_pct", 0.85) is True

    def test_same_value_returns_false(self, conn):
        conn.execute("INSERT INTO teams (name) VALUES ('Duke')")
        upsert_team_stat(conn, 1, 2026, "win_pct", 0.85)
        assert upsert_team_stat(conn, 1, 2026, "win_pct", 0.85) is False

    def test_changed_value_returns_true(self, conn):
        conn.execute("INSERT INTO teams (name) VALUES ('Duke')")
        upsert_team_stat(conn, 1, 2026, "win_pct", 0.85)
        assert upsert_team_stat(conn, 1, 2026, "win_pct", 0.90) is True

    def test_different_stat_name_returns_true(self, conn):
        conn.execute("INSERT INTO teams (name) VALUES ('Duke')")
        upsert_team_stat(conn, 1, 2026, "win_pct", 0.85)
        assert upsert_team_stat(conn, 1, 2026, "avg_margin", 5.0) is True

    def test_different_season_returns_true(self, conn):
        conn.execute("INSERT INTO teams (name) VALUES ('Duke')")
        upsert_team_stat(conn, 1, 2025, "win_pct", 0.85)
        assert upsert_team_stat(conn, 1, 2026, "win_pct", 0.85) is True

    def test_value_actually_updates_in_db(self, conn):
        conn.execute("INSERT INTO teams (name) VALUES ('Duke')")
        upsert_team_stat(conn, 1, 2026, "win_pct", 0.85)
        upsert_team_stat(conn, 1, 2026, "win_pct", 0.90)
        row = conn.execute(
            "SELECT stat_value FROM team_stats WHERE team_id=1 AND stat_name='win_pct'"
        ).fetchone()
        assert row["stat_value"] == 0.90


class TestUpsertGame:
    def test_insert_game(self, conn):
        conn.execute("INSERT INTO teams (name) VALUES ('Duke')")
        conn.execute("INSERT INTO teams (name) VALUES ('UNC')")
        upsert_game(conn, 2026, "2026-03-08", 1, 2, 80, 70)
        row = conn.execute("SELECT * FROM games").fetchone()
        assert row["home_score"] == 80
        assert row["away_score"] == 70
        assert row["neutral_site"] == 0

    def test_upsert_updates_score(self, conn):
        conn.execute("INSERT INTO teams (name) VALUES ('Duke')")
        conn.execute("INSERT INTO teams (name) VALUES ('UNC')")
        upsert_game(conn, 2026, "2026-03-08", 1, 2, 80, 70)
        upsert_game(conn, 2026, "2026-03-08", 1, 2, 85, 75)
        row = conn.execute("SELECT * FROM games").fetchone()
        assert row["home_score"] == 85

    def test_neutral_site_flag(self, conn):
        conn.execute("INSERT INTO teams (name) VALUES ('Duke')")
        conn.execute("INSERT INTO teams (name) VALUES ('UNC')")
        upsert_game(conn, 2026, "2026-03-08", 1, 2, 80, 70, neutral_site=True)
        row = conn.execute("SELECT * FROM games").fetchone()
        assert row["neutral_site"] == 1


class TestResolveTeam:
    def test_resolve_by_canonical_name(self, conn):
        conn.execute("INSERT INTO teams (name) VALUES ('Duke')")
        assert resolve_team(conn, "Duke") == 1

    def test_resolve_by_alias(self, conn):
        conn.execute("INSERT INTO teams (name) VALUES ('Duke')")
        conn.execute(
            "INSERT INTO team_aliases (team_id, alias, source) VALUES (1, 'Blue Devils', 'espn')"
        )
        assert resolve_team(conn, "Blue Devils") == 1

    def test_resolve_unknown_returns_none(self, conn):
        assert resolve_team(conn, "Nonexistent") is None


class TestGetOrCreateTeam:
    def test_creates_new_team(self, conn):
        tid = get_or_create_team(conn, "Duke", source="espn")
        assert tid == 1
        row = conn.execute("SELECT name FROM teams WHERE id=1").fetchone()
        assert row["name"] == "Duke"

    def test_returns_existing_by_alias(self, conn):
        get_or_create_team(conn, "Duke", source="kaggle")
        tid = get_or_create_team(conn, "Duke", source="espn")
        assert tid == 1  # same team, not duplicated

    def test_updates_conference(self, conn):
        get_or_create_team(conn, "Duke")
        get_or_create_team(conn, "Duke", conference="ACC")
        row = conn.execute("SELECT conference FROM teams WHERE id=1").fetchone()
        assert row["conference"] == "ACC"

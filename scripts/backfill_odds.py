#!/usr/bin/env python3
"""Backfill historical Vegas closing lines from ESPN's pickcenter API.

Fetches the ESPN scoreboard for each game day, matches games to our DB,
then pulls spread/total/moneyline from the summary endpoint.

Usage:
    python -u scripts/backfill_odds.py --seasons 2023        # single season
    python -u scripts/backfill_odds.py --seasons 2018-2023   # range
    python -u scripts/backfill_odds.py --seasons 2015-2023,2026  # multiple
    python -u scripts/backfill_odds.py --seasons 2023 --delay 1.0  # slower

Resumes automatically — skips games that already have odds.
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_predictions.db import get_db, resolve_team
from sports_predictions.odds import ESPN_SCOREBOARD, ESPN_SUMMARY

SPORT = "ncaa_basketball"

# ESPN shortDisplayName overrides (import from scraper)
from sports_predictions.scrapers.ncaa_basketball import ESPN_NAME_OVERRIDES


def parse_seasons(s: str) -> list[int]:
    """Parse season spec like '2018-2023,2026' into a list of ints."""
    seasons = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            seasons.extend(range(int(start), int(end) + 1))
        else:
            seasons.append(int(part))
    return sorted(set(seasons))


def get_season_dates(season: int) -> list[str]:
    """Generate all dates in a CBB season (Nov 1 through April 15)."""
    from datetime import timedelta
    start = date(season - 1, 11, 1)
    end = date(season, 4, 15)
    dates = []
    d = start
    while d <= end:
        dates.append(d.isoformat())
        d += timedelta(days=1)
    return dates


def fetch_scoreboard(date_str: str) -> list[dict]:
    """Fetch ESPN scoreboard for a date, return event info."""
    ymd = date_str.replace("-", "")
    try:
        resp = requests.get(ESPN_SCOREBOARD, params={
            "groups": "50", "limit": "200", "dates": ymd,
        }, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Scoreboard error for {date_str}: {e}")
        return []

    events = []
    for event in resp.json().get("events", []):
        comps = event.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        home_name = away_name = ""
        home_score = away_score = None
        for c in competitors:
            team = c.get("team", {})
            name = team.get("shortDisplayName", team.get("displayName", ""))
            score = c.get("score")
            if c.get("homeAway") == "home":
                home_name = name
                if score is not None:
                    try:
                        home_score = int(score)
                    except (ValueError, TypeError):
                        pass
            else:
                away_name = name
                if score is not None:
                    try:
                        away_score = int(score)
                    except (ValueError, TypeError):
                        pass

        events.append({
            "event_id": event.get("id"),
            "home": home_name,
            "away": away_name,
            "home_score": home_score,
            "away_score": away_score,
        })

    return events


def fetch_event_odds(event_id: str) -> dict | None:
    """Fetch pickcenter odds for an ESPN event."""
    try:
        resp = requests.get(ESPN_SUMMARY, params={"event": event_id},
                            timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    pickcenter = resp.json().get("pickcenter", [])
    if not pickcenter:
        return None

    # Pick the best provider (prefer DraftKings, then consensus, then any)
    pick = pickcenter[0]
    for p in pickcenter:
        provider = p.get("provider", {}).get("name", "")
        if "draft" in provider.lower() or "king" in provider.lower():
            pick = p
            break
        if "consensus" in provider.lower():
            pick = p

    provider = pick.get("provider", {}).get("name", "unknown")

    result = {"provider": provider}

    # Spread (from ESPN home perspective)
    spread_data = pick.get("pointSpread", {})
    if spread_data:
        home_line = spread_data.get("home", {}).get("close", {}).get("line")
        if home_line is not None:
            try:
                result["spread"] = float(home_line)
            except (ValueError, TypeError):
                pass  # "OFF" or other non-numeric values
    if "spread" not in result and "spread" in pick:
        try:
            result["spread"] = float(pick["spread"])
        except (ValueError, TypeError):
            pass

    # Total
    total_data = pick.get("total", {})
    if total_data:
        over_line = total_data.get("over", {}).get("close", {}).get("line", "")
        if over_line:
            try:
                result["total"] = float(str(over_line).lstrip("oO"))
            except (ValueError, TypeError):
                pass
    if "total" not in result and "overUnder" in pick:
        try:
            result["total"] = float(pick["overUnder"])
        except (ValueError, TypeError):
            pass

    # Moneyline
    def safe_int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    ml_data = pick.get("moneyline", {})
    if ml_data:
        home_ml = safe_int(ml_data.get("home", {}).get("close", {}).get("odds"))
        away_ml = safe_int(ml_data.get("away", {}).get("close", {}).get("odds"))
        if home_ml is not None:
            result["home_ml"] = home_ml
        if away_ml is not None:
            result["away_ml"] = away_ml

    # Fallback moneyline from homeTeamOdds/awayTeamOdds
    if "home_ml" not in result:
        home_odds = pick.get("homeTeamOdds", {})
        if home_odds and "moneyLine" in home_odds:
            ml = safe_int(home_odds["moneyLine"])
            if ml is not None:
                result["home_ml"] = ml
    if "away_ml" not in result:
        away_odds = pick.get("awayTeamOdds", {})
        if away_odds and "moneyLine" in away_odds:
            ml = safe_int(away_odds["moneyLine"])
            if ml is not None:
                result["away_ml"] = ml

    if "spread" not in result and "total" not in result:
        return None

    return result


def match_and_update(conn, date_str: str, season: int,
                     espn_events: list[dict],
                     delay: float) -> tuple[int, int]:
    """Match ESPN events to DB games and fetch odds. Returns (matched, updated)."""
    matched = 0
    updated = 0

    for event in espn_events:
        espn_home = ESPN_NAME_OVERRIDES.get(event["home"], event["home"])
        espn_away = ESPN_NAME_OVERRIDES.get(event["away"], event["away"])

        home_id = resolve_team(conn, espn_home)
        away_id = resolve_team(conn, espn_away)

        if not home_id or not away_id:
            continue

        # Try matching by date first (works for ESPN-sourced games)
        game = conn.execute(
            "SELECT id, vegas_spread FROM games "
            "WHERE date = ? AND home_team_id = ? AND away_team_id = ?",
            (date_str, home_id, away_id)
        ).fetchone()
        flipped = False

        if not game:
            game = conn.execute(
                "SELECT id, vegas_spread FROM games "
                "WHERE date = ? AND home_team_id = ? AND away_team_id = ?",
                (date_str, away_id, home_id)
            ).fetchone()
            flipped = True

        # Fallback: match by teams + scores within season
        # (Kaggle dates can be offset from ESPN dates)
        if not game and event.get("home_score") and event.get("away_score"):
            hs = event["home_score"]
            as_ = event["away_score"]
            game = conn.execute(
                "SELECT id, vegas_spread FROM games "
                "WHERE season = ? AND home_team_id = ? AND away_team_id = ? "
                "AND home_score = ? AND away_score = ? AND vegas_spread IS NULL",
                (season, home_id, away_id, hs, as_)
            ).fetchone()
            flipped = False

            if not game:
                game = conn.execute(
                    "SELECT id, vegas_spread FROM games "
                    "WHERE season = ? AND home_team_id = ? AND away_team_id = ? "
                    "AND home_score = ? AND away_score = ? AND vegas_spread IS NULL",
                    (season, away_id, home_id, as_, hs)
                ).fetchone()
                flipped = True

        if not game:
            continue

        matched += 1

        # Skip if already has odds
        if game["vegas_spread"] is not None:
            continue

        # Fetch odds
        odds = fetch_event_odds(event["event_id"])
        time.sleep(delay)

        if not odds or "spread" not in odds:
            continue

        # ESPN spread is from ESPN home perspective (negative = home favored).
        # Our DB stores spread from our home perspective.
        spread = odds["spread"]
        home_ml = odds.get("home_ml")
        away_ml = odds.get("away_ml")

        if flipped:
            # Our DB home = ESPN away, so negate spread and swap MLs
            spread = -spread
            home_ml, away_ml = away_ml, home_ml

        conn.execute(
            "UPDATE games SET vegas_spread = ?, vegas_total = ?, "
            "vegas_home_ml = ?, vegas_away_ml = ?, odds_provider = ? "
            "WHERE id = ?",
            (spread, odds.get("total"), home_ml, away_ml,
             odds["provider"], game["id"])
        )
        updated += 1

    return matched, updated


def main():
    parser = argparse.ArgumentParser(description="Backfill ESPN odds")
    parser.add_argument("--seasons", required=True,
                        help="Seasons to process (e.g. '2018-2023,2026')")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between summary requests (default: 0.5s)")
    args = parser.parse_args()

    seasons = parse_seasons(args.seasons)
    print(f"Backfilling odds for seasons: {seasons}")

    conn = get_db(SPORT)
    conn.execute("PRAGMA busy_timeout = 30000")  # Wait up to 30s for locks

    total_matched = 0
    total_updated = 0

    for season in seasons:
        dates = get_season_dates(season)
        print(f"\nSeason {season}: {len(dates)} days to scan")

        # Check how many games already have odds
        existing = conn.execute(
            "SELECT COUNT(*) as c FROM games "
            "WHERE season = ? AND vegas_spread IS NOT NULL",
            (season,)
        ).fetchone()["c"]
        total_games = conn.execute(
            "SELECT COUNT(*) as c FROM games WHERE season = ?",
            (season,)
        ).fetchone()["c"]
        print(f"  {existing}/{total_games} games already have odds")

        # Check if season is already fully backfilled
        pending = conn.execute(
            "SELECT COUNT(*) as c FROM games "
            "WHERE season = ? AND vegas_spread IS NULL",
            (season,)
        ).fetchone()["c"]
        if pending == 0:
            print("  All games already have odds, skipping")
            continue

        for i, date_str in enumerate(dates):
            espn_events = fetch_scoreboard(date_str)
            time.sleep(0.3)  # Be polite between scoreboard requests

            if not espn_events:
                continue

            matched, updated = match_and_update(
                conn, date_str, season, espn_events, args.delay
            )
            total_matched += matched
            total_updated += updated

            if updated > 0:
                conn.commit()

            # Progress
            sys.stdout.write(
                f"\r  {date_str} ({i+1}/{len(dates)}) - "
                f"matched: {matched}, updated: {updated}"
            )
            sys.stdout.flush()

        # Final commit for season
        conn.commit()
        print(f"\n  Season {season} done")

    # Summary
    final = conn.execute(
        "SELECT COUNT(*) as c FROM games WHERE vegas_spread IS NOT NULL"
    ).fetchone()["c"]
    print(f"\nBackfill complete: {total_updated} games updated, "
          f"{final} total games with odds")

    conn.close()


if __name__ == "__main__":
    main()

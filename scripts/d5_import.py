#!/usr/bin/env python3
"""Import all D5 Hockey data and compute player ratings.

Usage:
    python scripts/d5_import.py           # Full scrape + ratings
    python scripts/d5_import.py --ratings  # Just recompute ratings
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from d5_hockey.scraper import scrape_all
from d5_hockey.aliases import apply_aliases
from d5_hockey.ratings import print_ratings, compute_player_ratings
from d5_hockey.db import get_db


def show_summary():
    """Print a summary of what's in the database."""
    conn = get_db()

    seasons = conn.execute("SELECT COUNT(*) as n FROM seasons").fetchone()["n"]
    players = conn.execute("SELECT COUNT(*) as n FROM players").fetchone()["n"]
    games = conn.execute("SELECT COUNT(*) as n FROM games").fetchone()["n"]
    stats = conn.execute(
        "SELECT COUNT(*) as n FROM player_season_stats"
    ).fetchone()["n"]
    rosters = conn.execute(
        "SELECT COUNT(*) as n FROM roster_entries"
    ).fetchone()["n"]

    conn.close()

    print(f"\n--- Database Summary ---")
    print(f"  Seasons:        {seasons}")
    print(f"  Players:        {players}")
    print(f"  Games:          {games}")
    print(f"  Stat entries:   {stats}")
    print(f"  Roster entries: {rosters}")


def main():
    parser = argparse.ArgumentParser(description="Import D5 Hockey data")
    parser.add_argument(
        "--ratings", action="store_true",
        help="Only recompute ratings (skip scraping)"
    )
    parser.add_argument(
        "--top", type=int, default=30,
        help="Number of top players to show (default: 30)"
    )
    args = parser.parse_args()

    if not args.ratings:
        print("=== Scraping District Five Hockey ===")
        scrape_all()

        print("\n=== Applying player aliases ===")
        apply_aliases()

    show_summary()

    print("\n=== Computing player ratings ===")
    compute_player_ratings()
    print_ratings(top_n=args.top)


if __name__ == "__main__":
    main()

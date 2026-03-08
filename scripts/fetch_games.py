#!/usr/bin/env python3
"""Fetch recent game results from ESPN and update the database.

Usage:
    python scripts/fetch_games.py 2026-03-02              # single date
    python scripts/fetch_games.py 2026-03-02 2026-03-08   # date range
    python scripts/fetch_games.py 2026-03-02 --skip-train  # skip retrain
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_predictions.scrapers.ncaa_basketball import (
    fetch_espn_games, compute_season_stats, _date_to_season
)
from sports_predictions.model import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NCAA basketball games from ESPN"
    )
    parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", nargs="?", help="End date (YYYY-MM-DD)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip model retraining")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = (datetime.strptime(args.end_date, "%Y-%m-%d")
                if args.end_date else start_date)

    # Fetch games for each date in range
    total = 0
    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        count = fetch_espn_games(date_str)
        total += count
        current += timedelta(days=1)

    print(f"\nTotal: {total} games imported")

    if total > 0:
        season = _date_to_season(start_date.strftime("%Y-%m-%d"))
        print(f"\nRecomputing stats for season {season}...")
        compute_season_stats(season)

        if not args.skip_train:
            print("\nRetraining model...")
            train_model("ncaa_basketball")
        else:
            print("\nSkipping model retrain (--skip-train)")


if __name__ == "__main__":
    main()

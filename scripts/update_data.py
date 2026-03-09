#!/usr/bin/env python3
"""Nightly data update script.

Updates game results and team stats, then retrains the model.
Designed to be run via cron or launchd.

Usage:
    python scripts/update_data.py [--season YEAR]
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_predictions.scrapers.ncaa_basketball import (
    compute_season_stats, fetch_espn_games, fetch_kenpom_four_factors,
    fetch_kenpom_ratings
)
from sports_predictions.model import train_model


def main():
    parser = argparse.ArgumentParser(description="Update NCAA basketball data")
    parser.add_argument(
        "--season", type=int,
        default=datetime.now().year,
        help="Season year (default: current year)"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip model retraining"
    )
    parser.add_argument(
        "--skip-kenpom", action="store_true",
        help="Skip KenPom data fetch"
    )
    parser.add_argument(
        "--skip-espn", action="store_true",
        help="Skip ESPN game fetch"
    )
    args = parser.parse_args()

    print(f"=== Updating NCAA Basketball data for {args.season} ===")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Step 1: Fetch recent game results from ESPN
    if not args.skip_espn:
        print("\n--- Fetching ESPN games ---")
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        fetch_espn_games(yesterday)
        fetch_espn_games(today)

    # Step 2: Compute basic stats from game results already in the DB
    print("\n--- Computing season stats ---")
    compute_season_stats(args.season)

    # Step 2: Fetch KenPom ratings via API
    if not args.skip_kenpom:
        print("\n--- Fetching KenPom ratings ---")
        fetch_kenpom_ratings(args.season)

        print("\n--- Fetching KenPom four-factors ---")
        fetch_kenpom_four_factors(args.season)

    # Step 3: Retrain model
    if not args.skip_train:
        print("\n--- Training model ---")
        train_model("ncaa_basketball")

    print("\n=== Update complete ===")


if __name__ == "__main__":
    main()

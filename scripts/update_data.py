#!/usr/bin/env python3
"""Nightly data update script.

Updates game results and team stats, then retrains the model.
Designed to be run via cron or launchd.

Usage:
    python scripts/update_data.py [--season YEAR] [--kenpom-user EMAIL --kenpom-pass PASS]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_predictions.scrapers.ncaa_basketball import (
    compute_season_stats, scrape_kenpom
)
from sports_predictions.model import train_model


def main():
    parser = argparse.ArgumentParser(description="Update NCAA basketball data")
    parser.add_argument(
        "--season", type=int,
        default=datetime.now().year,
        help="Season year (default: current year)"
    )
    parser.add_argument("--kenpom-user", help="KenPom email")
    parser.add_argument("--kenpom-pass", help="KenPom password")
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip model retraining"
    )
    args = parser.parse_args()

    print(f"=== Updating NCAA Basketball data for {args.season} ===")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Step 1: Compute basic stats from game results already in the DB
    print("\n--- Computing season stats ---")
    compute_season_stats(args.season)

    # Step 2: Scrape KenPom for advanced metrics
    if args.kenpom_user and args.kenpom_pass:
        print("\n--- Scraping KenPom ---")
        scrape_kenpom(args.season, args.kenpom_user, args.kenpom_pass)
    else:
        print("\n--- Scraping KenPom (free tier) ---")
        scrape_kenpom(args.season)

    # Step 3: Retrain model
    if not args.skip_train:
        print("\n--- Training model ---")
        train_model("ncaa_basketball")

    print("\n=== Update complete ===")


if __name__ == "__main__":
    main()

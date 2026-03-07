#!/usr/bin/env python3
"""Import historical data from Kaggle March Machine Learning Mania dataset.

Download the dataset from:
https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data

Place the CSV files in data/kaggle/ and run:
    python scripts/import_kaggle.py

Expected files in data/kaggle/:
    - MTeams.csv (team ID to name mapping)
    - MRegularSeasonCompactResults.csv (game results)
    - MNCAATourneyCompactResults.csv (tournament results)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_predictions.scrapers.ncaa_basketball import (
    import_kaggle_games, import_kaggle_tourney, compute_season_stats
)
from sports_predictions.model import train_model

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kaggle"


def main():
    if not DATA_DIR.exists():
        print(f"Please download Kaggle data to: {DATA_DIR}")
        print("https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data")
        sys.exit(1)

    regular = DATA_DIR / "MRegularSeasonCompactResults.csv"
    tourney = DATA_DIR / "MNCAATourneyCompactResults.csv"

    if regular.exists():
        print("Importing regular season games...")
        import_kaggle_games(str(regular))
    else:
        print(f"Not found: {regular}")

    if tourney.exists():
        print("\nImporting tournament games...")
        import_kaggle_tourney(str(tourney))
    else:
        print(f"Not found: {tourney}")

    # Compute stats for each season
    from sports_predictions.db import get_db
    conn = get_db("ncaa_basketball")
    seasons = [r["season"] for r in conn.execute(
        "SELECT DISTINCT season FROM games ORDER BY season"
    ).fetchall()]
    conn.close()

    print(f"\nComputing stats for {len(seasons)} seasons...")
    for season in seasons:
        compute_season_stats(season)

    # Train model
    print("\nTraining model on all historical data...")
    train_model("ncaa_basketball")


if __name__ == "__main__":
    main()

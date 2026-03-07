#!/usr/bin/env python3
"""Predict the outcome of a game.

Usage:
    python scripts/predict.py "Duke" "North Carolina" --season 2025
    python scripts/predict.py "Duke" "North Carolina" --neutral
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_predictions.model import predict_game


def main():
    parser = argparse.ArgumentParser(description="Predict a game outcome")
    parser.add_argument("home_team", help="Home team name")
    parser.add_argument("away_team", help="Away team name")
    parser.add_argument(
        "--sport", default="ncaa_basketball",
        help="Sport database to use (default: ncaa_basketball)"
    )
    parser.add_argument(
        "--season", type=int, default=datetime.now().year,
        help="Season year (default: current year)"
    )
    parser.add_argument(
        "--neutral", action="store_true",
        help="Game is at a neutral site"
    )
    args = parser.parse_args()

    result = predict_game(
        args.sport, args.home_team, args.away_team,
        args.season, neutral_site=args.neutral
    )

    print(f"\n{'='*50}")
    print(f"  {result['home_team']} vs {result['away_team']}")
    if args.neutral:
        print(f"  (Neutral site)")
    print(f"{'='*50}")
    print(f"  {result['home_team']}: {result['home_win_prob']*100:.1f}%")
    print(f"  {result['away_team']}: {result['away_win_prob']*100:.1f}%")
    print(f"\n  Predicted winner: {result['predicted_winner']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

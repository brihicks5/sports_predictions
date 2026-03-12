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

    home = result['home_team']
    away = result['away_team']

    print(f"\n{'='*50}")
    print(f"  {home} vs {away}")
    if args.neutral:
        print(f"  (Neutral site)")
    print(f"{'='*50}")
    print(f"\n  Win probability:")
    print(f"    {home}: {result['home_win_prob']*100:.1f}%")
    print(f"    {away}: {result['away_win_prob']*100:.1f}%")
    print(f"\n  Predicted score:")
    print(f"    {home} {result['predicted_home_score']} - "
          f"{away} {result['predicted_away_score']}")
    print(f"    Margin: {abs(result['predicted_margin']):.1f} pts")
    print(f"\n  Predicted winner: {result['predicted_winner']}")

    # Show injuries if any
    home_injuries = result.get('home_injuries', [])
    away_injuries = result.get('away_injuries', [])
    if home_injuries or away_injuries:
        print(f"\n  Injuries:")
        for inj in home_injuries:
            status = inj['status'].upper()
            injury = inj.get('injury_type') or 'unknown'
            ret = inj.get('expected_return') or ''
            ret_str = f" ({ret})" if ret else ""
            print(f"    {home}: {inj['player_name']} - "
                  f"{status}, {injury}{ret_str}")
        for inj in away_injuries:
            status = inj['status'].upper()
            injury = inj.get('injury_type') or 'unknown'
            ret = inj.get('expected_return') or ''
            ret_str = f" ({ret})" if ret else ""
            print(f"    {away}: {inj['player_name']} - "
                  f"{status}, {injury}{ret_str}")
        print(f"\n  NOTE: Injuries are NOT factored into the prediction.")
        print(f"  Use your judgment to adjust the margin accordingly.")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

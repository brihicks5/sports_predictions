#!/usr/bin/env python3
"""Predict the outcome of a game.

Usage:
    python scripts/predict.py "Duke" "North Carolina" --season 2025
    python scripts/predict.py "Duke" "North Carolina" --neutral
"""

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_predictions.model import predict_game
from sports_predictions.odds import fetch_game_odds


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
    model_margin = result['predicted_margin']
    model_total = result.get('predicted_total',
                             result['predicted_home_score']
                             + result['predicted_away_score'])

    # Fetch Vegas odds
    odds = fetch_game_odds(home, away)

    # Blend model and Vegas if odds available
    BLEND_WEIGHT = 0.5  # 0.5 = equal weight model and Vegas
    if odds and "spread" in odds:
        vegas_spread = odds["spread"]
        blended_margin = (
            (1 - BLEND_WEIGHT) * model_margin
            + BLEND_WEIGHT * vegas_spread
        )
    else:
        vegas_spread = None
        blended_margin = model_margin

    if odds and "total" in odds:
        vegas_total = odds["total"]
        blended_total = (
            (1 - BLEND_WEIGHT) * model_total
            + BLEND_WEIGHT * vegas_total
        )
    else:
        vegas_total = None
        blended_total = model_total

    # Derive blended scores
    rounded_margin = round(blended_margin)
    rounded_total = round(blended_total)
    if rounded_total % 2 != rounded_margin % 2:
        rounded_total += 1
    blended_home_score = (rounded_total + rounded_margin) // 2
    blended_away_score = (rounded_total - rounded_margin) // 2

    # Blended win probability (reuse model's logistic k)
    k = result.get('margin_to_win_k')
    if k:
        blended_win_prob = 1.0 / (1.0 + math.exp(-k * blended_margin))
    else:
        blended_win_prob = result['home_win_prob']

    print(f"\n{'='*55}")
    print(f"  {home} vs {away}")
    if args.neutral:
        print(f"  (Neutral site)")
    print(f"{'='*55}")

    if vegas_spread is not None:
        # Format spreads in standard betting notation: "TEAM -X.X"
        def spread_str(margin, home_name, away_name):
            if margin > 0:
                return f"{home_name} -{abs(margin):.1f}"
            elif margin < 0:
                return f"{away_name} -{abs(margin):.1f}"
            else:
                return "PICK"

        m_str = spread_str(model_margin, home, away)
        v_str = spread_str(vegas_spread, home, away)
        b_str = spread_str(blended_margin, home, away)

        print(f"\n  {'':14s} {'Model':>18s} {'Vegas':>18s} {'Blended':>18s}")
        print(f"  {'':14s} {'-----':>18s} {'-----':>18s} {'-------':>18s}")
        print(f"  {'Spread':14s} {m_str:>18s} {v_str:>18s} {b_str:>18s}")

        # Total
        m_tot = f"{model_total:.1f}"
        v_tot = f"{vegas_total:.1f}" if vegas_total else "n/a"
        b_tot = f"{blended_total:.1f}"
        print(f"  {'Total':14s} {m_tot:>18s} {v_tot:>18s} {b_tot:>18s}")

        # Win prob
        m_wp = f"{result['home_win_prob']*100:.1f}%"
        b_wp = f"{blended_win_prob*100:.1f}%"
        print(f"  {'Win prob':14s} {m_wp:>18s} {'':>18s} {b_wp:>18s}")

        print(f"\n  Blended score: {home} {blended_home_score} - "
              f"{away} {blended_away_score}")
        print(f"  Blended winner: "
              f"{home if blended_margin > 0 else away}")

        # Confidence and ATS assessment
        uncertainty = result.get('uncertainty')
        diff = abs(model_margin - vegas_spread)
        if uncertainty is not None:
            print(f"\n  Uncertainty: ±{uncertainty:.1f} pts")

            # ATS edge assessment:
            # Best ATS picks are when model disagrees with Vegas AND
            # uncertainty is low relative to the disagreement.
            if diff >= 2:
                if model_margin > vegas_spread:
                    edge_team = home
                    edge_line = -vegas_spread
                else:
                    edge_team = away
                    edge_line = vegas_spread

                # Edge ratio: how much disagreement vs uncertainty
                edge_ratio = diff / uncertainty
                if edge_ratio > 0.6:
                    strength = "STRONG"
                elif edge_ratio > 0.35:
                    strength = "LEAN"
                else:
                    strength = None

                if strength:
                    print(f"  ** ATS {strength}: {edge_team} {edge_line:+.1f} "
                          f"(model disagrees by {diff:.1f} pts) **")
                else:
                    print(f"  Model and Vegas disagree by {diff:.1f} pts "
                          f"(low conviction)")
        elif diff >= 4:
            print(f"\n  ** Model and Vegas disagree by {diff:.1f} pts **")
    else:
        # No odds available — show original output
        print(f"\n  Win probability:")
        print(f"    {home}: {result['home_win_prob']*100:.1f}%")
        print(f"    {away}: {result['away_win_prob']*100:.1f}%")
        print(f"\n  Predicted score:")
        print(f"    {home} {result['predicted_home_score']} - "
              f"{away} {result['predicted_away_score']}")
        print(f"    Margin: {abs(result['predicted_margin']):.1f} pts")
        print(f"\n  Predicted winner: {result['predicted_winner']}")
        uncertainty = result.get('uncertainty')
        if uncertainty is not None:
            print(f"  Uncertainty: ±{uncertainty:.1f} pts")
        if odds is None:
            print(f"\n  (No Vegas odds available — game may not be on today's slate)")

    print(f"{'='*55}")


if __name__ == "__main__":
    main()

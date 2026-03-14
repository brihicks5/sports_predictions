#!/usr/bin/env python3
"""Predict all games for a given date using ESPN's scoreboard.

Usage:
    python scripts/predict_slate.py                    # today's games
    python scripts/predict_slate.py --date 2026-03-13  # specific date
    python scripts/predict_slate.py --season 2026
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_predictions.model import predict_game
from sports_predictions.odds import fetch_slate


def spread_str(margin, home_name, away_name):
    """Format a margin as a betting spread string."""
    if margin > 0:
        return f"{home_name} -{abs(margin):.1f}"
    elif margin < 0:
        return f"{away_name} -{abs(margin):.1f}"
    else:
        return "PICK"


def main():
    parser = argparse.ArgumentParser(description="Predict a day's slate")
    parser.add_argument(
        "--sport", default="ncaa_basketball",
        help="Sport database to use (default: ncaa_basketball)"
    )
    parser.add_argument(
        "--season", type=int, default=datetime.now().year,
        help="Season year (default: current year)"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date to predict (YYYY-MM-DD, default: today)"
    )
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else date.today()
    print(f"Fetching games for {target_date}...")

    slate = fetch_slate(target_date)
    if not slate:
        print("No games found.")
        return

    print(f"Found {len(slate)} games.\n")

    results = []
    for game in slate:
        home = game["home"]
        away = game["away"]
        neutral = game["neutral"]

        vegas_spread = game.get("spread")

        try:
            result = predict_game(
                args.sport, home, away, args.season, neutral_site=neutral,
                vegas_spread=vegas_spread
            )
        except (ValueError, FileNotFoundError):
            # Team not in our database (non-D1, etc.)
            continue

        model_margin = result["predicted_margin"]
        calibrated_margin = result.get("calibrated_margin", model_margin)
        win_prob = result["home_win_prob"]
        uncertainty = result.get("uncertainty")
        ats_cover = result.get("ats_cover_margin")

        # ATS pick (from ATS model)
        # vegas_spread is margin convention (positive = home favored)
        ats = ""
        if ats_cover is not None and vegas_spread is not None:
            if abs(ats_cover) >= 2:
                if ats_cover > 0:
                    # Home covers
                    edge_team = home
                    edge_line = -vegas_spread  # home's line
                else:
                    # Away covers
                    edge_team = away
                    edge_line = vegas_spread  # away's line
                if abs(ats_cover) >= 3:
                    ats = f"{edge_team} {edge_line:+.1f}"
                else:
                    ats = f"{edge_team} {edge_line:+.1f} (lean)"

        # Game result
        home_score = game.get("home_score")
        away_score = game.get("away_score")
        is_final = game["status"] == "STATUS_FINAL"

        score_str = ""
        winner_result = ""
        ats_result = ""
        if is_final and home_score is not None and away_score is not None:
            actual_margin = home_score - away_score
            score_str = f"{away} {away_score}-{home_score}"

            # Did model pick the winner correctly?
            model_correct = ((model_margin > 0 and actual_margin > 0)
                             or (model_margin < 0 and actual_margin < 0))
            if actual_margin == 0:
                winner_result = "PUSH"
            else:
                winner_result = "Y" if model_correct else "N"

            # ATS result (only if we flagged a pick)
            if ats and vegas_spread is not None and ats_cover is not None:
                # vegas_spread is margin convention (pos = home fav)
                # cover formula needs DB convention (neg = home fav)
                actual_cover = actual_margin - vegas_spread
                if actual_cover == 0:
                    ats_result = "PUSH"
                else:
                    # Did our pick direction match?
                    correct = (ats_cover > 0) == (actual_cover > 0)
                    ats_result = "Y" if correct else "N"

        n = " (N)" if neutral else ""
        results.append({
            "game": f"{away} vs {home}{n}",
            "model": spread_str(model_margin, home, away),
            "vegas": spread_str(vegas_spread, home, away) if vegas_spread is not None else "n/a",
            "win_prob": f"{max(win_prob, 1-win_prob)*100:.0f}%",
            "ats": ats,
            "score": score_str,
            "winner_result": winner_result,
            "ats_result": ats_result,
        })

    if not results:
        print("No matchable games found.")
        return

    # Print table
    headers = ["Game", "Model", "Vegas", "Win%", "ATS Pick", "Result", "Win?", "ATS?"]
    rows = [[r["game"], r["model"], r["vegas"], r["win_prob"], r["ats"],
             r["score"], r["winner_result"], r["ats_result"]]
            for r in results]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "  ".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    print()

    # Save to file
    slates_dir = Path(__file__).resolve().parent.parent / "data" / "slates"
    slates_dir.mkdir(exist_ok=True)
    out_path = slates_dir / f"{target_date}.txt"
    with open(out_path, "w") as f:
        f.write(f"{header_line}\n")
        f.write(f"{sep_line}\n")
        for row in rows:
            f.write("  ".join(cell.ljust(widths[i])
                              for i, cell in enumerate(row)) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

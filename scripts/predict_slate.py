#!/usr/bin/env python3
"""Predict all games for a given date using ESPN's scoreboard.

Usage:
    python scripts/predict_slate.py                    # today's games
    python scripts/predict_slate.py --date 2026-03-13  # specific date
    python scripts/predict_slate.py --season 2026
"""

import argparse
import re
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


def parse_slate_file(path):
    """Parse an existing slate file into header info and row dicts.

    Returns (headers, col_slices, rows) where each row is a dict with
    raw column values keyed by header name.
    """
    with open(path) as f:
        lines = f.readlines()

    if len(lines) < 2:
        return None, None, []

    header_line = lines[0]
    sep_line = lines[1]

    # Determine column boundaries from the separator line
    # Each column is a run of dashes, separated by two spaces
    col_slices = []
    i = 0
    while i < len(sep_line.rstrip()):
        # Skip spaces
        while i < len(sep_line.rstrip()) and sep_line[i] == ' ':
            i += 1
        if i >= len(sep_line.rstrip()):
            break
        start = i
        while i < len(sep_line.rstrip()) and sep_line[i] == '-':
            i += 1
        col_slices.append((start, i))

    headers = [header_line[s:e].strip() for s, e in col_slices]

    rows = []
    for line in lines[2:]:
        if not line.strip():
            continue
        row = {}
        for h, (s, e) in zip(headers, col_slices):
            row[h] = line[s:e].strip() if s < len(line) else ""
        rows.append(row)

    return headers, col_slices, rows


def parse_model_margin(model_str):
    """Extract model margin (home perspective) from spread string like 'Duke -2.6' or 'PICK'."""
    if model_str == "PICK":
        return 0.0
    # Format: "TeamName -X.X" — negative means that team is favored
    m = re.match(r'^(.+)\s+([+-]?\d+\.?\d*)$', model_str.strip())
    if not m:
        return None
    return float(m.group(2))


def parse_vegas_spread(vegas_str):
    """Extract Vegas spread value from string like 'Duke -6.5'."""
    if vegas_str in ("PICK", "n/a", ""):
        return None
    m = re.match(r'^(.+)\s+([+-]?\d+\.?\d*)$', vegas_str.strip())
    if not m:
        return None
    return float(m.group(2))


def parse_game_teams(game_str):
    """Parse 'Away vs Home (N)' into (away, home)."""
    neutral = game_str.endswith("(N)")
    clean = game_str.replace("(N)", "").strip()
    parts = clean.split(" vs ")
    if len(parts) != 2:
        return None, None
    return parts[0].strip(), parts[1].strip()


def update_results_only(target_date, slate_path):
    """Update an existing slate file with game results without re-predicting."""
    headers, col_slices, rows = parse_slate_file(slate_path)
    if not rows:
        print(f"No rows found in {slate_path}")
        return

    # Fetch scores from ESPN
    print(f"Fetching scores for {target_date}...")
    slate = fetch_slate(target_date)
    if not slate:
        print("No games found on ESPN.")
        return

    # Build lookup: normalize team names for matching
    def normalize(name):
        return name.lower().strip().replace("'", "'")

    espn_games = {}
    for g in slate:
        key = (normalize(g["away"]), normalize(g["home"]))
        espn_games[key] = g
        # Also store reverse for matching flexibility
        espn_games[(normalize(g["home"]), normalize(g["away"]))] = g

    updated = 0
    for row in rows:
        away, home = parse_game_teams(row.get("Game", ""))
        if not away or not home:
            continue

        # Find matching ESPN game
        key = (normalize(away), normalize(home))
        game = espn_games.get(key)
        if not game:
            continue

        is_final = game["status"] == "STATUS_FINAL"
        home_score = game.get("home_score")
        away_score = game.get("away_score")

        if not is_final or home_score is None or away_score is None:
            continue

        actual_margin = home_score - away_score
        score_str = f"{away} {away_score}-{home_score}"
        row["Result"] = score_str

        # Determine Win? from Model column
        model_str = row.get("Model", "")
        # The model spread string names the favored team; negative line = favored
        # We need to figure out model_margin in home perspective
        # "Home -X" means home favored, model_margin > 0
        # "Away -X" means away favored, model_margin < 0
        if model_str == "PICK":
            model_margin = 0.0
        else:
            m = re.match(r'^(.+?)\s+(-?\d+\.?\d*)$', model_str.strip())
            if m:
                fav_team = m.group(1).strip()
                line = float(m.group(2))
                # line is always negative in display (e.g. "Duke -2.6")
                # If fav_team == home, model_margin > 0
                if normalize(fav_team) == normalize(home):
                    model_margin = abs(line)
                else:
                    model_margin = -abs(line)
            else:
                model_margin = None

        if model_margin is not None:
            if actual_margin == 0:
                row["Win?"] = "PUSH"
            else:
                model_correct = ((model_margin > 0 and actual_margin > 0)
                                 or (model_margin < 0 and actual_margin < 0)
                                 or (model_margin == 0))  # PICK = no prediction
                row["Win?"] = "Y" if model_correct else "N"

        # Determine ATS? from ATS Pick and Vegas columns
        ats_pick = row.get("ATS Pick", "").strip()
        vegas_str = row.get("Vegas", "")
        if ats_pick and vegas_str not in ("n/a", ""):
            # Parse vegas spread (home perspective)
            # Vegas string: "TeamName -X.X"
            vm = re.match(r'^(.+?)\s+(-?\d+\.?\d*)$', vegas_str.strip())
            if vm:
                vfav = vm.group(1).strip()
                vline = float(vm.group(2))
                if normalize(vfav) == normalize(home):
                    vegas_spread = abs(vline)  # home favored, positive
                else:
                    vegas_spread = -abs(vline)  # away favored, negative

                actual_cover = actual_margin - vegas_spread

                # Parse ATS pick direction from Cover column
                cover_str = row.get("Cover", "").strip()
                if cover_str:
                    ats_cover = float(cover_str)
                    if actual_cover == 0:
                        row["ATS?"] = "PUSH"
                    else:
                        correct = (ats_cover > 0) == (actual_cover > 0)
                        row["ATS?"] = "Y" if correct else "N"

        updated += 1

    # Rewrite the file
    widths = [e - s for s, e in col_slices]
    # Recalculate widths in case result strings are wider
    for i, h in enumerate(headers):
        widths[i] = max(widths[i], len(h))
        for row in rows:
            widths[i] = max(widths[i], len(row.get(h, "")))

    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "  ".join("-" * widths[i] for i in range(len(headers)))

    with open(slate_path, "w") as f:
        f.write(f"{header_line}\n")
        f.write(f"{sep_line}\n")
        for row in rows:
            line = "  ".join(row.get(h, "").ljust(widths[i])
                            for i, h in enumerate(headers))
            f.write(f"{line}\n")

    print(f"Updated {updated} games in {slate_path}")

    # Print the updated file
    with open(slate_path) as f:
        print(f.read())


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
    parser.add_argument(
        "--results-only", action="store_true",
        help="Update an existing slate file with results only (no re-prediction)"
    )
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else date.today()

    if args.results_only:
        slates_dir = Path(__file__).resolve().parent.parent / "data" / "slates"
        slate_path = slates_dir / f"{target_date}.txt"
        if not slate_path.exists():
            print(f"No slate file found at {slate_path}")
            return
        update_results_only(target_date, slate_path)
        return

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
            "ats_cover": f"{ats_cover:+.1f}" if ats_cover is not None else "",
            "ats": ats,
            "score": score_str,
            "winner_result": winner_result,
            "ats_result": ats_result,
        })

    if not results:
        print("No matchable games found.")
        return

    # Print table
    headers = ["Game", "Model", "Vegas", "Win%", "Cover", "ATS Pick", "Result", "Win?", "ATS?"]
    rows = [[r["game"], r["model"], r["vegas"], r["win_prob"], r["ats_cover"],
             r["ats"], r["score"], r["winner_result"], r["ats_result"]]
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

#!/usr/bin/env python3
"""Fix neutral-site games where the winner was always assigned as home.

The Kaggle import assigned the winning team as 'home' for all neutral-site
games. This creates a bias where neutral-site home teams always win.

This script randomly swaps home/away for ~50% of neutral-site games in
seasons 2015-2023 (Kaggle data), so the model sees a realistic distribution.

Uses a fixed random seed for reproducibility.
"""

import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ncaa_basketball.db"


def main():
    random.seed(42)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Find all neutral-site games from Kaggle seasons
    games = cur.execute("""
        SELECT id, home_team_id, away_team_id, home_score, away_score,
               vegas_spread, vegas_home_ml, vegas_away_ml
        FROM games
        WHERE neutral_site = 1 AND season BETWEEN 2015 AND 2023
              AND home_score IS NOT NULL
    """).fetchall()

    flipped = 0
    for game in games:
        if random.random() < 0.5:
            continue

        # Swap home/away
        spread = game["vegas_spread"]
        cur.execute("""
            UPDATE games SET
                home_team_id = ?,
                away_team_id = ?,
                home_score = ?,
                away_score = ?,
                vegas_spread = ?,
                vegas_home_ml = ?,
                vegas_away_ml = ?
            WHERE id = ?
        """, (
            game["away_team_id"],
            game["home_team_id"],
            game["away_score"],
            game["home_score"],
            -spread if spread is not None else None,
            game["vegas_away_ml"],
            game["vegas_home_ml"],
            game["id"],
        ))
        flipped += 1

    conn.commit()
    conn.close()

    print(f"Processed {len(games)} neutral-site games (2015-2023)")
    print(f"Flipped {flipped} games ({100 * flipped / len(games):.1f}%)")

    # Verify
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) as total,
            SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) as home_wins
        FROM games
        WHERE neutral_site = 1 AND season BETWEEN 2015 AND 2023
              AND home_score IS NOT NULL
    """)
    total, hw = cur.fetchone()
    print(f"Neutral-site home win rate: {hw}/{total} ({100 * hw / total:.1f}%)")
    conn.close()


if __name__ == "__main__":
    main()

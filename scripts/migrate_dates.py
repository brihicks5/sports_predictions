#!/usr/bin/env python3
"""One-time migration: convert Kaggle ordinal dates to ISO dates.

Converts dates like "2026-118" (season 2026, day 118) to "2026-03-01"
using the formula: day 0 = first Monday in November of (season - 1).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_predictions.db import get_db
from sports_predictions.scrapers.ncaa_basketball import kaggle_day_to_date


def main():
    conn = get_db("ncaa_basketball")

    # Find all games with ordinal dates (format: "YYYY-NNN", not "YYYY-MM-DD")
    rows = conn.execute("""
        SELECT id, season, date FROM games
        WHERE date NOT LIKE '____-__-__'
    """).fetchall()

    print(f"Found {len(rows)} games with ordinal dates")

    updated = 0
    for row in rows:
        season = row["season"]
        old_date = row["date"]
        day_num = int(old_date.split("-")[1])
        new_date = kaggle_day_to_date(season, day_num)

        conn.execute("UPDATE games SET date = ? WHERE id = ?",
                      (new_date, row["id"]))
        updated += 1

        if updated % 10000 == 0:
            conn.commit()
            print(f"  Updated {updated} / {len(rows)}...")

    conn.commit()
    conn.close()
    print(f"Converted {updated} dates")

    # Verify
    conn = get_db("ncaa_basketball")
    remaining = conn.execute("""
        SELECT COUNT(*) as c FROM games WHERE date NOT LIKE '____-__-__'
    """).fetchone()["c"]
    sample = conn.execute("""
        SELECT date, season FROM games ORDER BY date LIMIT 5
    """).fetchall()
    conn.close()

    print(f"Remaining ordinal dates: {remaining}")
    print("Sample dates:")
    for s in sample:
        print(f"  {s['date']} (season {s['season']})")


if __name__ == "__main__":
    main()

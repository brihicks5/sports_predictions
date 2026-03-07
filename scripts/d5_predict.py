#!/usr/bin/env python3
"""Predict D5 Hockey matchups based on rosters.

Usage:
    python scripts/d5_predict.py  # Interactive mode — enter rosters

You can also import and call predict_matchup() programmatically.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from d5_hockey.db import get_db, resolve_player, get_or_create_player
from d5_hockey.ratings import (
    compute_player_ratings, predict_matchup, compute_team_strength, ROOKIE_RATING
)


def lookup_players(names: list, conn) -> list:
    """Resolve a list of player names to IDs, flagging unknowns."""
    ids = []
    for name in names:
        name = name.strip()
        if not name:
            continue
        pid = resolve_player(conn, name)
        if pid is None:
            print(f"    [ROOKIE] {name} — no history found")
            pid = get_or_create_player(conn, name)
        ids.append(pid)
    return ids


def main():
    conn = get_db()
    print("Computing player ratings...")
    ratings = compute_player_ratings()
    print(f"Rated {len(ratings)} players. Rookie baseline: {ROOKIE_RATING:.3f}")

    # Get player names for display
    players = {r["id"]: r["canonical_name"]
               for r in conn.execute("SELECT id, canonical_name FROM players")}

    while True:
        print(f"\n{'='*50}")
        team1_name = input("Team 1 name (or 'quit'): ").strip()
        if team1_name.lower() == "quit":
            break

        print("Enter Team 1 roster (one name per line, blank line to finish):")
        team1_names = []
        while True:
            name = input("  > ").strip()
            if not name:
                break
            team1_names.append(name)

        team2_name = input("\nTeam 2 name: ").strip()
        print("Enter Team 2 roster (one name per line, blank line to finish):")
        team2_names = []
        while True:
            name = input("  > ").strip()
            if not name:
                break
            team2_names.append(name)

        team1_ids = lookup_players(team1_names, conn)
        team2_ids = lookup_players(team2_names, conn)

        result = predict_matchup(
            team1_name, team1_ids,
            team2_name, team2_ids,
            player_ratings=ratings
        )

        print(f"\n{'='*50}")
        print(f"  {result['team1']} vs {result['team2']}")
        print(f"{'='*50}")
        print(f"\n  Win probability:")
        print(f"    {result['team1']}: {result['team1_win_prob']*100:.1f}%")
        print(f"    {result['team2']}: {result['team2_win_prob']*100:.1f}%")
        print(f"\n  Predicted winner: {result['predicted_winner']}")

        print(f"\n  Team strength breakdown:")
        for label, team_key in [("  " + result['team1'], "team1_strength"),
                                ("  " + result['team2'], "team2_strength")]:
            ts = result[team_key]
            print(f"\n  {label}:")
            print(f"    Overall: {ts['overall']:.3f} | "
                  f"Skaters: {ts['avg_skater']:.3f} | "
                  f"Goalie: {ts['avg_goalie']:.3f}")
            print(f"    Roster: {ts['num_skaters']} skaters, "
                  f"{ts['num_goalies']} goalies, "
                  f"{ts['num_rookies']} rookies")

            print(f"\n    Player ratings:")
            for p in ts["breakdown"][:10]:
                pname = players.get(p["player_id"], f"Player {p['player_id']}")
                role = "G" if p.get("is_goalie") else "S"
                rookie_tag = " [ROOKIE]" if p.get("rookie") else ""
                print(f"      {pname:<25} {role}  {p['rating']:.3f}{rookie_tag}")
            if len(ts["breakdown"]) > 10:
                print(f"      ... and {len(ts['breakdown']) - 10} more")

        print(f"{'='*50}")

    conn.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Manage team injuries.

Usage:
    # Add/update an injury
    python scripts/injuries.py add "Michigan" "L.J. Cason" --status out \
        --injury ACL --detail "Season-ending knee injury" \
        --date 2026-02-27 --return season-ending

    # List injuries for a team
    python scripts/injuries.py list "Michigan"

    # List all injuries
    python scripts/injuries.py list

    # Remove a player's injury (they're healthy)
    python scripts/injuries.py remove "Michigan" "L.J. Cason"

    # Clear all injuries for a team
    python scripts/injuries.py clear "Michigan"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_predictions.db import (
    get_db, resolve_team, upsert_injury, clear_injuries, get_team_injuries
)


def cmd_add(args):
    conn = get_db(args.sport)
    team_id = resolve_team(conn, args.team)
    if not team_id:
        print(f"Team not found: {args.team}")
        sys.exit(1)

    changed = upsert_injury(
        conn, team_id, args.player,
        status=args.status,
        injury_type=args.injury,
        detail=args.detail,
        date_reported=args.date,
        expected_return=args.returns,
    )
    conn.commit()
    conn.close()
    action = "Added" if changed else "Unchanged"
    print(f"{action}: {args.team} - {args.player} ({args.status})")


def cmd_list(args):
    conn = get_db(args.sport)
    if args.team:
        team_id = resolve_team(conn, args.team)
        if not team_id:
            print(f"Team not found: {args.team}")
            sys.exit(1)
        injuries = get_team_injuries(conn, team_id)
        if not injuries:
            print(f"No injuries for {args.team}")
        else:
            print(f"\n{args.team} injuries:")
            for inj in injuries:
                status = inj['status'].upper()
                injury = inj.get('injury_type') or '?'
                date = inj.get('date_reported') or '?'
                ret = inj.get('expected_return') or '?'
                detail = inj.get('detail') or ''
                print(f"  {inj['player_name']:20s} {status:12s} "
                      f"{injury:15s} reported {date}  return: {ret}")
                if detail:
                    print(f"    {detail}")
    else:
        # List all injuries across all teams
        rows = conn.execute("""
            SELECT t.name as team_name, i.player_name, i.status,
                   i.injury_type, i.detail, i.date_reported,
                   i.expected_return
            FROM injuries i
            JOIN teams t ON t.id = i.team_id
            ORDER BY t.name, i.status
        """).fetchall()
        if not rows:
            print("No injuries in database")
        else:
            print(f"\nAll injuries ({len(rows)} total):")
            current_team = None
            for r in rows:
                if r['team_name'] != current_team:
                    current_team = r['team_name']
                    print(f"\n  {current_team}:")
                status = r['status'].upper()
                injury = r['injury_type'] or '?'
                date = r['date_reported'] or '?'
                ret = r['expected_return'] or '?'
                print(f"    {r['player_name']:20s} {status:12s} "
                      f"{injury:15s} reported {date}  return: {ret}")
    conn.close()


def cmd_remove(args):
    conn = get_db(args.sport)
    team_id = resolve_team(conn, args.team)
    if not team_id:
        print(f"Team not found: {args.team}")
        sys.exit(1)

    deleted = conn.execute(
        "DELETE FROM injuries WHERE team_id=? AND player_name=?",
        (team_id, args.player)
    ).rowcount
    conn.commit()
    conn.close()
    if deleted:
        print(f"Removed: {args.team} - {args.player}")
    else:
        print(f"No injury found for {args.team} - {args.player}")


def cmd_clear(args):
    conn = get_db(args.sport)
    team_id = resolve_team(conn, args.team)
    if not team_id:
        print(f"Team not found: {args.team}")
        sys.exit(1)

    clear_injuries(conn, team_id)
    conn.commit()
    conn.close()
    print(f"Cleared all injuries for {args.team}")


def main():
    parser = argparse.ArgumentParser(description="Manage team injuries")
    parser.add_argument(
        "--sport", default="ncaa_basketball",
        help="Sport database (default: ncaa_basketball)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # add
    p_add = sub.add_parser("add", help="Add/update an injury")
    p_add.add_argument("team", help="Team name")
    p_add.add_argument("player", help="Player name")
    p_add.add_argument("--status", required=True,
                       choices=["out", "doubtful", "questionable",
                                "probable", "day-to-day"],
                       help="Injury status")
    p_add.add_argument("--injury", help="Injury type (e.g. ACL, ankle)")
    p_add.add_argument("--detail", help="Additional details")
    p_add.add_argument("--date", help="Date reported (YYYY-MM-DD)")
    p_add.add_argument("--returns", help="Expected return (date or description)")

    # list
    p_list = sub.add_parser("list", help="List injuries")
    p_list.add_argument("team", nargs="?", help="Team name (omit for all)")

    # remove
    p_remove = sub.add_parser("remove", help="Remove a player's injury")
    p_remove.add_argument("team", help="Team name")
    p_remove.add_argument("player", help="Player name")

    # clear
    p_clear = sub.add_parser("clear", help="Clear all injuries for a team")
    p_clear.add_argument("team", help="Team name")

    args = parser.parse_args()

    if args.command == "add":
        cmd_add(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "remove":
        cmd_remove(args)
    elif args.command == "clear":
        cmd_clear(args)


if __name__ == "__main__":
    main()

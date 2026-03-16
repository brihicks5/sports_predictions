#!/usr/bin/env python3
"""Fetch NCAA tournament bracket from ESPN and output bracket JSON.

Usage:
    python scripts/fetch_bracket.py                     # build initial bracket
    python scripts/fetch_bracket.py --update             # update with completed game results
    python scripts/fetch_bracket.py -o bracket.json     # custom output path
    python scripts/fetch_bracket.py --dry-run            # print JSON to stdout without writing
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_predictions.scrapers.ncaa_basketball import ESPN_NAME_OVERRIDES

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)

# First Four dates + R64 dates for 2026.  Adjust if needed.
FIRST_FOUR_DATES = ["20260317", "20260318"]
R64_DATES = ["20260319", "20260320"]

# Tournament start date for scanning completed games
TOURNAMENT_START = "2026-03-17"


def fetch_tournament_events(dates: list[str]) -> list[dict]:
    """Fetch all tournament events for the given ESPN-format dates."""
    events = []
    for d in dates:
        resp = requests.get(
            ESPN_SCOREBOARD_URL,
            params={"dates": d, "limit": 200, "groups": 100},
        )
        resp.raise_for_status()
        events.extend(resp.json().get("events", []))
    return events


def parse_note(event: dict) -> tuple[str, str]:
    """Extract (region, round) from the event notes headline.

    Returns e.g. ("West", "First Four") or ("South", "1st Round").
    """
    comp = event["competitions"][0]
    for note in comp.get("notes", []):
        headline = note.get("headline", "")
        # "NCAA Men's Basketball Championship - West Region - First Four"
        m = re.search(r"- (\w+) Region - (.+)", headline)
        if m:
            return m.group(1), m.group(2).strip()
        # Final Four / Championship have no region
        if "Final Four" in headline:
            return "", "Final Four"
        if "Championship" in headline and "Region" not in headline:
            return "", "Championship"
    return "", ""


def build_bracket(first_four_events: list[dict],
                  r64_events: list[dict],
                  season: int) -> dict:
    """Build bracket JSON from ESPN tournament events."""
    regions: dict[str, dict[str, str]] = {}
    first_four: list[dict] = []

    # Process First Four games
    for event in first_four_events:
        region, round_name = parse_note(event)
        if "First Four" not in round_name:
            continue

        comp = event["competitions"][0]
        competitors = comp["competitors"]

        teams = []
        seed = None
        for c in competitors:
            name = c["team"]["shortDisplayName"]
            name = ESPN_NAME_OVERRIDES.get(name, name)
            teams.append(name)
            seed = str(c.get("curatedRank", {}).get("current", ""))

        if region and seed and len(teams) == 2:
            first_four.append({
                "seed": seed,
                "region": region,
                "teams": teams,
            })

    # Collect First Four seed slots so we can mark them as TBD
    ff_slots = {(ff["region"], ff["seed"]) for ff in first_four}

    # Process R64 games to get all seeds per region
    for event in r64_events:
        region, round_name = parse_note(event)
        if not region or "1st Round" not in round_name:
            continue

        comp = event["competitions"][0]
        competitors = comp["competitors"]

        if region not in regions:
            regions[region] = {}

        for c in competitors:
            name = c["team"]["shortDisplayName"]
            name = ESPN_NAME_OVERRIDES.get(name, name)
            seed = str(c.get("curatedRank", {}).get("current", ""))
            # Skip TBD placeholders and First Four slots
            if not seed or not name or name == "TBD":
                continue
            if (region, seed) in ff_slots:
                continue
            if seed not in regions[region] or not regions[region][seed]:
                regions[region][seed] = name

    # Mark First Four seed slots as TBD — the simulator resolves
    # these by running the play-in games each iteration.
    for ff in first_four:
        region = ff["region"]
        seed = ff["seed"]
        if region not in regions:
            regions[region] = {}
        regions[region][seed] = "TBD"

    # Ensure all 16 seeds exist per region, remove any bogus keys
    for region in regions:
        # Remove keys that aren't valid seeds (e.g., "99" from TBD)
        valid_keys = {str(s) for s in range(1, 17)}
        for key in list(regions[region]):
            if key not in valid_keys:
                del regions[region][key]
        for s in range(1, 17):
            regions[region].setdefault(str(s), "")

    # Determine Final Four matchups from region names.
    # Standard NCAA pairings vary by year — user should verify.
    region_names = sorted(regions.keys())
    ff_matchups = []
    if len(region_names) == 4:
        ff_matchups = [
            [region_names[0], region_names[3]],
            [region_names[1], region_names[2]],
        ]

    bracket = {
        "season": season,
        "sport": "ncaa_basketball",
        "regions": dict(sorted(regions.items())),
        "first_four": first_four,
        "final_four_matchups": ff_matchups,
        "known_results": [],
    }

    return bracket


def fetch_completed_results() -> list[dict]:
    """Fetch all completed tournament game results from ESPN.

    Scans from tournament start through today.
    Returns list of {"winner": str, "loser": str} dicts.
    """
    start = datetime.strptime(TOURNAMENT_START, "%Y-%m-%d")
    today = datetime.now()
    results = []
    seen = set()

    d = start
    while d <= today:
        date_str = d.strftime("%Y%m%d")
        d += timedelta(days=1)

        try:
            events = fetch_tournament_events([date_str])
        except Exception as e:
            print(f"  Warning: failed to fetch {date_str}: {e}")
            continue

        for event in events:
            comp = event["competitions"][0]
            if comp.get("type", {}).get("abbreviation") != "TRNMNT":
                continue
            if not comp["status"]["type"].get("completed", False):
                continue

            event_id = event["id"]
            if event_id in seen:
                continue
            seen.add(event_id)

            competitors = comp["competitors"]
            # Winner has higher score
            teams = []
            for c in competitors:
                name = c["team"]["shortDisplayName"]
                name = ESPN_NAME_OVERRIDES.get(name, name)
                score = int(c.get("score", 0))
                teams.append((name, score))

            if len(teams) == 2:
                teams.sort(key=lambda x: -x[1])
                winner, loser = teams[0][0], teams[1][0]
                results.append({"winner": winner, "loser": loser})
                region, round_name = parse_note(event)
                label = f"{region} {round_name}" if region else round_name
                print(f"  {label}: {winner} def. {loser} "
                      f"({teams[0][1]}-{teams[1][1]})")

    return results


def cmd_build(args):
    """Build initial bracket from ESPN."""
    print("Fetching First Four games...")
    ff_events = fetch_tournament_events(FIRST_FOUR_DATES)
    ff_tournament = [e for e in ff_events
                     if e.get("competitions", [{}])[0]
                     .get("type", {}).get("abbreviation") == "TRNMNT"]
    print(f"  Found {len(ff_tournament)} First Four games")

    print("Fetching Round of 64 games...")
    r64_events = fetch_tournament_events(R64_DATES)
    r64_tournament = [e for e in r64_events
                      if e.get("competitions", [{}])[0]
                      .get("type", {}).get("abbreviation") == "TRNMNT"]
    print(f"  Found {len(r64_tournament)} R64 games")

    bracket = build_bracket(ff_tournament, r64_tournament, args.season)

    # Report what we found
    for region, seeds in bracket["regions"].items():
        filled = sum(1 for v in seeds.values() if v)
        empty = [s for s, v in seeds.items() if not v]
        status = f"{filled}/16 teams"
        if empty:
            status += f" (missing seeds: {', '.join(empty)})"
        print(f"  {region}: {status}")

    print(f"  First Four: {len(bracket['first_four'])} games")
    print(f"  Final Four matchups: {bracket['final_four_matchups']}")

    return bracket


def cmd_update(args):
    """Update existing bracket with completed game results."""
    path = Path(args.output)
    if not path.exists():
        print(f"Error: {args.output} not found. Run without --update first.")
        sys.exit(1)

    with open(path) as f:
        bracket = json.load(f)

    print("Fetching completed tournament results...")
    results = fetch_completed_results()

    bracket["known_results"] = results
    print(f"\n{len(results)} completed games recorded")

    return bracket


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NCAA tournament bracket from ESPN"
    )
    parser.add_argument(
        "-o", "--output", default="data/bracket.json",
        help="Output path for bracket JSON (default: data/bracket.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print bracket JSON to stdout without writing to file",
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Update existing bracket with completed game results",
    )
    parser.add_argument(
        "--season", type=int, default=2026,
        help="NCAA season year (default: 2026)",
    )
    args = parser.parse_args()

    if args.update:
        bracket = cmd_update(args)
    else:
        bracket = cmd_build(args)

    output = json.dumps(bracket, indent=2) + "\n"

    if args.dry_run:
        print("\n" + output)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\nWrote bracket to {args.output}")
        if not args.update:
            print("NOTE: Verify final_four_matchups — ESPN doesn't expose pairings directly.")
        print("Validate: python scripts/simulate_tournament.py --validate-only")


if __name__ == "__main__":
    main()

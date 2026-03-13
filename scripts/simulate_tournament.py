#!/usr/bin/env python3
"""Monte Carlo NCAA tournament bracket simulator.

Usage:
    python scripts/simulate_tournament.py --iterations 10000
    python scripts/simulate_tournament.py --bracket data/bracket.json -n 50000
    python scripts/simulate_tournament.py -n 10000 --seed 42
    python scripts/simulate_tournament.py -n 10000 --validate-only
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_predictions.db import get_db, resolve_team
from sports_predictions.model import predict_game

# Standard NCAA bracket matchup order within a region.
# Winners are paired sequentially: (1v16 winner) vs (8v9 winner), etc.
MATCHUP_ORDER = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]

ROUND_NAMES = {
    1: "First Four",
    2: "R64",
    3: "R32",
    4: "S16",
    5: "E8",
    6: "F4",
    7: "NCG",
    8: "Champ",
}


def load_bracket(path: str) -> dict:
    """Load and validate bracket structure from JSON file."""
    with open(path) as f:
        bracket = json.load(f)

    errors = []
    for field in ("season", "sport", "regions", "first_four", "final_four_matchups"):
        if field not in bracket:
            errors.append(f"Missing top-level field: {field}")

    if "regions" in bracket:
        regions = bracket["regions"]
        if len(regions) != 4:
            errors.append(f"Expected 4 regions, got {len(regions)}")
        for name, seeds in regions.items():
            if len(seeds) != 16:
                errors.append(f"Region {name}: expected 16 seeds, got {len(seeds)}")
            for seed_num in range(1, 17):
                if str(seed_num) not in seeds:
                    errors.append(f"Region {name}: missing seed {seed_num}")

    if "final_four_matchups" in bracket:
        ff = bracket["final_four_matchups"]
        if len(ff) != 2:
            errors.append(f"Expected 2 Final Four matchups, got {len(ff)}")
        if "regions" in bracket:
            region_names = set(bracket["regions"].keys())
            ff_regions = set()
            for pair in ff:
                ff_regions.update(pair)
            if ff_regions != region_names:
                errors.append(f"Final Four regions {ff_regions} don't match bracket regions {region_names}")

    if errors:
        print("Bracket validation errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    return bracket


def validate_teams(bracket: dict) -> bool:
    """Validate all team names resolve in the database."""
    sport = bracket["sport"]
    conn = get_db(sport)

    all_teams = []
    for region, seeds in bracket["regions"].items():
        for seed, team in seeds.items():
            if team:
                all_teams.append((f"{region} {seed}-seed", team))

    for ff in bracket["first_four"]:
        for team in ff["teams"]:
            if team:
                all_teams.append((f"First Four ({ff['region']} {ff['seed']})", team))

    errors = []
    for label, team in all_teams:
        team_id = resolve_team(conn, team)
        if not team_id:
            errors.append(f"  {label}: '{team}' not found")

    conn.close()

    if errors:
        print(f"Team name errors ({len(errors)}):")
        for e in errors:
            print(e)
        return False

    print(f"Validated {len(all_teams)} teams... OK")
    return True


# Win probability cache: (team_a, team_b) -> P(team_a wins)
# Keys are always sorted alphabetically for consistency
_prob_cache = {}


def get_win_prob(team_a: str, team_b: str, sport: str, season: int) -> float:
    """Get win probability for team_a vs team_b (neutral site), with caching."""
    key = tuple(sorted([team_a, team_b]))
    if key not in _prob_cache:
        result = predict_game(sport, key[0], key[1], season, neutral_site=True)
        _prob_cache[key] = result["home_win_prob"]
    prob_first = _prob_cache[key]
    return prob_first if team_a == key[0] else 1 - prob_first


def simulate_once(bracket: dict) -> dict:
    """Run a single tournament simulation.

    Returns dict mapping team name -> farthest round reached (2-8).
    Round 1 = First Four, Round 8 = Champion.
    """
    sport = bracket["sport"]
    season = bracket["season"]
    results = {}

    # Build the R64 field, starting from region seeds
    r64_teams = {}
    for region, seeds in bracket["regions"].items():
        r64_teams[region] = dict(seeds)

    # Simulate First Four — winners replace their region/seed slot
    for ff in bracket["first_four"]:
        t1, t2 = ff["teams"]
        if not t1 or not t2:
            continue
        p = get_win_prob(t1, t2, sport, season)
        winner = t1 if random.random() < p else t2
        loser = t2 if winner == t1 else t1
        results[loser] = 1  # eliminated in First Four
        r64_teams[ff["region"]][ff["seed"]] = winner

    # Simulate each region through Elite 8
    region_winners = {}
    for region, seeds in r64_teams.items():
        # Round of 64
        round_winners = []
        for s1, s2 in MATCHUP_ORDER:
            t1 = seeds[str(s1)]
            t2 = seeds[str(s2)]
            p = get_win_prob(t1, t2, sport, season)
            winner = t1 if random.random() < p else t2
            loser = t2 if winner == t1 else t1
            results[loser] = 2  # eliminated in R64
            round_winners.append(winner)

        # R32 (round 3), Sweet 16 (round 4), Elite 8 (round 5)
        round_num = 3
        while len(round_winners) > 1:
            next_round = []
            for i in range(0, len(round_winners), 2):
                t1, t2 = round_winners[i], round_winners[i + 1]
                p = get_win_prob(t1, t2, sport, season)
                winner = t1 if random.random() < p else t2
                loser = t2 if winner == t1 else t1
                results[loser] = round_num
                next_round.append(winner)
            round_winners = next_round
            round_num += 1

        region_winners[region] = round_winners[0]

    # Final Four (round 6)
    ff_winners = []
    for r1, r2 in bracket["final_four_matchups"]:
        t1 = region_winners[r1]
        t2 = region_winners[r2]
        p = get_win_prob(t1, t2, sport, season)
        winner = t1 if random.random() < p else t2
        loser = t2 if winner == t1 else t1
        results[loser] = 6  # eliminated in Final Four
        ff_winners.append(winner)

    # Championship game (round 7)
    t1, t2 = ff_winners
    p = get_win_prob(t1, t2, sport, season)
    winner = t1 if random.random() < p else t2
    loser = t2 if winner == t1 else t1
    results[loser] = 7  # lost championship
    results[winner] = 8  # champion

    return results


def run_simulations(bracket: dict, n: int) -> dict:
    """Run N simulations and aggregate results.

    Returns dict mapping team -> {round_num: count_reached_at_least}.
    """
    # Collect all team names
    all_teams = set()
    for seeds in bracket["regions"].values():
        all_teams.update(seeds.values())
    for ff in bracket["first_four"]:
        all_teams.update(ff["teams"])
    all_teams.discard("")

    # counts[team][round] = number of times team reached at least that round
    counts = {team: defaultdict(int) for team in all_teams}

    t0 = time.time()
    for i in range(n):
        sim = simulate_once(bracket)
        for team, farthest_round in sim.items():
            # Team reached all rounds up to and including farthest_round
            for r in range(1, farthest_round + 1):
                counts[team][r] += 1

    elapsed = time.time() - t0
    print(f"Simulated {n:,} brackets in {elapsed:.1f}s "
          f"({len(_prob_cache)} matchups cached)")

    return counts


def print_results(counts: dict, n: int, bracket: dict, top: int = 0):
    """Print simulation results as a formatted table."""
    # Build team info (region + seed)
    team_info = {}
    for region, seeds in bracket["regions"].items():
        for seed, team in seeds.items():
            if team:
                team_info[team] = f"{region[0]}{seed}"
    for ff in bracket["first_four"]:
        for team in ff["teams"]:
            if team and team not in team_info:
                team_info[team] = f"{ff['region'][0]}{ff['seed']}*"

    # Sort by championship probability, then by deepest expected run
    def sort_key(team):
        c = counts[team]
        return (-c[8], -c[7], -c[6], -c[5], -c[4], -c[3])

    sorted_teams = sorted(counts.keys(), key=sort_key)

    if top > 0:
        sorted_teams = sorted_teams[:top]

    # Header
    print(f"\n{'Team':<25s} {'Seed':>5s} {'R32':>7s} {'S16':>7s} "
          f"{'E8':>7s} {'F4':>7s} {'NCG':>7s} {'Champ':>7s}")
    print("-" * 78)

    for team in sorted_teams:
        c = counts[team]
        seed = team_info.get(team, "?")
        r32 = f"{100 * c[3] / n:.1f}%"
        s16 = f"{100 * c[4] / n:.1f}%"
        e8 = f"{100 * c[5] / n:.1f}%"
        f4 = f"{100 * c[6] / n:.1f}%"
        ncg = f"{100 * c[7] / n:.1f}%"
        champ = f"{100 * c[8] / n:.1f}%"
        print(f"  {team:<23s} {seed:>5s} {r32:>7s} {s16:>7s} "
              f"{e8:>7s} {f4:>7s} {ncg:>7s} {champ:>7s}")

    # Most likely champion
    best = max(counts.keys(), key=lambda t: counts[t][8])
    best_pct = 100 * counts[best][8] / n
    print(f"\nMost likely champion: {best} ({best_pct:.1f}%)")

    # Most likely Final Four
    f4_teams = sorted(counts.keys(), key=lambda t: -counts[t][6])[:4]
    print("Most likely Final Four: " +
          ", ".join(f"{t} ({100*counts[t][6]/n:.1f}%)" for t in f4_teams))

    # Cinderellas: teams seeded 10+ with >5% Sweet 16 probability
    cinderellas = []
    for team in counts:
        seed_str = team_info.get(team, "")
        # Extract numeric seed
        seed_num = "".join(c for c in seed_str if c.isdigit())
        if seed_num and int(seed_num) >= 10 and counts[team][4] / n > 0.05:
            cinderellas.append((team, seed_str, counts[team][4] / n))
    if cinderellas:
        cinderellas.sort(key=lambda x: -x[2])
        print("\nCinderella alert (10+ seeds with >5% Sweet 16 chance):")
        for team, seed, pct in cinderellas:
            print(f"  {team} ({seed}): {100*pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo NCAA tournament bracket simulator"
    )
    parser.add_argument(
        "--bracket", default="data/bracket.json",
        help="Path to bracket JSON file (default: data/bracket.json)"
    )
    parser.add_argument(
        "-n", "--iterations", type=int, default=10000,
        help="Number of simulations to run (default: 10000)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible results"
    )
    parser.add_argument(
        "--top", type=int, default=0,
        help="Only show top N teams (default: all)"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Just validate team names, don't simulate"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    bracket = load_bracket(args.bracket)

    if not validate_teams(bracket):
        sys.exit(1)

    if args.validate_only:
        print("Bracket is valid.")
        return

    print(f"Running {args.iterations:,} simulations...")
    counts = run_simulations(bracket, args.iterations)
    print_results(counts, args.iterations, bracket, top=args.top)


if __name__ == "__main__":
    main()

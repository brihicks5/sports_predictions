#!/usr/bin/env python3
"""Migrate teams table to use canonical names + team_aliases.

Identifies duplicate teams (Kaggle vs KenPom name variants),
merges them, and creates alias entries for all teams.

Run with --dry-run to preview changes without modifying the database.

Usage:
    python scripts/migrate_team_aliases.py
    python scripts/migrate_team_aliases.py --dry-run
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_predictions.db import get_db, DATA_DIR

# Explicit mapping of Kaggle names to KenPom names for teams that can't be
# matched by simple normalization (different abbreviation styles, etc.)
KAGGLE_TO_KENPOM = {
    # Abbreviated directions
    "C Michigan": "Central Michigan",
    "E Illinois": "Eastern Illinois",
    "E Kentucky": "Eastern Kentucky",
    "E Michigan": "Eastern Michigan",
    "E Washington": "Eastern Washington",
    "N Colorado": "Northern Colorado",
    "N Dakota St": "North Dakota St.",
    "N Illinois": "Northern Illinois",
    "N Kentucky": "Northern Kentucky",
    "W Carolina": "Western Carolina",
    "W Illinois": "Western Illinois",
    "W Michigan": "Western Michigan",
    "W Salem St": "Winston Salem St.",
    # State abbreviations / geographic short forms
    "CS Bakersfield": "Cal St. Bakersfield",
    "CS Fullerton": "Cal St. Fullerton",
    "CS Northridge": "CSUN",
    "CS Sacramento": "Sacramento St.",
    "FL Atlantic": "Florida Atlantic",
    "Ga Southern": "Georgia Southern",
    "IL Chicago": "Illinois Chicago",
    "MA Lowell": "UMass Lowell",
    "MD E Shore": "Maryland Eastern Shore",
    "MS Valley St": "Mississippi Valley St.",
    "NC A&T": "North Carolina A&T",
    "NC Central": "North Carolina Central",
    "NE Omaha": "Nebraska Omaha",
    "SC Upstate": "USC Upstate",
    "SE Louisiana": "Southeastern Louisiana",
    "SE Missouri St": "Southeast Missouri",
    "SF Austin": "Stephen F. Austin",
    "TX Southern": "Texas Southern",
    # Acronyms
    "ETSU": "East Tennessee St.",
    "FGCU": "Florida Gulf Coast",
    "MTSU": "Middle Tennessee",
    "ULM": "Louisiana Monroe",
    "WKU": "Western Kentucky",
    "UTRGV": "UT Rio Grande Valley",
    # Univ/University, suffix differences
    "American Univ": "American",
    "Boston Univ": "Boston University",
    "Southern Univ": "Southern",
    "G Washington": "George Washington",
    "Florida Intl": "FIU",
    # Punctuation / formatting
    "Bethune-Cookman": "Bethune Cookman",
    "Loyola-Chicago": "Loyola Chicago",
    "Loy Marymount": "Loyola Marymount",
    "St Joseph's PA": "Saint Joseph's",
    "St Louis": "Saint Louis",
    "St Mary's CA": "Saint Mary's",
    "St Peter's": "Saint Peter's",
    "Mt St Mary's": "Mount St. Mary's",
    "Monmouth NJ": "Monmouth",
    "F Dickinson": "Fairleigh Dickinson",
    # Shortened / abbreviated names
    "Ark Little Rock": "Little Rock",
    "Ark Pine Bluff": "Arkansas Pine Bluff",
    "Cent Arkansas": "Central Arkansas",
    "Central Conn": "Central Connecticut",
    "Charleston So": "Charleston Southern",
    "Citadel": "The Citadel",
    "Coastal Car": "Coastal Carolina",
    "Col Charleston": "Charleston",
    "Grambling": "Grambling St.",
    "Kent": "Kent St.",
    "Kennesaw": "Kennesaw St.",
    "Northwestern LA": "Northwestern St.",
    "Prairie View": "Prairie View A&M",
    "S Carolina St": "South Carolina St.",
    "S Dakota St": "South Dakota St.",
    "S Illinois": "Southern Illinois",
    "SUNY Albany": "Albany",
    "TAM C. Christi": "Texas A&M Corpus Chris",
    "TN Martin": "Tennessee Martin",
    "UT San Antonio": "UTSA",
    "WI Green Bay": "Green Bay",
    "WI Milwaukee": "Milwaukee",
    "Queens NC": "Queens",
    "St Thomas MN": "St. Thomas",
    "Missouri KC": "Kansas City",
    "PFW": "Purdue Fort Wayne",
}

# KenPom internal duplicates: old KenPom name -> current team name (with games).
# These are KenPom teams with 0 games whose stats should merge into an existing team.
KENPOM_RENAMES = {
    "Arkansas Little Rock": "Little Rock",
    "Cal St. Northridge": "CSUN",
    "College of Charleston": "Charleston",
    "Detroit Mercy": "Detroit",
    "Dixie St.": "Utah Tech",
    "Fort Wayne": "Purdue Fort Wayne",
    "Houston Baptist": "Houston Christian",
    "IPFW": "Purdue Fort Wayne",
    "IU Indy": "IUPUI",
    "LIU": "LIU Brooklyn",
    "Louisiana Lafayette": "Louisiana",
    "McNeese": "McNeese St.",
    "Nicholls": "Nicholls St.",
    "Saint Francis": "St. Francis PA",
    "SIU Edwardsville": "SIUE",
    "Southeast Missouri St.": "Southeast Missouri",
    "Texas Pan American": "UT Rio Grande Valley",
    "UMKC": "Kansas City",
}


def normalize_name(name: str) -> str:
    """Normalize a team name for fuzzy matching."""
    n = name.strip().lower()
    n = n.replace(".", "")
    # Expand abbreviations (only at end of string to avoid substring issues)
    if n.endswith(" chr"):
        n = n[:-4] + " christian"
    return n


def find_duplicates(conn):
    """Find duplicate team pairs (Kaggle name vs KenPom name).

    Uses both normalization-based matching and explicit mapping.
    Returns list of (kaggle_id, kaggle_name, kenpom_id, kenpom_name, conference).
    """
    teams = conn.execute(
        "SELECT id, name, conference FROM teams ORDER BY name"
    ).fetchall()

    teams_by_name = {t["name"]: t for t in teams}

    # Separate by source: KenPom teams have conference, Kaggle-only don't
    kenpom_by_norm = {}
    for t in teams:
        if t["conference"]:
            kenpom_by_norm[normalize_name(t["name"])] = t

    duplicates = []
    matched_kaggle = set()

    # Phase 1: Explicit mapping
    for kaggle_name, kenpom_name in KAGGLE_TO_KENPOM.items():
        kt = teams_by_name.get(kaggle_name)
        kp = teams_by_name.get(kenpom_name)
        if kt and kp and kt["name"] != kp["name"]:
            duplicates.append((
                kt["id"], kt["name"],
                kp["id"], kp["name"],
                kp["conference"],
            ))
            matched_kaggle.add(kaggle_name)

    # Phase 2: Normalization-based matching (for remaining teams)
    for t in teams:
        if t["conference"] or t["name"] in matched_kaggle:
            continue
        norm = normalize_name(t["name"])
        if norm in kenpom_by_norm:
            kp = kenpom_by_norm[norm]
            if t["name"] != kp["name"]:
                duplicates.append((
                    t["id"], t["name"],
                    kp["id"], kp["name"],
                    kp["conference"],
                ))

    return duplicates


def find_kenpom_renames(conn):
    """Find KenPom internal duplicates (name changes over time).

    Returns list of (old_id, old_name, current_id, current_name, conference).
    """
    teams = conn.execute(
        "SELECT id, name, conference FROM teams ORDER BY name"
    ).fetchall()
    teams_by_name = {t["name"]: t for t in teams}

    renames = []
    for old_name, current_name in KENPOM_RENAMES.items():
        old_t = teams_by_name.get(old_name)
        cur_t = teams_by_name.get(current_name)
        if old_t and cur_t:
            # Keep the one with more data (the current name, which has games)
            renames.append((
                old_t["id"], old_t["name"],
                cur_t["id"], cur_t["name"],
                old_t["conference"] or cur_t["conference"],
            ))
    return renames


def merge_teams(conn, loser_id, loser_name, keeper_id, keeper_name,
                conference, aliases, dry_run=False):
    """Merge two teams. Moves all data from loser to keeper, then deletes loser."""
    if dry_run:
        print(f"  Merge: '{loser_name}' (id={loser_id}) "
              f"-> '{keeper_name}' (id={keeper_id}), conf={conference}")
        return

    # Move team_stats from loser to keeper
    conn.execute(
        "UPDATE OR IGNORE team_stats SET team_id = ? WHERE team_id = ?",
        (keeper_id, loser_id)
    )
    conn.execute("DELETE FROM team_stats WHERE team_id = ?", (loser_id,))

    # Move any games
    conn.execute(
        "UPDATE games SET home_team_id = ? WHERE home_team_id = ?",
        (keeper_id, loser_id)
    )
    conn.execute(
        "UPDATE games SET away_team_id = ? WHERE away_team_id = ?",
        (keeper_id, loser_id)
    )

    # Delete the loser team
    conn.execute("DELETE FROM teams WHERE id = ?", (loser_id,))

    # Update keeper's name and conference if needed
    conn.execute(
        "UPDATE teams SET name = ?, conference = ? WHERE id = ?",
        (keeper_name, conference, keeper_id)
    )

    # Create aliases
    for alias, source in aliases:
        conn.execute(
            "INSERT OR IGNORE INTO team_aliases (team_id, alias, source) "
            "VALUES (?, ?, ?)",
            (keeper_id, alias, source)
        )


def create_aliases_for_remaining(conn, dry_run=False):
    """Create aliases for teams that weren't part of a duplicate pair."""
    teams_without_aliases = conn.execute("""
        SELECT t.id, t.name, t.conference FROM teams t
        LEFT JOIN team_aliases a ON t.id = a.team_id
        WHERE a.id IS NULL
    """).fetchall()

    if dry_run:
        print(f"\n  Would create aliases for {len(teams_without_aliases)} "
              f"non-duplicate teams")
        return

    for t in teams_without_aliases:
        source = "kenpom" if t["conference"] else "kaggle"
        conn.execute(
            "INSERT OR IGNORE INTO team_aliases (team_id, alias, source) "
            "VALUES (?, ?, 'canonical')",
            (t["id"], t["name"])
        )
        conn.execute(
            "INSERT OR IGNORE INTO team_aliases (team_id, alias, source) "
            "VALUES (?, ?, ?)",
            (t["id"], t["name"], source)
        )


def verify_migration(conn):
    """Run integrity checks after migration."""
    errors = []

    orphaned = conn.execute("""
        SELECT a.alias FROM team_aliases a
        LEFT JOIN teams t ON a.team_id = t.id
        WHERE t.id IS NULL
    """).fetchall()
    if orphaned:
        errors.append(f"Orphaned aliases: {[r['alias'] for r in orphaned]}")

    missing = conn.execute("""
        SELECT t.id, t.name FROM teams t
        LEFT JOIN team_aliases a ON t.id = a.team_id
        WHERE a.id IS NULL
    """).fetchall()
    if missing:
        errors.append(
            f"Teams without aliases: "
            f"{[(r['id'], r['name']) for r in missing]}"
        )

    dup_names = conn.execute("""
        SELECT name, COUNT(*) c FROM teams GROUP BY name HAVING c > 1
    """).fetchall()
    if dup_names:
        errors.append(
            f"Duplicate team names: {[r['name'] for r in dup_names]}"
        )

    dup_aliases = conn.execute("""
        SELECT alias, COUNT(*) c FROM team_aliases
        GROUP BY alias HAVING c > 1
    """).fetchall()
    if dup_aliases:
        errors.append(
            f"Duplicate aliases: {[r['alias'] for r in dup_aliases]}"
        )

    bad_games = conn.execute("""
        SELECT COUNT(*) FROM games g
        LEFT JOIN teams t1 ON g.home_team_id = t1.id
        LEFT JOIN teams t2 ON g.away_team_id = t2.id
        WHERE t1.id IS NULL OR t2.id IS NULL
    """).fetchone()[0]
    if bad_games:
        errors.append(f"Games with invalid team refs: {bad_games}")

    bad_stats = conn.execute("""
        SELECT COUNT(*) FROM team_stats ts
        LEFT JOIN teams t ON ts.team_id = t.id
        WHERE t.id IS NULL
    """).fetchone()[0]
    if bad_stats:
        errors.append(f"Stats with invalid team refs: {bad_stats}")

    team_count = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
    alias_count = conn.execute(
        "SELECT COUNT(*) FROM team_aliases"
    ).fetchone()[0]

    print(f"\nTeams: {team_count}")
    print(f"Aliases: {alias_count}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return False

    print("All verification checks passed!")
    return True


def main():
    dry_run = "--dry-run" in sys.argv
    sport = "ncaa_basketball"
    db_path = DATA_DIR / f"{sport}.db"

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    if not dry_run:
        backup_path = db_path.with_suffix(".db.bak")
        print(f"Backing up database to {backup_path}")
        shutil.copy2(db_path, backup_path)

    conn = get_db(sport)

    pre_teams = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
    pre_games = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    pre_stats = conn.execute("SELECT COUNT(*) FROM team_stats").fetchone()[0]
    print(f"Pre-migration: {pre_teams} teams, {pre_games} games, "
          f"{pre_stats} stats")

    # Phase 1: Kaggle <-> KenPom duplicates
    duplicates = find_duplicates(conn)
    print(f"\nPhase 1: {len(duplicates)} Kaggle/KenPom duplicate pairs")
    for kaggle_id, kaggle_name, kenpom_id, kenpom_name, conf in duplicates:
        # Keeper = Kaggle team (has game history), rename to KenPom name
        aliases = [
            (kenpom_name, "canonical"),
            (kaggle_name, "kaggle"),
            (kenpom_name, "kenpom"),
        ]
        merge_teams(
            conn, kenpom_id, kenpom_name, kaggle_id, kenpom_name,
            conf, aliases, dry_run=dry_run
        )

    # Phase 2: KenPom internal renames
    renames = find_kenpom_renames(conn)
    print(f"\nPhase 2: {len(renames)} KenPom internal renames")
    for old_id, old_name, cur_id, cur_name, conf in renames:
        # Keeper = current team (has games), loser = old name (0 games)
        aliases = [
            (cur_name, "canonical"),
            (old_name, "kenpom"),
        ]
        merge_teams(
            conn, old_id, old_name, cur_id, cur_name,
            conf, aliases, dry_run=dry_run
        )

    # Phase 3: Aliases for remaining teams
    create_aliases_for_remaining(conn, dry_run=dry_run)

    if not dry_run:
        conn.commit()

        post_games = conn.execute(
            "SELECT COUNT(*) FROM games"
        ).fetchone()[0]
        post_stats = conn.execute(
            "SELECT COUNT(*) FROM team_stats"
        ).fetchone()[0]
        print(f"\nPost-migration: {post_games} games, {post_stats} stats")

        if post_games != pre_games:
            print(f"WARNING: Game count changed! {pre_games} -> {post_games}")

        verify_migration(conn)
    else:
        print("\n(dry run — no changes made)")

    conn.close()


if __name__ == "__main__":
    main()

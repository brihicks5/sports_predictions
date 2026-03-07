"""Player name aliases for D5 Hockey.

Maps stat sheet names (nicknames, last names) to canonical roster names.
Add entries here as you discover mismatches between stats and rosters.
"""

from d5_hockey.db import get_db, resolve_player, get_or_create_player, add_alias

# Format: "stat_name_or_nickname": "Canonical Full Name"
# The canonical name should match how they appear on rosters.
KNOWN_ALIASES = {
    "Hicks": "Brian Hicks",
    "Meatbox": "Brian Hicks",
    "James D": "James Dziwura",
    # Add more as discovered. Examples:
    # "Everett": "Everett LastName",
    # "Craig": "Craig LastName",
}


def apply_aliases():
    """Apply all known aliases to the database."""
    conn = get_db()

    for alias, canonical in KNOWN_ALIASES.items():
        player_id = resolve_player(conn, canonical)
        if player_id is None:
            player_id = get_or_create_player(conn, canonical)

        add_alias(conn, player_id, alias)

        # If there's an existing player with the alias as canonical name,
        # merge their stats into the canonical player
        alias_player = conn.execute(
            "SELECT id FROM players WHERE canonical_name = ? AND id != ?",
            (alias, player_id)
        ).fetchone()

        if alias_player:
            alias_id = alias_player["id"]
            # Move stats to canonical player
            conn.execute("""
                UPDATE OR IGNORE player_season_stats
                SET player_id = ? WHERE player_id = ?
            """, (player_id, alias_id))
            # Move roster entries
            conn.execute("""
                UPDATE OR IGNORE roster_entries
                SET player_id = ? WHERE player_id = ?
            """, (player_id, alias_id))
            # Move aliases
            conn.execute("""
                UPDATE OR IGNORE player_aliases
                SET player_id = ? WHERE player_id = ?
            """, (player_id, alias_id))
            # Delete orphan stats/rosters that conflicted
            conn.execute(
                "DELETE FROM player_season_stats WHERE player_id = ?",
                (alias_id,)
            )
            conn.execute(
                "DELETE FROM roster_entries WHERE player_id = ?",
                (alias_id,)
            )
            conn.execute(
                "DELETE FROM player_aliases WHERE player_id = ?",
                (alias_id,)
            )
            conn.execute(
                "DELETE FROM players WHERE id = ?", (alias_id,)
            )

    conn.commit()
    conn.close()
    print(f"Applied {len(KNOWN_ALIASES)} player aliases")

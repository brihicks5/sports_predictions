"""Player rating system for D5 Hockey.

Computes a per-player rating from their historical stats across seasons.
Team strength is the aggregate of its roster's player ratings.
"""

import math
from collections import defaultdict

from d5_hockey.db import get_db


# How much to weight recent seasons vs older ones.
# Most recent season = 1.0, each prior season decays by this factor.
RECENCY_DECAY = 0.75

# Rating for players with no history (rookies).
# Calibrated against league average after first data import.
ROOKIE_RATING = None  # Set dynamically based on league data


def compute_player_ratings() -> dict:
    """Compute a rating for every player in the database.

    Returns dict of {player_id: {rating, seasons_played, ...}}
    """
    conn = get_db()

    # Get all seasons in chronological order
    seasons = conn.execute("""
        SELECT id, name, year, period FROM seasons
        ORDER BY year,
            CASE period
                WHEN 'spring' THEN 1
                WHEN 'summer' THEN 2
                WHEN 'fall' THEN 3
            END
    """).fetchall()

    season_order = {s["id"]: i for i, s in enumerate(seasons)}
    latest_idx = len(seasons) - 1

    # Get all player stats
    all_stats = conn.execute("""
        SELECT ps.*, p.canonical_name
        FROM player_season_stats ps
        JOIN players p ON p.id = ps.player_id
        ORDER BY ps.player_id, ps.season_id
    """).fetchall()

    # Group stats by player
    player_stats = defaultdict(list)
    for row in all_stats:
        player_stats[row["player_id"]].append(row)

    # Compute league averages for baseline
    league_pts_per_game = []
    league_goalie_gaa = []
    for stats_list in player_stats.values():
        for s in stats_list:
            if s["gp"] and s["gp"] > 0:
                if s["is_goalie"]:
                    if s["gaa"] is not None:
                        league_goalie_gaa.append(s["gaa"])
                else:
                    league_pts_per_game.append(s["points"] / s["gp"])

    avg_pts_per_game = (sum(league_pts_per_game) / len(league_pts_per_game)
                        if league_pts_per_game else 1.0)
    avg_gaa = (sum(league_goalie_gaa) / len(league_goalie_gaa)
               if league_goalie_gaa else 3.0)

    ratings = {}

    for player_id, stats_list in player_stats.items():
        is_goalie = any(s["is_goalie"] for s in stats_list)

        if is_goalie:
            rating = _rate_goalie(stats_list, season_order, latest_idx,
                                  avg_gaa)
        else:
            rating = _rate_skater(stats_list, season_order, latest_idx,
                                  avg_pts_per_game)

        ratings[player_id] = rating

    conn.close()

    # Set rookie rating to league average
    all_ratings = [r["rating"] for r in ratings.values()]
    global ROOKIE_RATING
    ROOKIE_RATING = sum(all_ratings) / len(all_ratings) if all_ratings else 0.0

    return ratings


def _rate_skater(stats_list, season_order, latest_idx, avg_ppg):
    """Rate a skater based on points per game, weighted by recency."""
    weighted_ppg = 0.0
    total_weight = 0.0
    total_gp = 0
    total_seasons = len(stats_list)
    total_goals = 0
    total_assists = 0
    total_gwg = 0

    for s in stats_list:
        if s["gp"] is None or s["gp"] == 0:
            continue

        season_idx = season_order.get(s["season_id"], 0)
        seasons_ago = latest_idx - season_idx
        weight = RECENCY_DECAY ** seasons_ago

        ppg = s["points"] / s["gp"]
        weighted_ppg += ppg * weight
        total_weight += weight
        total_gp += s["gp"]
        total_goals += s["goals"] or 0
        total_assists += s["assists"] or 0
        total_gwg += s["gwg"] or 0

    if total_weight == 0:
        return {"rating": 0.0, "seasons_played": total_seasons,
                "total_gp": 0, "is_goalie": False}

    # Recency-weighted points per game, normalized against league average
    norm_ppg = (weighted_ppg / total_weight) / avg_ppg if avg_ppg > 0 else 1.0

    # Bonus for clutch performance (game-winning goals per game)
    gwg_bonus = (total_gwg / total_gp) * 0.5 if total_gp > 0 else 0

    # Small experience bonus (diminishing returns)
    exp_bonus = min(0.2, math.log1p(total_seasons) * 0.1)

    rating = norm_ppg + gwg_bonus + exp_bonus

    return {
        "rating": round(rating, 4),
        "seasons_played": total_seasons,
        "total_gp": total_gp,
        "is_goalie": False,
        "avg_ppg": round(weighted_ppg / total_weight, 2) if total_weight else 0,
    }


def _rate_goalie(stats_list, season_order, latest_idx, avg_gaa):
    """Rate a goalie based on GAA and win rate, weighted by recency."""
    weighted_gaa = 0.0
    weighted_win_rate = 0.0
    total_weight = 0.0
    total_gp = 0
    total_seasons = len(stats_list)
    total_shutouts = 0

    for s in stats_list:
        if s["gp"] is None or s["gp"] == 0:
            continue

        season_idx = season_order.get(s["season_id"], 0)
        seasons_ago = latest_idx - season_idx
        weight = RECENCY_DECAY ** seasons_ago

        gaa = s["gaa"] if s["gaa"] is not None else avg_gaa
        win_rate = (s["wins"] or 0) / s["gp"]

        weighted_gaa += gaa * weight
        weighted_win_rate += win_rate * weight
        total_weight += weight
        total_gp += s["gp"]
        total_shutouts += s["shutouts"] or 0

    if total_weight == 0:
        return {"rating": 0.0, "seasons_played": total_seasons,
                "total_gp": 0, "is_goalie": True}

    avg_weighted_gaa = weighted_gaa / total_weight
    avg_weighted_wr = weighted_win_rate / total_weight

    # Lower GAA is better — invert relative to league average
    gaa_rating = avg_gaa / avg_weighted_gaa if avg_weighted_gaa > 0 else 1.0

    # Win rate component
    wr_rating = avg_weighted_wr * 2  # Scale so 0.500 = 1.0

    # Shutout bonus
    so_bonus = (total_shutouts / total_gp) * 0.3 if total_gp > 0 else 0

    rating = (gaa_rating * 0.5) + (wr_rating * 0.4) + so_bonus + 0.1

    return {
        "rating": round(rating, 4),
        "seasons_played": total_seasons,
        "total_gp": total_gp,
        "is_goalie": True,
        "avg_gaa": round(avg_weighted_gaa, 2) if total_weight else None,
    }


def compute_team_strength(roster_player_ids: list,
                          player_ratings: dict) -> dict:
    """Compute team strength from a list of player IDs.

    Returns dict with overall strength, skater strength, goalie strength,
    rookie count, and per-player breakdown.
    """
    rookie_rating = ROOKIE_RATING or 0.5
    skater_ratings = []
    goalie_ratings = []
    rookies = 0
    breakdown = []

    for pid in roster_player_ids:
        if pid in player_ratings:
            pr = player_ratings[pid]
            breakdown.append({"player_id": pid, **pr})
            if pr["is_goalie"]:
                goalie_ratings.append(pr["rating"])
            else:
                skater_ratings.append(pr["rating"])
        else:
            # Rookie — apply discounted average rating
            discounted = rookie_rating * 0.85
            skater_ratings.append(discounted)
            breakdown.append({
                "player_id": pid, "rating": discounted,
                "is_goalie": False, "rookie": True
            })
            rookies += 1

    avg_skater = (sum(skater_ratings) / len(skater_ratings)
                  if skater_ratings else 0)
    avg_goalie = (sum(goalie_ratings) / len(goalie_ratings)
                  if goalie_ratings else rookie_rating)

    # Overall: skaters matter more but goalie is high-impact
    overall = avg_skater * 0.65 + avg_goalie * 0.35

    return {
        "overall": round(overall, 4),
        "avg_skater": round(avg_skater, 4),
        "avg_goalie": round(avg_goalie, 4),
        "num_skaters": len(skater_ratings),
        "num_goalies": len(goalie_ratings),
        "num_rookies": rookies,
        "breakdown": sorted(breakdown, key=lambda x: -x["rating"]),
    }


def predict_matchup(team1_name: str, team1_players: list,
                    team2_name: str, team2_players: list,
                    player_ratings: dict = None) -> dict:
    """Predict a matchup between two teams given their rosters.

    team1_players and team2_players are lists of player IDs.
    """
    if player_ratings is None:
        player_ratings = compute_player_ratings()

    t1 = compute_team_strength(team1_players, player_ratings)
    t2 = compute_team_strength(team2_players, player_ratings)

    # Convert strength difference to win probability using logistic function
    strength_diff = t1["overall"] - t2["overall"]
    # Scale factor — calibrate based on historical data
    k = 3.0
    t1_win_prob = 1 / (1 + math.exp(-k * strength_diff))

    return {
        "team1": team1_name,
        "team2": team2_name,
        "team1_strength": t1,
        "team2_strength": t2,
        "team1_win_prob": round(t1_win_prob, 4),
        "team2_win_prob": round(1 - t1_win_prob, 4),
        "predicted_winner": team1_name if t1_win_prob > 0.5 else team2_name,
    }


def print_ratings(top_n: int = 30):
    """Print top player ratings."""
    conn = get_db()
    ratings = compute_player_ratings()

    # Get player names
    players = {r["id"]: r["canonical_name"]
               for r in conn.execute("SELECT id, canonical_name FROM players")}
    conn.close()

    sorted_ratings = sorted(ratings.items(), key=lambda x: -x[1]["rating"])

    print(f"\n{'='*60}")
    print(f"  D5 Hockey Player Ratings (Top {top_n})")
    print(f"{'='*60}")

    print(f"\n  SKATERS:")
    print(f"  {'Rank':<5} {'Player':<25} {'Rating':<8} {'Seasons':<8} {'PPG':<6}")
    print(f"  {'-'*52}")
    rank = 1
    for pid, r in sorted_ratings:
        if r["is_goalie"] or r["total_gp"] == 0:
            continue
        if rank > top_n:
            break
        name = players.get(pid, f"Player {pid}")
        print(f"  {rank:<5} {name:<25} {r['rating']:<8.3f} "
              f"{r['seasons_played']:<8} {r.get('avg_ppg', 0):<6}")
        rank += 1

    print(f"\n  GOALIES:")
    print(f"  {'Rank':<5} {'Player':<25} {'Rating':<8} {'Seasons':<8} {'GAA':<6}")
    print(f"  {'-'*52}")
    rank = 1
    for pid, r in sorted_ratings:
        if not r["is_goalie"] or r["total_gp"] == 0:
            continue
        name = players.get(pid, f"Player {pid}")
        gaa = r.get("avg_gaa", "N/A")
        print(f"  {rank:<5} {name:<25} {r['rating']:<8.3f} "
              f"{r['seasons_played']:<8} {gaa}")
        rank += 1

    print(f"\n  Rookie baseline rating: {ROOKIE_RATING:.3f}")
    print(f"{'='*60}")

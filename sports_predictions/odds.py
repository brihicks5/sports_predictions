"""Fetch betting odds from ESPN's pickcenter (no API key needed)."""

from datetime import date

import requests

from sports_predictions.db import get_db, resolve_team

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/scoreboard"
)
ESPN_SUMMARY = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/summary"
)


def fetch_slate(target_date: date = None) -> list[dict]:
    """Fetch all games and odds for a given date from ESPN.

    Returns list of dicts with keys: game_id, home, away, neutral,
    spread, total, home_moneyline, away_moneyline, status.
    Spread is from home team perspective (positive = home favored).
    """
    if target_date is None:
        target_date = date.today()

    try:
        resp = requests.get(ESPN_SCOREBOARD, params={
            "groups": "50", "limit": "200",
            "dates": target_date.strftime("%Y%m%d"),
        }, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    events = resp.json().get("events", [])
    games = []

    for event in events:
        comps = event.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        home_name = away_name = ""
        home_score = away_score = None
        for c in competitors:
            team = c.get("team", {})
            name = team.get("shortDisplayName", team.get("displayName", ""))
            score = c.get("score")
            if c.get("homeAway") == "home":
                home_name = name
                if score is not None:
                    home_score = int(score)
            else:
                away_name = name
                if score is not None:
                    away_score = int(score)

        neutral = comp.get("neutralSite", False)
        status = comp.get("status", {}).get("type", {}).get("name", "")
        game_id = event.get("id")

        game = {
            "game_id": game_id,
            "home": home_name,
            "away": away_name,
            "neutral": neutral,
            "status": status,
            "home_score": home_score,
            "away_score": away_score,
        }

        # Fetch odds from summary
        try:
            summary = requests.get(ESPN_SUMMARY, params={"event": game_id},
                                   timeout=10)
            summary.raise_for_status()
            pickcenter = summary.json().get("pickcenter", [])
            if pickcenter:
                odds = _extract_odds(pickcenter[0], teams_flipped=False)
                game.update(odds)
        except requests.RequestException:
            pass

        games.append(game)

    return games


def fetch_game_odds(home_team: str, away_team: str,
                    sport: str = "ncaa_basketball") -> dict | None:
    """Fetch current Vegas odds for a matchup via ESPN pickcenter.

    Looks up team ESPN IDs from the database, finds the matching game
    on today's scoreboard, then pulls spread/total from the summary.

    Returns dict with spread (home perspective), total, moneyline, or None.
    """
    # Resolve team IDs to find ESPN team IDs
    conn = get_db(sport)
    home_id = resolve_team(conn, home_team)
    away_id = resolve_team(conn, away_team)

    if not home_id or not away_id:
        conn.close()
        return None

    # Get ESPN aliases for matching
    home_aliases = _get_team_aliases(conn, home_id, "espn")
    away_aliases = _get_team_aliases(conn, away_id, "espn")
    conn.close()

    # Fetch today's scoreboard to find the game
    try:
        resp = requests.get(ESPN_SCOREBOARD, params={
                                "groups": "50", "limit": "200",
                                "dates": date.today().strftime("%Y%m%d"),
                            }, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    events = resp.json().get("events", [])
    match = _find_game(events, home_id, away_id,
                       home_aliases, away_aliases, home_team, away_team)
    if not match:
        return None
    game_id, teams_flipped = match

    # Fetch summary with pickcenter odds
    try:
        resp = requests.get(ESPN_SUMMARY, params={"event": game_id}, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    data = resp.json()
    pickcenter = data.get("pickcenter", [])
    if not pickcenter:
        return None

    return _extract_odds(pickcenter[0], teams_flipped)


def _get_team_aliases(conn, team_id: int, source: str) -> list[str]:
    """Get all aliases for a team from a specific source."""
    rows = conn.execute(
        "SELECT alias FROM team_aliases WHERE team_id = ? AND source = ?",
        (team_id, source)
    ).fetchall()
    return [r["alias"] for r in rows]


def _find_game(events, home_id, away_id,
               home_aliases, away_aliases,
               home_team, away_team) -> tuple[str, bool] | None:
    """Find ESPN game ID matching our teams from scoreboard events.

    Returns (game_id, teams_flipped) where teams_flipped is True if
    our "home" team is ESPN's "away" team (spread needs negating).
    """
    home_names = {a.lower() for a in home_aliases} | {home_team.lower()}
    away_names = {a.lower() for a in away_aliases} | {away_team.lower()}

    for event in events:
        comps = event.get("competitions", [])
        if not comps:
            continue
        competitors = comps[0].get("competitors", [])
        if len(competitors) != 2:
            continue

        # ESPN competitors: find which is home/away
        espn_home_names = set()
        espn_away_names = set()
        for c in competitors:
            team = c.get("team", {})
            names = {
                team.get("displayName", "").lower(),
                team.get("shortDisplayName", "").lower(),
                team.get("name", "").lower(),
                team.get("location", "").lower(),
            }
            if c.get("homeAway") == "home":
                espn_home_names = names
            else:
                espn_away_names = names

        all_espn = espn_home_names | espn_away_names

        # Check if both our teams are in this game
        our_home_in_game = bool(home_names & all_espn)
        our_away_in_game = bool(away_names & all_espn)

        if our_home_in_game and our_away_in_game:
            # Is our "home" team ESPN's away team?
            teams_flipped = bool(home_names & espn_away_names)
            return event.get("id"), teams_flipped

    return None


def _extract_odds(pick: dict, teams_flipped: bool) -> dict:
    """Extract spread, total, and moneyline from ESPN pickcenter data.

    The spread is returned from our caller's "home" team perspective
    (positive = our home team favored). If teams_flipped is True,
    our home team is ESPN's away team, so we negate the spread.
    """
    result = {
        "provider": pick.get("provider", {}).get("name", "unknown"),
    }

    # ESPN spread: negative = ESPN's home team favored.
    # Our model: positive margin = our home team wins.
    # So: negate ESPN's spread to convert to margin convention,
    # then negate again if our home ≠ ESPN's home.
    flip = -1 if teams_flipped else 1

    spread_data = pick.get("pointSpread", {})
    if spread_data:
        home_line = spread_data.get("home", {}).get("close", {}).get("line")
        if home_line is not None:
            # ESPN home line is negative when ESPN home is favored
            # Negate to get margin (positive = ESPN home wins)
            # Then apply flip for our perspective
            result["spread"] = -float(home_line) * flip

    # Fallback to top-level spread field
    if "spread" not in result and "spread" in pick:
        result["spread"] = -float(pick["spread"]) * flip

    # Total (same regardless of home/away)
    total_data = pick.get("total", {})
    if total_data:
        over_line = total_data.get("over", {}).get("close", {}).get("line", "")
        if over_line:
            result["total"] = float(str(over_line).lstrip("oO"))
    if "total" not in result and "overUnder" in pick:
        result["total"] = float(pick["overUnder"])

    # Moneyline (from our home team's perspective)
    ml_data = pick.get("moneyline", {})
    if ml_data:
        if teams_flipped:
            our_home_ml = ml_data.get("away", {}).get("close", {}).get("odds")
            our_away_ml = ml_data.get("home", {}).get("close", {}).get("odds")
        else:
            our_home_ml = ml_data.get("home", {}).get("close", {}).get("odds")
            our_away_ml = ml_data.get("away", {}).get("close", {}).get("odds")
        try:
            if our_home_ml:
                result["home_moneyline"] = int(our_home_ml)
        except (ValueError, TypeError):
            pass
        try:
            if our_away_ml:
                result["away_moneyline"] = int(our_away_ml)
        except (ValueError, TypeError):
            pass

    return result

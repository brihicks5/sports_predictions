"""Parlay suggestion engine.

Identifies candidate parlays from a slate of predicted games and
calculates expected value based on model probabilities.
"""

from itertools import combinations


def american_to_decimal(odds):
    """Convert American odds to decimal odds."""
    if odds > 0:
        return 1 + odds / 100
    else:
        return 1 + 100 / abs(odds)


def decimal_to_american(prob):
    """Convert a probability to fair American odds."""
    if prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return round(-100 * prob / (1 - prob))
    else:
        return round(100 * (1 - prob) / prob)


def calculate_parlay(legs):
    """Calculate true probability, fair odds, and EV for a parlay.

    Args:
        legs: list of dicts, each with:
            - prob: true probability of this leg hitting
            - ml: American odds for this leg (optional, used for EV calc)
                   For ATS legs, pass -110 or None to assume -110.

    Returns dict with:
        true_prob, fair_decimal, fair_american,
        estimated_decimal (from leg odds), ev (per dollar), legs
    """
    true_prob = 1.0
    for leg in legs:
        true_prob *= leg["prob"]

    fair_decimal = 1 / true_prob if true_prob > 0 else None
    fair_american = decimal_to_american(true_prob)

    # Estimate actual parlay odds from leg prices
    estimated_decimal = 1.0
    has_all_odds = True
    for leg in legs:
        ml = leg.get("ml")
        if ml is not None:
            estimated_decimal *= american_to_decimal(ml)
        else:
            # Default ATS pricing
            estimated_decimal *= american_to_decimal(-110)

    if not has_all_odds:
        ev = None
        estimated_decimal = None
    else:
        ev = true_prob * (estimated_decimal - 1) - (1 - true_prob)

    return {
        "true_prob": true_prob,
        "fair_decimal": fair_decimal,
        "fair_american": fair_american,
        "estimated_decimal": estimated_decimal,
        "ev": ev,
    }


def fair_odds(probs):
    """Calculate fair parlay odds from a list of leg probabilities.

    Args:
        probs: list of floats, each a probability (0-1)

    Returns dict with true_prob, fair_american, and the minimum
    American odds you should accept for this parlay to be +EV.
    """
    true_prob = 1.0
    for p in probs:
        true_prob *= p

    fair_american = decimal_to_american(true_prob)
    fair_decimal = 1 / true_prob if true_prob > 0 else None

    return {
        "true_prob": true_prob,
        "fair_american": fair_american,
        "fair_decimal": fair_decimal,
    }


def _classify_ats_leg(abs_cover, abs_spread):
    """Return estimated true probability for an ATS leg, or None if ineligible."""
    if abs_cover < 2:
        return None  # no edge
    # Sweet spot: spread 6.5-15, cover 3-6
    if 6.5 <= abs_spread <= 15 and 3 <= abs_cover <= 6:
        return 0.65
    # Strong pick outside sweet spot
    if abs_cover >= 3:
        return 0.58
    # Lean (cover 2-3): breakeven, skip for parlays
    return None


def _build_ml_legs(games, postseason):
    """Identify eligible moneyline parlay legs."""
    legs = []
    for g in games:
        win_prob = g.get("_win_prob")
        vegas_spread = g.get("_vegas_spread")
        if win_prob is None or vegas_spread is None:
            continue

        # Model's predicted winner
        if win_prob >= 0.5:
            model_fav_is_home = True
            fav_prob = win_prob
            fav_name = g["_home"]
            fav_ml = g.get("_home_ml")
        else:
            model_fav_is_home = False
            fav_prob = 1 - win_prob
            fav_name = g["_away"]
            fav_ml = g.get("_away_ml")

        # Require 70%+ win probability
        if fav_prob < 0.70:
            continue

        # Require Vegas agrees on winner (no contrarian picks)
        vegas_home_fav = vegas_spread > 0  # positive = home favored in slate
        if model_fav_is_home != vegas_home_fav:
            continue

        legs.append({
            "type": "ML",
            "game": g["game"],
            "team": fav_name,
            "prob": fav_prob,
            "ml": fav_ml,
            "description": f"{fav_name} ML"
                           + (f" ({fav_ml:+d})" if fav_ml else ""),
        })

    return legs


def _build_ats_legs(games, postseason):
    """Identify eligible ATS parlay legs."""
    if postseason:
        return []  # tournament ATS is unreliable (39% accuracy)

    legs = []
    for g in games:
        ats_cover = g.get("_ats_cover")
        vegas_spread = g.get("_vegas_spread")
        if ats_cover is None or vegas_spread is None:
            continue

        abs_cover = abs(ats_cover)
        abs_spread = abs(vegas_spread)

        prob = _classify_ats_leg(abs_cover, abs_spread)
        if prob is None:
            continue

        # Determine which team covers and their line
        if ats_cover > 0:
            # Home covers
            edge_team = g["_home"]
            edge_line = -vegas_spread  # home's line
        else:
            # Away covers
            edge_team = g["_away"]
            edge_line = vegas_spread  # away's line

        legs.append({
            "type": "ATS",
            "game": g["game"],
            "team": edge_team,
            "line": edge_line,
            "prob": prob,
            "ml": None,  # ATS legs use standard -110
            "description": f"{edge_team} {edge_line:+.1f}",
        })

    return legs


def find_parlay_candidates(games, postseason=False, max_legs=3):
    """Find 0-3 parlay suggestions from a slate of predicted games.

    Each game dict should include internal fields:
      _win_prob, _vegas_spread, _ats_cover, _home, _away,
      _home_ml, _away_ml

    Returns list of parlay dicts sorted by EV.
    """
    ml_legs = _build_ml_legs(games, postseason)
    ats_legs = _build_ats_legs(games, postseason)

    candidates = []

    # Generate ML-only parlays (2-3 legs)
    for size in (2, 3):
        for combo in combinations(ml_legs, size):
            candidates.append(("ML", list(combo)))

    # Generate ATS-only parlays (2 legs only)
    for combo in combinations(ats_legs, 2):
        candidates.append(("ATS", list(combo)))

    # Generate mixed parlays (1 ML + 1 ATS)
    for ml in ml_legs:
        for ats in ats_legs:
            if ml["game"] != ats["game"]:  # different games
                candidates.append(("Mixed", [ml, ats]))

    # Score and filter
    scored = []
    for parlay_type, legs in candidates:
        calc = calculate_parlay(legs)

        if calc["true_prob"] < 0.35:
            continue

        # Filter: only positive EV (or unknown EV)
        if calc["ev"] is not None and calc["ev"] <= 0:
            continue

        leg_descriptions = [leg["description"] for leg in legs]

        scored.append({
            "type": parlay_type,
            "legs": legs,
            "leg_descriptions": leg_descriptions,
            "true_prob": calc["true_prob"],
            "fair_american": calc["fair_american"],
            "estimated_odds": calc["estimated_decimal"],
            "ev": calc["ev"],
            "description": f"{parlay_type}: {' + '.join(leg_descriptions)}",
        })

    # Sort by EV (unknowns last), then by true_prob
    scored.sort(key=lambda p: (p["ev"] or -999, p["true_prob"]), reverse=True)

    # Return top 3 non-overlapping parlays
    result = []
    used_games = set()
    for p in scored:
        parlay_games = {leg["game"] for leg in p["legs"]}
        if parlay_games & used_games:
            continue
        result.append(p)
        used_games |= parlay_games
        if len(result) >= 3:
            break

    # Show best ML combo as a boost candidate if not already covered
    if ml_legs:
        best_combo = None
        best_prob = 0
        for size in (2, 3):
            for combo in combinations(ml_legs, size):
                prob = 1.0
                for leg in combo:
                    prob *= leg["prob"]
                if prob >= 0.35 and prob > best_prob:
                    best_prob = prob
                    best_combo = (size, list(combo), prob)

        if best_combo:
            size, legs, true_prob = best_combo
            combo_games = {leg["game"] for leg in legs}
            # Only show if these games aren't already in a positive-EV parlay
            if not (combo_games & used_games):
                fair_american = decimal_to_american(true_prob)
                leg_descriptions = [leg["description"] for leg in legs]
                result.append({
                    "type": "ML (boost candidate)",
                    "legs": legs,
                    "leg_descriptions": leg_descriptions,
                    "true_prob": true_prob,
                    "fair_american": fair_american,
                    "estimated_odds": None,
                    "ev": None,
                    "description": f"ML (look for boost): "
                                   + " + ".join(leg_descriptions),
                })

    return result


def format_parlays(parlays):
    """Format parlay suggestions as a string for display."""
    if not parlays:
        return ""

    lines = ["\n--- Parlay Suggestions ---"]
    for i, p in enumerate(parlays, 1):
        lines.append(f"\n  {i}. {p['description']}")
        parts = [f"True prob: {p['true_prob']:.0%}"]
        if p["fair_american"] is not None:
            parts.append(f"Fair odds: {p['fair_american']:+d}")
        if p["ev"] is not None:
            parts.append(f"EV: {p['ev']:+.2f}/dollar")
        lines.append(f"     {' | '.join(parts)}")

    return "\n".join(lines) + "\n"

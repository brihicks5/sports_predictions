"""Generic sports prediction model.

Uses team stats from the database to predict game outcomes.
The same model structure works for any sport — just needs different
stats in the database.
"""

import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from sports_predictions.db import get_db, resolve_team, DATA_DIR

# Top 8 features by importance. Using an explicit allowlist rather than
# auto-discovering all stats — extra features add noise without improving accuracy.
FEATURE_STATS = [
    "adj_efficiency_margin",
    "pythag",
    "avg_margin",
    "win_pct",
    "luck",
    "sos",
    "sos_offense",
]


def _get_team_features(conn, team_id: int, season: int,
                       stat_names: list) -> dict:
    """Get a team's stats as a feature dict."""
    rows = conn.execute("""
        SELECT stat_name, stat_value FROM team_stats
        WHERE team_id = ? AND season = ? AND stat_name IN ({})
    """.format(",".join("?" * len(stat_names))),
        (team_id, season, *stat_names)
    ).fetchall()
    return {r["stat_name"]: r["stat_value"] for r in rows}


def build_training_data(sport: str, seasons: list = None):
    """Build feature matrix from historical games.

    For each game, features are the difference between home team stats
    and away team stats (home - away).

    Returns X, y_win (1 if home won), y_margin (home - away score),
    y_total (total points), and feature_stats.
    """
    conn = get_db(sport)

    if seasons is None:
        rows = conn.execute(
            "SELECT DISTINCT season FROM games WHERE season >= 2010 "
            "ORDER BY season"
        ).fetchall()
        seasons = [r["season"] for r in rows]

    feature_stats = FEATURE_STATS

    X_rows = []
    y_win_rows = []
    y_margin_rows = []
    y_total_rows = []

    for season in seasons:
        games = conn.execute("""
            SELECT home_team_id, away_team_id, home_score, away_score,
                   neutral_site
            FROM games
            WHERE season = ? AND home_score IS NOT NULL
        """, (season,)).fetchall()

        for game in games:
            home_feats = _get_team_features(
                conn, game["home_team_id"], season, feature_stats
            )
            away_feats = _get_team_features(
                conn, game["away_team_id"], season, feature_stats
            )

            # Skip if either team has no stats
            if not home_feats or not away_feats:
                continue

            # Features: difference (home - away) for each stat + home court
            row = {}
            for stat in feature_stats:
                home_val = home_feats.get(stat, 0.0)
                away_val = away_feats.get(stat, 0.0)
                row[f"diff_{stat}"] = home_val - away_val
            row["neutral_site"] = game["neutral_site"]

            X_rows.append(row)
            y_win_rows.append(
                1 if game["home_score"] > game["away_score"] else 0
            )
            y_margin_rows.append(game["home_score"] - game["away_score"])
            y_total_rows.append(game["home_score"] + game["away_score"])

    conn.close()

    X = pd.DataFrame(X_rows)
    y_win = np.array(y_win_rows)
    y_margin = np.array(y_margin_rows)
    y_total = np.array(y_total_rows)
    return X, y_win, y_margin, y_total, feature_stats


# Stats available from the KenPom archive endpoint (point-in-time)
ARCHIVE_STATS = [
    "adj_efficiency_margin",
    "adj_offensive_efficiency",
    "adj_defensive_efficiency",
    "adj_tempo",
]


def _get_team_ratings_by_date(conn, team_id: int, date: str,
                               stat_names: list) -> dict:
    """Get a team's point-in-time ratings for a specific date."""
    rows = conn.execute("""
        SELECT stat_name, stat_value FROM team_ratings_by_date
        WHERE team_id = ? AND date = ? AND stat_name IN ({})
    """.format(",".join("?" * len(stat_names))),
        (team_id, date, *stat_names)
    ).fetchall()
    return {r["stat_name"]: r["stat_value"] for r in rows}


def build_training_data_pit(sport: str, seasons: list = None):
    """Build feature matrix using point-in-time KenPom ratings.

    Like build_training_data, but uses team_ratings_by_date to get
    each team's stats as of the game date, not end-of-season values.

    Returns X, y_win, y_margin, y_total, and feature_stats.
    """
    conn = get_db(sport)

    if seasons is None:
        rows = conn.execute(
            "SELECT DISTINCT season FROM games WHERE season >= 2010 "
            "ORDER BY season"
        ).fetchall()
        seasons = [r["season"] for r in rows]

    feature_stats = ARCHIVE_STATS

    X_rows = []
    y_win_rows = []
    y_margin_rows = []
    y_total_rows = []

    for season in seasons:
        games = conn.execute("""
            SELECT home_team_id, away_team_id, home_score, away_score,
                   neutral_site, date
            FROM games
            WHERE season = ? AND home_score IS NOT NULL
            ORDER BY date
        """, (season,)).fetchall()

        for game in games:
            home_feats = _get_team_ratings_by_date(
                conn, game["home_team_id"], game["date"], feature_stats
            )
            away_feats = _get_team_ratings_by_date(
                conn, game["away_team_id"], game["date"], feature_stats
            )

            # Skip if either team has no ratings for this date
            if not home_feats or not away_feats:
                continue

            row = {}
            for stat in feature_stats:
                home_val = home_feats.get(stat, 0.0)
                away_val = away_feats.get(stat, 0.0)
                row[f"diff_{stat}"] = home_val - away_val
            row["neutral_site"] = game["neutral_site"]

            X_rows.append(row)
            y_win_rows.append(
                1 if game["home_score"] > game["away_score"] else 0
            )
            y_margin_rows.append(game["home_score"] - game["away_score"])
            y_total_rows.append(game["home_score"] + game["away_score"])

    conn.close()

    X = pd.DataFrame(X_rows)
    y_win = np.array(y_win_rows)
    y_margin = np.array(y_margin_rows)
    y_total = np.array(y_total_rows)
    return X, y_win, y_margin, y_total, feature_stats


def _logistic(x, k):
    """Logistic function: P(home_win) = 1 / (1 + exp(-k * margin))."""
    return 1.0 / (1.0 + np.exp(np.clip(-k * x, -500, 500)))


def train_model(sport: str, seasons: list = None):
    """Train prediction models and save to disk.

    Trains two models:
    - Margin regressor (by how much)
    - Total regressor (combined score)

    Win probability is derived from predicted margin via a logistic function
    fitted on historical results, ensuring margin and win prob never disagree.

    Returns the margin model and cross-validation margin MAE.
    """
    X, y_win, y_margin, y_total, feature_stats = build_training_data(
        sport, seasons
    )
    print(f"Training on {len(X)} games with features: {feature_stats}")

    def _train_regressor(y, label):
        """Train a regressor with CV and return (model, scores)."""
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42,
        )
        scores = cross_val_score(
            model, X, y, cv=5, scoring="neg_mean_absolute_error"
        )
        print(f"{label} MAE: {-scores.mean():.2f} pts "
              f"(+/- {scores.std():.2f})")
        model.fit(X, y)
        return model, scores

    # Train margin and total regressors in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        margin_future = executor.submit(_train_regressor, y_margin, "Margin")
        total_future = executor.submit(_train_regressor, y_total, "Total")
        margin_model, margin_scores = margin_future.result()
        total_model, total_scores = total_future.result()

    # Derive logistic k from margin standard deviation
    # P(home_win | margin) = 1 / (1 + exp(-k * margin))
    # k = pi / (std * sqrt(3)) matches the logistic CDF to the margin distribution
    margin_std = np.std(y_margin)
    margin_to_win_k = np.pi / (margin_std * np.sqrt(3))
    print(f"Logistic k={margin_to_win_k:.4f} (margin std={margin_std:.2f}, "
          f"margin of 5 -> {_logistic(5, margin_to_win_k)*100:.1f}% win prob)")

    # Win accuracy: use margin model's training predictions as rough estimate
    train_margin_preds = margin_model.predict(X)
    win_accuracy = ((train_margin_preds > 0).astype(int) == y_win).mean()
    print(f"Win accuracy (from margin, train set): {win_accuracy:.4f}")

    # Save models
    model_path = DATA_DIR / f"{sport}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "margin_model": margin_model,
            "total_model": total_model,
            "margin_to_win_k": margin_to_win_k,
            "feature_stats": feature_stats,
            "feature_columns": list(X.columns),
        }, f)
    print(f"Models saved to {model_path}")

    # Feature importance (from margin model)
    print("\nFeature importance (margin model):")
    for name, imp in sorted(zip(X.columns, margin_model.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    return margin_model, -margin_scores.mean()


def predict_game(sport: str, home_team: str, away_team: str,
                 season: int, neutral_site: bool = False) -> dict:
    """Predict the outcome of a game.

    Returns dict with win probabilities, predicted winner, predicted margin,
    and predicted scores for each team.
    """
    model_path = DATA_DIR / f"{sport}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run train_model first."
        )

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    margin_model = saved["margin_model"]
    total_model = saved["total_model"]
    margin_to_win_k = saved["margin_to_win_k"]
    feature_stats = saved["feature_stats"]
    feature_columns = saved["feature_columns"]

    conn = get_db(sport)

    home_id = resolve_team(conn, home_team)
    away_id = resolve_team(conn, away_team)

    if not home_id:
        suggestions = conn.execute(
            "SELECT alias FROM team_aliases WHERE alias LIKE ?",
            (f"%{home_team}%",)
        ).fetchall()
        conn.close()
        msg = f"Team not found: {home_team}"
        if suggestions:
            msg += (". Did you mean: "
                    + ", ".join(s["alias"] for s in suggestions[:5]) + "?")
        raise ValueError(msg)
    if not away_id:
        suggestions = conn.execute(
            "SELECT alias FROM team_aliases WHERE alias LIKE ?",
            (f"%{away_team}%",)
        ).fetchall()
        conn.close()
        msg = f"Team not found: {away_team}"
        if suggestions:
            msg += (". Did you mean: "
                    + ", ".join(s["alias"] for s in suggestions[:5]) + "?")
        raise ValueError(msg)

    home_feats = _get_team_features(conn, home_id, season, feature_stats)
    away_feats = _get_team_features(conn, away_id, season, feature_stats)
    conn.close()

    row = {}
    for stat in feature_stats:
        row[f"diff_{stat}"] = home_feats.get(stat, 0.0) - away_feats.get(stat, 0.0)
    row["neutral_site"] = int(neutral_site)

    X = pd.DataFrame([row])[feature_columns]

    # Predicted margin (home - away) and total
    pred_margin = margin_model.predict(X)[0]
    pred_total = total_model.predict(X)[0]

    # Win probability derived from margin via logistic function
    home_win_prob = _logistic(pred_margin, margin_to_win_k)

    # Derive individual scores: total = home + away, margin = home - away
    # Round margin first, then derive scores so they stay consistent
    rounded_margin = round(pred_margin)
    rounded_total = round(pred_total)
    # Ensure total and margin have the same parity (both even or both odd)
    if rounded_total % 2 != rounded_margin % 2:
        rounded_total += 1
    pred_home_score = (rounded_total + rounded_margin) // 2
    pred_away_score = (rounded_total - rounded_margin) // 2

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_prob": round(home_win_prob, 4),
        "away_win_prob": round(1 - home_win_prob, 4),
        "predicted_winner": home_team if home_win_prob > 0.5 else away_team,
        "predicted_margin": round(pred_margin, 1),
        "predicted_home_score": round(pred_home_score),
        "predicted_away_score": round(pred_away_score),
    }

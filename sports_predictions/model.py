"""Generic sports prediction model.

Uses team stats from the database to predict game outcomes.
The same model structure works for any sport — just needs different
stats in the database.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from sports_predictions.db import get_db, DATA_DIR

# Stats to exclude from features (e.g. metadata that shouldn't be model inputs).
# Everything else in team_stats is used automatically.
EXCLUDED_STATS = {"games_played"}


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
            "SELECT DISTINCT season FROM games ORDER BY season"
        ).fetchall()
        seasons = [r["season"] for r in rows]

    # Use all stats in the database except excluded ones
    available = conn.execute(
        "SELECT DISTINCT stat_name FROM team_stats ORDER BY stat_name"
    ).fetchall()
    feature_stats = [
        r["stat_name"] for r in available
        if r["stat_name"] not in EXCLUDED_STATS
    ]

    if not feature_stats:
        conn.close()
        raise ValueError(
            "No feature stats found in database. "
            "Import team stats before training."
        )

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


def train_model(sport: str, seasons: list = None):
    """Train prediction models and save to disk.

    Trains three models:
    - Win classifier (who wins)
    - Margin regressor (by how much)
    - Total regressor (combined score)

    Returns the win model and cross-validation accuracy.
    """
    X, y_win, y_margin, y_total, feature_stats = build_training_data(
        sport, seasons
    )
    print(f"Training on {len(X)} games with features: {feature_stats}")

    # Win probability classifier
    win_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42,
    )
    win_scores = cross_val_score(win_model, X, y_win, cv=5, scoring="accuracy")
    print(f"Win classifier accuracy: {win_scores.mean():.4f} "
          f"(+/- {win_scores.std():.4f})")
    win_model.fit(X, y_win)

    # Score margin regressor (home - away)
    margin_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42,
    )
    margin_scores = cross_val_score(
        margin_model, X, y_margin, cv=5, scoring="neg_mean_absolute_error"
    )
    print(f"Margin MAE: {-margin_scores.mean():.2f} pts "
          f"(+/- {margin_scores.std():.2f})")
    margin_model.fit(X, y_margin)

    # Total points regressor
    total_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42,
    )
    total_scores = cross_val_score(
        total_model, X, y_total, cv=5, scoring="neg_mean_absolute_error"
    )
    print(f"Total MAE: {-total_scores.mean():.2f} pts "
          f"(+/- {total_scores.std():.2f})")
    total_model.fit(X, y_total)

    # Save all models
    model_path = DATA_DIR / f"{sport}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "win_model": win_model,
            "margin_model": margin_model,
            "total_model": total_model,
            "feature_stats": feature_stats,
            "feature_columns": list(X.columns),
        }, f)
    print(f"Models saved to {model_path}")

    # Feature importance (from win model)
    print("\nFeature importance (win model):")
    for name, imp in sorted(zip(X.columns, win_model.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    return win_model, win_scores.mean()


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

    win_model = saved["win_model"]
    margin_model = saved["margin_model"]
    total_model = saved["total_model"]
    feature_stats = saved["feature_stats"]
    feature_columns = saved["feature_columns"]

    conn = get_db(sport)

    home_row = conn.execute(
        "SELECT id FROM teams WHERE name = ?", (home_team,)
    ).fetchone()
    away_row = conn.execute(
        "SELECT id FROM teams WHERE name = ?", (away_team,)
    ).fetchone()

    if not home_row:
        conn.close()
        raise ValueError(f"Team not found: {home_team}")
    if not away_row:
        conn.close()
        raise ValueError(f"Team not found: {away_team}")

    home_feats = _get_team_features(conn, home_row["id"], season, feature_stats)
    away_feats = _get_team_features(conn, away_row["id"], season, feature_stats)
    conn.close()

    row = {}
    for stat in feature_stats:
        row[f"diff_{stat}"] = home_feats.get(stat, 0.0) - away_feats.get(stat, 0.0)
    row["neutral_site"] = int(neutral_site)

    X = pd.DataFrame([row])[feature_columns]

    # Win probability
    prob = win_model.predict_proba(X)[0]
    home_win_prob = prob[1]

    # Predicted margin (home - away) and total
    pred_margin = margin_model.predict(X)[0]
    pred_total = total_model.predict(X)[0]

    # Derive individual scores: total = home + away, margin = home - away
    pred_home_score = (pred_total + pred_margin) / 2
    pred_away_score = (pred_total - pred_margin) / 2

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

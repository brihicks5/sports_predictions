"""Generic sports prediction model.

Uses team stats from the database to predict game outcomes.
The same model structure works for any sport — just needs different
stats in the database.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from sports_predictions.db import get_db, DATA_DIR

# Stats to use as features, in priority order.
# The model uses whichever of these are available in the database.
FEATURE_STATS = [
    "adj_efficiency_margin",
    "adj_offensive_efficiency",
    "adj_defensive_efficiency",
    "adj_tempo",
    "win_pct",
    "avg_margin",
    "avg_points_for",
    "avg_points_against",
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
    and away team stats (home - away). Target is 1 if home team won.
    """
    conn = get_db(sport)

    if seasons is None:
        rows = conn.execute(
            "SELECT DISTINCT season FROM games ORDER BY season"
        ).fetchall()
        seasons = [r["season"] for r in rows]

    # Figure out which stats are available
    available = conn.execute(
        "SELECT DISTINCT stat_name FROM team_stats"
    ).fetchall()
    available_stats = [r["stat_name"] for r in available]
    feature_stats = [s for s in FEATURE_STATS if s in available_stats]

    if not feature_stats:
        conn.close()
        raise ValueError(
            "No feature stats found in database. "
            "Import team stats before training."
        )

    X_rows = []
    y_rows = []

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
            y_rows.append(1 if game["home_score"] > game["away_score"] else 0)

    conn.close()

    X = pd.DataFrame(X_rows)
    y = np.array(y_rows)
    return X, y, feature_stats


def train_model(sport: str, seasons: list = None):
    """Train a prediction model and save it to disk.

    Returns the trained model and cross-validation accuracy.
    """
    X, y, feature_stats = build_training_data(sport, seasons)
    print(f"Training on {len(X)} games with features: {feature_stats}")

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )

    # Cross-validate
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Train on full data
    model.fit(X, y)

    # Save model and metadata
    model_path = DATA_DIR / f"{sport}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_stats": feature_stats,
            "feature_columns": list(X.columns),
        }, f)
    print(f"Model saved to {model_path}")

    # Feature importance
    print("\nFeature importance:")
    for name, imp in sorted(zip(X.columns, model.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    return model, scores.mean()


def predict_game(sport: str, home_team: str, away_team: str,
                 season: int, neutral_site: bool = False) -> dict:
    """Predict the outcome of a game.

    Returns dict with home_win_prob, away_win_prob, and predicted_winner.
    """
    model_path = DATA_DIR / f"{sport}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run train_model first."
        )

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
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
    prob = model.predict_proba(X)[0]

    home_win_prob = prob[1]
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_prob": round(home_win_prob, 4),
        "away_win_prob": round(1 - home_win_prob, 4),
        "predicted_winner": home_team if home_win_prob > 0.5 else away_team,
    }

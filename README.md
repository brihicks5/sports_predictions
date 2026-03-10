# sports_predictions

Predict the outcomes of NCAA basketball games using historical data and machine learning.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your API tokens
```

## Quick Start

### 1. Import historical data (Kaggle)

Download the [March Machine Learning Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data) dataset. Place the CSV files in `data/kaggle/`, then:

```bash
python scripts/import_kaggle.py
```

This imports all historical games (converting Kaggle ordinal dates to ISO format), computes team stats, and trains the model.

### 2. Add KenPom advanced metrics (optional but recommended)

```bash
python scripts/update_data.py --season 2026 --skip-espn
```

### 3. Fetch recent game results

```bash
python scripts/fetch_games.py 2026-03-02              # single date
python scripts/fetch_games.py 2026-03-02 2026-03-08   # date range
```

### 4. Predict a game

```bash
python scripts/predict.py "Duke" "North Carolina" --season 2026
python scripts/predict.py "Duke" "North Carolina" --neutral  # neutral site
```

## Data Refresh

Run `scripts/update_data.py` to fetch ESPN games, refresh KenPom stats, and retrain the model:

```bash
python scripts/update_data.py --season 2026
```

The script automatically skips model retraining when no new data is detected (no new ESPN games and no KenPom stat changes). You can also control behavior with flags:

```bash
python scripts/update_data.py --season 2026 --skip-espn     # skip ESPN fetch
python scripts/update_data.py --season 2026 --skip-kenpom   # skip KenPom fetch
python scripts/update_data.py --season 2026 --skip-train    # skip model training
```

Set this up as a cron job for nightly updates.

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests cover:
- **update_data** — skip-training logic, CLI flag combinations
- **db** — upsert change detection, game inserts, team resolution/dedup
- **scrapers** — NCAA season date mapping, Kaggle ordinal-to-ISO conversion

## Project Structure

```
sports_predictions/
├── sports_predictions/       # Python package
│   ├── db.py                 # Generic SQLite schema (works for any sport)
│   ├── model.py              # Gradient boosted margin + total regressors
│   └── scrapers/
│       └── ncaa_basketball.py  # Kaggle, KenPom, ESPN importers
├── scripts/
│   ├── import_kaggle.py      # Bootstrap with historical data
│   ├── update_data.py        # Nightly update script
│   ├── fetch_games.py        # Fetch ESPN games by date/range
│   ├── predict.py            # CLI for predictions
│   ├── migrate_team_aliases.py  # One-time team deduplication
│   └── migrate_dates.py       # One-time ordinal-to-ISO date conversion
├── tests/                    # Unit tests
├── data/                     # SQLite DBs & models (gitignored)
└── requirements.txt
```

---

Built with [Claude Code](https://claude.com/claude-code)

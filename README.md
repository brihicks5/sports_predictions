# sports_predictions

Predict the outcomes of sports games using historical data and machine learning. Currently supports NCAA basketball, with a generic architecture designed to expand to other sports (NFL, NBA, NHL, college football).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1. Import historical data (Kaggle)

Download the [March Machine Learning Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data) dataset. Place the CSV files in `data/kaggle/`, then:

```bash
python scripts/import_kaggle.py
```

This imports all historical games, computes team stats, and trains the model.

### 2. Add KenPom advanced metrics (optional but recommended)

```bash
python scripts/update_data.py --season 2026 --kenpom-user you@email.com --kenpom-pass yourpass
```

### 3. Predict a game

```bash
python scripts/predict.py "Duke" "North Carolina" --season 2026
python scripts/predict.py "Duke" "North Carolina" --neutral  # neutral site
```

### 4. Nightly updates

Run `scripts/update_data.py` via cron to keep stats and the model current during the season.

## Project Structure

```
sports_predictions/
├── sports_predictions/       # Python package
│   ├── db.py                 # Generic SQLite schema (works for any sport)
│   ├── model.py              # Gradient boosted classifier
│   └── scrapers/
│       └── ncaa_basketball.py
├── scripts/
│   ├── import_kaggle.py      # Bootstrap with historical data
│   ├── update_data.py        # Nightly update script
│   └── predict.py            # CLI for predictions
├── data/                     # SQLite DBs & models (gitignored)
└── requirements.txt
```

## Adding a New Sport

1. Create a new scraper in `sports_predictions/scrapers/`
2. Use the same `db.py` helpers — each sport gets its own `.db` file
3. The model in `model.py` is generic and works with any sport's stats

---

Built with [Claude Code](https://claude.com/claude-code)

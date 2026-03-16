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

### 5. Predict a full day's slate

Fetches all games and Vegas lines from ESPN, runs the model, and flags ATS picks:

```bash
python scripts/predict_slate.py                    # today's games
python scripts/predict_slate.py --date 2026-03-14  # specific date
python scripts/predict_slate.py --date 2026-03-14 --results-only  # fill in scores only
```

Results are saved to `data/slates/YYYY-MM-DD.txt`. Use `--results-only` to update an existing slate with final scores without re-running predictions.

### 6. Simulate the NCAA Tournament

After Selection Sunday, fetch the bracket from ESPN and run simulations:

```bash
# 1. Fetch bracket from ESPN (teams, seeds, regions, First Four)
python scripts/fetch_bracket.py

# 2. Validate team names resolve correctly
python scripts/simulate_tournament.py --validate-only

# 3. Print game-by-game picks (for filling in a bracket on a website)
python scripts/simulate_tournament.py --pick-bracket

# 4. Run Monte Carlo simulation for probability tables
python scripts/simulate_tournament.py -n 10000
python scripts/simulate_tournament.py -n 10000 --seed 42    # reproducible
python scripts/simulate_tournament.py -n 10000 --top 20     # top 20 only

# 5. As the tournament progresses, update with real results and re-run
python scripts/fetch_bracket.py --update          # fetches completed game results
python scripts/simulate_tournament.py --pick-bracket   # locked-in games show *
python scripts/simulate_tournament.py -n 10000         # updated probabilities
```

**`--pick-bracket`** runs through the bracket deterministically (always picking the higher-probability team) and prints every game round by round with win percentages. Use this to fill in a bracket.

**Monte Carlo mode** (`-n`) simulates the full 67-game tournament thousands of times, randomly resolving each game weighted by win probability. The output shows how often each team reached each round across all simulations.

**`--update`** fetches completed tournament games from ESPN and records them as known results. The simulator locks in these results and only simulates remaining games.

Note: Verify `final_four_matchups` in `data/bracket.json` after the initial fetch — ESPN doesn't expose which regions are paired, so the script guesses.

### 7. Backfill historical Vegas odds

Scrapes ESPN's pickcenter for closing lines across historical seasons:

```bash
python -u scripts/backfill_odds.py --seasons 2023        # single season
python -u scripts/backfill_odds.py --seasons 2018-2023   # range
python -u scripts/backfill_odds.py --seasons 2015-2023 --delay 1.0  # slower
```

Resumes automatically — skips games that already have odds.

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

## Model

- **Architecture**: 3 MLP regressors — margin (64-32), total (64-32), and ATS/cover (256-128)
- **Features**: 7 KenPom stat diffs + neutral_site + 2 point-in-time features (consensus rank, adj efficiency margin) + same_conference + avg_tempo
- **Win probability**: Derived from predicted margin via logistic function
- **Vegas blending**: Model predictions blended 50/50 with live ESPN odds when available
- **Vegas odds persistence**: ESPN pickcenter odds (spread, total, moneyline) stored in DB during game fetch
- **Margin calibration**: Cross-validation calibration scale (~1.4x) corrects for margin compression when comparing to Vegas lines
- **ATS model**: Dedicated model predicts cover margin using PIT features + Vegas spread; picks flagged at |cover| >= 2 (lean) or >= 3 (pick)

### Performance (2026 season)

| Metric | Value |
|--------|-------|
| Win accuracy | 81.3% |
| Margin MAE | 7.52 pts |
| Total MAE | 13.78 pts |
| Training time | ~10 sec |

## Data Sources

- **Kaggle**: March Machine Learning Mania dataset (1985-2026, ~200k games)
- **KenPom**: Advanced metrics via REST API (2010-2026). Requires `KENPOM_API_TOKEN`.
- **ESPN**: Free scoreboard API for current-season games and pickcenter odds (no auth needed)
- **Massey Ratings**: Composite rankings from CSV export

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
│   ├── model.py              # MLP margin + total regressors
│   ├── odds.py               # ESPN pickcenter odds fetcher
│   └── scrapers/
│       └── ncaa_basketball.py  # Kaggle, KenPom, ESPN importers
├── scripts/
│   ├── import_kaggle.py      # Bootstrap with historical data
│   ├── update_data.py        # Nightly update script
│   ├── fetch_games.py        # Fetch ESPN games by date/range
│   ├── predict.py            # CLI for predictions (model + Vegas blending)
│   ├── predict_slate.py      # Batch predictions for a day's games
│   ├── backfill_odds.py      # Historical Vegas odds scraper
│   ├── fetch_bracket.py       # Fetch tournament bracket from ESPN
│   ├── simulate_tournament.py # Monte Carlo bracket simulator + pick-bracket
│   ├── migrate_team_aliases.py  # One-time team deduplication
│   └── migrate_dates.py       # One-time ordinal-to-ISO date conversion
├── tests/                    # Unit tests
├── data/                     # SQLite DBs, models, slates, bracket JSON (gitignored)
├── bets.md                   # ATS bet tracking
├── bracket_template.json     # Tournament bracket template
└── requirements.txt
```

---

Built with [Claude Code](https://claude.com/claude-code)

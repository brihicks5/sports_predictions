# Claude Code Instructions

## Project
NCAA basketball (and eventually other sports) game outcome prediction using historical data and ML.

## Setup
- Python 3.13 (Homebrew) with venv in `venv/`
- API tokens in `.env` — see `.env.example` for required vars
- `data/` is gitignored — contains SQLite DBs, model pickles, and Kaggle CSVs

## Common Commands
- **Always activate the venv first**: `source venv/bin/activate`
- `python scripts/predict.py "Team A" "Team B" --season 2026` — predict a game
- `python scripts/predict_slate.py --date 2026-03-14` — predict a full day's slate
- `python scripts/predict_slate.py --date 2026-03-14 --results-only` — fill in scores without re-predicting
- `python scripts/update_data.py --season 2026` — refresh ESPN games, KenPom, and retrain
- `python scripts/import_kaggle.py` — one-time bulk import from Kaggle
- `python scripts/fetch_bracket.py` — fetch tournament bracket from ESPN
- `python scripts/fetch_bracket.py --update` — update bracket with completed game results
- `python scripts/simulate_tournament.py --pick-bracket` — print game-by-game picks for filling in a bracket
- `python scripts/simulate_tournament.py -n 10000` — Monte Carlo tournament simulation

## Code Conventions
- Use `python -u` for scripts that print progress (avoids output buffering)
- Source `.env` before running scripts that need API tokens: `set -a && source .env && set +a`
- KenPom scraper uses the API (not web scraping) — see `kenpom-api.md` in memory for reference
- Run tests with `python -m pytest tests/ -v`

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
- `python scripts/update_data.py --season 2026` — refresh KenPom + retrain
- `python scripts/import_kaggle.py` — one-time bulk import from Kaggle

## Code Conventions
- Use `python -u` for scripts that print progress (avoids output buffering)
- Source `.env` before running scripts that need API tokens: `set -a && source .env && set +a`
- KenPom scraper uses the API (not web scraping) — see `kenpom-api.md` in memory for reference
- Run tests with `python -m pytest tests/ -v`

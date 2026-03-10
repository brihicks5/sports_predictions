"""Tests for NCAA basketball scrapers."""

import pytest

from sports_predictions.scrapers.ncaa_basketball import _date_to_season, kaggle_day_to_date


class TestDateToSeason:
    def test_january_same_year(self):
        assert _date_to_season("2026-01-15") == 2026

    def test_march_same_year(self):
        assert _date_to_season("2026-03-08") == 2026

    def test_november_next_year(self):
        assert _date_to_season("2025-11-15") == 2026

    def test_december_next_year(self):
        assert _date_to_season("2025-12-01") == 2026

    def test_october_same_year(self):
        assert _date_to_season("2025-10-31") == 2025

    def test_april_same_year(self):
        assert _date_to_season("2026-04-07") == 2026


class TestKaggleDayToDate:
    def test_season_2026_day_0(self):
        # 2026 season day 0 = Mon Nov 3, 2025
        assert kaggle_day_to_date(2026, 0) == "2025-11-03"

    def test_season_2025_day_0(self):
        # 2025 season day 0 = Mon Nov 4, 2024
        assert kaggle_day_to_date(2025, 0) == "2024-11-04"

    def test_season_2026_day_118(self):
        # Day 118 = Nov 3 + 118 = Mar 1, 2026
        assert kaggle_day_to_date(2026, 118) == "2026-03-01"

    def test_season_2024_day_0(self):
        # 2024 season day 0 = Mon Nov 6, 2023
        assert kaggle_day_to_date(2024, 0) == "2023-11-06"

    def test_season_2023_day_0(self):
        # 2023 season day 0 = Mon Nov 7, 2022
        assert kaggle_day_to_date(2023, 0) == "2022-11-07"

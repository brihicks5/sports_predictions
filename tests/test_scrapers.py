"""Tests for NCAA basketball scrapers."""

import pytest

from sports_predictions.scrapers.ncaa_basketball import _date_to_season


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

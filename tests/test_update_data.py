"""Tests for the update_data script's training and skip-flag behavior."""

import sys
from pathlib import Path
from unittest.mock import patch, call

import pytest

# Add project root so we can import the script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_main(*cli_args):
    """Run update_data.main() with the given CLI args."""
    with patch("sys.argv", ["update_data.py", *cli_args]):
        from scripts import update_data
        update_data.main()


# ---------------------------------------------------------------------------
# Skip-training logic
# ---------------------------------------------------------------------------

class TestSkipTraining:
    """Test that model training is skipped/triggered based on data changes."""

    @patch("scripts.update_data.train_model")
    @patch("scripts.update_data.compute_season_stats")
    @patch("scripts.update_data.fetch_kenpom_four_factors", return_value=0)
    @patch("scripts.update_data.fetch_kenpom_ratings", return_value=0)
    @patch("scripts.update_data.fetch_espn_games", return_value=0)
    def test_no_changes_still_trains(self, mock_espn, mock_ratings,
                                      mock_ff, mock_stats, mock_train):
        """Training always runs even when no data changed."""
        run_main("--season", "2026")
        mock_train.assert_called_once_with("ncaa_basketball")

    @patch("scripts.update_data.train_model")
    @patch("scripts.update_data.compute_season_stats")
    @patch("scripts.update_data.fetch_kenpom_four_factors", return_value=0)
    @patch("scripts.update_data.fetch_kenpom_ratings", return_value=0)
    @patch("scripts.update_data.fetch_espn_games", return_value=5)
    def test_espn_changes_triggers_training(self, mock_espn, mock_ratings,
                                             mock_ff, mock_stats, mock_train):
        """When ESPN imports games, training should run."""
        run_main("--season", "2026")
        mock_train.assert_called_once_with("ncaa_basketball")

    @patch("scripts.update_data.train_model")
    @patch("scripts.update_data.compute_season_stats")
    @patch("scripts.update_data.fetch_kenpom_four_factors", return_value=0)
    @patch("scripts.update_data.fetch_kenpom_ratings", return_value=10)
    @patch("scripts.update_data.fetch_espn_games", return_value=0)
    def test_kenpom_ratings_changes_triggers_training(self, mock_espn,
                                                       mock_ratings, mock_ff,
                                                       mock_stats, mock_train):
        """When KenPom ratings change, training should run."""
        run_main("--season", "2026")
        mock_train.assert_called_once_with("ncaa_basketball")

    @patch("scripts.update_data.train_model")
    @patch("scripts.update_data.compute_season_stats")
    @patch("scripts.update_data.fetch_kenpom_four_factors", return_value=5)
    @patch("scripts.update_data.fetch_kenpom_ratings", return_value=0)
    @patch("scripts.update_data.fetch_espn_games", return_value=0)
    def test_kenpom_four_factors_changes_triggers_training(self, mock_espn,
                                                            mock_ratings,
                                                            mock_ff,
                                                            mock_stats,
                                                            mock_train):
        """When KenPom four-factors change, training should run."""
        run_main("--season", "2026")
        mock_train.assert_called_once_with("ncaa_basketball")

    @patch("scripts.update_data.train_model")
    @patch("scripts.update_data.compute_season_stats")
    @patch("scripts.update_data.fetch_kenpom_four_factors", return_value=3)
    @patch("scripts.update_data.fetch_kenpom_ratings", return_value=7)
    @patch("scripts.update_data.fetch_espn_games", return_value=10)
    def test_all_sources_changed_trains_once(self, mock_espn, mock_ratings,
                                              mock_ff, mock_stats, mock_train):
        """When all sources have changes, training should still run exactly once."""
        run_main("--season", "2026")
        mock_train.assert_called_once_with("ncaa_basketball")


# ---------------------------------------------------------------------------
# --skip-train flag
# ---------------------------------------------------------------------------

class TestSkipTrainFlag:
    @patch("scripts.update_data.train_model")
    @patch("scripts.update_data.compute_season_stats")
    @patch("scripts.update_data.fetch_kenpom_four_factors", return_value=10)
    @patch("scripts.update_data.fetch_kenpom_ratings", return_value=10)
    @patch("scripts.update_data.fetch_espn_games", return_value=10)
    def test_skip_train_flag_prevents_training(self, mock_espn, mock_ratings,
                                                mock_ff, mock_stats,
                                                mock_train):
        """--skip-train should prevent training even when data changed."""
        run_main("--season", "2026", "--skip-train")
        mock_train.assert_not_called()


# ---------------------------------------------------------------------------
# --skip-espn / --skip-kenpom flags
# ---------------------------------------------------------------------------

class TestSkipSourceFlags:
    @patch("scripts.update_data.train_model")
    @patch("scripts.update_data.compute_season_stats")
    @patch("scripts.update_data.fetch_kenpom_four_factors", return_value=0)
    @patch("scripts.update_data.fetch_kenpom_ratings", return_value=0)
    @patch("scripts.update_data.fetch_espn_games", return_value=0)
    def test_skip_espn_does_not_call_espn(self, mock_espn, mock_ratings,
                                           mock_ff, mock_stats, mock_train):
        """--skip-espn should skip ESPN fetch entirely."""
        run_main("--season", "2026", "--skip-espn")
        mock_espn.assert_not_called()

    @patch("scripts.update_data.train_model")
    @patch("scripts.update_data.compute_season_stats")
    @patch("scripts.update_data.fetch_kenpom_four_factors", return_value=0)
    @patch("scripts.update_data.fetch_kenpom_ratings", return_value=0)
    @patch("scripts.update_data.fetch_espn_games", return_value=0)
    def test_skip_kenpom_does_not_call_kenpom(self, mock_espn, mock_ratings,
                                               mock_ff, mock_stats,
                                               mock_train):
        """--skip-kenpom should skip KenPom fetch entirely."""
        run_main("--season", "2026", "--skip-kenpom")
        mock_ratings.assert_not_called()
        mock_ff.assert_not_called()

    @patch("scripts.update_data.train_model")
    @patch("scripts.update_data.compute_season_stats")
    @patch("scripts.update_data.fetch_kenpom_four_factors", return_value=0)
    @patch("scripts.update_data.fetch_kenpom_ratings", return_value=0)
    @patch("scripts.update_data.fetch_espn_games", return_value=0)
    def test_skip_espn_still_computes_stats(self, mock_espn, mock_ratings,
                                             mock_ff, mock_stats, mock_train):
        """--skip-espn should still compute season stats (for manual imports)."""
        run_main("--season", "2026", "--skip-espn")
        mock_stats.assert_called_once_with(2026)

    @patch("scripts.update_data.train_model")
    @patch("scripts.update_data.compute_season_stats")
    @patch("scripts.update_data.fetch_kenpom_four_factors", return_value=0)
    @patch("scripts.update_data.fetch_kenpom_ratings", return_value=0)
    @patch("scripts.update_data.fetch_espn_games", return_value=0)
    def test_no_espn_changes_still_computes_stats(self, mock_espn, mock_ratings,
                                                     mock_ff, mock_stats, mock_train):
        """Stats always recompute even when ESPN returns 0 changes."""
        run_main("--season", "2026")
        mock_stats.assert_called_once_with(2026)

    @patch("scripts.update_data.train_model")
    @patch("scripts.update_data.compute_season_stats")
    @patch("scripts.update_data.fetch_kenpom_four_factors", return_value=5)
    @patch("scripts.update_data.fetch_kenpom_ratings", return_value=0)
    @patch("scripts.update_data.fetch_espn_games", return_value=0)
    def test_skip_espn_with_kenpom_changes_triggers_training(
            self, mock_espn, mock_ratings, mock_ff, mock_stats, mock_train):
        """--skip-espn + KenPom changes should still trigger training."""
        run_main("--season", "2026", "--skip-espn")
        mock_train.assert_called_once_with("ncaa_basketball")

    @patch("scripts.update_data.train_model")
    @patch("scripts.update_data.compute_season_stats")
    @patch("scripts.update_data.fetch_kenpom_four_factors", return_value=0)
    @patch("scripts.update_data.fetch_kenpom_ratings", return_value=0)
    @patch("scripts.update_data.fetch_espn_games", return_value=0)
    def test_both_skipped_still_trains(self, mock_espn, mock_ratings,
                                        mock_ff, mock_stats, mock_train):
        """--skip-espn + --skip-kenpom still trains (use --skip-train to prevent)."""
        run_main("--season", "2026", "--skip-espn", "--skip-kenpom")
        mock_train.assert_called_once_with("ncaa_basketball")

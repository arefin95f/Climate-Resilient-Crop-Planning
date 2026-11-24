import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.models.train_crops import (
    _evaluate_observed_bbs,
    _training_frame,
)


class ObservedYieldValidationTests(unittest.TestCase):
    def test_training_keeps_every_stored_year(self):
        frame = pd.DataFrame(
            {
                "year": [2004, 2023, 2024, 2026],
                "yield_ton_ha": [4.1, 4.2, 4.3, 4.4],
            }
        )

        kept = _training_frame(frame, "bbs")

        self.assertEqual(kept["year"].tolist(), [2004, 2023, 2024, 2026])

    def test_observed_backtest_separates_error_band_evaluation_years(self):
        years = np.repeat(np.arange(2015, 2024), 2)
        features = pd.DataFrame({"feature": np.arange(len(years))})
        target = years.astype(float) + np.tile([0.0, 0.4], 9)

        metrics = _evaluate_observed_bbs(
            DummyRegressor,
            features,
            target,
            years,
        )
        band = metrics["empirical_error_band"]

        self.assertEqual(metrics["test_years"], [2019, 2020, 2021, 2022, 2023])
        self.assertEqual(metrics["n_test"], 10)
        self.assertEqual(band["calibration_n"], 6)
        self.assertEqual(band["evaluation_years"], [2022, 2023])
        self.assertEqual(band["evaluation_n"], 4)
        self.assertGreater(band["half_width_t_ha"], 0)
        self.assertGreaterEqual(band["evaluation_coverage"], 0)
        self.assertLessEqual(band["evaluation_coverage"], 1)


if __name__ == "__main__":
    unittest.main()

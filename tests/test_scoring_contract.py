from __future__ import annotations

import unittest

from bgc2activity.scoring_contract import (
    BASE_SCORE_COLUMN,
    ScoreInputError,
    UPDATE_FEATURE_COLUMNS,
    require_finite_after_fold_preprocessing,
    select_update_inputs,
    validate_update_feature_selection,
)


class ScoringContractTests(unittest.TestCase):
    def test_accepts_exact_final_update_inputs(self):
        expected = (BASE_SCORE_COLUMN, *UPDATE_FEATURE_COLUMNS)
        self.assertEqual(validate_update_feature_selection(expected), expected)

    def test_rejects_bgc_metadata_as_an_activity_input(self):
        with self.assertRaisesRegex(ScoreInputError, "BGC metadata"):
            validate_update_feature_selection((BASE_SCORE_COLUMN, *UPDATE_FEATURE_COLUMNS, "bgc_score"))

    def test_extracts_only_sbase_and_nine_features(self):
        record = {"candidate_library_id": "BNP_000001", BASE_SCORE_COLUMN: 0.4}
        record.update({name: index + 1 for index, name in enumerate(UPDATE_FEATURE_COLUMNS)})
        record["reconstruction_confidence"] = 0.99
        score, features = select_update_inputs(record)
        self.assertEqual(score, 0.4)
        self.assertEqual(features, tuple(range(1, 10)))

    def test_preserves_missing_values_for_fold_specific_preprocessing(self):
        record = {"candidate_library_id": "BNP_000001", BASE_SCORE_COLUMN: 0.4}
        record.update({name: 1.0 for name in UPDATE_FEATURE_COLUMNS})
        record["GRAVY"] = "NA"
        _, features = select_update_inputs(record)
        with self.assertRaisesRegex(ScoreInputError, "after fold-specific preprocessing"):
            require_finite_after_fold_preprocessing(features)

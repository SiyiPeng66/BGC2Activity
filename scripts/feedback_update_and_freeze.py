#!/usr/bin/env python3
"""Round-1 feedback regression update, monotone regression-scale calibration
and freeze-package manifest.

Implements the final main-text formulation (sections 2.3, 3.3; Fig. 3a) plus
the Method-17 freeze manifest as one auditable command:

1. Load the frozen Round-1 table (candidate_library_id, sbase = predicted
   log10(MIC), nine pre-specified features, observed_log10_mic) and validate
   the input boundary (BGC metadata is rejected as an activity input).
2. Five-fold stratified cross-fitting produces one out-of-fold Supd per
   candidate and one development out-of-fold calibrated score per candidate;
   diagnostics are written alongside.
3. The final residual model is fitted on all 40 candidates; the final
   monotone (isotonic) regression-scale mapping is fitted on the 40
   out-of-fold Supd values and their observed targets.
4. If --freeze-artifacts supplies the ten Method-17 role files, a
   content-addressed freeze manifest is written via bgc2activity.release.

Input TSV columns: candidate_library_id, sbase, observed_log10_mic plus the
nine feature columns fixed by the scoring contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from bgc2activity.activity.evaluation import mean_absolute_error, root_mean_squared_error, spearman
from bgc2activity.calibration import (
    cross_fit_out_of_fold_calibrated,
    fit_monotone_regression_calibration,
    fit_residual_update,
)
from bgc2activity.release.freeze import REQUIRED_ROLES, FreezeManifestError, write_freeze_manifest
from bgc2activity.scoring_contract import (
    BASE_SCORE_COLUMN,
    FORBIDDEN_ACTIVITY_INPUTS,
    REQUIRED_COLUMNS,
    UPDATE_FEATURE_COLUMNS,
    select_update_inputs,
    validate_update_feature_selection,
)

TARGET_COLUMN = "observed_log10_mic"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round1-table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--freeze-artifacts", nargs="*", default=[],
                        help="role=path pairs for the ten Method-17 freeze roles")
    parser.add_argument("--code-commit", default=None, help="code commit recorded in the freeze manifest")
    parser.add_argument("--study-version", default=None, help="study version recorded in the freeze manifest")
    args = parser.parse_args()

    table = pd.read_csv(args.round1_table, sep="\t")
    columns = set(table.columns)
    forbidden = columns & set(FORBIDDEN_ACTIVITY_INPUTS)
    if forbidden:
        raise SystemExit(f"BGC metadata cannot be an activity input: {sorted(forbidden)}")
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise SystemExit(f"missing required columns: {sorted(missing)}")
    if TARGET_COLUMN not in columns:
        raise SystemExit(f"missing required column: {TARGET_COLUMN}")
    # The update may only ever consume the frozen Sbase-plus-nine-feature selection.
    validate_update_feature_selection((BASE_SCORE_COLUMN, *UPDATE_FEATURE_COLUMNS))
    if len(table) != 40:
        raise SystemExit(f"Round-1 cohort must contain 40 candidates, got {len(table)}")

    sbase_list: list[float] = []
    feature_rows: list[tuple[float, ...]] = []
    for record in table.to_dict(orient="records"):
        score, features_row = select_update_inputs(record)
        sbase_list.append(score)
        feature_rows.append(features_row)
    sbase = np.asarray(sbase_list, dtype=float)
    features = np.asarray(feature_rows, dtype=float)
    if not np.isfinite(features).all():
        raise SystemExit("missing feature values must be imputed inside training folds; "
                         "provide fold-specific imputation via the training entry point")
    targets = table[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(targets).all():
        raise SystemExit(f"{TARGET_COLUMN} must be finite for all Round-1 candidates")

    supd_oof, scal_dev_oof = cross_fit_out_of_fold_calibrated(sbase, features, targets)
    final_model = fit_residual_update(sbase, features, targets)
    supd_final = final_model.predict_supd(sbase, features)
    final_calibration = fit_monotone_regression_calibration(supd_oof, targets)
    scal_oof = final_calibration.predict_calibrated(supd_oof, training_supd=supd_oof)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame({
        "candidate_library_id": table["candidate_library_id"],
        "sbase": sbase,
        "supd_oof": supd_oof,
        "supd_final": supd_final,
        "scal_dev_oof": scal_dev_oof,
        "scal_oof": scal_oof,
        TARGET_COLUMN: targets,
    })
    scores.to_csv(output_dir / "round1_scores.tsv", sep="\t", index=False)

    diagnostics = {
        "n_candidates": int(len(table)),
        "target": "log10_mic",
        "regularization_c": final_model.regularization_c,
        "coefficients": final_model.coefficients.tolist(),
        "oof_calibrated_mae": float(mean_absolute_error(targets, scal_oof)),
        "oof_calibrated_rmse": float(root_mean_squared_error(targets, scal_oof)),
        "oof_calibrated_spearman": float(spearman(targets, scal_oof)),
        "dev_calibrated_rmse": float(root_mean_squared_error(targets, scal_dev_oof)),
    }
    (output_dir / "update_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"wrote scores and diagnostics to {output_dir}")

    if args.freeze_artifacts:
        if not args.code_commit or not args.study_version:
            raise SystemExit("--code-commit and --study-version are required with --freeze-artifacts")
        artifacts = {}
        for pair in args.freeze_artifacts:
            role, _, path = pair.partition("=")
            if role not in REQUIRED_ROLES:
                raise SystemExit(f"unknown freeze role: {role}")
            artifacts[role] = Path(path)
        missing_roles = REQUIRED_ROLES - set(artifacts)
        if missing_roles:
            raise SystemExit(f"missing freeze roles: {sorted(missing_roles)}")
        try:
            manifest_path, metadata_path = write_freeze_manifest(
                artifacts, output_dir / "freeze_package",
                code_commit=args.code_commit, study_version=args.study_version,
            )
        except FreezeManifestError as error:
            raise SystemExit(str(error))
        print(f"wrote freeze manifest: {manifest_path}")
        print(f"wrote freeze metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

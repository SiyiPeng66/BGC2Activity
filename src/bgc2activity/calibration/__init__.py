"""Round-1 feedback regression updating and monotone regression-scale
calibration (main text sections 2.3 and 3.3)."""

from bgc2activity.calibration.feedback_update import (
    C_GRID,
    MAX_EFFECTIVE_DEGREES_OF_FREEDOM,
    N_FOLDS,
    MonotoneRegressionCalibration,
    ResidualUpdateModel,
    cross_fit_out_of_fold,
    cross_fit_out_of_fold_calibrated,
    fit_monotone_regression_calibration,
    fit_residual_update,
    quantile_strata,
    select_regularization,
    stratified_fold_indices,
)

__all__ = [
    "C_GRID",
    "MAX_EFFECTIVE_DEGREES_OF_FREEDOM",
    "N_FOLDS",
    "MonotoneRegressionCalibration",
    "ResidualUpdateModel",
    "cross_fit_out_of_fold",
    "cross_fit_out_of_fold_calibrated",
    "fit_monotone_regression_calibration",
    "fit_residual_update",
    "quantile_strata",
    "select_regularization",
    "stratified_fold_indices",
]

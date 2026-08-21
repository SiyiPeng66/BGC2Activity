"""Round-1 feedback regression update and monotone regression-scale calibration.

Implements the final main-text formulation (sections 2.3 and 3.3, Fig. 3a):
measured Round-1 log10(MIC) values are used only for a low-capacity residual
regression update followed by monotone regression-scale calibration under
five-fold cross-fitting, reducing optimistic bias from the small update set.
Prespecified MIC thresholds define experimental endpoints only and never
enter the update target.

Residual regression update. The updated score is a residual correction on
the log10(MIC) scale,

    Supd_i = Sbase_i + beta^T x_i ,

with the coefficient of Sbase fixed at 1. beta is an L2-regularised linear
(ridge) coefficient vector over exactly nine pre-specified features,
selected from C in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1} with effective
degrees of freedom constrained to <= 4, by lowest inner-CV RMSE (RD-10).
Missing-value handling and standardisation are strictly nested within
training folds. Five folds, stratified on quartile bins of the observed
target (RD-11), cross-fit the 40-candidate Round-1 cohort; development
evaluation uses only the 40 out-of-fold scores; the final model is then
fitted on all 40 candidates and frozen for Round 2.

Monotone regression-scale calibration. Isotonic (pool-adjacent-violators)
regression maps Supd onto the calibrated predicted log10(MIC) scale,
monotone non-decreasing by construction. Inputs are truncated to the
0.5-99.5 percentiles of the fitting distribution. During development, the
calibrated value for each outer held-out candidate comes from the isotonic
mapping fitted on the out-of-fold Supd of the 32 training records within the
same fold (RD-12, the regression analogue of the former RD-1); the final
mapping is fitted on the 40 outer out-of-fold Supd values and their observed
log10(MIC) targets.

Sbase, Supd and the calibrated score (Scal) are stored independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
MAX_EFFECTIVE_DEGREES_OF_FREEDOM = 4.0
N_FOLDS = 5
N_STRATA_BINS = 4  # quartile bins of the observed target (RD-11)
INPUT_PERCENTILE_TRUNCATION = (0.5, 99.5)


def quantile_strata(targets: np.ndarray, n_bins: int = N_STRATA_BINS) -> np.ndarray:
    """Stratum ids from quantile bins of the observed log10(MIC) target (RD-11)."""
    targets = np.asarray(targets, dtype=float)
    quantiles = np.quantile(targets, np.linspace(0.0, 1.0, n_bins + 1))
    strata = np.digitize(targets, quantiles[1:-1], right=False)
    return strata


def stratified_fold_indices(
    targets: np.ndarray, n_folds: int = N_FOLDS, seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Single source of truth for the stratified five-fold split.

    Both the Supd cross-fitting and the per-fold development calibration
    fits must use identical folds; every cross-fitting entry point therefore
    consumes this generator.
    """
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    strata = quantile_strata(targets)
    features_placeholder = np.zeros((len(targets), 1))
    return [
        (training_index, validation_index)
        for training_index, validation_index in splitter.split(features_placeholder, strata)
    ]


@dataclass(frozen=True)
class ResidualUpdateModel:
    coefficients: np.ndarray  # beta over the nine features
    feature_means: np.ndarray
    feature_standard_deviations: np.ndarray
    regularization_c: float

    def predict_supd(self, sbase: np.ndarray, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.feature_means) / self.feature_standard_deviations
        return np.asarray(sbase, dtype=float) + standardized @ self.coefficients


def _ridge_coefficients(x: np.ndarray, residual: np.ndarray, c: float) -> np.ndarray:
    """Solve (X^T X / n + I / c) beta = X^T r / n for the residual r = y - Sbase.

    The offset carries the frozen Sbase prediction, so the coefficient of
    Sbase is exactly 1 by construction.
    """
    n = len(residual)
    gram = x.T @ x / n + np.eye(x.shape[1]) / c
    rhs = x.T @ residual / n
    return np.linalg.solve(gram, rhs)


def _effective_dof(x: np.ndarray, c: float) -> float:
    """Effective degrees of freedom trace(S) of the ridge smoother S = X (X^T X / n + I / c)^-1 X^T / n."""
    n = len(x)
    gram = x.T @ x / n + np.eye(x.shape[1]) / c
    smoother = x @ np.linalg.solve(gram, x.T) / n
    return float(np.trace(smoother))


def select_regularization(
    sbase: np.ndarray, features: np.ndarray, targets: np.ndarray, seed: int = 42,
) -> float:
    """Choose C by lowest inner-CV RMSE under the effective-dof cap (RD-10).

    The outer stratified folds of the 40-candidate cohort provide the
    validation sets; within each outer training fold an inner stratified
    split selects C. Preprocessing is nested inside every training fold.
    """
    best_c = None
    best_loss = float("inf")
    for c in C_GRID:
        losses = []
        for training_index, validation_index in stratified_fold_indices(targets, seed=seed):
            x_train_raw, y_train, s_train = features[training_index], targets[training_index], sbase[training_index]
            mean = x_train_raw.mean(axis=0)
            std = x_train_raw.std(axis=0)
            std[std == 0] = 1.0
            x_train = (x_train_raw - mean) / std
            fitted = _ridge_coefficients(x_train, y_train - s_train, c)
            if _effective_dof(x_train, c) > MAX_EFFECTIVE_DEGREES_OF_FREEDOM:
                continue
            x_val = (features[validation_index] - mean) / std
            predicted = sbase[validation_index] + x_val @ fitted
            losses.append(float(np.mean((predicted - targets[validation_index]) ** 2)) ** 0.5)
        if not losses:
            continue
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_c = c
    if best_c is None:
        raise RuntimeError("no regularization strength satisfied the degrees-of-freedom cap")
    return best_c


def fit_residual_update(
    sbase: np.ndarray, features: np.ndarray, targets: np.ndarray, seed: int = 42,
) -> ResidualUpdateModel:
    """Final residual regression update fitted on all Round-1 candidates."""
    c = select_regularization(sbase, features, targets, seed=seed)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    x = (features - mean) / std
    coefficients = _ridge_coefficients(x, np.asarray(targets, dtype=float) - np.asarray(sbase, dtype=float), c)
    return ResidualUpdateModel(
        coefficients=coefficients,
        feature_means=mean,
        feature_standard_deviations=std,
        regularization_c=c,
    )


def cross_fit_out_of_fold(
    sbase: np.ndarray, features: np.ndarray, targets: np.ndarray,
    n_folds: int = N_FOLDS, seed: int = 42,
) -> np.ndarray:
    """One out-of-fold Supd per candidate under stratified five-fold cross-fitting."""
    supd_oof = np.empty(len(targets), dtype=float)
    for training_index, validation_index in stratified_fold_indices(targets, n_folds, seed):
        model = fit_residual_update(
            sbase[training_index], features[training_index], targets[training_index], seed=seed,
        )
        supd_oof[validation_index] = model.predict_supd(sbase[validation_index], features[validation_index])
    return supd_oof


def _pav_monotone_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators on x-sorted data; returns (x anchors, fitted
    monotone non-decreasing values)."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = np.asarray(x)[order], np.asarray(y)[order]
    blocks: list[list[float]] = []  # [x_sum, y_sum, count]
    for xi, yi in zip(xs, ys):
        blocks.append([float(xi), float(yi), 1.0])
        while len(blocks) > 1 and blocks[-1][1] / blocks[-1][2] < blocks[-2][1] / blocks[-2][2]:
            last = blocks.pop()
            blocks[-1][0] += last[0]
            blocks[-1][1] += last[1]
            blocks[-1][2] += last[2]
    anchors = np.array([block[0] / block[2] for block in blocks])
    values = np.array([block[1] / block[2] for block in blocks])
    return anchors, values


@dataclass(frozen=True)
class MonotoneRegressionCalibration:
    anchors: np.ndarray  # sorted Supd anchors (after percentile truncation)
    values: np.ndarray  # fitted calibrated log10(MIC) at the anchors
    input_low: float
    input_high: float

    def predict_calibrated(self, supd: np.ndarray, training_supd: np.ndarray | None = None) -> np.ndarray:
        values = np.asarray(supd, dtype=float)
        if training_supd is not None and len(training_supd) > 1:
            low, high = np.percentile(training_supd, INPUT_PERCENTILE_TRUNCATION)
            values = np.clip(values, low, high)
        else:
            values = np.clip(values, self.input_low, self.input_high)
        return np.interp(values, self.anchors, self.values)


def fit_monotone_regression_calibration(supd: np.ndarray, targets: np.ndarray) -> MonotoneRegressionCalibration:
    """Monotone (isotonic) regression-scale calibration of Supd onto the
    observed log10(MIC) scale; inputs truncated to the 0.5-99.5 percentiles."""
    low, high = np.percentile(supd, INPUT_PERCENTILE_TRUNCATION)
    truncated = np.clip(np.asarray(supd, dtype=float), low, high)
    anchors, values = _pav_monotone_fit(truncated, np.asarray(targets, dtype=float))
    return MonotoneRegressionCalibration(anchors, values, float(low), float(high))


def cross_fit_out_of_fold_calibrated(
    sbase: np.ndarray, features: np.ndarray, targets: np.ndarray,
    n_folds: int = N_FOLDS, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Development-stage out-of-fold Supd and calibrated scores (RD-12).

    The five stratified folds first produce one out-of-fold Supd per
    candidate. Then, for each fold, the monotone regression-scale mapping is
    fitted on the out-of-fold Supd and observed targets of the 32 training
    records of that same fold only, and the held-out candidates receive their
    development calibrated score from that fold-specific mapping. Returns
    ``(supd_oof, scal_dev_oof)``.
    """
    supd_oof = cross_fit_out_of_fold(sbase, features, targets, n_folds=n_folds, seed=seed)
    scal_dev_oof = np.empty(len(targets), dtype=float)
    for training_index, validation_index in stratified_fold_indices(targets, n_folds, seed):
        fold_calibration = fit_monotone_regression_calibration(
            supd_oof[training_index], targets[training_index],
        )
        scal_dev_oof[validation_index] = fold_calibration.predict_calibrated(
            supd_oof[validation_index], training_supd=supd_oof[training_index],
        )
    return supd_oof, scal_dev_oof

"""Base MIC Regression Model: quantitative MIC curation, regression model,
clustering, training protocol and regression evaluation.

Main-text section 2.3 defines the activity model as a quantitative
log10(MIC) regression model; the freeze boundary, chemical-feature fusion
and cluster-split details are taken from Supplementary Methods 11-12.
Torch-dependent members are exported lazily so the dependency-free curation,
clustering and evaluation members import without torch.
"""

_LAZY_EXPORTS = {
    "BaseActivityModel": "bgc2activity.activity.base_model",
    "ChemistryEncoder": "bgc2activity.activity.base_model",
    "encode_sequences_to_ensemble_sbase": "bgc2activity.activity.base_model",
    "MicMeasurement": "bgc2activity.activity.mic_curation",
    "QuantitativeTarget": "bgc2activity.activity.mic_curation",
    "log10_mic_target": "bgc2activity.activity.mic_curation",
    "aggregate_sequence_targets": "bgc2activity.activity.mic_curation",
    "strict_label_for_measurement": "bgc2activity.activity.mic_curation",
    "aggregate_sequence_labels": "bgc2activity.activity.mic_curation",
    "replicate_consistency_groups": "bgc2activity.activity.mic_curation",
    "assign_structural_clusters": "bgc2activity.activity.clustering",
    "effective_training_mask": "bgc2activity.activity.clustering",
    "global_sequence_similarity": "bgc2activity.activity.clustering",
    "sequences_are_clustered": "bgc2activity.activity.clustering",
    "split_clusters": "bgc2activity.activity.clustering",
    "tanimoto_similarity": "bgc2activity.activity.clustering",
    "build_parameter_groups": "bgc2activity.activity.training",
    "ensemble_sbase": "bgc2activity.activity.training",
    "huber_loss": "bgc2activity.activity.training",
    "select_best_epoch": "bgc2activity.activity.training",
    "train_one_seed": "bgc2activity.activity.training",
    "mean_absolute_error": "bgc2activity.activity.evaluation",
    "root_mean_squared_error": "bgc2activity.activity.evaluation",
    "spearman": "bgc2activity.activity.evaluation",
    "observed_predicted_agreement": "bgc2activity.activity.evaluation",
    "residual_structure": "bgc2activity.activity.evaluation",
    "regression_report": "bgc2activity.activity.evaluation",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value

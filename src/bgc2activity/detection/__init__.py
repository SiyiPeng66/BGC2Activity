"""BGC detection module reconstructed from Supplementary Methods 3-7.

Torch-dependent members (model, losses, dataset) are exported lazily so that
the dependency-free contract members (families, regions constants, annotation
records, evaluation) can be imported without torch.
"""

_LAZY_EXPORTS = {
    "MultitaskBGCDetector": "bgc2activity.detection.model",
    "DetectorOutput": "bgc2activity.detection.model",
    "load_balancing_loss": "bgc2activity.detection.model",
    "multitask_loss": "bgc2activity.detection.losses",
    "MultitaskLoss": "bgc2activity.detection.losses",
    "composite_bgc_distance": "bgc2activity.detection.families",
    "single_linkage_families": "bgc2activity.detection.families",
    "pairwise_distance_matrix": "bgc2activity.detection.families",
    "family_heldout_partition": "bgc2activity.detection.families",
    "random_partition": "bgc2activity.detection.families",
    "family_overlap_fraction": "bgc2activity.detection.families",
    "call_regions": "bgc2activity.detection.regions",
    "region_is_nrps": "bgc2activity.detection.regions",
    "aggregate_window_probabilities": "bgc2activity.detection.regions",
    "window_starts": "bgc2activity.detection.regions",
    "PfamSequenceDataset": "bgc2activity.detection.data",
    "RegionSample": "bgc2activity.detection.data",
    "collate_regions": "bgc2activity.detection.data",
    "load_pfam2vec": "bgc2activity.detection.data",
    "GeneCall": "bgc2activity.detection.annotation",
    "PfamHit": "bgc2activity.detection.annotation",
    "ordered_pfam_sequence": "bgc2activity.detection.annotation",
    "predict_genes": "bgc2activity.detection.annotation",
    "scan_pfam": "bgc2activity.detection.annotation",
    "BENCHMARK_GENE_THRESHOLD": "bgc2activity.detection.evaluation",
    "BenchmarkGene": "bgc2activity.detection.evaluation",
    "PredictedRegion": "bgc2activity.detection.evaluation",
    "RecoveryReport": "bgc2activity.detection.evaluation",
    "TruthRegion": "bgc2activity.detection.evaluation",
    "aggregate_position_scores_to_genes": "bgc2activity.detection.evaluation",
    "assign_gene_labels": "bgc2activity.detection.evaluation",
    "auroc": "bgc2activity.detection.evaluation",
    "auprc": "bgc2activity.detection.evaluation",
    "binary_metrics_at_threshold": "bgc2activity.detection.evaluation",
    "classify_recovery": "bgc2activity.detection.evaluation",
    "evaluate_gene_level": "bgc2activity.detection.evaluation",
    "merge_product_class": "bgc2activity.detection.evaluation",
    "minimum_training_distance": "bgc2activity.detection.evaluation",
    "project_regions_to_gene_scores": "bgc2activity.detection.evaluation",
    "region_level_metrics": "bgc2activity.detection.evaluation",
    "stratified_counts": "bgc2activity.detection.evaluation",
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

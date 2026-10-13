"""BGC detection module reconstructed from Supplementary Methods 3-7.

Torch-dependent members (model, losses, dataset) are exported lazily so that
the dependency-free contract members (families, regions constants, annotation
records, evaluation) can be imported without torch.
"""

_LAZY_EXPORTS = {
    "MultitaskBGCDetector": "bgc2activity_reconstructed.detection.model",
    "DetectorOutput": "bgc2activity_reconstructed.detection.model",
    "load_balancing_loss": "bgc2activity_reconstructed.detection.model",
    "multitask_loss": "bgc2activity_reconstructed.detection.losses",
    "MultitaskLoss": "bgc2activity_reconstructed.detection.losses",
    "composite_bgc_distance": "bgc2activity_reconstructed.detection.families",
    "single_linkage_families": "bgc2activity_reconstructed.detection.families",
    "pairwise_distance_matrix": "bgc2activity_reconstructed.detection.families",
    "family_heldout_partition": "bgc2activity_reconstructed.detection.families",
    "random_partition": "bgc2activity_reconstructed.detection.families",
    "family_overlap_fraction": "bgc2activity_reconstructed.detection.families",
    "call_regions": "bgc2activity_reconstructed.detection.regions",
    "region_is_nrps": "bgc2activity_reconstructed.detection.regions",
    "aggregate_window_probabilities": "bgc2activity_reconstructed.detection.regions",
    "window_starts": "bgc2activity_reconstructed.detection.regions",
    "PfamSequenceDataset": "bgc2activity_reconstructed.detection.data",
    "RegionSample": "bgc2activity_reconstructed.detection.data",
    "collate_regions": "bgc2activity_reconstructed.detection.data",
    "load_pfam2vec": "bgc2activity_reconstructed.detection.data",
    "GeneCall": "bgc2activity_reconstructed.detection.annotation",
    "PfamHit": "bgc2activity_reconstructed.detection.annotation",
    "ordered_pfam_sequence": "bgc2activity_reconstructed.detection.annotation",
    "predict_genes": "bgc2activity_reconstructed.detection.annotation",
    "scan_pfam": "bgc2activity_reconstructed.detection.annotation",
    "BENCHMARK_GENE_THRESHOLD": "bgc2activity_reconstructed.detection.evaluation",
    "BenchmarkGene": "bgc2activity_reconstructed.detection.evaluation",
    "PredictedRegion": "bgc2activity_reconstructed.detection.evaluation",
    "RecoveryReport": "bgc2activity_reconstructed.detection.evaluation",
    "TruthRegion": "bgc2activity_reconstructed.detection.evaluation",
    "aggregate_position_scores_to_genes": "bgc2activity_reconstructed.detection.evaluation",
    "assign_gene_labels": "bgc2activity_reconstructed.detection.evaluation",
    "auroc": "bgc2activity_reconstructed.detection.evaluation",
    "auprc": "bgc2activity_reconstructed.detection.evaluation",
    "binary_metrics_at_threshold": "bgc2activity_reconstructed.detection.evaluation",
    "classify_recovery": "bgc2activity_reconstructed.detection.evaluation",
    "evaluate_gene_level": "bgc2activity_reconstructed.detection.evaluation",
    "merge_product_class": "bgc2activity_reconstructed.detection.evaluation",
    "minimum_training_distance": "bgc2activity_reconstructed.detection.evaluation",
    "project_regions_to_gene_scores": "bgc2activity_reconstructed.detection.evaluation",
    "region_level_metrics": "bgc2activity_reconstructed.detection.evaluation",
    "stratified_counts": "bgc2activity_reconstructed.detection.evaluation",
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

"""Comparative evaluation on the nine-genome benchmark (Supplementary Method 7).

The proposed detector and all comparators (antiSMASH, ClusterFinder,
BGC-Prophet, DeepBGC) are evaluated on the same nine anonymous genomes
(genome_0..genome_8; the BGC-Prophet anonymous test set: 67,072 genes, of
which 3,696 are positive and 54,285 carry Pfam annotations), with gene-level
labels and unified evaluation rules:

* external region outputs are projected to gene level by coordinate overlap;
* external per-position scores are reduced to the maximum within a gene;
* genes without Pfam annotations receive score 0.0 for the proposed model;
* product classes are merged into NRPS, PKS, RiPP, terpene and other;
* binary gene-level metrics are computed at a uniform threshold of 0.5 on the
  continuous gene-level scores, while AUROC/AUPRC use the continuous scores;
  genome-scale region calling separately retains the 0.45 position threshold
  of Supplementary Method 6;
* a reference-backed true positive requires a predicted region to satisfy the
  prespecified coordinate-overlap rule with a labelled BGC;
* each labelled BGC is classified as model-exclusive, comparator-exclusive,
  jointly recovered, or missed by all methods; exclusive true positives are
  reported separately from unmatched predictions;
* for model-distribution analysis, each labelled BGC is assigned its minimum
  composite evolutionary distance (Supplementary Method 5) to the training
  positive set, and distributions are compared between model-exclusive and
  jointly recovered true positives without changing the detection threshold or
  evaluation partition.

The module is dependency-free; the distance analysis imports
``bgc2activity.detection.families`` lazily so that it is only required when
actually used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GENOME_IDENTIFIERS = tuple(f"genome_{index}" for index in range(9))
PRODUCT_CLASSES = ("NRPS", "PKS", "RiPP", "terpene", "other")
BENCHMARK_GENE_THRESHOLD = 0.5
REGION_POSITION_THRESHOLD = 0.45
BENCHMARK_TOTAL_GENES = 67_072
BENCHMARK_POSITIVE_GENES = 3_696
BENCHMARK_PFAM_ANNOTATED_GENES = 54_285

_PRODUCT_CLASS_RULES = (
    ("NRPS", ("nrps", "non-ribosomal", "nonribosomal")),
    ("PKS", ("pks", "polyketide")),
    ("RiPP", ("ripp", "ribosomally")),
    ("terpene", ("terpene", "terpenoid", "isoprenoid")),
)


def merge_product_class(raw_label: str) -> str:
    """Merge an arbitrary product label into the five benchmark categories."""
    lowered = (raw_label or "").lower()
    for product_class, keywords in _PRODUCT_CLASS_RULES:
        if any(keyword in lowered for keyword in keywords):
            return product_class
    return "other"


@dataclass(frozen=True)
class BenchmarkGene:
    gene_id: str
    genome_id: str
    start: int
    end: int
    pfam_annotated: bool
    label: int  # 1 if the gene falls inside a labelled BGC, else 0


@dataclass(frozen=True)
class TruthRegion:
    genome_id: str
    start: int
    end: int
    product_class: str = "other"


@dataclass(frozen=True)
class PredictedRegion:
    genome_id: str
    start: int
    end: int


def assign_gene_labels(genes: list[BenchmarkGene], truth_regions: list[TruthRegion]) -> list[BenchmarkGene]:
    """Return gene records with labels set by coordinate overlap with labelled BGCs."""
    by_genome: dict[str, list[TruthRegion]] = {}
    for region in truth_regions:
        by_genome.setdefault(region.genome_id, []).append(region)
    relabelled = []
    for gene in genes:
        positive = any(
            gene.start <= region.end and region.start <= gene.end
            for region in by_genome.get(gene.genome_id, [])
        )
        relabelled.append(
            BenchmarkGene(gene.gene_id, gene.genome_id, gene.start, gene.end, gene.pfam_annotated, int(positive))
        )
    return relabelled


def project_regions_to_gene_scores(
    predictions: list[PredictedRegion], genes: list[BenchmarkGene],
) -> dict[str, float]:
    """Project region-level tool output to gene scores by coordinate overlap.

    A gene receives 1.0 when it overlaps at least one predicted region and 0.0
    otherwise (unified projection rule of Supplementary Method 7).
    """
    by_genome: dict[str, list[PredictedRegion]] = {}
    for region in predictions:
        by_genome.setdefault(region.genome_id, []).append(region)
    scores = {}
    for gene in genes:
        covered = any(
            gene.start <= region.end and region.start <= gene.end
            for region in by_genome.get(gene.genome_id, [])
        )
        scores[gene.gene_id] = 1.0 if covered else 0.0
    return scores


def aggregate_position_scores_to_genes(
    position_scores: dict[tuple[str, int], float],
    genes: list[BenchmarkGene],
    default_score: float = 0.0,
) -> dict[str, float]:
    """Reduce per-position scores to gene scores by taking the maximum within
    each gene interval.

    Positions are keyed by ``(genome_id, position)`` with 1-based coordinates.
    Genes without covered positions (including genes without Pfam annotations,
    which the proposed model assigns 0.0) receive ``default_score``.
    """
    scores = {}
    for gene in genes:
        best = None
        for position in range(gene.start, gene.end + 1):
            value = position_scores.get((gene.genome_id, position))
            if value is not None and (best is None or value > best):
                best = value
        scores[gene.gene_id] = best if best is not None else default_score
    return scores


def binary_metrics_at_threshold(
    scores: list[float], labels: list[int], threshold: float = BENCHMARK_GENE_THRESHOLD,
) -> dict[str, float]:
    """Gene-level binary metrics at the uniform 0.5 benchmark threshold."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        predicted = score >= threshold
        if predicted and label == 1:
            tp += 1
        elif predicted and label == 0:
            fp += 1
        elif not predicted and label == 1:
            fn += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def auroc(scores: list[float], labels: list[int]) -> float:
    """Area under the ROC curve via the tie-corrected Mann-Whitney U statistic."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        raise ValueError("AUROC requires both positive and negative samples")
    ordered = sorted(zip(scores, labels))
    rank_sum_positive = 0.0
    index = 0
    while index < len(ordered):
        block_end = index
        while block_end + 1 < len(ordered) and ordered[block_end + 1][0] == ordered[index][0]:
            block_end += 1
        average_rank = (index + block_end) / 2.0 + 1.0
        for position in range(index, block_end + 1):
            if ordered[position][1] == 1:
                rank_sum_positive += average_rank
        index = block_end + 1
    n_pos, n_neg = len(positives), len(negatives)
    u_statistic = rank_sum_positive - n_pos * (n_pos + 1) / 2.0
    return u_statistic / (n_pos * n_neg)


def auprc(scores: list[float], labels: list[int]) -> float:
    """Area under the precision-recall curve with linear interpolation between
    distinct score thresholds."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    total_positive = sum(labels)
    if total_positive == 0 or total_positive == len(labels):
        raise ValueError("AUPRC requires both positive and negative samples")
    ordered = sorted(zip(scores, labels), key=lambda item: -item[0])
    points = [(0.0, 1.0)]
    true_positives = 0
    index = 0
    while index < len(ordered):
        block_end = index
        while block_end + 1 < len(ordered) and ordered[block_end + 1][0] == ordered[index][0]:
            block_end += 1
        for position in range(index, block_end + 1):
            true_positives += ordered[position][1]
        examined = block_end + 1
        recall = true_positives / total_positive
        precision = true_positives / examined
        points.append((recall, precision))
        index = block_end + 1
    area = 0.0
    for (recall_a, precision_a), (recall_b, precision_b) in zip(points, points[1:]):
        area += (recall_b - recall_a) * (precision_a + precision_b) / 2.0
    return area


def evaluate_gene_level(
    scores: list[float], labels: list[int], threshold: float = BENCHMARK_GENE_THRESHOLD,
) -> dict[str, float]:
    """Full gene-level metric block reported for each method on the benchmark."""
    metrics = binary_metrics_at_threshold(scores, labels, threshold)
    metrics["auroc"] = auroc(scores, labels)
    metrics["auprc"] = auprc(scores, labels)
    return metrics


@dataclass
class RecoveryReport:
    """Reference-backed recovery classification of labelled BGCs.

    ``categories`` maps each truth-region index to one of ``model_exclusive``,
    ``comparator_exclusive``, ``jointly_recovered`` or ``missed_by_all``;
    ``recovering_methods`` records the full recovering set, and
    ``unmatched_predictions`` lists per-method predictions without truth
    support (reported separately and never counted as exclusive true
    positives).
    """

    categories: list[str] = field(default_factory=list)
    recovering_methods: list[tuple[str, ...]] = field(default_factory=list)
    unmatched_predictions: dict[str, list[PredictedRegion]] = field(default_factory=dict)


def regions_overlap(region_a_start: int, region_a_end: int, region_b_start: int, region_b_end: int) -> bool:
    return region_a_start <= region_b_end and region_b_start <= region_a_end


def classify_recovery(
    truth_regions: list[TruthRegion],
    predictions_by_method: dict[str, list[PredictedRegion]],
    proposed_method: str,
) -> RecoveryReport:
    """Classify each labelled BGC by which methods recover it under the
    prespecified coordinate-overlap rule (Supplementary Method 7)."""
    if proposed_method not in predictions_by_method:
        raise ValueError("proposed_method must be present in predictions_by_method")
    report = RecoveryReport(unmatched_predictions={method: [] for method in predictions_by_method})
    for truth in truth_regions:
        recovering = []
        for method, predictions in predictions_by_method.items():
            if any(
                prediction.genome_id == truth.genome_id
                and regions_overlap(prediction.start, prediction.end, truth.start, truth.end)
                for prediction in predictions
            ):
                recovering.append(method)
        model_recovered = proposed_method in recovering
        comparator_recovered = [method for method in recovering if method != proposed_method]
        if model_recovered and not comparator_recovered:
            category = "model_exclusive"
        elif model_recovered and comparator_recovered:
            category = "jointly_recovered"
        elif comparator_recovered:
            category = "comparator_exclusive"
        else:
            category = "missed_by_all"
        report.categories.append(category)
        report.recovering_methods.append(tuple(sorted(recovering)))
    for method, predictions in predictions_by_method.items():
        for prediction in predictions:
            matched = any(
                truth.genome_id == prediction.genome_id
                and regions_overlap(prediction.start, prediction.end, truth.start, truth.end)
                for truth in truth_regions
            )
            if not matched:
                report.unmatched_predictions[method].append(prediction)
    return report


def stratified_counts(report: RecoveryReport, truth_regions: list[TruthRegion]) -> dict[tuple[str, str], int]:
    """Count recovery categories stratified by truth product class (used for
    product-class and host-taxonomy stratification of exclusive true
    positives)."""
    if len(report.categories) != len(truth_regions):
        raise ValueError("report and truth_regions must have the same length")
    counts: dict[tuple[str, str], int] = {}
    for category, truth in zip(report.categories, truth_regions):
        key = (category, truth.product_class)
        counts[key] = counts.get(key, 0) + 1
    return counts


def minimum_training_distance(
    query_pfam_sets: list[frozenset[str]],
    query_centroids,
    training_pfam_sets: list[frozenset[str]],
    training_centroids,
) -> list[float]:
    """Minimum composite evolutionary distance (Supplementary Method 5) from
    each labelled BGC to the training positive set, used for the
    model-exclusive vs jointly-recovered distribution comparison.

    Distances are computed with the fixed Pfam-set/pfam2vec metric; the
    detection threshold and evaluation partition are not modified here.
    """
    from bgc2activity.detection.families import composite_bgc_distance

    minima = []
    for query_set, query_centroid in zip(query_pfam_sets, query_centroids):
        best = None
        for training_set, training_centroid in zip(training_pfam_sets, training_centroids):
            distance = composite_bgc_distance(query_set, training_set, query_centroid, training_centroid)
            if best is None or distance < best:
                best = distance
        minima.append(float(best))
    return minima


def region_level_metrics(
    predictions: list[PredictedRegion], truth_regions: list[TruthRegion],
) -> dict[str, float]:
    """Reference-backed region-level precision/recall.

    A predicted region is a true positive only when it satisfies the
    coordinate-overlap rule with a labelled BGC; unmatched predictions count
    as false positives and unmatched truth regions as false negatives.
    """
    true_positives = 0
    matched_truth = set()
    for prediction in predictions:
        hit = False
        for index, truth in enumerate(truth_regions):
            if truth.genome_id == prediction.genome_id and regions_overlap(
                prediction.start, prediction.end, truth.start, truth.end
            ):
                hit = True
                matched_truth.add(index)
        if hit:
            true_positives += 1
    precision = true_positives / len(predictions) if predictions else 0.0
    recall = len(matched_truth) / len(truth_regions) if truth_regions else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": float(true_positives),
        "fp": float(len(predictions) - true_positives),
        "fn": float(len(truth_regions) - len(matched_truth)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def product_class_metrics(
    predicted_classes: list[str], labels: list[str], classes: tuple[str, ...] = PRODUCT_CLASSES,
) -> dict[str, object]:
    """Product-class prediction metrics (Methods 7 and 24).

    Reports per-class precision/recall/F1, macro-averaged F1 and the
    confusion matrix (rows = true class, columns = predicted class).
    """
    if len(predicted_classes) != len(labels):
        raise ValueError("predicted_classes and labels must have the same length")
    confusion = {true_class: {pred_class: 0 for pred_class in classes} for true_class in classes}
    for predicted, label in zip(predicted_classes, labels):
        if label not in confusion or predicted not in confusion[label]:
            raise ValueError(f"class outside the five benchmark categories: {label!r}/{predicted!r}")
        confusion[label][predicted] += 1
    per_class: dict[str, dict[str, float]] = {}
    f1_values = []
    for class_name in classes:
        tp = confusion[class_name][class_name]
        fp = sum(confusion[other][class_name] for other in classes if other != class_name)
        fn = sum(confusion[class_name][other] for other in classes if other != class_name)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[class_name] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        f1_values.append(f1)
    return {
        "per_class": per_class,
        "macro_f1": sum(f1_values) / len(f1_values),
        "confusion": confusion,
    }

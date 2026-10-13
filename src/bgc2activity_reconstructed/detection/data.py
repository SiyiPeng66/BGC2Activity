"""Input representation and batching for the BGC detector.

Implements Supplementary Methods 3-4 of the final manuscript. Each region is an
ordered Pfam domain sequence (order fixed by genomic coordinates, gene
orientation and intra-protein domain position). Every Pfam position is
represented by a frozen 100-dimensional pfam2vec vector recovered from the
workspace; domains outside the
pfam2vec vocabulary map to a dedicated OOV token. Samples are padded to a
uniform length within a batch and padding positions are excluded from the loss,
pooling and metric computation by an explicit validity mask. Training samples
are truncated to the first 256 Pfam positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import gzip

# The manuscript says 102 dimensions, but every recoverable table and the
# historical 100pfamdim pipeline are 100-dimensional. No synthetic padding is
# introduced; the loaded width is recorded in each training manifest.
PFAM2VEC_DIMENSION = 100
MAX_TRAIN_POSITIONS = 256
OOV_TOKEN = "<OOV>"


def load_pfam2vec(path: str | Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load the frozen pfam2vec table (CSV: pfam_id followed by 100 values).

    Returns (vocabulary mapping, embedding matrix) where row 0 of the matrix is
    reserved for the OOV token.
    """
    table_path = Path(path)
    vocabulary: dict[str, int] = {OOV_TOKEN: 0}
    rows: list[np.ndarray] = []
    opener = gzip.open if table_path.suffix == ".gz" else open
    with opener(table_path, "rt", encoding="utf-8") as handle:
        header = handle.readline()
        n_columns = header.rstrip("\n").split(",")
        dimension = len(n_columns) - 1
        if dimension < 1:
            raise ValueError(f"unexpected pfam2vec column count: {len(n_columns)}")
        for line in handle:
            parts = line.rstrip("\n").split(",")
            token, values = parts[0], parts[1:]
            vocabulary.setdefault(token, len(vocabulary))
            if len(values) != dimension:
                raise ValueError(f"inconsistent pfam2vec row width for {token}")
            rows.append(np.asarray(values, dtype=np.float32))
    oov_row = np.zeros(dimension, dtype=np.float32)
    matrix = np.vstack([oov_row[None, :], np.stack(rows)])
    return vocabulary, matrix


@dataclass(frozen=True)
class RegionSample:
    sample_id: str
    pfam_ids: tuple[str, ...]
    position_labels: tuple[int, ...]
    class_label: int
    family_label: int


class PfamSequenceDataset(Dataset):
    """Batchable dataset of ordered Pfam sequences with position labels."""

    def __init__(
        self,
        samples: list[RegionSample],
        vocabulary: dict[str, int],
        embedding_matrix: np.ndarray,
        max_positions: int = MAX_TRAIN_POSITIONS,
    ) -> None:
        self.samples = samples
        self.vocabulary = vocabulary
        self.embeddings = torch.as_tensor(embedding_matrix, dtype=torch.float32)
        self.max_positions = max_positions

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> RegionSample:
        return self.samples[index]


def collate_regions(
    batch: list[RegionSample],
    vocabulary: dict[str, int],
    embedding_matrix: torch.Tensor,
    max_positions: int = MAX_TRAIN_POSITIONS,
) -> dict[str, torch.Tensor]:
    """Pad a batch and build embeddings plus validity/label masks."""
    lengths = [min(len(sample.pfam_ids), max_positions) for sample in batch]
    width = max(lengths)
    token_indices = torch.zeros(len(batch), width, dtype=torch.long)
    valid = torch.zeros(len(batch), width, dtype=torch.float32)
    labels = torch.zeros(len(batch), width, dtype=torch.float32)
    class_labels = torch.zeros(len(batch), dtype=torch.long)
    family_labels = torch.zeros(len(batch), dtype=torch.long)
    for row, (sample, length) in enumerate(zip(batch, lengths)):
        for position in range(length):
            token = sample.pfam_ids[position]
            token_indices[row, position] = vocabulary.get(token, vocabulary[OOV_TOKEN])
            valid[row, position] = 1.0
            labels[row, position] = float(sample.position_labels[position])
        class_labels[row] = sample.class_label
        family_labels[row] = sample.family_label
    embeddings = embedding_matrix[token_indices]
    return {
        "embeddings": embeddings,
        "valid": valid,
        "position_labels": labels,
        "class_labels": class_labels,
        "family_labels": family_labels,
    }

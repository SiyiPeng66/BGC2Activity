#!/usr/bin/env python3
"""Train the chemical-aware peptide pretrainer under the manuscript protocol.

Entry point for Supplementary Method 10. All inputs are explicit CLI paths
(no legacy defaults). The protocol is frozen by the final manuscript:
AdamW (lr 8e-4, weight decay 1e-3), gradient clipping 1.0, batch size 128,
30 epochs, seed 20260802; 30% of valid positions are masked with the
80/10/10 scheme (at least one per sequence); the objective is SmoothL1 +
0.20 * cosine distance at the selected positions; the checkpoint with the
lowest validation total loss is taken as the final encoder.

Input corpus TSV columns:
  entity_id, monomers (';'-separated monomer names),
  descriptors (';'-separated groups of 8 comma-separated values, one group
  per monomer; missing descriptors as 'NA' groups are median-imputed per
  batch column)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader, Dataset

from bgc2activity.representation.pretraining import (
    CHEMICAL_TARGET_DIMENSION,
    DESCRIPTOR_DIMENSION,
    ChemicalAwarePretrainer,
    apply_masking,
    blake2b_text_vector,
    pretraining_loss,
)

PROTOCOL = {
    "optimizer": "AdamW",
    "learning_rate": 8e-4,
    "weight_decay": 1e-3,
    "gradient_clip": 1.0,
    "batch_size": 128,
    "max_epochs": 30,
    "seed": 20260802,
    "mask_fraction": 0.30,
    "mask_scheme": [0.80, 0.10, 0.10],
    "loss": "smooth_l1 + 0.20 * cosine_distance",
    "checkpoint_selection": "lowest_validation_total_loss",
}


class PretrainingEntity:
    def __init__(self, entity_id: str, monomers: list[str], descriptors: list[list[float]]):
        self.entity_id = entity_id
        self.monomers = monomers
        self.descriptors = descriptors


def load_corpus(path: Path) -> list[PretrainingEntity]:
    corpus = []
    with path.open("rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            monomers = row["monomers"].split(";")
            descriptor_groups = []
            for group in row["descriptors"].split(";"):
                if group.strip().upper() == "NA":
                    descriptor_groups.append([float("nan")] * DESCRIPTOR_DIMENSION)
                else:
                    values = [float(value) for value in group.split(",")]
                    if len(values) != DESCRIPTOR_DIMENSION:
                        raise ValueError(f"{row['entity_id']}: descriptor groups must have 8 values")
                    descriptor_groups.append(values)
            if len(descriptor_groups) != len(monomers):
                raise ValueError(f"{row['entity_id']}: monomer/descriptor length mismatch")
            corpus.append(PretrainingEntity(row["entity_id"], monomers, descriptor_groups))
    return corpus


class CorpusDataset(Dataset):
    def __init__(self, entities: list[PretrainingEntity]):
        self.entities = entities
        self.text_vectors = [
            torch.stack([blake2b_text_vector(monomer) for monomer in entity.monomers])
            for entity in entities
        ]

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, index: int):
        entity = self.entities[index]
        return self.text_vectors[index], torch.tensor(entity.descriptors, dtype=torch.float32)


def collate_corpus(batch):
    text_vectors, descriptors = zip(*batch)
    length = max(tensor.shape[0] for tensor in text_vectors)
    attention_mask = torch.zeros(len(batch), length)
    padded_text = torch.zeros(len(batch), length, text_vectors[0].shape[1])
    padded_descriptors = torch.zeros(len(batch), length, DESCRIPTOR_DIMENSION)
    for row, (text, descriptor) in enumerate(zip(text_vectors, descriptors)):
        attention_mask[row, : text.shape[0]] = 1.0
        padded_text[row, : text.shape[0]] = text
        padded_descriptors[row, : text.shape[0]] = descriptor
    # median-impute missing descriptors per batch column
    missing = torch.isnan(padded_descriptors)
    if missing.any():
        column_medians = torch.nanmedian(padded_descriptors, dim=0).values
        column_medians = torch.where(torch.isnan(column_medians), torch.zeros_like(column_medians), column_medians)
        padded_descriptors = torch.where(missing, column_medians.expand_as(padded_descriptors), padded_descriptors)
    return padded_text, padded_descriptors, attention_mask


def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    generator = torch.Generator(device=device).manual_seed(PROTOCOL["seed"])
    with torch.no_grad():
        for text_vectors, descriptors, attention_mask in loader:
            text_vectors = text_vectors.to(device)
            descriptors = descriptors.to(device)
            attention_mask = attention_mask.to(device)
            hidden = model.embed(text_vectors, descriptors, attention_mask)
            masked_hidden, selection = apply_masking(hidden, attention_mask, model.mask_token, generator=generator)
            prediction = _forward_from_hidden(model, masked_hidden, attention_mask)
            target = torch.cat([text_vectors, descriptors], dim=-1)
            losses = pretraining_loss(prediction, target, selection)
            total_loss += float(losses["total"]) * int(selection.sum())
            total_tokens += int(selection.sum())
    return total_loss / max(total_tokens, 1)


def _forward_from_hidden(model, masked_hidden, attention_mask):
    encoded = model.encoder(masked_hidden, attention_mask)
    return model.recovery_head(model.output_norm(encoded))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-corpus", required=True, help="training corpus TSV")
    parser.add_argument("--validation-corpus", required=True, help="validation corpus TSV")
    parser.add_argument("--output-dir", required=True, help="checkpoint and manifest output directory")
    args = parser.parse_args()

    torch.manual_seed(PROTOCOL["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(CorpusDataset(load_corpus(Path(args.train_corpus))),
                              batch_size=PROTOCOL["batch_size"], shuffle=True, collate_fn=collate_corpus)
    validation_loader = DataLoader(CorpusDataset(load_corpus(Path(args.validation_corpus))),
                                   batch_size=PROTOCOL["batch_size"], shuffle=False, collate_fn=collate_corpus)

    model = ChemicalAwarePretrainer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL["learning_rate"],
                                  weight_decay=PROTOCOL["weight_decay"])

    best_validation_loss = float("inf")
    best_state = None
    history = []
    generator = torch.Generator(device=device).manual_seed(PROTOCOL["seed"])
    for epoch in range(1, PROTOCOL["max_epochs"] + 1):
        model.train()
        for text_vectors, descriptors, attention_mask in train_loader:
            text_vectors = text_vectors.to(device)
            descriptors = descriptors.to(device)
            attention_mask = attention_mask.to(device)
            optimizer.zero_grad()
            hidden = model.embed(text_vectors, descriptors, attention_mask)
            masked_hidden, selection = apply_masking(hidden, attention_mask, model.mask_token, generator=generator)
            prediction = _forward_from_hidden(model, masked_hidden, attention_mask)
            target = torch.cat([text_vectors, descriptors], dim=-1)
            losses = pretraining_loss(prediction, target, selection)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL["gradient_clip"])
            optimizer.step()
        validation_loss = evaluate(model, validation_loader, device)
        history.append({"epoch": epoch, "validation_total_loss": validation_loss})
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "pretrainer_checkpoint.pt")
    manifest = {"protocol": PROTOCOL, "history": history,
                "best_validation_total_loss": best_validation_loss}
    (output_dir / "pretraining_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote pretrainer checkpoint and manifest to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

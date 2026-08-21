#!/usr/bin/env python3
"""Audit the committed release data against manuscript-fixed invariants.

Dependency-free (standard library only) so it runs in the base `make test`
target. Every check mirrors a count or rule fixed by the final manuscript
(Tables S1-S5, Supplementary Methods 6, 8, 13, 18). Exits non-zero on any
violation.
"""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
BGC_ID_RE = re.compile(r"^(GCF_\d+\.\d+)__(c\d+)_GCF_.*\.\.\.(region\d+)$")


class AuditError(RuntimeError):
    pass


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        return reader.fieldnames or [], rows


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def audit_genomes() -> None:
    _fields, rows = read_tsv(DATA_ROOT / "genomes" / "refseq_genome_inventory.tsv")
    expect(len(rows) == 56709, f"genome inventory rows {len(rows)} != 56709")
    accessions = [row["genome_accession"] for row in rows]
    expect(len(set(accessions)) == 56709, "genome accessions are not unique")


def audit_bgc_indices() -> None:
    _fields, nrps = read_tsv(DATA_ROOT / "bgc_detection" / "nrps_positive_bgc_index.tsv")
    expect(len(nrps) == 258750, f"NRPS-positive BGC rows {len(nrps)} != 258750")
    ids = [row["bgc_id"] for row in nrps]
    expect(len(set(ids)) == len(ids), "NRPS-positive BGC identifiers are not unique")
    for row in nrps[:1000]:
        match = BGC_ID_RE.match(row["bgc_id"])
        expect(match is not None, f"unparseable bgc_id {row['bgc_id']!r}")
        expect(match.group(1) == row["genome_accession"], "bgc_id/genome_accession disagree")

    _fields, substrate = read_tsv(DATA_ROOT / "bgc_detection" / "substrate_predicted_bgc_index.tsv")
    expect(len(substrate) == 77587, f"substrate-predicted BGC rows {len(substrate)} != 77587")


def audit_candidate_library() -> None:
    _fields, rows = read_tsv(DATA_ROOT / "candidate_library" / "candidate_library.tsv")
    expect(len(rows) == 23540, f"candidate library rows {len(rows)} != 23540")
    entry_ids = [row["entry_id"] for row in rows]
    expect(len(set(entry_ids)) == 23540, "entry_id values are not unique")

    bgc_ids = {row["bgc_id"] for row in rows}
    expect(len(bgc_ids) == 12270, f"library-wide unique BGC identifiers {len(bgc_ids)} != 12270")
    systematic = [row for row in rows if row["is_constrained_variant"] == "False"]
    constrained = [row for row in rows if row["is_constrained_variant"] == "True"]
    expect(len({row["bgc_id"] for row in systematic}) == 12266,
           "systematic parent BGC count != 12266")
    expect(len(constrained) == 29, f"constrained variants {len(constrained)} != 29")
    expect(len({row["bgc_id"] for row in constrained}) == 19, "constrained parent BGCs != 19")
    expect(len({row["core_sequence"] for row in rows}) == 3713, "unique core sequences != 3713")
    expect(len({row["displayed_sequence"] for row in rows}) == 5303, "unique displayed sequences != 5303")

    oh = sum(row["c_terminal"] == "OH" for row in rows)
    nh2 = sum(row["c_terminal"] == "NH2" for row in rows)
    expect(oh == 14233 and nh2 == 9307, f"C-terminal counts OH={oh}/NH2={nh2} != 14233/9307")
    nonstandard = sum(row["is_nonstandard_or_modified"] == "True" for row in rows)
    lipidated = sum(row["is_lipidated"] == "True" for row in rows)
    expect(nonstandard == 20579, f"nonstandard/modified records {nonstandard} != 20579")
    expect(lipidated == 1625, f"lipidated records {lipidated} != 1625")
    expect(all(int(row["effective_module_count"]) >= 5 for row in rows),
           "candidate with fewer than five effective modules")


def audit_funnel() -> None:
    _fields, rows = read_tsv(DATA_ROOT / "candidate_library" / "library_construction_funnel.tsv")
    counts = {row["stage"]: row["count"] for row in rows}
    expected = {
        "RefSeq genomes screened": "56709",
        "Genomes with NRPS-positive BGCs": "33157",
        "NRPS-positive BGCs": "258750",
        "BGCs with substrate predictions": "77587",
        "Systematic parent BGC candidates": "12266",
        "Systematically generated candidate records": "23540",
        "Total candidate records": "23540",
        "Unique BGCs represented in final library": "12266",
        "Unique core sequences": "3713",
        "Unique display sequences": "5303",
    }
    for stage, count in expected.items():
        expect(counts.get(stage) == count, f"funnel stage {stage!r}: {counts.get(stage)} != {count}")


def audit_screening() -> None:
    _fields, mic = read_tsv(DATA_ROOT / "screening" / "two_round_mic.tsv")
    expect(len(mic) == 400, f"MIC rows {len(mic)} != 400")
    rounds = {row["screening_round"] for row in mic}
    expect(rounds == {"Round 1", "Round 2"}, f"unexpected rounds {rounds}")
    r1 = {row["screened_candidate_id"] for row in mic if row["screening_round"] == "Round 1"}
    r2 = {row["screened_candidate_id"] for row in mic if row["screening_round"] == "Round 2"}
    expect(len(r1) == 40 and len(r2) == 40, "rounds must contain 40 candidates each")
    expect(not (r1 & r2), "Round 1 and Round 2 candidates overlap")

    def minimum_mic(rows: list[dict[str, str]]) -> dict[str, float]:
        minima: dict[str, float] = {}
        for row in rows:
            value = row["mic_ug_mL"]
            numeric = float("inf") if value.startswith(">") else float(value)
            candidate = row["screened_candidate_id"]
            minima[candidate] = min(minima.get(candidate, float("inf")), numeric)
        return minima

    for rnd, expected_hits, expected_potent in (("Round 1", 8, 3), ("Round 2", 18, 12)):
        minima = minimum_mic([row for row in mic if row["screening_round"] == rnd])
        hits = sum(value <= 256 for value in minima.values())
        potent = sum(value <= 64 for value in minima.values())
        expect(hits == expected_hits, f"{rnd} screening hits {hits} != {expected_hits}")
        expect(potent == expected_potent, f"{rnd} potent hits {potent} != {expected_potent}")

    _fields, mapping = read_tsv(DATA_ROOT / "screening" / "screening_to_library_mapping.tsv")
    expect(len(mapping) == 80, f"mapping rows {len(mapping)} != 80")
    expect(all(int(row["n_matched_entries"]) >= 1 for row in mapping),
           "screened candidate without library match")


def audit_checksums() -> None:
    manifest_path = DATA_ROOT / "manifests" / "data_checksums.tsv"
    _fields, rows = read_tsv(manifest_path)
    for row in rows:
        target = DATA_ROOT / row["file"]
        expect(target.is_file(), f"checksum manifest references missing file {row['file']}")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        expect(digest == row["sha256"], f"checksum mismatch for {row['file']}")


def main() -> int:
    audits = (
        audit_genomes,
        audit_bgc_indices,
        audit_candidate_library,
        audit_funnel,
        audit_screening,
        audit_checksums,
    )
    for audit in audits:
        audit()
        print(f"  pass: {audit.__name__}")
    print("release data audit: all manuscript-fixed invariants hold")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AuditError as error:
        print(f"release data audit FAILED: {error}", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""Build the release-ready core data tables for the BGC2Activity repository.

Every exported table is derived strictly from the final-manuscript source
workbooks listed below. The script is deterministic: it reads the frozen
legacy sources, validates manuscript-fixed counts, writes UTF-8 TSV files and
records SHA-256 checksums. No modelling output is produced here.

Source workbooks (legacy workspace, passed via --workspace):
  CHAIN   = BGC全链条关键信息_0721(1).xlsx        (genome inventory, BGC indices, wet-lab MIC wide table)
  LIB     = 最终待筛选多肽文库_0722.xlsx           (23,540-record candidate library with variant classes)
  SUPP    = Manuscripts/BGC-20260811-V2/文稿v3/Supplementary_Tables.xlsx  (final Tables S1-S5)
  MICV3   = Manuscripts/V3/补充材料/最终附表/Supplementary_Tables_V3_core.xlsx  (80-candidate MIC long table)
  FIG4    = Figure4_traceability_package/          (NRP45 source-genome traceability)

Subcommands: genomes, bgc-index, candidate-library, supplementary-tables,
screening-mic, nrp45, checksums, all
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

MANUSCRIPT = "Peng et al., Hierarchical NRPS-guided global-to-local discovery and optimisation of synthetic anti-MRSA lipopeptides (final manuscript Manuscript_SB_0813.docx)"

STD_MONOMERS = frozenset(
    "Ala Arg Asn Asp Cys Gln Glu Gly His Ile Leu Lys Met Phe Pro Ser Thr Trp Tyr Val".split()
)
TOKEN_RE = re.compile(r"[A-Za-z0-9β,]+(?:\([^)]*\))?")
LIPID_RE = re.compile(r"Lys\(|N-Myr|N-Lau|Myr-|Lau-|3R-OH")
BGC_ID_RE = re.compile(r"^(?P<genome>GCF_\d+\.\d+)__(?P<contig>c\d+)_(?P<rest>GCF_.*)\.\.\.(?P<region>region\d+)$")


class ContractError(RuntimeError):
    """Raised when a manuscript-fixed count or invariant is violated."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def workspace_path(args: argparse.Namespace, *parts: str) -> Path:
    return Path(args.workspace).joinpath(*parts)


def out_path(args: argparse.Namespace, *parts: str) -> Path:
    target = Path(args.output).joinpath(*parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")


def display_tokens(displayed_sequence: str) -> list[str]:
    body = displayed_sequence[2:] if displayed_sequence.startswith("H-") else displayed_sequence
    body = re.sub(r"-(OH|NH2)$", "", body)
    return TOKEN_RE.findall(body)


def parse_bgc_id(bgc_id: str) -> tuple[str, str, str]:
    m = BGC_ID_RE.match(bgc_id)
    if not m:
        raise ContractError(f"unparseable frozen BGC_ID: {bgc_id!r}")
    return m.group("genome"), m.group("contig"), m.group("region")


# --------------------------------------------------------------------------
# Data units
# --------------------------------------------------------------------------

def export_genomes(args: argparse.Namespace) -> list[Path]:
    chain = pd.read_excel(workspace_path(args, "BGC全链条关键信息_0721(1).xlsx"), sheet_name="1.基因组来源")
    supp = pd.read_excel(
        workspace_path(args, "Manuscripts/BGC-20260811-V2/文稿v3/Supplementary_Tables.xlsx"),
        sheet_name="Table S3", header=3,
    )
    check(len(chain) == 56709, f"genome inventory must have 56709 rows, got {len(chain)}")
    check(set(chain["ID"]) == set(supp["ID"]), "chain workbook and final Table S3 genome sets differ")
    out = pd.DataFrame({
        "source": chain["Source"],
        "genome_accession": chain["ID"],
        "keyword": chain["Keyword"],
        "ncbi_taxonomy": chain["NCBI-Tax"],
    })
    target = out_path(args, "genomes", "refseq_genome_inventory.tsv")
    write_tsv(out, target)
    return [target]


def export_bgc_index(args: argparse.Namespace) -> list[Path]:
    chain_file = workspace_path(args, "BGC全链条关键信息_0721(1).xlsx")
    outputs: list[Path] = []

    nrps = pd.read_excel(chain_file, sheet_name="2.NRPS+聚类")
    check(len(nrps) == 258750, f"NRPS-positive BGC index must have 258750 rows, got {len(nrps)}")
    parsed = nrps["BGC_ID"].astype(str).map(parse_bgc_id)
    genomes_ok = all(g == gid for (g, _c, _r), gid in zip(parsed, nrps["GenomeID"]))
    check(genomes_ok, "BGC_ID genome prefix disagrees with GenomeID column")
    out = pd.DataFrame({
        "genome_accession": nrps["GenomeID"],
        "bgc_id": nrps["BGC_ID"],
        "contig_label": [c for _g, c, _r in parsed],
        "region_label": [r for _g, _c, r in parsed],
        "nearest_known_bgc": nrps["MostSimilarBGC"],
        "reference_relative_distance": nrps["Distance"],
    })
    target = out_path(args, "bgc_detection", "nrps_positive_bgc_index.tsv")
    write_tsv(out, target)
    outputs.append(target)

    subs = pd.read_excel(chain_file, sheet_name="3.NRPS-BGC-NRPStransformer")
    check(len(subs) == 77587, f"substrate-predicted BGC index must have 77587 rows, got {len(subs)}")
    parsed = subs["BGC_ID"].astype(str).map(parse_bgc_id)
    out = pd.DataFrame({
        "genome_accession": subs["GenomeID"],
        "bgc_id": subs["BGC_ID"],
        "contig_label": [c for _g, c, _r in parsed],
        "region_label": [r for _g, _c, r in parsed],
        "nearest_known_bgc": subs["MostSimilarBGC"],
        "reference_relative_distance": subs["Distance"],
        "predicted_product": subs["Product"],
    })
    target = out_path(args, "bgc_detection", "substrate_predicted_bgc_index.tsv")
    write_tsv(out, target)
    outputs.append(target)
    return outputs


def export_candidate_library(args: argparse.Namespace) -> list[Path]:
    wb = pd.read_excel(workspace_path(args, "最终待筛选多肽文库_0722.xlsx"), sheet_name="待筛选多肽文库")
    check(len(wb) == 23540, f"candidate library must have 23540 rows, got {len(wb)}")
    check(wb["Entry_ID"].is_unique, "Entry_ID must be unique")

    parsed = wb["BGC_ID"].astype(str).map(parse_bgc_id)
    tokens = wb["Displayed_sequence"].astype(str).map(display_tokens)
    is_nonstandard = [any(t not in STD_MONOMERS for t in ts) for ts in tokens]
    is_lipidated = wb["Displayed_sequence"].astype(str).str.contains(LIPID_RE)

    check(sum(is_nonstandard) == 20579,
          f"nonstandard-or-modified records must be 20579, got {sum(is_nonstandard)}")
    check(int(is_lipidated.sum()) == 1625,
          f"explicitly lipidated records must be 1625, got {int(is_lipidated.sum())}")
    check(wb["BGC_ID"].nunique() == 12270, "library-wide unique BGC_ID count must be 12270")
    systematic = wb[wb["Variant_class"] != "最终测试候选"]
    check(systematic["BGC_ID"].nunique() == 12266,
          "systematic parent BGC count must be 12266 (manuscript parent set)")
    constrained = wb[wb["Variant_class"] == "最终测试候选"]
    check(len(constrained) == 29 and constrained["BGC_ID"].nunique() == 19,
          "constrained variants must be 29 records from 19 parent BGCs")
    check(wb["Core_sequence"].nunique() == 3713, "unique core sequences must be 3713")
    check(wb["Displayed_sequence"].nunique() == 5303, "unique displayed sequences must be 5303")
    ct = wb["C_terminal"].value_counts()
    check(ct.get("OH") == 14233 and ct.get("NH2") == 9307, "C-terminal counts must be OH=14233, NH2=9307")

    out = pd.DataFrame({
        "entry_id": wb["Entry_ID"],
        "bgc_id": wb["BGC_ID"],
        "genome_accession": [g for g, _c, _r in parsed],
        "contig_label": [c for _g, c, _r in parsed],
        "region_label": [r for _g, _c, r in parsed],
        "nrps_gene_ids": wb["NRPS_Gene_ID"],
        "a_domain_coordinates": wb["Coordinate_of_A_domains"],
        "original_predicted_substrates": wb["Original_predicted_substrates"],
        "effective_module_count": wb["Effective_module_count"],
        "core_sequence": wb["Core_sequence"],
        "displayed_sequence": wb["Displayed_sequence"],
        "c_terminal": wb["C_terminal"],
        "variant_class": wb["Variant_class"],
        "variant_detail": wb["Variant_detail"],
        "is_constrained_variant": wb["Variant_class"] == "最终测试候选",
        "is_nonstandard_or_modified": is_nonstandard,
        "is_lipidated": is_lipidated,
    })
    target = out_path(args, "candidate_library", "candidate_library.tsv")
    write_tsv(out, target)
    return [target]


def export_supplementary_tables(args: argparse.Namespace) -> list[Path]:
    supp_file = workspace_path(args, "Manuscripts/BGC-20260811-V2/文稿v3/Supplementary_Tables.xlsx")
    outputs: list[Path] = []

    def block(sheet: str, title_prefix: str) -> pd.DataFrame:
        raw = pd.read_excel(supp_file, sheet_name=sheet, header=None)
        starts = [i for i in range(len(raw)) if str(raw.iloc[i, 0]).startswith(title_prefix)]
        if not starts:
            raise ContractError(f"{title_prefix} not found in {sheet}")
        start = starts[0]
        end = len(raw)
        for j in range(start + 1, len(raw)):
            v = str(raw.iloc[j, 0])
            if v.startswith("Table S") and not v.startswith(title_prefix):
                end = j
                break
        body = raw.iloc[start:end].dropna(how="all")
        # first non-empty row after the sub-table title row is the header
        header_index = 1
        header = body.iloc[header_index]
        df = body.iloc[header_index + 1:].copy()
        df.columns = [str(h) for h in header]
        return df.reset_index(drop=True)

    funnel = block("Table S1", "Table S1a")
    target = out_path(args, "candidate_library", "library_construction_funnel.tsv")
    write_tsv(funnel, target)
    outputs.append(target)

    rules = block("Table S1", "Table S1b")
    target = out_path(args, "candidate_library", "admission_filtering_rules.tsv")
    write_tsv(rules, target)
    outputs.append(target)

    variants = block("Table S1", "Table S1c")
    target = out_path(args, "candidate_library", "constrained_variant_rules.tsv")
    write_tsv(variants, target)
    outputs.append(target)

    resources = block("Table S2", "Table S2a")
    target = out_path(args, "bgc_detection", "data_resources.tsv")
    write_tsv(resources, target)
    outputs.append(target)

    partitions = block("Table S2", "Table S2b")
    target = out_path(args, "bgc_detection", "partition_composition.tsv")
    write_tsv(partitions, target)
    outputs.append(target)

    mic_datasets = block("Table S2", "Table S2c")
    target = out_path(args, "activity_modelling", "strict_mic_dataset_summary.tsv")
    write_tsv(mic_datasets, target)
    outputs.append(target)

    groups = pd.read_excel(supp_file, sheet_name="Table S5", header=3)
    target = out_path(args, "animal_model", "group_assignment.tsv")
    write_tsv(groups, target)
    outputs.append(target)
    return outputs


def export_screening_mic(args: argparse.Namespace) -> list[Path]:
    mic = pd.read_excel(
        workspace_path(args, "Manuscripts/V3/补充材料/最终附表/Supplementary_Tables_V3_core.xlsx"),
        sheet_name="Table S4", header=3,
    )
    check(len(mic) == 400, f"two-round MIC table must have 400 rows, got {len(mic)}")
    check(set(mic["screening_round"]) == {"Round 1", "Round 2"}, "screening rounds must be Round 1 and Round 2")
    r1 = mic[mic["screening_round"] == "Round 1"]["candidate_id"].unique()
    r2 = mic[mic["screening_round"] == "Round 2"]["candidate_id"].unique()
    check(len(r1) == 40 and len(r2) == 40, "each round must contain exactly 40 candidates")
    check(not set(r1) & set(r2), "Round 1 and Round 2 candidate sets must not overlap")

    wide = pd.read_excel(workspace_path(args, "BGC全链条关键信息_0721(1).xlsx"), sheet_name="湿实验测试多肽")
    wide_by_seq = {row["Sequence"]: row for _i, row in wide.iterrows()}
    check(set(mic.groupby("candidate_id")["sequence"].first()) == set(wide_by_seq),
          "screened sequence sets differ between the V3 MIC table and the chain wet-lab sheet")
    org_col = {
        "A. baumannii": "鲍曼不动杆菌\nMIC (μg/mL)",
        "E. coli": "大肠杆菌\nMIC (μg/mL)",
        "P. aeruginosa": "铜绿假单胞菌\nMIC (μg/mL)",
        "S. aureus": "金黄色葡萄球菌\nMIC (μg/mL)",
        "S. enterica": "沙门氏菌\nMIC (μg/mL)",
    }
    for _i, row in mic.iterrows():
        src = wide_by_seq.get(row["sequence"])
        check(src is not None, f"screened sequence missing from chain wet-lab sheet: {row['sequence']!r}")
        expected = str(src[org_col[row["test_organism"]]])
        check(expected == str(row["MIC_ug_mL"]),
              f"MIC mismatch for {row['candidate_id']} {row['test_organism']}: {expected} vs {row['MIC_ug_mL']}")

    out = pd.DataFrame({
        "screened_candidate_id": mic["candidate_id"],
        "candidate_library_source_id": mic["source_candidate_id"],
        "screening_round": mic["screening_round"],
        "sequence": mic["sequence"],
        "test_organism": mic["test_organism"],
        "mic_ug_mL": mic["MIC_ug_mL"],
        "active_at_mic_le_256": mic["active_at_MIC_le_256"].astype(str).str.upper() == "TRUE",
    })
    target = out_path(args, "screening", "two_round_mic.tsv")
    write_tsv(out, target)

    lib = pd.read_excel(workspace_path(args, "最终待筛选多肽文库_0722.xlsx"), sheet_name="待筛选多肽文库")
    lib_tokens = lib["Displayed_sequence"].astype(str).map(
        lambda x: tuple(t.lower() for t in display_tokens(x)))
    groups: dict[tuple, list[int]] = {}
    for i, t in enumerate(lib_tokens):
        groups.setdefault(t, []).append(i)

    aliases = {"24dab": "dab", "asp-gamma": "beta-asp"}

    def screened_tokens(seq: str) -> tuple[str, ...]:
        toks = []
        for raw in re.findall(r"[A-Za-z0-9]+(?:\([^)]*\))?", seq):
            low = aliases.get(raw.lower(), raw.lower())
            low = re.sub(r"lys\(myr\)", "lys(ε-myr)", low)
            low = re.sub(r"lys\(lau\)", "lys(ε-lau)", low)
            toks.append(low)
        return tuple(toks)

    mapping_rows = []
    v3_unique = mic.drop_duplicates("candidate_id")
    for _i, row in v3_unique.iterrows():
        idx = groups.get(screened_tokens(row["sequence"]), [])
        rows = lib.iloc[idx]
        src = wide_by_seq[row["sequence"]]
        bgc_id = str(src["BGCid"])
        if len(rows) == 1:
            entry_ids = [rows["Entry_ID"].iloc[0]]
            status = "unique_token_match"
        else:
            same_bgc = rows[rows["BGC_ID"] == bgc_id]
            entry_ids = list(same_bgc["Entry_ID"]) if len(same_bgc) else list(rows["Entry_ID"])
            status = "ambiguous_terminal_forms" if len(entry_ids) > 1 else "token_plus_bgc_match"
        mapping_rows.append({
            "screened_candidate_id": row["candidate_id"],
            "screening_round": row["screening_round"],
            "sequence": row["sequence"],
            "wetlab_bgc_id": bgc_id,
            "matched_entry_ids": ";".join(entry_ids),
            "n_matched_entries": len(entry_ids),
            "match_status": status,
        })
    mapping = pd.DataFrame(mapping_rows)
    n_unique = int((mapping["match_status"] != "ambiguous_terminal_forms").sum())
    check(int((mapping["n_matched_entries"] >= 1).all()), "every screened candidate must match the library")
    target = out_path(args, "screening", "screening_to_library_mapping.tsv")
    write_tsv(mapping, target)

    def numeric(v: str) -> float:
        s = str(v).strip()
        if s.startswith(">"):
            return float("inf")
        try:
            return float(s)
        except ValueError:
            return float("inf")

    summary_rows = []
    for rnd in ("Round 1", "Round 2"):
        sub = mic[mic["screening_round"] == rnd]
        mins = sub.assign(num=sub["MIC_ug_mL"].map(numeric)).groupby("candidate_id")["num"].min()
        n_hit = int((mins <= 256).sum())
        n_potent = int((mins <= 64).sum())
        active_counts = sub.assign(num=sub["MIC_ug_mL"].map(numeric)).groupby("candidate_id").apply(
            lambda d: int((d["num"] <= 256).sum()), include_groups=False)
        n_broad = int((active_counts >= 2).sum())
        best = mins[mins <= 256].min() if (mins <= 256).any() else None
        summary_rows.append({
            "screening_round": rnd,
            "candidates": len(mins),
            "screening_hits_mic_le_256": n_hit,
            "screening_hit_rate": n_hit / len(mins),
            "broad_spectrum_hits_ge2_organisms": n_broad,
            "potent_hits_mic_le_64": n_potent,
            "potent_hit_rate": n_potent / len(mins),
            "best_observed_mic_ug_mL": best,
        })
    summary = pd.DataFrame(summary_rows)
    hit1, hit2 = summary_rows[0]["screening_hits_mic_le_256"], summary_rows[1]["screening_hits_mic_le_256"]
    pot1, pot2 = summary_rows[0]["potent_hits_mic_le_64"], summary_rows[1]["potent_hits_mic_le_64"]
    check((hit1, hit2) == (8, 18), f"screening hits must be 8/40 then 18/40, got {hit1}, {hit2}")
    check((pot1, pot2) == (3, 12), f"potent hits must be 3/40 then 12/40, got {pot1}, {pot2}")
    target = out_path(args, "screening", "screening_endpoint_summary.tsv")
    write_tsv(summary, target)
    return [out_path(args, "screening", "two_round_mic.tsv"),
            out_path(args, "screening", "screening_to_library_mapping.tsv"), target]


def export_nrp45(args: argparse.Namespace) -> list[Path]:
    fig4 = workspace_path(args, "Figure4_traceability_package")
    outputs: list[Path] = []
    for name in ("figure4_antismash_domains.tsv", "figure4_antismash_genes.tsv",
                 "figure4_antismash_provenance.txt", "figure4_family_source.tsv",
                 "figure4_lead_candidate_source.tsv", "figure4_round_summary.tsv",
                 "figure4_activity_matrix_source.tsv"):
        src = fig4 / "04_processed_tables" / name
        target = out_path(args, "nrp45", name)
        target.write_bytes(src.read_bytes())
        outputs.append(target)
    for name in ("FILE_MANIFEST.tsv", "SHA256SUMS.tsv"):
        src = fig4 / "00_index" / name
        target = out_path(args, "nrp45", f"traceability_package_{name}")
        target.write_bytes(src.read_bytes())
        outputs.append(target)
    return outputs


def export_checksums(args: argparse.Namespace) -> list[Path]:
    data_root = Path(args.output)
    rows = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".tsv", ".txt"}:
            continue
        if "manifests" in path.parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({
            "file": str(path.relative_to(data_root)),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        })
    manifest = pd.DataFrame(rows)
    target = data_root / "manifests" / "data_checksums.tsv"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(manifest, target)
    return [target]


SUBCOMMANDS = {
    "genomes": export_genomes,
    "bgc-index": export_bgc_index,
    "candidate-library": export_candidate_library,
    "supplementary-tables": export_supplementary_tables,
    "screening-mic": export_screening_mic,
    "nrp45": export_nrp45,
    "checksums": export_checksums,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subcommand", choices=[*SUBCOMMANDS, "all"])
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[2]),
                        help="legacy workspace root containing the source workbooks")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "data"),
                        help="release data directory to write")
    args = parser.parse_args()

    names = list(SUBCOMMANDS) if args.subcommand == "all" else [args.subcommand]
    if args.subcommand == "all":
        names.remove("checksums")
        names.append("checksums")
    written: list[Path] = []
    for name in names:
        print(f"[build_release_data] {name}")
        written.extend(SUBCOMMANDS[name](args))
    for path in written:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

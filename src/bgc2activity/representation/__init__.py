"""Chemical-aware peptide representation learning (Supplementary Methods 9-10).

Torch-dependent pretraining members are exported lazily so the
dependency-free corpus curation members can be imported without torch.
"""

_LAZY_EXPORTS = {
    "ChemicalAwarePretrainer": "bgc2activity.representation.pretraining",
    "CandidatePooling": "bgc2activity.representation.pretraining",
    "MonomerInputProjection": "bgc2activity.representation.pretraining",
    "blake2b_text_vector": "bgc2activity.representation.pretraining",
    "apply_masking": "bgc2activity.representation.pretraining",
    "pretraining_loss": "bgc2activity.representation.pretraining",
    "SOURCE_DATABASES": "bgc2activity.representation.corpus",
    "ChemicalEntity": "bgc2activity.representation.corpus",
    "PeptideRecord": "bgc2activity.representation.corpus",
    "build_vocabulary": "bgc2activity.representation.corpus",
    "exclude_mic_heldout_neighbours": "bgc2activity.representation.corpus",
    "merge_chemical_entities": "bgc2activity.representation.corpus",
    "modification_flags": "bgc2activity.representation.corpus",
    "normalize_monomer": "bgc2activity.representation.corpus",
    "split_corpus": "bgc2activity.representation.corpus",
    "vocabulary_status": "bgc2activity.representation.corpus",
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

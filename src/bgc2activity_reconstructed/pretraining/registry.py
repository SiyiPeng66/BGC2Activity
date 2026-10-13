"""Versioned additional monomer definitions, serialized in each new checkpoint.

label_only records preserve a source name, not a verified chemical structure.
Numeric properties stay missing unless explicitly supplied.
"""
import copy,math
_REGISTRY={}

def install(entries):
    clean={}
    for key,value in entries.items():
        if not key or key!=key.lower() or ';' in key:raise ValueError('Registry keys must be lowercase single tokens')
        if value.get('status') not in ['label_only','structure_defined']:raise ValueError('Registry status required')
        if not value.get('description') or not value.get('source'):raise ValueError('Description and provenance required')
        desc=value.get('descriptors')
        if not isinstance(desc,list) or len(desc)!=8:raise ValueError('Eight descriptor slots required')
        if any(v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)) for v in desc):raise ValueError('Finite numbers or null descriptors only')
        clean[key]=copy.deepcopy(value)
    _REGISTRY.clear();_REGISTRY.update(clean)

def entries():return copy.deepcopy(_REGISTRY)
def get(key):return _REGISTRY.get(key)

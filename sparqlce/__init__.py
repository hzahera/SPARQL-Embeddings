"""sparqlce package initializer.

Expose the main public API for easy imports from the repository.

Imports are resolved lazily (PEP 562). The data-preparation scripts
(`build_component_splits.py`, `build_rq1_dbpedia.py`, ...) need only
`sparqlce.sparql_utils`, which is pure-stdlib; eagerly importing `Config` here
would drag in torch and make split construction impossible outside the training
environment. Attribute access is unchanged for every existing caller.
"""
import importlib

_EXPORTS = {
    "Config": ".config",
    "SPARQLGraphBuilder": ".graph_builder",
    "HybridSPARQLEncoder": ".encoder",
    "SemanticSPARQLDataset": ".dataset",
    "semantic_collate_fn": ".dataset",
    "semantic_collate_fn_inbatch": ".dataset",
    "SPARQLEmbedder": ".embedder",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value          # cache, so this runs once per attribute
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))

"""SPARQL utilities for parsing and validation."""

import json
import os
import re


_VAR_RE = re.compile(r'\?[a-zA-Z_]\w*')
_TRUE_RE = re.compile(r'^\s*true\s*$', re.IGNORECASE)

# ── Wikidata label expansion ──────────────────────────────────────────────────
# Shared by the SPARQLCE graph builder and the text baselines (E5) so every
# model reads the SAME label-expanded query text — otherwise a comparison
# credits SPARQLCE for a preprocessing step the baselines never received.
_WD_LABELS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "datasets", "wikidata_labels.json"
)
_WD_LABELS_CACHE = None

# Matches a Wikidata id as a full URI (<.../P31>), a prefixed qname (wdt:P31),
# or a bare id (P31); captures the bare id in group 1 or 2.
_WD_EXPAND_RE = re.compile(
    r'<[^>]*?[/#]([PQ]\d+)>'                 # full URI  <http://.../P31>
    r'|(?:[A-Za-z_][\w.\-]*:)?([PQ]\d+)\b'   # qname (wdt:P31) or bare id (P31)
)


# Global kill-switch: set SPARQLCE_NO_WIKIDATA_LABELS=1 to disable label
# expansion everywhere (text baselines AND the graph builder), so every model
# reads the raw opaque P/Q ids. This is the only *fair* way to turn expansion
# off -- disabling it for the baselines alone while the graph builder still
# resolves labels would re-introduce the original unfairness. Unset to restore.
def _labels_disabled() -> bool:
    return os.environ.get("SPARQLCE_NO_WIKIDATA_LABELS", "").strip().lower() \
        in ("1", "true", "yes", "on")


def _wikidata_labels() -> dict:
    """Lazily load the id->label map; empty dict (no-op) if absent or disabled."""
    global _WD_LABELS_CACHE
    if _labels_disabled():
        return {}
    if _WD_LABELS_CACHE is None:
        try:
            with open(_WD_LABELS_PATH, encoding="utf-8") as f:
                _WD_LABELS_CACHE = json.load(f)
        except (OSError, ValueError):
            _WD_LABELS_CACHE = {}
    return _WD_LABELS_CACHE


def expand_wikidata_labels(query: str) -> str:
    """Rewrite opaque Wikidata ids in a query string to ``label (ID)`` form.

    e.g. ``?v0 wdt:P31 wde:Q5`` -> ``?v0 instance of (P31) human (Q5)``.
    No-op on DBpedia queries (no P/Q ids) and when the label map is absent or a
    given id has no English label. Mirrors the expansion the SPARQLCE graph
    builder applies to node/edge labels, so text baselines get identical
    semantic content.
    """
    labels = _wikidata_labels()
    if not labels:
        return query

    def repl(m):
        wid = m.group(1) or m.group(2)
        lab = labels.get(wid)
        return f"{lab} ({wid})" if lab else m.group(0)

    return _WD_EXPAND_RE.sub(repl, query)


def normalize_query(query: str) -> str:
    return " ".join(query.split())


def is_valid_sparql(query: str) -> bool:
    query = normalize_query(query)
    return bool(query and query.lower().startswith(("select", "ask", "construct", "describe")))


def canonicalize_variables(query: str) -> str:
    """Rename SPARQL variables to ?v0, ?v1, ... by first-appearance order.

    Bakes variable-name invariance into the input so two queries that differ
    only by variable names produce identical graphs (and therefore identical
    embeddings).
    """
    seen = {}

    def repl(match):
        var = match.group(0)
        if var not in seen:
            seen[var] = f"?v{len(seen)}"
        return seen[var]

    return _VAR_RE.sub(repl, query)


def is_noop_filter_text(filter_body: str) -> bool:
    """True for FILTER(true) and trivially-true forms.

    Conservative on purpose: only matches the literal boolean ``true``
    (case-insensitive), with optional surrounding whitespace and one layer of
    parentheses. More aggressive simplification (e.g. ``1=1``, ``?x = ?x``)
    can silently change semantics on edge cases, so it is not attempted here.
    """
    body = filter_body.strip()
    # strip a single outer pair of parentheses if present
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1].strip()
    return bool(_TRUE_RE.match(body))

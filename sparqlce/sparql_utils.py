"""SPARQL utilities for parsing and validation."""

import re


_VAR_RE = re.compile(r'\?[a-zA-Z_]\w*')
_TRUE_RE = re.compile(r'^\s*true\s*$', re.IGNORECASE)


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

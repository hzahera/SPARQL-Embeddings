"""dbpedia.py
DBpedia-specific input construction: IRI humanization, a graph builder, and the
edge-type vocabulary the typed GATv2 layers key on.

Why a DBpedia-specific path exists at all
-----------------------------------------
Measured over the shipped corpus (4,713 anchors / 39,520 positives, all 4,727
distinct query strings):

* **74.3%** of positives share their anchor's `structural_template`, i.e. they
  are the SAME query shape with a different constant. Masking only the
  `ENT:`/`LIT:` node labels makes **71.9%** of anchor-positive graph pairs
  byte-identical; erasing node labels entirely makes **84.0%** identical.
* Graphs are tiny and near-uniform: mean 4.42 nodes / 8.45 edges, 46.5% have
  <= 3 nodes, and the whole corpus collapses to **84** distinct label-free
  shapes (top 5 cover 73.7% of queries).
* **60%** of all edges carry the single label `join`, from the pairwise
  (clique) join wiring.

So on DBpedia the discriminating variable is *which constants a query mentions*
and whether those constants have the same extension in the graph -- a knowledge
question -- not the query's shape. Structure-heavy machinery is invariant on
three quarters of the supervision, and mean-pooling a 4-node graph dilutes the
one node that actually differs to ~1/4 of the signal. This module attacks that
directly:

1. `DBpediaGraphBuilder` **humanizes** IRI local names (`dbpr:United_States` ->
   `United States (dbpr:United_States)`, `dbpo:birthPlace` -> `birth place
   (dbpo:birthPlace)`), so the text encoder's pretraining knowledge about the
   *entity* is reachable instead of being shredded into subword fragments of a
   snake_cased identifier. This is what lets a model learn that two country
   constants have the same `rdf:type` extension.
2. It defaults to `join_hubs=True`. Measured on the 4,713 anchors: pairwise
   join wiring leaves **38.3%** of DBpedia graphs at diameter <= 1, where the
   second GATv2 layer has nothing to propagate; hub wiring drops that to
   **3.4%**.
3. It labels every edge with a **type id** (`edge_type_id`), which is what
   `DBpediaSPARQLEncoder`'s typed edge embeddings consume. The pre-existing
   `HybridSPARQLEncoder` also has typed-edge code, but nothing in the repo ever
   populated `data.edge_type`, so those parameters have never trained.

FAIRNESS: humanization is a preprocessing step, not an architecture. If
SparqlCE gets it, the text baselines MUST get it too, or the comparison credits
the model for the preprocessing -- the same trap the Wikidata label expansion
already documents in `graph_builder.py`. Use `humanize_dbpedia_query()` to
apply the identical transform to a baseline's input text, and the shared
kill-switch `SPARQLCE_NO_DBPEDIA_HUMANIZE=1` to disable it for *all* models at
once (the only fair way to run the "without humanization" ablation).
"""
from __future__ import annotations

import os
import re
from urllib.parse import unquote

from .graph_builder import SPARQLGraphBuilder

# Shared kill-switch, mirroring SPARQLCE_NO_WIKIDATA_LABELS. Read at call time
# rather than import time so a notebook can flip it between runs.
_HUMANIZE_ENV = "SPARQLCE_NO_DBPEDIA_HUMANIZE"


def humanization_disabled() -> bool:
    return os.environ.get(_HUMANIZE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


# ── IRI humanization ─────────────────────────────────────────────────────────

_SEPARATORS_RE = re.compile(r"[_\-]+")
# A camel hump inside a word: the capital starts an ordinary word, so it is
# lowercased (`birthPlace` -> `birth place`). Kept separate from the two rules
# below so that capitals which are NOT humps keep their case.
_CAMEL_WORD_RE = re.compile(r"(?<=[a-z0-9])([A-Z])(?=[a-z])")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ACRONYM_RE = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_WS_RE = re.compile(r"\s+")


def local_name_to_words(local: str) -> str:
    """`United_States` -> `United States`; `birthPlace` -> `birth place`.

    Percent-escapes are decoded first (DBpedia resource names carry them, e.g.
    `Ma%C3%ADra`), then underscores/hyphens become spaces and camel-case and
    acronym boundaries are split. Case is preserved except at a camel hump, so
    `USPresident` stays `US President` while `birthPlace` becomes `birth place`.
    Returns the input unchanged when there is nothing to split, which is how the
    caller detects a no-op. (E5's tokenizer is uncased, so casing is cosmetic to
    the model -- it is kept tidy for the humans reading node labels in a debug
    dump.)
    """
    s = unquote(local)
    s = _SEPARATORS_RE.sub(" ", s)
    s = _CAMEL_WORD_RE.sub(lambda m: " " + m.group(1).lower(), s)
    s = _ACRONYM_RE.sub(" ", _CAMEL_RE.sub(" ", s))
    return _WS_RE.sub(" ", s).strip()


def humanize_term(cleaned: str) -> str:
    """Append a word-split rendering of an already-`_clean`ed term.

    The original token is KEPT in parentheses, exactly as the Wikidata label
    expansion does: the prose feeds the encoder's world knowledge, the raw token
    stays available as an exact-match anchor, and two distinct entities that
    happen to humanize to the same words remain distinguishable.

    No-ops (returns `cleaned` unchanged) when humanization is disabled, when the
    term already contains whitespace (a literal, not an identifier), or when
    splitting changes nothing (`rdf:type`, `owl:Class`).
    """
    if humanization_disabled() or not cleaned or " " in cleaned:
        return cleaned
    local = cleaned.split(":", 1)[1] if ":" in cleaned else cleaned
    words = local_name_to_words(local)
    if not words or words == local:
        return cleaned
    return f"{words} ({cleaned})"


_IRI_RE = re.compile(r"<[^>\s]+>")
_QNAME_RE = re.compile(r"\b([A-Za-z][\w.\-]*):([A-Za-z0-9_%\-.()]+)")
_PREFIX_DECL_RE = re.compile(r"PREFIX\s+\w*:\s*<[^>]*>", re.IGNORECASE)


def humanize_dbpedia_query(query: str) -> str:
    """Apply the graph builder's humanization to a raw query STRING.

    This is the function a text baseline calls so that it reads the same
    vocabulary SparqlCE reads. It rewrites IRIs and qnames in place and leaves
    PREFIX declarations alone (rewriting a declaration would change what the
    qnames resolve to, and the declaration is boilerplate no model needs).

    The result is NOT parseable SPARQL -- `dbpr:X` becomes `dbpr:X (words)`, so
    the term is followed by a parenthesised gloss. It is input for a text
    encoder only: never hand it to an endpoint, a graph builder, or
    `structural_template`. The graph path does not use this function; it
    humanizes per-term inside `DBpediaGraphBuilder._clean`, where the query has
    already been parsed.
    """
    if humanization_disabled():
        return query

    decls = []

    def _stash(m):
        decls.append(m.group(0))
        return f"\x00{len(decls) - 1}\x00"

    body = _PREFIX_DECL_RE.sub(_stash, query)

    def _iri(m):
        raw = m.group(0)[1:-1]
        local = raw.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        words = local_name_to_words(local)
        return m.group(0) if not words or words == local else f"<{raw}> ({words})"

    def _qname(m):
        words = local_name_to_words(m.group(2))
        return m.group(0) if words == m.group(2) else f"{m.group(0)} ({words})"

    body = _QNAME_RE.sub(_qname, _IRI_RE.sub(_iri, body))
    return re.sub(r"\x00(\d+)\x00", lambda m: decls[int(m.group(1))], body)


# ── Edge-type vocabulary ─────────────────────────────────────────────────────
# Index 0 is "other" so an unmapped label degrades to a learned generic offset
# rather than silently colliding with a real relation. Keep this list APPEND-ONLY:
# the ids are baked into trained `edge_type_emb` rows, so reordering it
# reinterprets every checkpoint.
DBPEDIA_EDGE_TYPES = [
    "other", "triple", "join", "path", "filter", "optional", "union",
    "values", "bind", "project", "group", "minus", "exists", "token", "self",
]
NUM_DBPEDIA_EDGE_TYPES = len(DBPEDIA_EDGE_TYPES)
_EDGE_TYPE2IDX = {t: i for i, t in enumerate(DBPEDIA_EDGE_TYPES)}

_EDGE_LABEL_TO_TYPE = {
    "join": "join", "join_in": "join", "join_out": "join",
    "path_in": "path", "path_out": "path",
    "uses_filter": "filter",
    "optional_scope": "optional",
    "union_scope": "union",
    "binds": "values", "value": "values",
    "defines": "bind", "used_in_bind": "bind", "bind_value": "bind",
    "projects": "project", "aggregated_by": "project",
    "groups_by": "group",
    "minus_scope": "minus",
    "exists_scope": "exists",
    "next": "token",
    "self": "self",
}


def edge_type_id(label: str) -> int:
    """Map a graph-builder edge label to its `DBPEDIA_EDGE_TYPES` index.

    Predicate edges are emitted as `PRED:<predicate>`, so they are matched by
    prefix; everything else is a fixed structural label.
    """
    if label.startswith("PRED:"):
        return _EDGE_TYPE2IDX["triple"]
    return _EDGE_TYPE2IDX.get(_EDGE_LABEL_TO_TYPE.get(label, "other"), 0)


# Node kinds that carry a CONSTANT -- the axis-C variable. `DBpediaSPARQLEncoder`
# pools exactly these nodes into its dedicated constant stream. PATH nodes are
# predicates, not constants, and VAR nodes are canonicalized away by design.
CONSTANT_NODE_PREFIXES = ("ENT:", "LIT:")


def is_constant_label(label: str) -> bool:
    return label.startswith(CONSTANT_NODE_PREFIXES)


# ── Graph builder ────────────────────────────────────────────────────────────


class DBpediaGraphBuilder(SPARQLGraphBuilder):
    """`SPARQLGraphBuilder` with humanized terms and hub-wired joins.

    Only `_clean` is overridden, so every construct the base builder
    understands (paths, FILTER, OPTIONAL, VALUES, BIND, projection, modifiers,
    negation scopes) keeps its exact structure -- terms inside them are simply
    rendered as words as well as ids. Property paths inherit this for free,
    because the base `_clean_path` cleans atom-wise through `_clean`.

    `join_hubs` defaults to True here (the base default is False, kept for
    checkpoint compatibility). Both settings change the graph, so a model must
    be trained and evaluated with the same one -- `graph_config` in the
    checkpoint records it.
    """

    def __init__(self, join_hubs: bool = True):
        super().__init__(join_hubs=join_hubs)

    def _clean(self, term):
        return humanize_term(super()._clean(term))

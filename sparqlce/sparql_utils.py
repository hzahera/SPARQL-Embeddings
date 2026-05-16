"""SPARQL utilities for parsing and validation."""

import re


def normalize_query(query: str) -> str:
    return " ".join(query.split())


def is_valid_sparql(query: str) -> bool:
    query = normalize_query(query)
    return bool(query and query.lower().startswith(("select", "ask", "construct", "describe")))

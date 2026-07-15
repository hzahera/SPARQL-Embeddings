#!/usr/bin/env python3
"""
Extract unique SPARQL queries from hard triplet datasets.

Usage:
  python scripts/extract_unique_queries.py --input data/hard_triplets_tfidf_dbpedia.json \
    --output data/unique_queries_dbpedia.json
"""
import json
import argparse


def extract_unique_queries(input_path, output_path):
    """Extract unique queries from a hard triplets JSON file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    unique = set()
    for item in data:
        if 'query' in item:
            unique.add(item['query'])
        for key in ('positive_queries', 'negative_queries'):
            for entry in item.get(key, []):
                if isinstance(entry, dict):
                    unique.add(entry.get('query', ''))
                else:
                    unique.add(str(entry))

    # Remove empty strings if any
    unique.discard('')

    # Write as JSON array
    with open(output_path, 'w', encoding='utf-8') as out:
        json.dump(sorted(list(unique)), out, indent=2, ensure_ascii=False)

    print(f'Extracted {len(unique)} unique queries from {input_path}')
    print(f'Output written to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract unique SPARQL queries from hard triplet datasets.')
    parser.add_argument('--input', required=True, help='Path to input hard triplets JSON file')
    parser.add_argument('--output', required=True, help='Path to output unique queries JSON file')
    args = parser.parse_args()

    extract_unique_queries(args.input, args.output)

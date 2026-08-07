#!/usr/bin/env python3
"""Smoke-test the --arch dbpedia path end to end on a handful of real queries.

Runs on CPU in well under a minute and checks the things a shape bug would
break silently: the collate emits the new tensors, every ablation forwards,
embeddings come out L2-normalised at the requested width, the calibrated loss
backpropagates into the constant/graph branches, and a save/load round-trip
through SPARQLEmbedder reproduces the same vectors.

    python scripts/smoke_test_dbpedia_arch.py
"""
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from torch_geometric.data import Batch                              # noqa: E402

from sparqlce.dbpedia import DBpediaGraphBuilder, edge_type_id      # noqa: E402
from sparqlce.dbpedia_encoder import (                              # noqa: E402
    DEFAULT_OUTPUT_DIM, ABLATIONS, DBpediaSPARQLEncoder, calibrated_infonce)
from sparqlce.dataset import SemanticSPARQLDataset                  # noqa: E402

DATA = ROOT_DIR / "datasets" / "dbpedia-triplets-dataset.json"
OUT_DIM = 64          # small, so the test stays fast; the real default is 768


def main():
    if not DATA.exists():
        raise SystemExit(f"{DATA} not found (see datasets/README.md)")

    # A 4-anchor slice of the real corpus, written to a temp file so the real
    # dataset class does the loading -- testing a hand-built Batch would not
    # exercise the collate path that a training run actually uses.
    with open(DATA) as f:
        subset = json.load(f)[:4]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "subset.json"
        path.write_text(json.dumps(subset))

        ds = SemanticSPARQLDataset(
            str(path), graph_builder=DBpediaGraphBuilder(join_hubs=True),
            in_batch_negatives=True, deterministic=True,
            query_stream=True, constant_stream=True, edge_type_fn=edge_type_id)
        items = [ds[i] for i in range(len(ds))]
        anchors = Batch.from_data_list([a for a, _, _ in items])
        positives = Batch.from_data_list([p for _, p, _ in items])
        pos_j = torch.tensor([j for _, _, j in items], dtype=torch.float32)

        for field in ("x_is_const", "edge_type", "q_ids", "q_len"):
            assert hasattr(anchors, field), f"collate did not emit {field}"
        assert anchors.x_is_const.numel() == anchors.num_nodes
        assert anchors.edge_type.numel() == anchors.edge_index.size(1)
        assert int(anchors.q_len.sum()) == anchors.q_ids.numel()
        print(f"collate OK: {anchors.num_graphs} graphs, "
              f"{anchors.num_nodes} nodes ({int(anchors.x_is_const.sum())} constants), "
              f"{anchors.edge_index.size(1)} edges")

        for ablation in ABLATIONS:
            torch.manual_seed(0)
            model = DBpediaSPARQLEncoder(ablation=ablation, output_dim=OUT_DIM)
            model.eval()
            with torch.no_grad():
                emb, share = model(anchors)
            assert emb.shape == (len(items), OUT_DIM), emb.shape
            norms = emb.norm(dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), norms
            print(f"  {ablation:<10} -> {tuple(emb.shape)}  unit-norm OK  "
                  f"non-text share {share:.3f}")

        # The loss must reach the branches that exist for DBpedia; a zero grad
        # here means the constant/graph stream is decorative.
        torch.manual_seed(0)
        model = DBpediaSPARQLEncoder(ablation="full", output_dim=OUT_DIM)
        a_emb, _ = model(anchors)
        p_emb, _ = model(positives)
        loss, ce, reg = calibrated_infonce(a_emb, p_emb, pos_j, 0.05, 0.3,
                                           calibrate_inbatch=True)
        loss.backward()
        for name in ("gamma_c", "gamma_g", "const_pool.q.weight", "graph_proj.weight",
                     "edge_type_emb.weight"):
            g = dict(model.named_parameters())[name].grad
            assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, \
                f"no gradient reached {name}"
        print(f"loss OK: total={loss.item():.4f} ce={ce.item():.4f} reg={reg.item():.4f}; "
              f"gradients reach the constant, graph and edge-type parameters")

        # Round-trip: what training writes must be what evaluation rebuilds.
        ckpt = Path(tmp) / "best_model.pt"
        torch.save({"model_state_dict": model.state_dict(), "ablation": "full",
                    "arch": "dbpedia", "output_dim": OUT_DIM,
                    "graph_config": {"query_stream": True, "join_hubs": True,
                                     "query_max_length": 384,
                                     "constant_stream": True, "edge_types": True,
                                     "humanize": True}}, ckpt)
        from sparqlce.embedder import SPARQLEmbedder
        emb_wrapper = SPARQLEmbedder(str(ckpt))
        queries = [g["anchor"] for g in ds.groups]
        vecs = emb_wrapper.encode(queries)
        model.eval()
        with torch.no_grad():
            direct, _ = model(anchors)
        delta = (torch.as_tensor(vecs) - direct).abs().max().item()
        assert delta < 1e-4, f"SPARQLEmbedder disagrees with the trained model by {delta}"
        print(f"checkpoint round-trip OK: max |Δ| = {delta:.2e}")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()

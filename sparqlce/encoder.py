"""hybrid_encoder.py
HybridSPARQLEncoder: combines E5-base-v2 mean-pooled embeddings with a
relation-typed GATv2 graph encoder and a gated fusion layer to produce
L2-normalised result-aware query embeddings.

Improvements over the CodeBERT+CLS baseline:
  1. Backbone: E5-base-v2 with mean pooling (not CLS) — matches the
     strong text baseline and uses E5's query: prefix convention.
  2. Typed edge embeddings: SPARQL algebra relation types (triple,
     filter, optional, union, path, coref) are encoded as learned
     offsets added to the text-encoded edge embedding, giving the GNN
     structured relational inductive bias without extra parameters.
  3. Two GATv2 layers with residual connection: deeper message passing
     for multi-hop SPARQL join patterns; residual prevents collapse.
  4. Gated fusion: a learned sigmoid gate decides per-query how much
     weight to give the text stream vs the graph stream, rather than
     a fixed concatenation + projection.
  5. Jaccard-calibration hook: forward() returns (embedding, gate_mean)
     so the training loop can log gate statistics without architectural
     changes.

``ablation`` modes (unchanged API):
  - "full"    (default): E5 node/edge embeddings → typed GATv2 × 2 →
    gated fusion with E5 text embedding → projected L2-normalised output.
  - "no_gnn": E5 mean-pooled text embedding only, no graph branch.
    Isolates whether GNN message passing adds anything.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import GATv2Conv, global_mean_pool

from .config import Config

# ── SPARQL algebra relation types ────────────────────────────────────────────
# Keep this list stable across runs; index = edge-type id stored in data.edge_type
EDGE_TYPES = ["triple", "filter", "optional", "union", "path", "coref"]
NUM_EDGE_TYPES = len(EDGE_TYPES)
EDGE_TYPE2IDX = {t: i for i, t in enumerate(EDGE_TYPES)}


class HybridSPARQLEncoder(nn.Module):
    """E5-base-v2 + typed GATv2 encoder with gated fusion for SPARQL queries.

    Changes vs. previous version
    ----------------------------
    * Text encoding: CLS → mean pooling over non-padding tokens (E5 style).
    * Edge encoding: raw text embedding + learned edge-type offset vector.
    * GNN depth: 1 GATv2 layer → 2 GATv2 layers with residual projection.
    * Fusion: concat + linear → sigmoid gate over (text, graph) streams.
    * Ablation API: identical to previous — ``ablation`` kwarg unchanged.
    """

    def __init__(self, ablation: str = "full", output_dim: int | None = None):
        """
        Parameters
        ----------
        output_dim : int | None
            Final embedding width. Defaults to ``Config.OUTPUT_DIM``. Pass it
            explicitly when loading a checkpoint trained at a different width —
            the encoder's projection/gate/LayerNorm shapes depend on it, so a
            mismatch makes ``load_state_dict`` fail. Training checkpoints record
            the value they used (see scripts/train_sparqlce.py).
        """
        super().__init__()
        assert ablation in ("full", "no_gnn"), f"unknown ablation: {ablation}"
        self.ablation = ablation

        # ── 1. Text backbone: E5-base-v2, fully trainable ───────────────────
        self.text_enc = AutoModel.from_pretrained(Config.MODEL_NAME)
        for p in self.text_enc.parameters():
            p.requires_grad = True

        d = Config.EMBEDDING_DIM      # 768
        h = Config.HIDDEN_DIM         # 256
        heads = Config.GNN_HEADS      # 4
        out = Config.OUTPUT_DIM if output_dim is None else output_dim
        self.output_dim = out
        gnn_dim = h * heads           # 1024

        if ablation == "full":
            # ── 2. Typed edge embedding offsets ─────────────────────────────
            # Each SPARQL relation type gets a learned d-dim offset added to
            # the text-encoded edge label before GATv2 message passing.
            self.edge_type_emb = nn.Embedding(NUM_EDGE_TYPES, d)
            nn.init.zeros_(self.edge_type_emb.weight)   # start as identity

            # ── 3. Two GATv2 layers with residual ───────────────────────────
            self.gnn1 = GATv2Conv(d, h, heads=heads, edge_dim=d, concat=True)
            self.gnn2 = GATv2Conv(gnn_dim, h, heads=heads, edge_dim=d, concat=True)
            # Residual alignment: project node input d → gnn_dim so we can
            # add the skip connection after gnn1 without dimension mismatch.
            self.res_proj = nn.Linear(d, gnn_dim, bias=False)

            # ── 4. Gated fusion ──────────────────────────────────────────────
            # gate ∈ (0,1)^out decides text vs graph contribution per query.
            self.gate_fc = nn.Linear(d + gnn_dim, out)
            self.text_proj  = nn.Linear(d,       out, bias=False)
            self.graph_proj = nn.Linear(gnn_dim, out, bias=False)
        else:
            # no_gnn: text-only projection
            self.text_proj = nn.Linear(d, out, bias=False)

        # Final layer norm before L2 normalisation (stabilises training)
        self.ln = nn.LayerNorm(out)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _mean_pool(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """E5-style mean pooling over non-padding token positions."""
        out = self.text_enc(input_ids=ids, attention_mask=mask)
        h = out.last_hidden_state                     # [B, L, d]
        m = mask.unsqueeze(-1).float()                # [B, L, 1]
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)   # [B, d]

    def _query_stream_pool(self, data) -> torch.Tensor | None:
        """Mean-pool the whole-query token stream, padded per batch.

        Returns None when the batch carries no query stream, so callers fall
        back to the node-label mean and pre-existing checkpoints keep scoring
        exactly as they did.

        q_ids arrives flat (sum of all lengths) with q_len giving the split, so
        each batch is padded only to ITS longest query rather than to a fixed
        budget. The padding VALUE is irrelevant because attention_mask is zero
        there and mean pooling divides by the mask sum -- which is why 0 is safe
        without consulting the tokenizer for its pad id.
        """
        q_ids = getattr(data, "q_ids", None)
        if q_ids is None:
            return None
        lengths = data.q_len.view(-1)
        chunks = torch.split(q_ids, lengths.tolist())
        padded = torch.nn.utils.rnn.pad_sequence(
            chunks, batch_first=True, padding_value=0)          # [B, Lmax]
        mask = (torch.arange(padded.size(1), device=padded.device)[None, :]
                < lengths.view(-1, 1)).long()                   # [B, Lmax]
        return self._mean_pool(padded, mask)

    def _encode_edges(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        edge_type: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mean-pool edge label tokens and add typed relation offset."""
        e = self._mean_pool(ids, mask)                # [E, d]
        if edge_type is not None:
            e = e + self.edge_type_emb(edge_type)     # broadcast typed offset
        return e

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, data) -> tuple[torch.Tensor, float]:
        """
        Parameters
        ----------
        data : torch_geometric.data.Batch
            Must contain:
              x_ids, x_mask          — node token ids / attention mask
              edge_attr_ids,
              edge_attr_mask         — edge label token ids / mask
              edge_index             — [2, E] COO edge index
              batch                  — node-to-graph assignment
            Optional (new):
              edge_type              — [E] LongTensor of EDGE_TYPE2IDX values

        Returns
        -------
        emb : Tensor [B, OUTPUT_DIM], L2-normalised
        gate_mean : float  (mean gate value across batch; log for diagnostics)
        """
        # Node text embeddings
        z_text_nodes = self._mean_pool(data.x_ids, data.x_mask)  # [N, d]

        if self.ablation == "no_gnn":
            # Text-only path. With q_ids this is a flat encoder over the query
            # string -- the graph-free control, trained by our own recipe.
            z_q = self._query_stream_pool(data)
            z_pooled = (z_q if z_q is not None
                        else global_mean_pool(z_text_nodes, data.batch))  # [B, d]
            emb = self.ln(self.text_proj(z_pooled))
            return F.normalize(emb, p=2, dim=1), 0.5  # gate_mean = 0.5 sentinel

        # ── Graph branch ─────────────────────────────────────────────────────
        edge_type = getattr(data, "edge_type", None)
        e = self._encode_edges(
            data.edge_attr_ids, data.edge_attr_mask, edge_type
        )                                                          # [E, d]

        # GATv2 layer 1 + residual
        x1 = F.elu(self.gnn1(z_text_nodes, data.edge_index, edge_attr=e))
        x1 = x1 + self.res_proj(z_text_nodes)                     # residual

        # GATv2 layer 2 + residual
        x2 = F.elu(self.gnn2(x1, data.edge_index, edge_attr=e))
        x2 = x2 + x1                                              # residual

        # Graph embedding: global mean pool over nodes
        z_graph = global_mean_pool(x2, data.batch)                 # [B, gnn_dim]

        # Text embedding at query level.
        #
        # Preferred: encode the whole query STRING (data.q_ids), the way a flat
        # text baseline does. Falling back to a mean over node-label embeddings
        # -- the original behaviour, kept so pre-existing checkpoints still load
        # and score identically -- makes the text stream a bag of node labels
        # with no token order, no syntax and no query-level context, i.e.
        # strictly less informed than the E5-FT control it is measured against.
        # That also made the two fusion inputs near-duplicates on graphs whose
        # diameter is 1, since z_graph then pools the same node embeddings.
        # With q_ids present the gate saturating at 1 reproduces the flat text
        # encoder exactly, so the graph branch can only add to it.
        z_q = self._query_stream_pool(data)
        z_text = (z_q if z_q is not None
                  else global_mean_pool(z_text_nodes, data.batch))  # [B, d]

        # ── Gated fusion ─────────────────────────────────────────────────────
        gate = torch.sigmoid(self.gate_fc(
            torch.cat([z_text, z_graph], dim=-1)
        ))                                                         # [B, out]
        emb = gate * self.text_proj(z_text) + (1.0 - gate) * self.graph_proj(z_graph)
        gate_mean = gate.mean().item()

        emb = self.ln(emb)
        return F.normalize(emb, p=2, dim=1), gate_mean
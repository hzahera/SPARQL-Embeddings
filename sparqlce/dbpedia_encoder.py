"""dbpedia_encoder.py
`DBpediaSPARQLEncoder` — a DBpedia-specific query encoder, plus the calibrated
contrastive loss it is meant to be trained with.

What it changes, and why each change is there
---------------------------------------------
Every item below answers a measured failure of `HybridSPARQLEncoder` on the
DBpedia corpus (measurements in `sparqlce/dbpedia.py`'s docstring).

1. **Query text is the trunk, not one of two peers.** The text stream reads the
   whole query string (`q_ids`); there is no node-label-mean fallback. In
   `HybridSPARQLEncoder` that fallback is the DEFAULT (`query_stream=False`),
   which makes the text side a bag of truncated node labels -- no token order,
   no syntax -- i.e. strictly less informed than the flat E5 baseline it is
   scored against. Here it is not an option.

2. **Residual fusion instead of a convex gate.** The output is
   `text_proj(q) + g_c * const + g_g * graph`, with `g_*` learned per-dimension
   LayerScale parameters initialised small. The convex gate
   `g * text + (1-g) * graph` spends capacity *trading off* two streams that on
   DBpedia are near-duplicates (a mean over 4 node embeddings vs. a mean-pooled
   GNN over the same 4 nodes), and it can only reach the flat text encoder in
   the limit `g -> 1`, which also switches the graph off. A residual starts
   near the flat encoder and moves away from it only where that pays.

3. **A dedicated constant stream.** ENT/LIT nodes are pooled separately by
   attention conditioned on the query embedding. This is the whole point: with
   71.9% of anchor-positive graph pairs identical once constants are masked,
   the constant is the discriminating variable, and `global_mean_pool` over a
   4-node graph dilutes it to ~1/4 of the vector. Attention conditioned on the
   query lets the model ask "which constant matters *for this template*",
   which is exactly the axis-C question.

4. **Attention pooling instead of mean pooling** on the graph stream, for the
   same dilution reason.

5. **Typed edge embeddings that are actually populated.** `edge_type` is
   consumed here and emitted by `SemanticSPARQLDataset(edge_type_fn=...)`.
   The equivalent code in `HybridSPARQLEncoder` reads `getattr(data,
   "edge_type", None)`, and nothing in the repo ever sets it, so its
   `edge_type_emb` has never received a gradient.

6. **Default width 768**, matching the fine-tuned E5 baseline. The 128-d
   default elsewhere compares a 128-d representation against a 6x wider one,
   which is a capacity difference reported as an architectural one.

Ablations (`ablation=`) isolate each addition against the same trunk:
  * `full`       — text + constant + graph
  * `no_const`   — text + graph (isolates the constant stream)
  * `no_gnn`     — text + constant (isolates message passing)
  * `text_only`  — the trunk alone; a flat E5 encoder trained by OUR recipe,
                   which is the correct architecture-controlled baseline.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch_geometric.utils import softmax as scatter_softmax
from transformers import AutoModel

from .config import Config
from .dbpedia import NUM_DBPEDIA_EDGE_TYPES

# Deliberately not Config.OUTPUT_DIM (128): the fine-tuned E5 baseline embeds at
# 768, and matching it is what makes the comparison architectural.
DEFAULT_OUTPUT_DIM = 768

ABLATIONS = ("full", "no_const", "no_gnn", "text_only")


class ConditionedAttentionPool(nn.Module):
    """Multi-head attention pooling of a node set, conditioned on a context.

    Pools rows of ``h`` into one vector per graph, weighting each row by its
    agreement with that graph's context vector (here: the query embedding).
    Graphs with no rows pool to zeros, which is the correct neutral element for
    a residual branch -- a query with no constants simply contributes nothing
    through the constant stream instead of contributing noise.
    """

    def __init__(self, dim: int, ctx_dim: int, heads: int = 4):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.q = nn.Linear(ctx_dim, dim)

    def forward(self, h: torch.Tensor, index: torch.Tensor,
                ctx: torch.Tensor, num_graphs: int) -> torch.Tensor:
        if h.numel() == 0:
            return ctx.new_zeros(num_graphs, self.dim)
        m, H, D = h.size(0), self.heads, self.head_dim
        k = self.k(h).view(m, H, D)
        v = self.v(h).view(m, H, D)
        q = self.q(ctx).view(-1, H, D)[index]                    # [M, H, D]
        score = (q * k).sum(-1) / math.sqrt(D)                   # [M, H]
        alpha = scatter_softmax(score, index, num_nodes=num_graphs)
        weighted = (alpha.unsqueeze(-1) * v).reshape(m, H * D)
        return global_add_pool(weighted, index, size=num_graphs)


class DBpediaSPARQLEncoder(nn.Module):
    """E5 trunk + conditioned constant stream + typed GATv2, fused residually."""

    def __init__(
        self,
        ablation: str = "full",
        output_dim: int | None = None,
        num_edge_types: int = NUM_DBPEDIA_EDGE_TYPES,
        heads: int = 4,
        gamma_init: float = 0.1,
    ):
        super().__init__()
        assert ablation in ABLATIONS, f"unknown ablation: {ablation}"
        self.ablation = ablation
        self.use_const = ablation in ("full", "no_gnn")
        self.use_graph = ablation in ("full", "no_const")

        self.text_enc = AutoModel.from_pretrained(Config.MODEL_NAME)

        d = Config.EMBEDDING_DIM                 # 768
        h = Config.HIDDEN_DIM                    # 256
        out = DEFAULT_OUTPUT_DIM if output_dim is None else output_dim
        self.output_dim = out
        gnn_dim = h * heads                      # 1024

        self.text_proj = nn.Linear(d, out, bias=False)

        if self.use_const:
            self.const_pool = ConditionedAttentionPool(d, d, heads=heads)
            self.const_proj = nn.Linear(d, out, bias=False)
            # LayerScale: start close to the flat text encoder and earn the way
            # out. A large init here is how the graph branch drowned the text
            # branch in the gated design before it had learned anything.
            self.gamma_c = nn.Parameter(torch.full((out,), gamma_init))

        if self.use_graph:
            self.edge_type_emb = nn.Embedding(num_edge_types, d)
            nn.init.zeros_(self.edge_type_emb.weight)   # starts as identity
            self.gnn1 = GATv2Conv(d, h, heads=heads, edge_dim=d, concat=True)
            self.gnn2 = GATv2Conv(gnn_dim, h, heads=heads, edge_dim=d, concat=True)
            self.res_proj = nn.Linear(d, gnn_dim, bias=False)
            self.graph_pool = ConditionedAttentionPool(gnn_dim, d, heads=heads)
            self.graph_proj = nn.Linear(gnn_dim, out, bias=False)
            self.gamma_g = nn.Parameter(torch.full((out,), gamma_init))

        self.ln = nn.LayerNorm(out)

    def enable_gradient_checkpointing(self) -> None:
        """Trade ~30% step time for a large cut in backbone activation memory.

        Effectively mandatory when training with hard negatives. This encoder
        pushes the WHOLE QUERY through the backbone for every graph in the
        batch, so a step with B anchors, B positives and B*max_neg negatives
        encodes B*(2 + max_neg) query streams at up to `query_max_length`
        tokens each, on top of their node and edge labels. At B=32,
        max_neg=16 that is 576 query streams -- ~196k tokens of retained
        activations through a 12-layer backbone, which exhausts an 80GB A100
        on the first step. Checkpointing keeps only layer boundaries.

        `use_reentrant=False` is required: the reentrant implementation drops
        gradients for inputs that do not require grad, which is exactly the
        case here (token ids).
        """
        self.text_enc.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(self.text_enc, "config"):
            self.text_enc.config.use_cache = False

    # ── helpers ─────────────────────────────────────────────────────────────

    def _mean_pool(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.text_enc(input_ids=ids, attention_mask=mask)
        m = mask.unsqueeze(-1).float()
        return (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def _query_pool(self, data) -> torch.Tensor:
        """Mean-pool the whole-query token stream, padded per batch.

        Unlike `HybridSPARQLEncoder` this raises instead of falling back to a
        node-label mean: the fallback is a silently different (and weaker)
        model, and every result produced under it would be attributed to the
        architecture rather than to the input it was handed.
        """
        q_ids = getattr(data, "q_ids", None)
        if q_ids is None:
            raise ValueError(
                "DBpediaSPARQLEncoder requires the whole-query text stream. "
                "Build the dataset with query_stream=True (train with "
                "--query-stream, which --arch dbpedia sets for you)."
            )
        lengths = data.q_len.view(-1)
        chunks = torch.split(q_ids, lengths.tolist())
        padded = nn.utils.rnn.pad_sequence(chunks, batch_first=True, padding_value=0)
        mask = (torch.arange(padded.size(1), device=padded.device)[None, :]
                < lengths.view(-1, 1)).long()
        return self._mean_pool(padded, mask)

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, data) -> tuple[torch.Tensor, float]:
        """Returns (L2-normalised embedding [B, out], graph-share diagnostic).

        The second element replaces `HybridSPARQLEncoder`'s `gate_mean` and is
        the fraction of the output's norm contributed by the non-text streams —
        log it: if it stays near 0 the extra machinery is not being used, and
        the honest report is that a flat encoder does the same job.
        """
        z_q = self._query_pool(data)                              # [B, d]
        num_graphs = z_q.size(0)
        emb = self.text_proj(z_q)
        # Diagnostics only -- detached, so they neither retain activations nor
        # force a device sync inside the autograd graph.
        with torch.no_grad():
            text_norm = emb.detach().norm(dim=1).mean().clamp(min=1e-9)
            extra_norm = emb.new_zeros(())

        need_nodes = self.use_const or self.use_graph
        z_nodes = self._mean_pool(data.x_ids, data.x_mask) if need_nodes else None

        if self.use_const:
            sel = getattr(data, "x_is_const", None)
            if sel is None:
                raise ValueError(
                    "the constant stream needs per-node constant flags; build "
                    "the dataset with constant_stream=True")
            sel = sel.bool().view(-1)
            z_c = self.const_pool(z_nodes[sel], data.batch[sel], z_q, num_graphs)
            contrib = self.gamma_c * self.const_proj(z_c)
            emb = emb + contrib
            extra_norm = extra_norm + contrib.detach().norm(dim=1).mean()

        if self.use_graph:
            edge_type = getattr(data, "edge_type", None)
            e = self._mean_pool(data.edge_attr_ids, data.edge_attr_mask)
            if edge_type is not None:
                e = e + self.edge_type_emb(edge_type.view(-1))
            x1 = F.elu(self.gnn1(z_nodes, data.edge_index, edge_attr=e))
            x1 = x1 + self.res_proj(z_nodes)
            x2 = F.elu(self.gnn2(x1, data.edge_index, edge_attr=e))
            x2 = x2 + x1
            z_g = self.graph_pool(x2, data.batch, z_q, num_graphs)
            contrib = self.gamma_g * self.graph_proj(z_g)
            emb = emb + contrib
            extra_norm = extra_norm + contrib.detach().norm(dim=1).mean()

        emb = self.ln(emb)
        share = (float(extra_norm / (text_norm + extra_norm)) if need_nodes else 0.0)
        return F.normalize(emb, p=2, dim=1), share


# ── loss ─────────────────────────────────────────────────────────────────────


def calibrated_infonce(
    a_emb: torch.Tensor,
    p_emb: torch.Tensor,
    pos_j: torch.Tensor,
    temperature: float,
    lambda_reg: float,
    n_emb: torch.Tensor | None = None,
    neg_slices=None,
    neg_j: torch.Tensor | None = None,
    calibrate_inbatch: bool = False,
    share_negatives: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """InfoNCE + a **two-sided** Jaccard calibration term.

    The existing recipe regresses cosine onto Jaccard over POSITIVE pairs only.
    On DBpedia 78.9% of positives carry `answer_jaccard == 1.0` (mean 0.970), so
    that term is `MSE(cos, 1)` on four fifths of the batch: a pure pull-together
    with no matching push-apart. It is consistent with the geometry the repo
    already measured — SparqlCE sits at mean cosine 0.991 across every
    perturbation pair in `results/robustness/robustness_results.json`, which is
    high invariance and low discriminability written as one number.

    Here the same regression also covers the labelled hard negatives (target =
    their true Jaccard, typically 0), so the calibration has a direction to push
    in as well as one to pull in.

    `calibrate_inbatch` additionally regresses every off-diagonal in-batch pair
    to 0. That is only sound on the component splits, where two anchors from
    different components provably have disjoint answers; on a group split two
    in-batch anchors can be result-equivalent and the target would be wrong.
    Off by default for that reason.

    Returns (total, ce, reg) so the training loop can log the parts separately —
    a reg term that flatlines while ce falls means the calibration is doing
    nothing, which is precisely the failure this function exists to make
    visible.
    """
    a = F.normalize(a_emb, dim=1)
    p = F.normalize(p_emb, dim=1)
    b = a.size(0)
    logits = a @ p.T / temperature                      # [B, B], diag = positive

    sq_err = [(( a * p).sum(dim=1) - pos_j).pow(2)]

    if n_emb is not None and n_emb.size(0) > 0:
        n = F.normalize(n_emb, dim=1)
        if share_negatives:
            # Every anchor scored against EVERY mined negative in the batch, the
            # way finetune_e5_baseline.py's info_nce_loss already works. Without
            # it the baseline sees BATCH mined negatives per step and SparqlCE
            # sees --max-neg, a 64x gap at batch 128 with --max-neg 2. Free: the
            # negatives are already encoded, so this is one [B, N] matmul.
            logits = torch.cat([logits, a @ n.T / temperature], dim=1)
        else:
            max_neg = max((e - s) for s, e in neg_slices)
            hard = a.new_full((b, max_neg), float("-inf"))
            for i, (s, e) in enumerate(neg_slices):
                if e > s:
                    hard[i, : e - s] = a[i] @ n[s:e].T / temperature
            logits = torch.cat([logits, hard], dim=1)
        # NB the calibration below stays per-anchor regardless of
        # share_negatives: neg_j is the measured Jaccard of an anchor
        # against ITS OWN negatives. Another anchor's negative has no
        # known target here, so regressing it would invent one.
        if neg_j is not None:
            cos_neg = torch.cat([a[i] @ n[s:e].T for i, (s, e) in enumerate(neg_slices)
                                 if e > s])
            if cos_neg.numel():
                sq_err.append((cos_neg - neg_j.to(cos_neg.device)).pow(2))

    if calibrate_inbatch and b > 1:
        off = ~torch.eye(b, dtype=torch.bool, device=a.device)
        sq_err.append((a @ p.T)[off].pow(2))            # target 0

    ce = F.cross_entropy(logits, torch.arange(b, device=a.device))
    reg = torch.cat(sq_err).mean()
    return ce + lambda_reg * reg, ce, reg

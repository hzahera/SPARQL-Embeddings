from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .preprocessing import get_transform
from .utils import ensure_dir


_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


class HFEmbeddingModel(nn.Module):
    """A common trainable embedding interface over Hugging Face transformers."""

    def __init__(
        self,
        model_cfg: dict[str, Any],
        device: str | torch.device = "cuda",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required. Install with: pip install -r requirements.txt"
            ) from exc

        self.cfg = dict(model_cfg)
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        source = str(checkpoint_path or model_cfg["model_name_or_path"])
        trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
        dtype_name = str(model_cfg.get("torch_dtype", "float32")).lower()
        dtype = _DTYPE_MAP.get(dtype_name, torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer.padding_side = model_cfg.get("padding_side", "right")
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.transformer = AutoModel.from_pretrained(
            source,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )
        # If a pad token had to be added, make embeddings large enough.
        if len(self.tokenizer) > self.transformer.get_input_embeddings().num_embeddings:
            self.transformer.resize_token_embeddings(len(self.tokenizer))
        self.transformer.to(self.device)

        self.pooling = str(model_cfg.get("pooling", "mean"))
        self.prefix = str(model_cfg.get("prefix", ""))
        self.max_length = int(model_cfg.get("max_length", 512))
        self.input_transform_name = str(model_cfg.get("input_transform", "identity"))
        self.input_transform = get_transform(self.input_transform_name)

    def preprocess(self, texts: Iterable[str]) -> list[str]:
        return [self.prefix + self.input_transform(t) for t in texts]

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.preprocess(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-9)
        return summed / denom

    @staticmethod
    def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Works for both left and right padding by locating the final non-pad token.
        seq_len = attention_mask.size(1)
        last_idx = seq_len - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)
        rows = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        return last_hidden_state[rows, last_idx]

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            pooled = self._mean_pool(last_hidden_state, attention_mask)
        elif self.pooling == "cls":
            pooled = last_hidden_state[:, 0]
        elif self.pooling == "last_token":
            pooled = self._last_token_pool(last_hidden_state, attention_mask)
        else:
            raise ValueError(f"Unsupported pooling={self.pooling!r}")
        return F.normalize(pooled, p=2, dim=-1)

    def forward_texts(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenize(texts)
        outputs = self.transformer(**tokens, return_dict=True)
        if not hasattr(outputs, "last_hidden_state"):
            raise RuntimeError(
                f"Model output for {self.cfg.get('name')} has no last_hidden_state. "
                "Use a base AutoModel-compatible checkpoint or add a custom adapter."
            )
        return self._pool(outputs.last_hidden_state, tokens["attention_mask"])

    @torch.inference_mode()
    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        chunks: list[torch.Tensor] = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Encoding {self.cfg.get('name', 'model')}")
        for start in iterator:
            emb = self.forward_texts(texts[start : start + batch_size])
            chunks.append(emb.detach().cpu())
        if was_training:
            self.train()
        if not chunks:
            hidden = int(getattr(self.transformer.config, "hidden_size", 0))
            return torch.empty((0, hidden), dtype=torch.float32)
        return torch.cat(chunks, dim=0)

    def enable_gradient_checkpointing(self) -> None:
        fn = getattr(self.transformer, "gradient_checkpointing_enable", None)
        if callable(fn):
            fn()
        # Most decoder models should not cache during training.
        if hasattr(self.transformer.config, "use_cache"):
            self.transformer.config.use_cache = False

    def save_pretrained(self, path: str | Path) -> Path:
        path = ensure_dir(path)
        self.transformer.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        with open(path / "result_aware_model_config.json", "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2)
        return path

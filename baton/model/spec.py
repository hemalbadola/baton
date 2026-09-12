"""Model geometry and checkpoint tensor naming (PRD 5.1).

The engine implements its own decoder (decision D2), so one dataclass must carry
every number and flag the layer code needs. Per-model differences are flags here,
not subclasses, which keeps `DecoderLayer` a single class.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Hugging Face tensor names for the Llama / Qwen2 module tree. `{i}` is the
# absolute layer index. Both families share this tree, so one table covers both.
DEFAULT_TENSOR_NAMES: dict[str, str] = {
    "embed": "model.embed_tokens.weight",
    "input_layernorm": "model.layers.{i}.input_layernorm.weight",
    "q_proj": "model.layers.{i}.self_attn.q_proj.weight",
    "k_proj": "model.layers.{i}.self_attn.k_proj.weight",
    "v_proj": "model.layers.{i}.self_attn.v_proj.weight",
    "o_proj": "model.layers.{i}.self_attn.o_proj.weight",
    "q_bias": "model.layers.{i}.self_attn.q_proj.bias",
    "k_bias": "model.layers.{i}.self_attn.k_proj.bias",
    "v_bias": "model.layers.{i}.self_attn.v_proj.bias",
    "post_attention_layernorm": "model.layers.{i}.post_attention_layernorm.weight",
    "gate_proj": "model.layers.{i}.mlp.gate_proj.weight",
    "up_proj": "model.layers.{i}.mlp.up_proj.weight",
    "down_proj": "model.layers.{i}.mlp.down_proj.weight",
    "final_norm": "model.norm.weight",
    "lm_head": "lm_head.weight",
}

# Keys that exist only when `attn_bias` is set (Qwen 2.5).
BIAS_KEYS = ("q_bias", "k_bias", "v_bias")


@dataclass(frozen=True)
class ModelSpec:
    """Everything the decoder needs to know about one checkpoint."""

    n_layers: int
    hidden: int
    intermediate: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    rms_eps: float
    rope_theta: float
    rope_scaling: dict[str, Any] | None
    attn_bias: bool
    tie_embeddings: bool
    tensor_names: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TENSOR_NAMES))
    # Context length the RoPE tables are built for. Not a per-model difference in
    # the maths, but the layer code needs a bound to precompute cos/sin (PRD 5.2).
    max_position: int = 8192

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads:
            raise ValueError(f"n_heads {self.n_heads} not divisible by n_kv_heads {self.n_kv_heads}")
        if self.head_dim % 2:
            raise ValueError(f"head_dim {self.head_dim} must be even for RoPE")

    # --- derived geometry ------------------------------------------------

    @property
    def n_kv_groups(self) -> int:
        """Query heads per key/value head (grouped-query attention)."""
        return self.n_heads // self.n_kv_heads

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def rope_type(self) -> str:
        """`"default"`, or the scaling family named in config.json."""
        if not self.rope_scaling:
            return "default"
        scaling = self.rope_scaling
        return str(scaling.get("rope_type") or scaling.get("type") or "default")

    # --- tensor names ----------------------------------------------------

    def layer_names(self, i: int) -> dict[str, str]:
        """Checkpoint tensor names for absolute layer `i`, bias keys included only
        when the architecture has them."""
        out = {}
        for key, template in self.tensor_names.items():
            if "{i}" not in template:
                continue
            if key in BIAS_KEYS and not self.attn_bias:
                continue
            out[key] = template.format(i=i)
        return out

    def head_names(self) -> dict[str, str]:
        """Names of the tensors only Nk holds: final norm and `lm_head`."""
        names = {"final_norm": self.tensor_names["final_norm"]}
        # A tied head has no `lm_head.weight` of its own; the embedding is reused.
        names["lm_head"] = (
            self.tensor_names["embed"] if self.tie_embeddings else self.tensor_names["lm_head"]
        )
        return names

    def range_names(self, start: int, end: int, *, embed: bool = False, head: bool = False) -> list[str]:
        """Every checkpoint tensor a worker holding layers `[start, end)` must fetch."""
        names: list[str] = []
        if embed:
            names.append(self.tensor_names["embed"])
        for i in range(start, end):
            names.extend(self.layer_names(i).values())
        if head:
            for name in self.head_names().values():
                if name not in names:
                    names.append(name)
        return names

    # --- construction ----------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any], **overrides: Any) -> ModelSpec:
        """Parse a Hugging Face `config.json` dict."""
        hidden = int(config["hidden_size"])
        n_heads = int(config["num_attention_heads"])
        head_dim = int(config.get("head_dim") or hidden // n_heads)
        spec = cls(
            n_layers=int(config["num_hidden_layers"]),
            hidden=hidden,
            intermediate=int(config["intermediate_size"]),
            n_heads=n_heads,
            n_kv_heads=int(config.get("num_key_value_heads", n_heads)),
            head_dim=head_dim,
            vocab=int(config["vocab_size"]),
            rms_eps=float(config.get("rms_norm_eps", 1e-6)),
            rope_theta=float(config.get("rope_theta", 10000.0)),
            rope_scaling=config.get("rope_scaling") or None,
            attn_bias=bool(config.get("attention_bias", False)),
            tie_embeddings=bool(config.get("tie_word_embeddings", False)),
            max_position=int(config.get("max_position_embeddings", 8192)),
        )
        if overrides:
            spec = replace_spec(spec, **overrides)
        return spec

    @classmethod
    def from_pretrained(cls, path: str | Path, **overrides: Any) -> ModelSpec:
        """Parse `config.json` from a checkpoint directory, or the file itself."""
        p = Path(path)
        if p.is_dir():
            p = p / "config.json"
        return cls.from_config(json.loads(p.read_text()), **overrides)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form, for the `plan` frame the head sends each worker."""
        from dataclasses import asdict

        return asdict(self)


def replace_spec(spec: ModelSpec, **changes: Any) -> ModelSpec:
    """`dataclasses.replace` under a name that says what it is."""
    from dataclasses import replace

    return replace(spec, **changes)

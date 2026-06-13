"""Core library for the minimal gate-feature replication.

Self-contained helpers for:
  * loading GemmaScope-2 transcoders (JumpReLU) for any layer,
  * loading/building concept steering vectors (concept_vectors.py method),
  * injecting a steering vector into the residual stream (injection setting),
  * capturing transcoder feature activations at the answer position,
  * direct logit attribution (DLA) to rank gate candidates,
  * causally ablating a chosen set of transcoder features.

Design notes
------------
* Transcoder input site: GemmaScope-2 ``transcoder_all`` maps the output of a
  layer's ``pre_feedforward_layernorm`` (normalized pre-MLP residual) to the
  MLP block's residual contribution. We therefore capture/encode at that
  submodule and, for ablation, subtract the decoded feature write from the
  decoder layer's residual output.
* Layer convention mirrors the existing codebase: a steering vector is
  extracted from ``hidden_states[steering_layer]`` (concept_vectors.py,
  BASELINE_LAYER=37) and injected at ``get_layer_module(steering_layer)``
  output (model_utils steering hook). This off-by-one is intentional — it
  reproduces the pipeline whose L37 injection was empirically validated.
"""

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

STEERING_LAYER = 37          # injection layer (and vector extraction layer)
TRANSCODER_WIDTH = "262k"    # GemmaScope-2 transcoder width
TRANSCODER_L0 = "big"        # GemmaScope-2 transcoder L0 bucket
EXTRACTION_TEMPLATE = "Tell me about {word}."

# 50 baseline concepts (capitalized; from 02b_run_500_concepts.py).
BASELINE_CONCEPTS = [
    "Dust", "Satellites", "Trumpets", "Origami", "Illusions",
    "Cameras", "Lightning", "Constellations", "Treasures", "Phones",
    "Trees", "Avalanches", "Mirrors", "Fountains", "Quarries",
    "Sadness", "Xylophones", "Secrecy", "Oceans", "Happiness",
    "Deserts", "Kaleidoscopes", "Sugar", "Vegetables", "Poetry",
    "Aquariums", "Bags", "Peace", "Caverns", "Memories",
    "Frosts", "Volcanoes", "Boulders", "Harmonies", "Masquerades",
    "Rubber", "Plastic", "Blood", "Amphitheaters", "Contraptions",
    "Youths", "Dynasties", "Snow", "Dirigibles", "Algorithms",
    "Denim", "Monoliths", "Milk", "Bread", "Silver",
]

# Neutral baseline words for vector mean-subtraction (from concept_vectors.py).
BASELINE_WORDS = [
    "Table", "Chair", "Bed", "Shelf", "Cabinet", "Drawer", "Lamp", "Clock", "Mirror", "Carpet",
    "Curtain", "Blanket", "Pillow", "Towel", "Basin", "Bottle", "Glass", "Plate", "Bowl", "Cup",
    "Shirt", "Pants", "Shoes", "Hat", "Belt", "Bag", "Wallet", "Watch", "Ring", "Necklace",
    "Button", "Zipper", "Thread", "Fabric", "Leather", "Cotton", "Wool", "Silk", "Linen", "Denim",
    "Bread", "Rice", "Pasta", "Sugar", "Salt", "Oil", "Milk", "Egg", "Butter", "Cheese",
    "Apple", "Orange", "Banana", "Potato", "Carrot", "Onion", "Garlic", "Pepper", "Tomato", "Lettuce",
    "Tree", "Grass", "Flower", "Leaf", "Branch", "Root", "Soil", "Sand", "Rock", "Stone",
    "Water", "River", "Lake", "Ocean", "Mountain", "Hill", "Valley", "Field", "Forest", "Garden",
    "Wood", "Metal", "Plastic", "Paper", "Rubber", "Paint", "Glue", "Tape", "Wire", "Brick",
    "Hammer", "Screwdriver", "Nail", "Screw", "Bolt", "Wrench", "Saw", "Drill", "Knife", "Scissors",
    "House", "Door", "Window", "Wall", "Floor", "Ceiling", "Roof", "Stair", "Hall", "Room",
    "Book", "Box", "Jar", "Can", "Key", "Coin", "Card", "Ticket", "Envelope", "Map",
]


# ─────────────────────────────────────────────────────────────────────────────
# Transcoder (JumpReLU) — adapted from 09_circuit_analysis.py
# ─────────────────────────────────────────────────────────────────────────────

class JumpReLUSAE(torch.nn.Module):
    """JumpReLU transcoder: encode(pre-MLP-LN) -> sparse features -> decode(MLP write)."""

    def __init__(self, d_in: int, d_sae: int, affine_skip_connection: bool = False):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.w_enc = torch.nn.Parameter(torch.zeros(d_in, d_sae))
        self.w_dec = torch.nn.Parameter(torch.zeros(d_sae, d_in))
        self.threshold = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))
        if affine_skip_connection:
            self.affine_skip_connection = torch.nn.Parameter(torch.zeros(d_in, d_in))
        else:
            self.affine_skip_connection = None

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        return mask * torch.nn.functional.relu(pre_acts)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.w_dec + self.b_dec


def load_transcoder(
    layer: int,
    width: str = TRANSCODER_WIDTH,
    l0: str = TRANSCODER_L0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> JumpReLUSAE:
    """Load a GemmaScope-2 27B-it transcoder for ``layer``."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    repo_id = "google/gemma-scope-2-27b-it"
    filename = f"transcoder_all/layer_{layer}_width_{width}_l0_{l0}_affine/params.safetensors"
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    params = load_file(path, device=device)

    d_model, d_sae = params["w_enc"].shape
    tc = JumpReLUSAE(d_model, d_sae, affine_skip_connection=True)
    tc.load_state_dict(params)
    tc = tc.to(device=device, dtype=dtype)
    tc.eval()
    return tc


# ─────────────────────────────────────────────────────────────────────────────
# Module resolution
# ─────────────────────────────────────────────────────────────────────────────

def get_transcoder_input_module(model_w, layer_idx: int):
    """Return the submodule whose output is the transcoder encoder input
    (``pre_feedforward_layernorm`` for Gemma; fall back to common names)."""
    layer = model_w.get_layer_module(layer_idx)
    for attr in ("pre_feedforward_layernorm", "post_attention_layernorm"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    raise AttributeError(
        f"Could not find transcoder-input layernorm on layer {layer_idx}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Concept steering vectors (concept_vectors.py method)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_residual_last_token(model_w, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Run a forward pass and return ``hidden_states[layer_idx][0, -1, :]`` (CPU)."""
    with torch.no_grad():
        out = model_w.model(
            input_ids=input_ids.to(model_w.device),
            use_cache=False,
            output_hidden_states=True,
        )
    return out.hidden_states[layer_idx][0, -1, :].detach().float().cpu()


def load_or_build_concept_vectors(
    model_w,
    concepts: Sequence[str],
    layer_idx: int,
    cache_path: Path,
    baseline_words: Sequence[str] = BASELINE_WORDS,
) -> Dict[str, torch.Tensor]:
    """Load concept vectors from ``cache_path``; build any missing ones.

    Vector(concept) = act("Tell me about {concept}.")[layer] - mean_w act(baseline_w)[layer],
    extracted at the last token of ``hidden_states[layer_idx]`` (concept_vectors.py).
    Returns a dict {concept: tensor[d_model]} restricted to ``concepts``.
    """
    from prompts import format_extraction_prompt  # local module

    cache_path = Path(cache_path)
    vectors: Dict[str, torch.Tensor] = {}
    if cache_path.exists():
        vectors = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"[gate_lib] Loaded {len(vectors)} cached concept vectors from {cache_path}")

    missing = [c for c in concepts if c not in vectors]
    if not missing:
        return {c: vectors[c] for c in concepts}

    print(f"[gate_lib] Building {len(missing)} concept vectors at layer {layer_idx} ...")
    tokenizer = model_w.tokenizer

    # Baseline mean (compute once, reused across all missing concepts).
    baseline_acts = []
    for w in baseline_words:
        ids = format_extraction_prompt(tokenizer, w)
        baseline_acts.append(_extract_residual_last_token(model_w, ids, layer_idx))
    baseline_mean = torch.stack(baseline_acts).mean(dim=0)

    for c in missing:
        ids = format_extraction_prompt(tokenizer, c)
        act = _extract_residual_last_token(model_w, ids, layer_idx)
        vectors[c] = act - baseline_mean

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, cache_path)
    print(f"[gate_lib] Saved {len(vectors)} concept vectors to {cache_path}")
    gc.collect()
    torch.cuda.empty_cache()
    return {c: vectors[c] for c in concepts}


# ─────────────────────────────────────────────────────────────────────────────
# Injection hook (mirrors model_utils steering_hook; all positions)
# ─────────────────────────────────────────────────────────────────────────────

def make_injection_hook(steering_vec: torch.Tensor, strength: float):
    """Add ``strength * steering_vec`` to a decoder layer's residual output at
    all positions."""
    def hook(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        vec = steering_vec.to(out.device, out.dtype) * float(strength)
        modified = out + vec.view(1, 1, -1)
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
    return hook


# ─────────────────────────────────────────────────────────────────────────────
# Transcoder feature capture (answer position)
# ─────────────────────────────────────────────────────────────────────────────

class TranscoderCapture:
    """Context manager capturing each target layer's transcoder-input (pre-MLP
    LN output) so callers can encode it into feature activations."""

    def __init__(self, model_w, layers: Sequence[int]):
        self.model_w = model_w
        self.layers = list(layers)
        self.captured: Dict[int, torch.Tensor] = {}
        self._handles: List = []

    def _make_hook(self, layer_idx: int):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            self.captured[layer_idx] = out.detach()
        return hook

    def __enter__(self):
        for L in self.layers:
            mod = get_transcoder_input_module(self.model_w, L)
            self._handles.append(mod.register_forward_hook(self._make_hook(L)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.captured.clear()


def encode_last_token(transcoder: JumpReLUSAE, pre_ln: torch.Tensor) -> torch.Tensor:
    """Encode the last-token pre-MLP-LN activation into feature acts (fp32 CPU)."""
    x = pre_ln[0, -1, :].to(transcoder.w_enc.device, transcoder.w_enc.dtype)
    acts = transcoder.encode(x.unsqueeze(0)).squeeze(0)
    return acts.float().cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Unembedding & direct logit attribution (DLA)
# ─────────────────────────────────────────────────────────────────────────────

def get_unembedding(model_w) -> torch.Tensor:
    """Return the (tied) unembedding matrix [vocab, d_model] on CPU fp32."""
    emb = model_w.model.get_input_embeddings().weight
    return emb.detach().float().cpu()


def compute_delta_u(
    model_w,
    yes_tokens: Iterable[str] = ("yes", "Yes", "YES"),
    no_tokens: Iterable[str] = ("no", "No", "NO"),
) -> torch.Tensor:
    """delta_u = mean(unembed[Yes]) - mean(unembed[No]) over single-token forms."""
    tokenizer = model_w.tokenizer
    unembed = get_unembedding(model_w)

    def _avg(tokens):
        vecs = []
        for t in tokens:
            ids = tokenizer.encode(t, add_special_tokens=False)
            if len(ids) == 1:
                vecs.append(unembed[ids[0]])
        if not vecs:
            raise ValueError(f"No single-token forms among {list(tokens)}")
        return torch.stack(vecs).mean(dim=0)

    return _avg(yes_tokens) - _avg(no_tokens)


def compute_dla(
    transcoder: JumpReLUSAE,
    delta_u: torch.Tensor,
    mean_acts: torch.Tensor,
) -> torch.Tensor:
    """DLA(f) = (unit(w_dec[f]) . delta_u) * mean_acts[f].  Returns fp32 [n_features].

    Negative DLA => feature pushes the Yes-No logit toward "No" (gate candidate).
    """
    w_dec = transcoder.w_dec.detach().float().cpu()                  # [F, d]
    w_dec_unit = w_dec / (w_dec.norm(dim=-1, keepdim=True) + 1e-10)
    scores = w_dec_unit @ delta_u.float().cpu()                      # [F]
    return scores * mean_acts.float().cpu()


def top_gate_features(dla: torch.Tensor, n_top: int = 200) -> List[int]:
    """Indices of the ``n_top`` most-negative-DLA features (gate candidates)."""
    order = torch.argsort(dla)  # ascending => most negative first
    return order[:n_top].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Causal feature ablation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureSlice:
    """Per-layer slice of a transcoder containing only the columns for a chosen
    set of features. Decouples ablation/patch hooks from the full SAE so the
    transcoder can be freed after extraction.
    """

    layer_idx: int
    feature_ids: torch.Tensor   # [k] long, sorted unique
    w_enc_sel: torch.Tensor     # [d, k]
    b_enc_sel: torch.Tensor     # [k]
    thr_sel: torch.Tensor       # [k]
    w_dec_sel: torch.Tensor     # [k, d]

    @classmethod
    def from_transcoder(
        cls,
        layer_idx: int,
        transcoder: "JumpReLUSAE",
        feature_ids: Sequence[int],
    ) -> "FeatureSlice":
        ids = torch.tensor(sorted(set(int(f) for f in feature_ids)), dtype=torch.long)
        dev = transcoder.w_enc.device
        idx = ids.to(dev)
        return cls(
            layer_idx=layer_idx,
            feature_ids=ids,
            w_enc_sel=transcoder.w_enc.index_select(1, idx).detach().clone(),
            b_enc_sel=transcoder.b_enc.index_select(0, idx).detach().clone(),
            thr_sel=transcoder.threshold.index_select(0, idx).detach().clone(),
            w_dec_sel=transcoder.w_dec.index_select(0, idx).detach().clone(),
        )

    @torch.no_grad()
    def encode_last_token(self, pre_ln: torch.Tensor) -> torch.Tensor:
        """Encode the last-token pre-LN activation through the slice only.
        Returns the selected features' activations (fp32 CPU, ordered like ``feature_ids``).
        """
        x = pre_ln[0, -1, :].to(self.w_enc_sel.device, self.w_enc_sel.dtype)
        pre = x @ self.w_enc_sel + self.b_enc_sel
        mask = pre > self.thr_sel
        return (mask * torch.nn.functional.relu(pre)).float().cpu()

    def __len__(self) -> int:
        return int(self.feature_ids.numel())


class GateAblation:
    """Context manager that ablates a set of transcoder features at one layer.

    For each forward pass it (1) captures the layer's pre-MLP-LN output,
    (2) recomputes the selected features' activations and their decoded write,
    and (3) subtracts that write from the decoder layer's residual output —
    removing those features' causal contribution downstream.
    """

    def __init__(
        self,
        model_w,
        slice_or_layer,
        transcoder: Optional["JumpReLUSAE"] = None,
        feature_ids: Optional[Sequence[int]] = None,
    ):
        # Two construction modes:
        #   GateAblation(model_w, feature_slice)
        #   GateAblation(model_w, layer_idx, transcoder, feature_ids)
        if isinstance(slice_or_layer, FeatureSlice):
            self._slice = slice_or_layer
        else:
            assert transcoder is not None and feature_ids is not None, (
                "legacy construction needs (model_w, layer_idx, transcoder, feature_ids)"
            )
            self._slice = FeatureSlice.from_transcoder(int(slice_or_layer), transcoder, feature_ids)
        self.model_w = model_w
        self.layer_idx = self._slice.layer_idx
        self.feature_ids = self._slice.feature_ids
        self._handles: List = []
        self._contribution: Dict[str, torch.Tensor] = {}

    def _pre_ln_hook(self, module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        x = out.to(self._slice.w_enc_sel.dtype)
        pre = x @ self._slice.w_enc_sel + self._slice.b_enc_sel              # [B, seq, k]
        mask = pre > self._slice.thr_sel
        acts = mask * torch.nn.functional.relu(pre)
        self._contribution["c"] = (acts @ self._slice.w_dec_sel).detach()    # [B, seq, d]

    def _layer_out_hook(self, module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        c = self._contribution.get("c")
        if c is None:
            return output
        modified = out - c.to(out.device, out.dtype)
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    def __enter__(self):
        if len(self.feature_ids) > 0:
            pre_mod = get_transcoder_input_module(self.model_w, self.layer_idx)
            out_mod = self.model_w.get_layer_module(self.layer_idx)
            self._handles.append(pre_mod.register_forward_hook(self._pre_ln_hook))
            self._handles.append(out_mod.register_forward_hook(self._layer_out_hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._contribution.clear()

    def encode_selected_last(self, pre_ln: torch.Tensor) -> torch.Tensor:
        """Encode pre-LN's last token via this layer's slice (delegates to FeatureSlice)."""
        return self._slice.encode_last_token(pre_ln)


class GatePatch:
    """Context manager that patches selected features' last-token activations to
    their steered values inside an *unsteered* forward pass (knock-in).

    At the answer position it adds ``(steered_sel - unsteered_sel) @ w_dec_sel``
    to the layer's residual output, so the gate features carry their steered
    activation while everything else stays unsteered. ``steered_sel`` must be
    ordered like the slice's ``feature_ids``.
    """

    def __init__(
        self,
        model_w,
        slice_or_layer,
        transcoder_or_steered=None,
        feature_ids: Optional[Sequence[int]] = None,
        steered_sel: Optional[torch.Tensor] = None,
    ):
        # Two construction modes:
        #   GatePatch(model_w, feature_slice, steered_sel=...)
        #   GatePatch(model_w, layer_idx, transcoder, feature_ids, steered_sel=...)
        if isinstance(slice_or_layer, FeatureSlice):
            self._slice = slice_or_layer
            assert steered_sel is None or transcoder_or_steered is None, \
                "pass steered_sel as keyword in slice mode"
            steered = steered_sel if steered_sel is not None else transcoder_or_steered
        else:
            assert transcoder_or_steered is not None and feature_ids is not None and steered_sel is not None, (
                "legacy construction needs (model_w, layer_idx, transcoder, feature_ids, steered_sel=...)"
            )
            self._slice = FeatureSlice.from_transcoder(int(slice_or_layer), transcoder_or_steered, feature_ids)
            steered = steered_sel
        self.model_w = model_w
        self.layer_idx = self._slice.layer_idx
        self.feature_ids = self._slice.feature_ids
        self._handles: List = []
        self._contribution: Dict[str, torch.Tensor] = {}
        self._steered_sel = steered.to(self._slice.w_enc_sel.device, self._slice.w_enc_sel.dtype)

    def _pre_ln_hook(self, module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        x = out[0, -1, :].to(self._slice.w_enc_sel.dtype)                # [d]
        pre = x @ self._slice.w_enc_sel + self._slice.b_enc_sel          # [k]
        mask = pre > self._slice.thr_sel
        unsteered_sel = mask * torch.nn.functional.relu(pre)
        delta = self._steered_sel - unsteered_sel                        # [k]
        self._contribution["c"] = (delta @ self._slice.w_dec_sel).detach()  # [d]

    def _layer_out_hook(self, module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        c = self._contribution.get("c")
        if c is None:
            return output
        modified = out.clone()
        modified[:, -1, :] = modified[:, -1, :] + c.to(out.device, out.dtype)
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    def __enter__(self):
        if len(self.feature_ids) > 0:
            pre_mod = get_transcoder_input_module(self.model_w, self.layer_idx)
            out_mod = self.model_w.get_layer_module(self.layer_idx)
            self._handles.append(pre_mod.register_forward_hook(self._pre_ln_hook))
            self._handles.append(out_mod.register_forward_hook(self._layer_out_hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._contribution.clear()


def selected_last_token_acts(
    transcoder: JumpReLUSAE,
    pre_ln: torch.Tensor,
    feature_ids: Sequence[int],
) -> torch.Tensor:
    """Encode the last-token pre-LN activation and return the selected features'
    activations ordered like ``sorted(set(feature_ids))`` (fp32 CPU)."""
    ids = torch.tensor(sorted(set(int(f) for f in feature_ids)), dtype=torch.long)
    feats = encode_last_token(transcoder, pre_ln)                       # [F] fp32 cpu
    return feats.index_select(0, ids)


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass
# ─────────────────────────────────────────────────────────────────────────────

def forward_last_logits(model_w, input_ids: torch.Tensor) -> torch.Tensor:
    """Single forward pass; returns last-position logits [1, vocab] on CPU."""
    with torch.no_grad():
        out = model_w.model(input_ids=input_ids.to(model_w.device), use_cache=False)
    return out.logits[:, -1, :].float().cpu()

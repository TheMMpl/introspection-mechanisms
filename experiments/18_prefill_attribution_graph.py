#!/usr/bin/env python3
"""
Prefill Attribution Graph (18): Attribution Graph for Prefill Injection Scenarios

Builds multi-hop attribution graphs for prefill-based concept injection (appending
concept words to prompts), analogous to the steering attribution graph (16/17) but
using prompt interpolation instead of activation steering.

Prefill Gradient (PG) replaces Steering Gradient (SG):
    PA(f)  = GA(f) × PG(f)           per SAE feature
    GA     = gradient attribution     — dL/dx at each SAE site (backward pass)
    PG     = prefill gradient         — d(h)/dα at each SAE site (JVP from embedding diff)
    IPA(f) = ∫₀¹ PA(f, α) dα         — integrated via Simpson's rule

The perturbation direction δ = dirty_emb − clean_emb replaces the steering vector.
PG propagates this tangent forward through the model's Jacobian (same JVP mechanism
as in 16_steering_attribution.py), but anchored at the embedding layer so every
transformer layer has a valid gradient path.

Two prompt variants from 06b_prefill_attribution.py:
    --variant append_user         concept words appended to user turn
    --variant replace_assistant   concept words placed in assistant turn

Parquet schema is identical to 16: "steering_attribution" stores PA values,
"steering_grad" stores PG values, α ∈ [0, 1] stored as injection_strength.
Directories use "strength_X_XX" naming for full compatibility with compute_isa
(from 16) and the graph/edge-weight machinery (from 17).

Paper sections supported:
    - Section 5.2 ("Prefill Activation Patching") — compare prefill vs. steering circuits
    - Section 5.3 ("Gate and Evidence Carrier Features") — feature overlap analysis

Model: Gemma-3 27B with Gemma Scope 2 SAEs/Transcoders (262k, big)

Usage:
    # Extract PA at all α ∈ [0,1] + compute IPA
    python 18_prefill_attribution_graph.py extract-all \\
        --concept Bread --variant append_user --n-extra-words 3

    # Build attribution graph from PA/IPA data
    python 18_prefill_attribution_graph.py build-graph \\
        --concept Bread --variant append_user --direction both

    # Re-render graph from saved JSON
    python 18_prefill_attribution_graph.py visualize \\
        --concept Bread --variant append_user

    # Full pipeline: extract → IPA → graph → visualize
    python 18_prefill_attribution_graph.py all \\
        --concept Bread --variant append_user --n-extra-words 3
"""

import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from model_utils import ModelWrapper, load_model

# ─────────────────────────────────────────────────────────────────────────────
# Dynamic imports (avoid circular imports; keep file standalone)
# ─────────────────────────────────────────────────────────────────────────────

from importlib.util import spec_from_file_location, module_from_spec

_HERE = Path(__file__).resolve().parent


def _load_module(name: str, fname: str):
    spec = spec_from_file_location(name, str(_HERE / fname))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_06b = _load_module("prefill_attribution_06b", "06b_prefill_attribution.py")
_16  = _load_module("steering_attribution_16",  "16_steering_attribution.py")
_17  = _load_module("attribution_graph_17",      "17_attribution_graph.py")

# ── From 06b: prompt construction ────────────────────────────────────────────
build_prompt_pair    = _06b.build_prompt_pair
get_yes_no_token_ids = _06b.get_yes_no_token_ids

# ── From 16: SAE infrastructure ──────────────────────────────────────────────
JumpReLUSAE              = _16.JumpReLUSAE
load_sae                 = _16.load_sae
ActivationHooks          = _16.ActivationHooks
FeatureNode              = _16.FeatureNode
FeatureEdge              = _16.FeatureEdge
AttributionGraph         = _16.AttributionGraph
SAE_TYPES                = _16.SAE_TYPES
SAE_TYPE_KEYS            = _16.SAE_TYPE_KEYS
SAE_TYPE_LOAD_MAP        = _16.SAE_TYPE_LOAD_MAP
JVP_SITE_SUFFIXES        = _16.JVP_SITE_SUFFIXES
CAPTURE_TYPES_SAE        = _16.CAPTURE_TYPES_SAE
TRACE_SAE_TYPES          = _16.TRACE_SAE_TYPES
get_layers               = _16.get_layers
preload_saes             = _16.preload_saes
combine_ga_sg_to_sa      = _16.combine_ga_sg_to_sa
make_logit_loss_fn       = _16.make_logit_loss_fn
make_feature_target_loss_fn = _16.make_feature_target_loss_fn
save_tangent_data        = _16.save_tangent_data
load_tangent_data        = _16.load_tangent_data
compute_isa              = _16.compute_isa   # parquet format is identical
_act_key                 = _16._act_key
_model_name_to_sae_size  = _16._model_name_to_sae_size
_get_from_list           = _16._get_from_list

# ── From 17: graph/edge machinery + visualization ────────────────────────────
compute_edge_weights              = _17.compute_edge_weights
select_features_from_edge_weights = _17.select_features_from_edge_weights
_load_curve_from_root_sa          = _17._load_curve_from_root_sa
export_graph_json                 = _17.export_graph_json
render_interactive                = _17.render_interactive
write_graph_summary               = _17.write_graph_summary
render_pdf_from_html              = _17.render_pdf_from_html

warnings.filterwarnings("ignore", message="Glyph .* missing from font")


# =============================================================================
# Constants & defaults
# =============================================================================

DEFAULT_MODEL         = "gemma3_27b"
DEFAULT_N_ALPHAS      = 21          # α ∈ [0, 1] integration points
DEFAULT_TRACE_DEPTH   = 2
DEFAULT_MAX_PER_TYPE  = [8, 5, 3, 2]
DEFAULT_FRAC_OF_MAX   = 0.10
DEFAULT_OUTPUT_DIR    = "analysis/18_prefill_attribution"
DEFAULT_DEVICE        = "cuda"
DEFAULT_DTYPE         = "bfloat16"
DEFAULT_SEED          = 42
DEFAULT_SAE_WIDTH     = "262k"
DEFAULT_SAE_L0        = "big"
DEFAULT_DIRECTION     = "both"
DEFAULT_VARIANT       = "append_user"
DEFAULT_N_EXTRA_WORDS = 3
DEFAULT_BASELINE_WORD = "thing"
DEFAULT_CLEAN_MODE    = "filler_word"
DEFAULT_TRIAL_NUM     = 1

# Token IDs for the detection logit-gap (Gemma-3 27B):
#   At α=0 (clean) the model says "No"   (neg); at α=1 (dirty) it says "Oh" (pos).
#   These match the steering attribution defaults in 16.
DEFAULT_POS_TOKEN_ID  = 12932   # "Oh" — detected
DEFAULT_NEG_TOKEN_ID  = 3771    # "No" — not detected


# =============================================================================
# Embedding utilities
# =============================================================================

def get_embed_module(model: nn.Module) -> nn.Module:
    """Navigate to the token embedding module (embed_tokens)."""
    mdl = model
    for _ in range(6):
        if hasattr(mdl, "embed_tokens"):
            return mdl.embed_tokens
        if hasattr(mdl, "model"):
            mdl = mdl.model
        elif hasattr(mdl, "language_model"):
            mdl = mdl.language_model
        else:
            break
    raise ValueError("Cannot find embed_tokens in model hierarchy")


def compute_delta_emb(
    model_inner: nn.Module,
    clean_ids: torch.Tensor,
    dirty_ids: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """Return dirty_emb − clean_emb as the prefill perturbation direction.

    Both tensors must have shape [1, seq_len] with identical lengths.
    Returns [seq_len, d_model] float32 on CPU.
    """
    assert clean_ids.shape == dirty_ids.shape, (
        f"clean/dirty token shapes must match: {clean_ids.shape} vs {dirty_ids.shape}"
    )
    embed_mod = get_embed_module(model_inner)
    with torch.no_grad():
        c = embed_mod(clean_ids.to(device)).float()
        d = embed_mod(dirty_ids.to(device)).float()
    return (d - c)[0].cpu()   # [seq, d_model]


# =============================================================================
# EmbeddingInterpolationCut  (drop-in for AdditiveLayerCut from 16)
# =============================================================================

class EmbeddingInterpolationCut:
    """Prefill injection via embedding interpolation.

    Adds α * δ to the embedding output, where δ = dirty_emb − clean_emb.
      α = 0  →  clean prompt (no concept)
      α = 1  →  dirty prompt (full concept word(s))

    Because the injection is at the embedding level (before all transformer
    layers), every layer has a valid gradient / JVP path, so has_grad_path
    always returns True.
    """

    def __init__(self, embed_module: nn.Module, delta_emb: torch.Tensor):
        """
        Args:
            embed_module: The model's embed_tokens nn.Module.
            delta_emb: [seq_len, d_model] float32 on CPU.
        """
        self.embed_module = embed_module
        self._delta_cpu   = delta_emb.float()

    def make_steering_hook(self, device: str = "cuda"):
        """Return hook_factory(alpha) → forward hook for embed_module.

        Matches the hook_factory(strength) interface of AdditiveLayerCut.
        """
        delta = self._delta_cpu.to(device)

        def hook_factory(alpha):
            def hook(module, args, output):
                h    = output[0] if isinstance(output, tuple) else output
                rest = output[1:] if isinstance(output, tuple) else ()
                interp = h + alpha * delta.unsqueeze(0)   # [1, seq, d_model]
                return (interp,) + rest if isinstance(output, tuple) else interp
            return hook

        return hook_factory

    def has_grad_path(self, layer_idx: int, sae_type: str, injection_layer: int = -1) -> bool:
        """All transformer layers are downstream of the embedding."""
        return True


# =============================================================================
# Pass 1 (GA) — forward + backward from loss, embedding-interpolated at α
# =============================================================================

def compute_ga_pass_prefill(
    model_inner: nn.Module,
    clean_ids: torch.Tensor,
    embed_cut: EmbeddingInterpolationCut,
    alpha: float,
    n_layers: int,
    loss_fn,
    device: str = "cuda",
    layer_indices: Optional[Set[int]] = None,
) -> Tuple[Dict, Dict]:
    """GA pass for prefill: forward+backward with embedding interpolated at α.

    Mirrors compute_ga_pass() from 16 but hooks the embedding module instead
    of a transformer layer, so all downstream layers receive grads.

    Returns:
        grad_data: {(layer_idx, sae_type): [seq, d_model]} gradient tensors (CPU).
        act_data:  {(layer_idx, sae_type): [seq, d_model]} activation tensors (CPU).
    """
    attn_mask   = torch.ones_like(clean_ids).to(device)
    hook_layers = sorted(layer_indices) if layer_indices is not None else list(range(n_layers))

    hf           = embed_cut.make_steering_hook(device)
    embed_handle = embed_cut.embed_module.register_forward_hook(hf(alpha))
    act_hooks    = ActivationHooks(model_inner, retain_grad=True)
    act_hooks.register_hooks(layer_indices=hook_layers, capture_types=list(CAPTURE_TYPES_SAE))

    try:
        with act_hooks:
            model_inner.eval()
            outputs = model_inner(
                input_ids=clean_ids,
                attention_mask=attn_mask,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        embed_handle.remove()
        act_hooks.remove_hooks()

    raw  = act_hooks.get_activations()
    loss = loss_fn(outputs, raw)
    loss.sum().backward()

    grad_data: Dict[Tuple[int, str], torch.Tensor] = {}
    act_data:  Dict[Tuple[int, str], torch.Tensor] = {}

    for li in hook_layers:
        for st in SAE_TYPES:
            in_sfx, tgt_sfx = SAE_TYPE_KEYS[st]
            grad_sfx = tgt_sfx if (st == "transcoder_all" and tgt_sfx) else in_sfx

            gk = _act_key(li, grad_sfx)
            if gk not in raw or not raw[gk]:
                continue
            t = raw[gk][0]
            if isinstance(t, (list, tuple)):
                t = t[0]
            if t.grad is None:
                continue
            grad_data[(li, st)] = t.grad.detach().float()[0].cpu()

            ek = _act_key(li, in_sfx)
            if ek not in raw or not raw[ek]:
                continue
            et = raw[ek][0]
            if isinstance(et, (list, tuple)):
                et = et[0]
            act_data[(li, st)] = et[0].detach().float().cpu()

    del raw, outputs
    torch.cuda.empty_cache()
    return grad_data, act_data


# =============================================================================
# Pass 2 (PG) — JVP from embedding diff, analogous to compute_sg_pass in 16
# =============================================================================

def compute_pg_pass(
    model_inner: nn.Module,
    clean_ids: torch.Tensor,
    embed_cut: EmbeddingInterpolationCut,
    alpha: float,
    n_layers: int,
    device: str = "cuda",
) -> Dict[Tuple[int, str], torch.Tensor]:
    """PG pass: JVP w.r.t. α through the embedding interpolation.

    Propagates the tangent (d_embedding/dα = δ = dirty_emb − clean_emb)
    forward through the model Jacobian to yield d(h_layer)/dα at every SAE site.

    This is mathematically equivalent to integrated-gradients' path tangent, but
    computed via forward-mode AD rather than finite differences.

    Returns:
        tangent_data: {(layer_idx, suffix): tensor} — same format as SG from 16.
    """
    alpha_primal  = torch.tensor(alpha, dtype=torch.float32, device=device)
    alpha_tangent = torch.tensor(1.0,   dtype=torch.float32, device=device)
    attn_mask     = torch.ones_like(clean_ids).to(device)

    def jvp_forward(a):
        hf     = embed_cut.make_steering_hook(device)
        handle = embed_cut.embed_module.register_forward_hook(hf(a))
        hooks  = ActivationHooks(model_inner, retain_grad=False)
        hooks.register_hooks(list(range(n_layers)), list(CAPTURE_TYPES_SAE))
        try:
            with hooks:
                model_inner.eval()
                _ = model_inner(
                    input_ids=clean_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )
        finally:
            handle.remove()
            hooks.remove_hooks()

        raw    = hooks.get_activations()
        result = []
        for li in range(n_layers):
            for sfx in JVP_SITE_SUFFIXES:
                t = raw.get(_act_key(li, sfx), [None])[0]
                if t is None:
                    t = torch.zeros(1, device=device)
                if isinstance(t, (list, tuple)):
                    t = t[0]
                result.append(t[0] if t.dim() == 3 else t)
        return result

    _, site_tangents = torch.func.jvp(jvp_forward, (alpha_primal,), (alpha_tangent,))

    tangent_data: Dict[Tuple[int, str], torch.Tensor] = {}
    n_sfx = len(JVP_SITE_SUFFIXES)
    for li in range(n_layers):
        for j, sfx in enumerate(JVP_SITE_SUFFIXES):
            tangent_data[(li, sfx)] = site_tangents[li * n_sfx + j].detach()

    del site_tangents
    torch.cuda.empty_cache()
    return tangent_data


# =============================================================================
# Forward JVP pass (for forward tracing from a specific source feature)
# =============================================================================

def compute_forward_jvp_pass_prefill(
    model_inner: nn.Module,
    clean_ids: torch.Tensor,
    embed_cut: EmbeddingInterpolationCut,
    alpha: float,
    source_layer: int,
    source_decoder_vec: torch.Tensor,
    source_sae_type: str,
    n_layers: int,
    device: str = "cuda",
) -> Dict[Tuple[int, str], torch.Tensor]:
    """Forward JVP: propagate source feature's decoder direction downstream.

    Identical to compute_forward_jvp_pass in 16, except the operating-point
    background uses embedding interpolation at α rather than a steering hook.

    Returns tangent_data keyed by (layer_idx, suffix) for layers >= source_layer.
    """
    layers_module = get_layers(model_inner)
    t_primal      = torch.tensor(0.0, dtype=torch.float32, device=device)
    t_tangent     = torch.tensor(1.0, dtype=torch.float32, device=device)
    src_vec       = source_decoder_vec.to(device).float()
    attn_mask     = torch.ones_like(clean_ids).to(device)

    # Determine perturbation hook site (mirrors 16's compute_forward_jvp_pass)
    use_pre_hook = False
    if source_sae_type == "attn_out_all":
        src_module   = layers_module[source_layer].self_attn.o_proj
        use_pre_hook = True
    elif source_sae_type in ("transcoder_all", "mlp_out_all"):
        src_module = getattr(
            layers_module[source_layer], "post_feedforward_layernorm",
            layers_module[source_layer],
        )
    else:
        src_module = layers_module[source_layer]

    # Fixed embedding interpolation hook (not differentiated)
    hf_embed     = embed_cut.make_steering_hook(device)
    embed_handle = embed_cut.embed_module.register_forward_hook(hf_embed(alpha))

    def jvp_forward(t_val):
        perturbation = src_vec * t_val
        if use_pre_hook:
            def src_hook(module, args):
                x = args[0]
                p = perturbation.unsqueeze(0).unsqueeze(0).expand_as(x)
                return (x + p,) + args[1:]
            src_h = src_module.register_forward_pre_hook(src_hook)
        else:
            def src_hook(module, inp, output):
                h    = output[0] if isinstance(output, tuple) else output
                rest = output[1:] if isinstance(output, tuple) else ()
                p    = perturbation.unsqueeze(0).unsqueeze(0).expand_as(h)
                return (h + p,) + rest if isinstance(output, tuple) else h + p
            src_h = src_module.register_forward_hook(src_hook)

        hooks = ActivationHooks(model_inner, retain_grad=False)
        hooks.register_hooks(list(range(n_layers)), list(CAPTURE_TYPES_SAE))
        try:
            with hooks:
                model_inner.eval()
                _ = model_inner(
                    input_ids=clean_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )
        finally:
            src_h.remove()
            hooks.remove_hooks()

        raw    = hooks.get_activations()
        result = []
        for li in range(n_layers):
            for sfx in JVP_SITE_SUFFIXES:
                t = raw.get(_act_key(li, sfx), [None])[0]
                if t is None:
                    t = torch.zeros(1, device=device)
                if isinstance(t, (list, tuple)):
                    t = t[0]
                result.append(t[0] if t.dim() == 3 else t)
        return result

    try:
        _, site_tangents = torch.func.jvp(jvp_forward, (t_primal,), (t_tangent,))
    finally:
        embed_handle.remove()

    tangent_data: Dict[Tuple[int, str], torch.Tensor] = {}
    n_sfx = len(JVP_SITE_SUFFIXES)
    for li in range(source_layer, n_layers):
        for j, sfx in enumerate(JVP_SITE_SUFFIXES):
            tangent_data[(li, sfx)] = site_tangents[li * n_sfx + j].detach()

    del site_tangents
    torch.cuda.empty_cache()
    return tangent_data


# =============================================================================
# Helpers
# =============================================================================

def _alpha_to_strength_str(alpha: float) -> str:
    """Convert α ∈ [0, 1] to the "strength_X_XX" directory name used by 16/17."""
    i = int(alpha)
    f = int(round((alpha % 1) * 100))
    return f"strength_{i}_{f:02d}"


def _build_prompt_pair_for_trial(
    tokenizer,
    concept: str,
    trial_num: int,
    variant: str,
    n_extra_words: int,
    baseline_word: str,
    clean_mode: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Wrapper around 06b's build_prompt_pair for a given trial number."""
    result = build_prompt_pair(
        tokenizer=tokenizer,
        trial_n=trial_num,
        variant=variant,
        concept=concept,
        baseline_word=baseline_word,
        n_extra_words=n_extra_words,
        clean_mode=clean_mode,
    )
    if result is None:
        print(f"  WARNING: clean/dirty token lengths don't match for '{concept}' "
              f"(variant={variant}, K={n_extra_words}) — skipping trial {trial_num}")
    return result


# =============================================================================
# Root PA extraction — extract_pa_for_alpha
# =============================================================================

def extract_pa_for_alpha(
    model_wrapper: ModelWrapper,
    concept: str,
    clean_ids: torch.Tensor,
    dirty_ids: torch.Tensor,
    alpha: float,
    output_dir: Path,
    trial_num: int = 1,
    pos_token_id: int = DEFAULT_POS_TOKEN_ID,
    neg_token_id: int = DEFAULT_NEG_TOKEN_ID,
    sae_width: str = DEFAULT_SAE_WIDTH,
    sae_l0: str = DEFAULT_SAE_L0,
    device: str = "cuda",
    pg_cache_dir: Optional[Path] = None,
    save_pg_cache: bool = False,
    layer_indices: Optional[Set[int]] = None,
    compute_remainder: bool = True,
    loss_fn_override=None,
    extra_meta: Optional[Dict] = None,
    output_filename: str = "sa_trial{trial_num}.parquet",
) -> Optional[Path]:
    """Extract Prefill Attribution (PA) for one interpolation point α.

    Saves a parquet with schema identical to 16's SA parquets:
        layer, sae_type, feature_id, token_pos,
        activation, gradient_attribution, steering_grad, steering_attribution
    — so 16's compute_isa and 17's graph building work without modification.

    The α ∈ [0, 1] is stored as injection_strength; injection_layer is -1
    (embedding level).  Output directory uses "strength_X_XX" naming.

    Args:
        pg_cache_dir:    Directory to cache PG tangent data (optional).
        save_pg_cache:   Write PG to cache if True.
        loss_fn_override: Custom loss callable; defaults to logit-gap loss.
    """
    inner    = model_wrapper.model
    n_layers = model_wrapper.n_layers
    seq_len  = clean_ids.shape[1]
    clean_d  = clean_ids.to(device)

    delta_emb  = compute_delta_emb(inner, clean_ids, dirty_ids, device)
    embed_cut  = EmbeddingInterpolationCut(get_embed_module(inner), delta_emb)
    sae_cache  = preload_saes(n_layers, model_wrapper.model_name,
                               sae_width, sae_l0, "cpu", layer_indices)
    loss_fn    = loss_fn_override or make_logit_loss_fn(pos_token_id, neg_token_id)

    base_meta = {
        "concept":          concept,
        "injection_layer":  -1,        # embedding-level
        "injection_strength": alpha,
        "trial_num":        trial_num,
    }
    if extra_meta:
        base_meta.update(extra_meta)

    hook_layers = sorted(layer_indices) if layer_indices else None

    # ── Pass 1: GA ───────────────────────────────────────────────────────────
    grad_data, act_data = compute_ga_pass_prefill(
        inner, clean_d, embed_cut, alpha, n_layers, loss_fn, device, hook_layers,
    )

    # ── Pass 2: PG (with optional caching) ──────────────────────────────────
    alpha_str = _alpha_to_strength_str(alpha)
    pg_file   = (pg_cache_dir / alpha_str / "tangent_data.safetensors") if pg_cache_dir else None

    if pg_file and pg_file.exists():
        tangent_data = load_tangent_data(pg_file)
    else:
        tangent_data = compute_pg_pass(inner, clean_d, embed_cut, alpha, n_layers, device)
        if save_pg_cache and pg_file:
            save_tangent_data(tangent_data, pg_file)
            print(f"  PG cached: {pg_file}")

    # ── Combine PA = GA × PG ─────────────────────────────────────────────────
    # combine_ga_sg_to_sa is fully reusable: it calls embed_cut.has_grad_path()
    # which returns True for all layers (embedding-level injection).
    rows = combine_ga_sg_to_sa(
        grad_data, act_data, tangent_data, sae_cache, embed_cut,
        injection_layer=-1, n_layers=n_layers, seq_len=seq_len,
        base_meta=base_meta, device=device, compute_remainder=compute_remainder,
    )

    out_path = output_dir / alpha_str / output_filename.format(trial_num=trial_num)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"  [{alpha:.3f}] Saved {len(rows)} rows → {out_path.relative_to(output_dir.parent)}")
    torch.cuda.empty_cache()
    return out_path


# =============================================================================
# Feature-targeted PA — for backward hop-1+ tracing
# =============================================================================

def extract_feature_target_pa(
    model_wrapper: ModelWrapper,
    concept: str,
    clean_ids: torch.Tensor,
    dirty_ids: torch.Tensor,
    alpha: float,
    target_layer: int,
    target_sae_type: str,
    target_feature_id: int,
    target_token_pos: int,
    output_dir: Path,
    trial_num: int = 1,
    sae_width: str = DEFAULT_SAE_WIDTH,
    sae_l0: str = DEFAULT_SAE_L0,
    device: str = "cuda",
    pg_cache_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Feature-targeted PA for backward tracing: loss = target feature activation.

    Mirrors extract_feature_target_sa from 16 but uses embedding interpolation.
    Saves under output_dir / <target_subdir> / strength_X_XX / sa_trial{n}.parquet.
    """
    seq_len = clean_ids.shape[1]
    loss_fn = make_feature_target_loss_fn(
        target_layer, target_sae_type, target_feature_id, target_token_pos,
        sae_width, sae_l0, model_wrapper.model_name, device, seq_len,
    )

    # Use the same subdirectory naming as 17_attribution_graph.py's compute_edge_weights:
    #   feat_sa/<sae_type>_L{layer}_F{fid}_T{tok}/strength_X_XX/feat_sa_trial{n}.parquet
    tgt_name   = (f"{target_sae_type}_L{target_layer}"
                  f"_F{target_feature_id}_T{target_token_pos}")
    out_subdir = output_dir / "feat_sa" / tgt_name

    return extract_pa_for_alpha(
        model_wrapper, concept, clean_ids, dirty_ids, alpha,
        output_dir=out_subdir, trial_num=trial_num,
        sae_width=sae_width, sae_l0=sae_l0, device=device,
        pg_cache_dir=pg_cache_dir,
        loss_fn_override=loss_fn,
        compute_remainder=False,
        output_filename="feat_sa_trial{trial_num}.parquet",
        extra_meta={
            "target_layer":       target_layer,
            "target_sae_type":    target_sae_type,
            "target_feature_id":  target_feature_id,
            "target_token_pos":   target_token_pos,
        },
    )


# =============================================================================
# Forward PA — for forward hop-1+ tracing
# =============================================================================

def extract_forward_pa_for_alpha(
    model_wrapper: ModelWrapper,
    concept: str,
    clean_ids: torch.Tensor,
    dirty_ids: torch.Tensor,
    alpha: float,
    source_layer: int,
    source_sae_type: str,
    source_feature_id: int,
    source_token_pos: int,
    output_dir: Path,
    trial_num: int = 1,
    sae_width: str = DEFAULT_SAE_WIDTH,
    sae_l0: str = DEFAULT_SAE_L0,
    device: str = "cuda",
) -> Optional[Path]:
    """Forward PA: JVP from source feature's decoder direction downstream.

    Mirrors extract_forward_sa_for_strength from 16 but uses embedding
    interpolation at α as the operating point.

    The forward JVP propagates the source feature's decoder direction through
    the model Jacobian.  Downstream SA is then: forward_jvp × GA_root,
    where GA_root is loaded from the root PA parquets at the same α.

    Saves under:
        output_dir / feat_sa / fwd_<source> / strength_X_XX / feat_sa_trial{n}.parquet
    """
    inner    = model_wrapper.model
    n_layers = model_wrapper.n_layers
    clean_d  = clean_ids.to(device)

    delta_emb = compute_delta_emb(inner, clean_ids, dirty_ids, device)
    embed_cut = EmbeddingInterpolationCut(get_embed_module(inner), delta_emb)

    # Load source SAE decoder vector
    src_load   = SAE_TYPE_LOAD_MAP.get(source_sae_type, source_sae_type)
    model_size = _model_name_to_sae_size(model_wrapper.model_name)
    src_sae    = load_sae(source_layer, sae_width, sae_l0, src_load, model_size, True, device)
    with torch.no_grad():
        src_dec = src_sae.w_dec[source_feature_id].float().cpu()
    del src_sae
    torch.cuda.empty_cache()

    # Forward JVP pass
    fwd_tangent = compute_forward_jvp_pass_prefill(
        inner, clean_d, embed_cut, alpha,
        source_layer, src_dec, source_sae_type, n_layers, device,
    )

    # Preload SAEs for downstream layers only
    downstream_layers = set(range(source_layer + 1, n_layers))
    sae_cache = preload_saes(n_layers, model_wrapper.model_name,
                              sae_width, sae_l0, "cpu", downstream_layers)

    # Load GA_root for this α from root PA parquets
    alpha_str    = _alpha_to_strength_str(alpha)
    root_parquet = output_dir / alpha_str / f"sa_trial{trial_num}.parquet"
    root_df      = None
    if root_parquet.exists():
        root_df = pd.read_parquet(root_parquet, columns=[
            "layer", "sae_type", "feature_id", "token_pos", "gradient_attribution",
        ])

    # Combine: forward_jvp × GA_root  (same logic as 16's extract_forward_sa_for_strength)
    base_meta = {
        "concept":            concept,
        "injection_layer":    -1,
        "injection_strength": alpha,
        "trial_num":          trial_num,
        "source_layer":       source_layer,
        "source_sae_type":    source_sae_type,
        "source_feature_id":  source_feature_id,
        "source_token_pos":   source_token_pos,
    }

    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for li in sorted(downstream_layers):
            for st in SAE_TYPES:
                in_sfx  = SAE_TYPE_KEYS[st][0]
                lt      = SAE_TYPE_LOAD_MAP.get(st, st)
                tng_key = (li, in_sfx)
                if (li, lt) not in sae_cache or tng_key not in fwd_tangent:
                    continue

                sae = sae_cache[(li, lt)].to(device=device, dtype=torch.float32).eval()
                dx  = fwd_tangent[tng_key].to(device).float()
                if dx.dim() == 3:
                    dx = dx[0]
                fwd_jvp = dx @ sae.w_enc  # [seq, d_sae]

                # Build GA_root lookup for this layer/type
                ga_root: Dict[Tuple[int, int], float] = {}
                if root_df is not None:
                    lr = root_df[(root_df["layer"] == li) & (root_df["sae_type"] == st)]
                    for _, row in lr.iterrows():
                        ga_root[(int(row["token_pos"]), int(row["feature_id"]))] = float(
                            row["gradient_attribution"]
                        )

                nonzero = (fwd_jvp.abs() > 1e-8).nonzero(as_tuple=True)
                if len(nonzero[0]) > 0:
                    toks    = nonzero[0].cpu().tolist()
                    feats   = nonzero[1].cpu().tolist()
                    jvp_vals = fwd_jvp[nonzero[0], nonzero[1]].cpu().tolist()
                    for tok, fid, jvp_val in zip(toks, feats, jvp_vals):
                        ga_val = ga_root.get((tok, fid), 0.0)
                        if abs(ga_val) < 1e-10:
                            continue
                        rows.append({
                            **base_meta,
                            "layer":             li,
                            "token_pos":         tok,
                            "sae_type":          st,
                            "feature_id":        fid,
                            "forward_jvp":       jvp_val,
                            "gradient_attribution": ga_val,
                            "steering_attribution": jvp_val * ga_val,
                        })
                sae.to("cpu")
                torch.cuda.empty_cache()

    del fwd_tangent
    torch.cuda.empty_cache()

    # Save with same path convention as 17 expects for forward tracing:
    # feat_sa/fwd_<sae_type>_L{layer}_F{fid}_T{tok}/strength_X_XX/feat_sa_trial{n}.parquet
    src_name  = (f"fwd_{source_sae_type}"
                 f"_L{source_layer}_F{source_feature_id}_T{source_token_pos}")
    out_path  = output_dir / "feat_sa" / src_name / alpha_str / f"feat_sa_trial{trial_num}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_parquet(out_path, index=False)
    return out_path


# =============================================================================
# IPA computation — thin wrapper; reuses 16's compute_isa unchanged
# =============================================================================

def compute_ipa(pa_dir: Path, trial_nums: List[int] = None) -> None:
    """Compute Integrated Prefill Attribution (IPA) via trapezoidal integration.

    Delegates directly to compute_isa from 16 since the parquet schema is
    identical.  IPA(f) = ∫₀¹ PA(f, α) dα stored as integrated_steering_attribution
    in isa_trial{n}.parquet within pa_dir.
    """
    compute_isa(pa_dir, trial_nums)


# =============================================================================
# Attribution Graph construction (prefill-specific)
# =============================================================================

def build_attribution_graph_prefill(
    model_wrapper: ModelWrapper,
    concept: str,
    clean_dirty_fn,           # callable(trial_num) -> (clean_ids, dirty_ids) | None
    output_dir: Path,
    trial_nums: List[int] = None,
    trace_depth: int = DEFAULT_TRACE_DEPTH,
    n_alphas_feat_pa: int = DEFAULT_N_ALPHAS,
    max_per_type: List[int] = None,
    frac_of_max: float = DEFAULT_FRAC_OF_MAX,
    direction: str = DEFAULT_DIRECTION,
    pos_token_id: int = DEFAULT_POS_TOKEN_ID,
    neg_token_id: int = DEFAULT_NEG_TOKEN_ID,
    device: str = DEFAULT_DEVICE,
) -> AttributionGraph:
    """Build multi-hop attribution graph for prefill injection.

    Mirrors build_attribution_graph from 17 but calls prefill-specific SA
    extraction functions (extract_feature_target_pa, extract_forward_pa_for_alpha)
    and uses injection_layer=-1 (embedding level) throughout.

    Algorithm:
        Hop 0: edge weights from root PA parquets (already extracted by
               extract-all command).
        Hop 1+: feature-targeted PA (backward) and/or forward PA (forward),
                using Simpson's-rule-weighted edge weights.
        Per-hop feature cap via max_per_type and frac_of_max.
        Only ATTN+TC features are traced to next hop.

    Args:
        clean_dirty_fn: callable(trial_num) -> (clean_ids, dirty_ids) or None.
                        Called once per trial to get the prompt pair.
        output_dir:     Root directory containing strength_X_XX/sa_trial*.parquet.
    """
    if trial_nums is None:
        trial_nums = [1]
    if max_per_type is None:
        max_per_type = DEFAULT_MAX_PER_TYPE

    # injection_layer=-1 means: all layers ≥ 0 are "downstream" (correct for embed)
    injection_layer = -1
    root_pa_dir     = output_dir

    # ── Hop 0: edge weights from root PA ─────────────────────────────────────
    print("\n  Computing hop-0 edge weights (root → features)...")
    hop0_ew = compute_edge_weights(root_pa_dir, trial_nums, optimal_strength=1.0)
    if not hop0_ew:
        print("  ERROR: No edge weights found. Run 'extract-all' first.")
        return AttributionGraph(nodes={}, edges=[], optimal_strength=1.0)

    mpt0         = _get_from_list(max_per_type, 0)
    hop0_selected = select_features_from_edge_weights(
        hop0_ew, mpt0, frac_of_max, injection_layer
    )

    graph = AttributionGraph(nodes={}, edges=[], optimal_strength=1.0)
    root  = FeatureNode(layer=-1, sae_type="root", feature_id=-1, token_pos=-1,
                        isa_value=0, hop=-1)
    graph.nodes[root.key] = root

    visited_bwd: Set[Tuple] = set()
    visited_fwd: Set[Tuple] = set()

    for key, ew in hop0_selected:
        node = FeatureNode(key[0], key[1], key[2], key[3], ew, hop=0)
        graph.nodes[key] = node
        visited_bwd.add(key)
        visited_fwd.add(key)
        graph.edges.append(FeatureEdge(
            source_key=key, target_key=graph.root_key, weight=ew, hop=0
        ))

    # Integration points for hop-1+ feature SA
    feat_alphas  = np.linspace(0.0, 1.0, n_alphas_feat_pa).tolist()
    do_backward  = direction in ("backward", "both")
    do_forward   = direction in ("forward", "both")

    # ── Backward tracing (hop-1+) ─────────────────────────────────────────────
    if do_backward:
        bwd_targets = [
            graph.nodes[k] for k, _ in hop0_selected
            if graph.nodes[k].sae_type in TRACE_SAE_TYPES
            and graph.nodes[k].feature_id >= 0
            and graph.nodes[k].layer > injection_layer   # all layers ≥ 0
        ]

        # Build (trial_num → (clean_ids, dirty_ids)) map for fast lookup
        trial_prompts = {}
        for tn in trial_nums:
            result = clean_dirty_fn(tn)
            if result is not None:
                trial_prompts[tn] = result

        pg_cache = output_dir / "pg_cache"

        for hop in range(trace_depth):
            if not bwd_targets:
                break
            mpt = _get_from_list(max_per_type, hop + 1)
            print(f"\n  Backward hop {hop+1}: "
                  f"{len(bwd_targets)} targets (cap {mpt}/type)...")

            for target in bwd_targets:
                print(f"    {target.short_name()}...")
                for a in tqdm(feat_alphas, desc="    PA α", leave=False):
                    for tn, (c_ids, d_ids) in trial_prompts.items():
                        extract_feature_target_pa(
                            model_wrapper, concept, c_ids, d_ids, a,
                            target.layer, target.sae_type,
                            target.feature_id, target.token_pos,
                            output_dir=output_dir,
                            trial_num=tn, device=device,
                            pg_cache_dir=pg_cache,
                        )

            next_targets = []
            for target in bwd_targets:
                # compute_edge_weights from 17 resolves the subdir from target node
                ew = compute_edge_weights(
                    output_dir, trial_nums,
                    optimal_strength=1.0,
                    target=target,
                    root_sa_dir=root_pa_dir,
                )
                if not ew:
                    continue
                selected = select_features_from_edge_weights(
                    ew, mpt, frac_of_max, injection_layer
                )
                for key, w in selected:
                    if key not in visited_bwd:
                        node = FeatureNode(key[0], key[1], key[2], key[3], w, hop=hop + 1)
                        graph.nodes[key] = node
                        visited_bwd.add(key)
                        if (node.sae_type in TRACE_SAE_TYPES
                                and node.layer > injection_layer):
                            next_targets.append(node)
                    graph.edges.append(FeatureEdge(
                        source_key=key, target_key=target.key, weight=w, hop=hop + 1
                    ))

            bwd_targets = next_targets
            print(f"  Backward hop {hop+1}: {len(next_targets)} new features")

    # ── Forward tracing (hop-1+) ──────────────────────────────────────────────
    if do_forward:
        max_layer   = model_wrapper.n_layers - 1
        fwd_sources = [
            graph.nodes[k] for k, _ in hop0_selected
            if graph.nodes[k].sae_type in TRACE_SAE_TYPES
            and graph.nodes[k].feature_id >= 0
            and graph.nodes[k].layer < max_layer
        ]

        # Build trial prompts for forward tracing (separate call in case forward
        # is run without backward; both use the same clean_dirty_fn).
        trial_prompts_fwd = {}
        for tn in trial_nums:
            result = clean_dirty_fn(tn)
            if result is not None:
                trial_prompts_fwd[tn] = result

        for hop in range(trace_depth):
            if not fwd_sources:
                break
            mpt = _get_from_list(max_per_type, hop + 1)
            print(f"\n  Forward hop {hop+1}: "
                  f"{len(fwd_sources)} sources (cap {mpt}/type)...")

            for source in fwd_sources:
                print(f"    {source.short_name()}...")
                for a in tqdm(feat_alphas, desc="    fwd PA α", leave=False):
                    for tn, (c_ids, d_ids) in trial_prompts_fwd.items():
                        extract_forward_pa_for_alpha(
                            model_wrapper, concept, c_ids, d_ids, a,
                            source.layer, source.sae_type,
                            source.feature_id, source.token_pos,
                            output_dir=output_dir,
                            trial_num=tn, device=device,
                        )

            # Compute PG_root(source) weighting curve and edge weights
            next_sources = []
            for source in fwd_sources:
                pg_curve = _load_curve_from_root_sa(
                    root_pa_dir, trial_nums, source, "steering_grad"
                )  # PG stored as "steering_grad" column

                src_name = (f"fwd_{source.sae_type}"
                            f"_L{source.layer}_F{source.feature_id}_T{source.token_pos}")
                # Forward parquets use a node with "fwd_" prefix so compute_edge_weights
                # finds the correct subdirectory (matches 17's convention)
                fwd_node = FeatureNode(
                    source.layer, f"fwd_{source.sae_type}",
                    source.feature_id, source.token_pos, 0, 0,
                )
                ew = compute_edge_weights(
                    output_dir, trial_nums,
                    optimal_strength=1.0,
                    target=fwd_node,
                    weighting_curve=pg_curve,
                )
                if not ew:
                    continue

                selected = select_features_from_edge_weights(
                    ew, mpt, frac_of_max, source.layer + 1
                )
                for key, w in selected:
                    if key not in visited_fwd:
                        node = FeatureNode(key[0], key[1], key[2], key[3], w, hop=hop + 1)
                        if key not in graph.nodes:
                            graph.nodes[key] = node
                        visited_fwd.add(key)
                        if node.sae_type in TRACE_SAE_TYPES and node.layer < max_layer:
                            next_sources.append(node)
                    graph.edges.append(FeatureEdge(
                        source_key=source.key, target_key=key, weight=w, hop=hop + 1
                    ))

            fwd_sources = next_sources
            print(f"  Forward hop {hop+1}: {len(next_sources)} new features")

    # Dedup edges: keep max |weight| per (source, target) pair
    seen_edges: Dict[Tuple, FeatureEdge] = {}
    for e in graph.edges:
        edge_key = (e.source_key, e.target_key)
        if (edge_key not in seen_edges
                or abs(e.weight) > abs(seen_edges[edge_key].weight)):
            seen_edges[edge_key] = e
    graph.edges = list(seen_edges.values())

    print(f"\n  Graph complete: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    return graph


# =============================================================================
# Simplified PA: single-point and multi-K discrete-sweep approaches
#
# Motivation (from paper §Q.3 & §Q.2):
#   The full approach in extract-all uses path integration over α ∈ [0,1] —
#   an artificial linear interpolation between clean and dirty embeddings —
#   to account for operating-point dependence caused by JumpReLU/softmax
#   nonlinearities.  This costs 4×N_ALPHAS forward passes (default ≈ 84).
#
#   For prefill injection, a more natural "strength axis" already exists:
#   K ∈ {0, 1, 2, 3}, the number of appended concept words.  Each K is a
#   real experimental condition with its own prompt, its own δ_K, and its
#   own operating point.  We can evaluate SA_K directly at each real prompt
#   and integrate over K — no artificial interpolation needed.
#
#   Single-point (K fixed): costs 4 forward passes — ~21× cheaper.
#   Multi-K (K=0..K_max): costs 4×(K_max+1) passes — still ~5× cheaper.
#
# Key difference from path-integral (18 extract-all):
#   • Each K uses a DIFFERENT δ_K = emb(dirty_K) − emb(clean_K) as the PG
#     tangent direction, so PG changes at each K (no cross-K caching).
#   • The "strength" axis K is discrete and bounded by the number of
#     APPEND_TEMPLATES (max 3), not a continuous 0→1 ramp.
#   • Features that appear only at high K are "signal-strength-dependent";
#     features that appear at K=1 already are likely more fundamental.
#
# What the comparison reveals:
#   • High overlap with path-integral results → the nonlinear correction from
#     integration is small; single-point is sufficient for this concept.
#   • New features in multi-K but not path-integral (or vice versa) → the
#     real prompt trajectory differs meaningfully from the linear embedding
#     interpolation; some features only activate on the genuine path.
#   • Features stable across K=1,2,3 → robust circuit nodes regardless of
#     injection "dose".
#
# SA_K(f) = GA_K(f) × PG_K(f)   at the dirty_K prompt operating point
# discrete-IPA(f) = Σ_K SA_K(f) × ΔK   (trapezoidal over K axis)
#
# Directories use "strength_K_00" naming (K treated as integer strength) so
# that compute_isa from 16 and 17's graph machinery work without changes.
# =============================================================================

def extract_pa_for_k(
    model_wrapper: ModelWrapper,
    concept: str,
    k: int,
    variant: str,
    baseline_word: str,
    clean_mode: str,
    trial_num: int,
    output_dir: Path,
    pos_token_id: int = DEFAULT_POS_TOKEN_ID,
    neg_token_id: int = DEFAULT_NEG_TOKEN_ID,
    sae_width: str = DEFAULT_SAE_WIDTH,
    sae_l0: str = DEFAULT_SAE_L0,
    device: str = "cuda",
    layer_indices: Optional[Set[int]] = None,
    compute_remainder: bool = True,
    loss_fn_override=None,
) -> Optional[Path]:
    """Extract single-point PA at a genuine K-word dirty prompt.

    Unlike extract_pa_for_alpha (which interpolates embeddings), this uses the
    *real* dirty prompt with K appended concept words as the operating point.
    The PG tangent direction is δ_K = emb(dirty_K) − emb(clean_K).

    Saves to output_dir / strength_{K}_00 / sa_trial{trial_num}.parquet so
    the results are fully compatible with compute_isa and the graph builder
    when ``optimal_strength`` is set to K_max and strengths are K values.

    Args:
        k:  Number of appended word templates (1..3).  K=0 gives SA≡0 and is
            skipped (clean prompt has no concept signal, δ=0 exactly).
    """
    if k == 0:
        # δ=0 at K=0 → PG=0 → SA=0 everywhere; nothing to save
        return None

    pair = _build_prompt_pair_for_trial(
        model_wrapper.tokenizer, concept, trial_num, variant,
        k, baseline_word, clean_mode,
    )
    if pair is None:
        return None
    clean_ids, dirty_ids = pair

    # K is stored as injection_strength so directories use "strength_K_00"
    return extract_pa_for_alpha(
        model_wrapper, concept, clean_ids, dirty_ids,
        alpha=float(k),             # K treated as integer "strength"
        output_dir=output_dir,
        trial_num=trial_num,
        pos_token_id=pos_token_id,
        neg_token_id=neg_token_id,
        sae_width=sae_width,
        sae_l0=sae_l0,
        device=device,
        layer_indices=layer_indices,
        compute_remainder=compute_remainder,
        loss_fn_override=loss_fn_override,
        extra_meta={"prefill_k": k},
    )


def extract_pa_multi_k(
    model_wrapper: ModelWrapper,
    concept: str,
    variant: str,
    k_values: List[int],
    baseline_word: str,
    clean_mode: str,
    trial_nums: List[int],
    output_dir: Path,
    pos_token_id: int = DEFAULT_POS_TOKEN_ID,
    neg_token_id: int = DEFAULT_NEG_TOKEN_ID,
    sae_width: str = DEFAULT_SAE_WIDTH,
    sae_l0: str = DEFAULT_SAE_L0,
    device: str = "cuda",
) -> None:
    """Extract single-point PA for each K in k_values using real dirty prompts.

    Each K uses its own δ_K = emb(dirty_K) − emb(clean_K) so the PG tangent
    direction reflects the genuine prompt difference at that K level.

    Saves to output_dir/strength_{K}_00/sa_trial{n}.parquet for each (K, trial).
    Then computes discrete-IPA over K via compute_ipa from 16 (treating K as
    the strength axis, so integration is ∫_0^K_max SA_K dK ≈ Σ_K SA_K).
    """
    for trial_num in trial_nums:
        for k in k_values:
            print(f"\n  K={k}, trial={trial_num}")
            extract_pa_for_k(
                model_wrapper, concept, k, variant,
                baseline_word, clean_mode, trial_num, output_dir,
                pos_token_id=pos_token_id, neg_token_id=neg_token_id,
                sae_width=sae_width, sae_l0=sae_l0, device=device,
            )

    # Discrete-IPA: treat K axis like the strength axis in 16's compute_isa
    print("\n  Computing discrete-K IPA...")
    compute_ipa(output_dir, trial_nums)


def build_attribution_graph_multi_k(
    model_wrapper: ModelWrapper,
    concept: str,
    variant: str,
    k_values: List[int],
    baseline_word: str,
    clean_mode: str,
    output_dir: Path,
    trial_nums: List[int] = None,
    trace_depth: int = DEFAULT_TRACE_DEPTH,
    max_per_type: List[int] = None,
    frac_of_max: float = DEFAULT_FRAC_OF_MAX,
    direction: str = DEFAULT_DIRECTION,
    pos_token_id: int = DEFAULT_POS_TOKEN_ID,
    neg_token_id: int = DEFAULT_NEG_TOKEN_ID,
    device: str = DEFAULT_DEVICE,
) -> AttributionGraph:
    """Build attribution graph using the multi-K discrete approach.

    Reuses build_attribution_graph_prefill from this file but supplies a
    clean_dirty_fn keyed to the *maximum K* (the strongest real condition),
    and sets optimal_strength = max(k_values) so the graph builder correctly
    scans strength_1_00..strength_{K_max}_00 directories for edge weights.

    The IPA integration over K values must already have been run (via
    extract_pa_multi_k).  Hop-1+ backward/forward tracing extracts
    feature-targeted SA at each K as operating points.
    """
    if trial_nums is None:
        trial_nums = [1]
    if max_per_type is None:
        max_per_type = DEFAULT_MAX_PER_TYPE

    k_max = max(k_values)

    # clean_dirty_fn for hop-1+ tracing uses K=k_max (strongest signal)
    def clean_dirty_fn(trial_num: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return _build_prompt_pair_for_trial(
            model_wrapper.tokenizer, concept, trial_num, variant,
            k_max, baseline_word, clean_mode,
        )

    # Patch: build_attribution_graph_prefill uses optimal_strength=1.0 which
    # would limit compute_edge_weights to strength ≤ 1.  We need K_max here.
    # Override by subclassing won't work cleanly, so we use the underlying
    # functions directly.
    injection_layer = -1
    root_pa_dir     = output_dir

    print("\n  Computing hop-0 edge weights from K-sweep IPA...")
    hop0_ew = compute_edge_weights(root_pa_dir, trial_nums, optimal_strength=float(k_max))
    if not hop0_ew:
        print("  ERROR: No edge weights. Run multi-k first.")
        return AttributionGraph(nodes={}, edges=[], optimal_strength=float(k_max))

    mpt0          = _get_from_list(max_per_type, 0)
    hop0_selected = select_features_from_edge_weights(
        hop0_ew, mpt0, frac_of_max, injection_layer,
    )

    graph = AttributionGraph(nodes={}, edges=[], optimal_strength=float(k_max))
    root  = FeatureNode(layer=-1, sae_type="root", feature_id=-1, token_pos=-1,
                        isa_value=0, hop=-1)
    graph.nodes[root.key] = root
    visited_bwd: Set[Tuple] = set()
    visited_fwd: Set[Tuple] = set()

    for key, ew in hop0_selected:
        node = FeatureNode(key[0], key[1], key[2], key[3], ew, hop=0)
        graph.nodes[key] = node
        visited_bwd.add(key)
        visited_fwd.add(key)
        graph.edges.append(FeatureEdge(
            source_key=key, target_key=graph.root_key, weight=ew, hop=0,
        ))

    do_backward = direction in ("backward", "both")
    do_forward  = direction in ("forward", "both")

    # Pre-build trial prompt cache for hop-1+ extraction
    trial_prompts: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for tn in trial_nums:
        result = clean_dirty_fn(tn)
        if result is not None:
            trial_prompts[tn] = result

    # ── Backward tracing ─────────────────────────────────────────────────────
    if do_backward:
        bwd_targets = [
            graph.nodes[k] for k, _ in hop0_selected
            if graph.nodes[k].sae_type in TRACE_SAE_TYPES
            and graph.nodes[k].feature_id >= 0
        ]
        pg_cache = output_dir / "pg_cache_multi_k"

        for hop in range(trace_depth):
            if not bwd_targets:
                break
            mpt = _get_from_list(max_per_type, hop + 1)
            print(f"\n  [multi-K] Backward hop {hop+1}: "
                  f"{len(bwd_targets)} targets (cap {mpt}/type)...")

            # Extract feature-targeted PA at each K for each target
            for target in bwd_targets:
                print(f"    {target.short_name()}...")
                for k in k_values:
                    for tn, (c_ids, d_ids) in trial_prompts.items():
                        # Rebuild prompt pair at this K (different δ)
                        pair_k = _build_prompt_pair_for_trial(
                            model_wrapper.tokenizer, concept, tn, variant,
                            k, baseline_word, clean_mode,
                        )
                        if pair_k is None:
                            continue
                        c_k, d_k = pair_k
                        seq_len = c_k.shape[1]
                        loss_fn = make_feature_target_loss_fn(
                            target.layer, target.sae_type, target.feature_id,
                            target.token_pos,
                            DEFAULT_SAE_WIDTH, DEFAULT_SAE_L0,
                            model_wrapper.model_name, device, seq_len,
                        )
                        tgt_name = (f"{target.sae_type}_L{target.layer}"
                                    f"_F{target.feature_id}_T{target.token_pos}")
                        tgt_dir  = output_dir / "feat_sa" / tgt_name
                        extract_pa_for_alpha(
                            model_wrapper, concept, c_k, d_k, float(k),
                            output_dir=tgt_dir, trial_num=tn,
                            device=device, pg_cache_dir=pg_cache,
                            loss_fn_override=loss_fn, compute_remainder=False,
                            output_filename="feat_sa_trial{trial_num}.parquet",
                        )

            next_targets = []
            for target in bwd_targets:
                ew = compute_edge_weights(
                    output_dir, trial_nums,
                    optimal_strength=float(k_max),
                    target=target, root_sa_dir=root_pa_dir,
                )
                if not ew:
                    continue
                selected = select_features_from_edge_weights(
                    ew, mpt, frac_of_max, injection_layer,
                )
                for key, w in selected:
                    if key not in visited_bwd:
                        node = FeatureNode(key[0], key[1], key[2], key[3], w, hop=hop + 1)
                        graph.nodes[key] = node
                        visited_bwd.add(key)
                        if node.sae_type in TRACE_SAE_TYPES:
                            next_targets.append(node)
                    graph.edges.append(FeatureEdge(
                        source_key=key, target_key=target.key, weight=w, hop=hop + 1,
                    ))
            bwd_targets = next_targets

    # ── Forward tracing ───────────────────────────────────────────────────────
    if do_forward:
        max_layer   = model_wrapper.n_layers - 1
        fwd_sources = [
            graph.nodes[k] for k, _ in hop0_selected
            if graph.nodes[k].sae_type in TRACE_SAE_TYPES
            and graph.nodes[k].feature_id >= 0
            and graph.nodes[k].layer < max_layer
        ]
        for hop in range(trace_depth):
            if not fwd_sources:
                break
            mpt = _get_from_list(max_per_type, hop + 1)
            print(f"\n  [multi-K] Forward hop {hop+1}: "
                  f"{len(fwd_sources)} sources (cap {mpt}/type)...")

            for source in fwd_sources:
                print(f"    {source.short_name()}...")
                for k in k_values:
                    for tn in trial_prompts:
                        pair_k = _build_prompt_pair_for_trial(
                            model_wrapper.tokenizer, concept, tn, variant,
                            k, baseline_word, clean_mode,
                        )
                        if pair_k is None:
                            continue
                        c_k, d_k = pair_k
                        extract_forward_pa_for_alpha(
                            model_wrapper, concept, c_k, d_k, float(k),
                            source.layer, source.sae_type,
                            source.feature_id, source.token_pos,
                            output_dir=output_dir,
                            trial_num=tn, device=device,
                        )

            next_sources = []
            for source in fwd_sources:
                pg_curve = _load_curve_from_root_sa(
                    root_pa_dir, trial_nums, source, "steering_grad",
                )
                fwd_node = FeatureNode(
                    source.layer, f"fwd_{source.sae_type}",
                    source.feature_id, source.token_pos, 0, 0,
                )
                ew = compute_edge_weights(
                    output_dir, trial_nums,
                    optimal_strength=float(k_max),
                    target=fwd_node, weighting_curve=pg_curve,
                )
                if not ew:
                    continue
                selected = select_features_from_edge_weights(
                    ew, mpt, frac_of_max, source.layer + 1,
                )
                for key, w in selected:
                    if key not in visited_fwd:
                        node = FeatureNode(key[0], key[1], key[2], key[3], w, hop=hop + 1)
                        if key not in graph.nodes:
                            graph.nodes[key] = node
                        visited_fwd.add(key)
                        if node.sae_type in TRACE_SAE_TYPES and node.layer < max_layer:
                            next_sources.append(node)
                    graph.edges.append(FeatureEdge(
                        source_key=source.key, target_key=key, weight=w, hop=hop + 1,
                    ))
            fwd_sources = next_sources

    # Dedup edges
    seen_edges: Dict[Tuple, FeatureEdge] = {}
    for e in graph.edges:
        edge_key = (e.source_key, e.target_key)
        if (edge_key not in seen_edges
                or abs(e.weight) > abs(seen_edges[edge_key].weight)):
            seen_edges[edge_key] = e
    graph.edges = list(seen_edges.values())

    print(f"\n  [multi-K] Graph complete: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    return graph


# =============================================================================
# CLI commands
# =============================================================================

def _run_name(concept: str, variant: str) -> str:
    return f"{concept.replace(' ', '_')}_{variant}"


def run_single_point(args) -> None:
    """Extract PA at the real dirty prompt (K=n_extra_words, α=1-equivalent).

    ~21× cheaper than extract-all (4 forward passes instead of 4×N_ALPHAS).
    Saves as strength_{K}_00/sa_trial{n}.parquet.  After extraction you can
    run build-graph with --mode multi-k --k-values K to visualize.
    """
    concept    = args.concept
    variant    = args.variant
    run        = _run_name(concept, variant)
    output_dir = Path(args.output_dir) / args.model / (run + "_multiK")
    output_dir.mkdir(parents=True, exist_ok=True)

    k = args.n_extra_words
    trial_nums = list(range(1, args.n_trials + 1))

    print("=" * 70)
    print(f"SINGLE-POINT: {concept} | variant={variant} | K={k}")
    print(f"  ~21× cheaper than extract-all (no path integration)")
    print("=" * 70)

    mw = load_model(args.model, device=args.device, dtype=args.dtype,
                    quantization=getattr(args, "quantization", None))

    for trial_num in trial_nums:
        print(f"\n  Trial {trial_num}...")
        extract_pa_for_k(
            mw, concept, k, variant, args.baseline_word, args.clean_mode,
            trial_num, output_dir,
            pos_token_id=args.pos_token_id, neg_token_id=args.neg_token_id,
            sae_width=args.sae_width, sae_l0=args.sae_l0, device=args.device,
        )

    # compute_isa treats a single strength as degenerate (needs ≥ 2 points)
    # so we record a "trivial IPA" equal to SA directly
    print("\n  NOTE: single-point has no integration axis; ISA = SA at K.")
    print(f"  To compare with extract-all, run multi-k with --k-values 1 2 3")

    config = vars(args)
    config["mode"] = "single_point"
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"\n  Done. Output: {output_dir}")


def run_multi_k(args) -> None:
    """Extract PA at each K ∈ {1..n_extra_words} using real dirty prompts.

    Each K uses its own δ_K = emb(dirty_K) − emb(clean_K) as the PG tangent,
    so the operating point and perturbation direction are both genuine.
    Integrates over K axis (trapezoidal) → discrete-IPA.

    Cost: 4×K_max forward passes  (vs. 4×21 for extract-all with N_ALPHAS=21)

    Key comparison with extract-all:
      • Features stable across K=1,2,3 → robust circuit nodes
      • Features only at K=3 → signal-strength-dependent
      • Overlap with extract-all → path-integral correction is negligible
      • Divergence from extract-all → nonlinear path effects are important
    """
    concept    = args.concept
    variant    = args.variant
    run        = _run_name(concept, variant)
    output_dir = Path(args.output_dir) / args.model / (run + "_multiK")
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values   = list(range(1, args.n_extra_words + 1))
    trial_nums = list(range(1, args.n_trials + 1))

    print("=" * 70)
    print(f"MULTI-K: {concept} | variant={variant} | K={k_values}")
    print(f"  Each K uses a genuine dirty prompt (no embedding interpolation)")
    print("=" * 70)

    mw = load_model(args.model, device=args.device, dtype=args.dtype,
                    quantization=getattr(args, "quantization", None))

    extract_pa_multi_k(
        mw, concept, variant, k_values, args.baseline_word, args.clean_mode,
        trial_nums, output_dir,
        pos_token_id=args.pos_token_id, neg_token_id=args.neg_token_id,
        sae_width=args.sae_width, sae_l0=args.sae_l0, device=args.device,
    )

    config = vars(args)
    config["mode"] = "multi_k"
    config["k_values"] = k_values
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"\n  Done. Output: {output_dir}")


def run_build_graph_multi_k(args) -> None:
    """Build attribution graph from multi-K discrete PA data."""
    concept    = args.concept
    variant    = args.variant
    run        = _run_name(concept, variant)
    output_dir = Path(args.output_dir) / args.model / (run + "_multiK")

    # Load config
    k_values      = list(range(1, args.n_extra_words + 1))
    baseline_word = args.baseline_word
    clean_mode    = args.clean_mode
    cfg_path = output_dir / "run_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        k_values      = cfg.get("k_values", k_values)
        baseline_word = cfg.get("baseline_word", baseline_word)
        clean_mode    = cfg.get("clean_mode", clean_mode)

    trial_nums = list(range(1, args.n_trials + 1))

    print("=" * 70)
    print(f"BUILD-GRAPH (multi-K): {concept} | variant={variant} | K={k_values}")
    print("=" * 70)

    mw = load_model(args.model, device=args.device, dtype=args.dtype,
                    quantization=getattr(args, "quantization", None))

    graph = build_attribution_graph_multi_k(
        model_wrapper=mw,
        concept=concept,
        variant=variant,
        k_values=k_values,
        baseline_word=baseline_word,
        clean_mode=clean_mode,
        output_dir=output_dir,
        trial_nums=trial_nums,
        trace_depth=args.trace_depth,
        max_per_type=args.max_per_type,
        frac_of_max=args.frac_of_max,
        direction=args.direction,
        pos_token_id=args.pos_token_id,
        neg_token_id=args.neg_token_id,
        device=args.device,
    )

    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    export_graph_json(graph, graphs_dir / "attribution_graph.json")
    render_interactive(graph, graphs_dir / "attribution_graph.html")
    render_pdf_from_html(
        graphs_dir / "attribution_graph.html",
        graphs_dir / "attribution_graph.pdf",
    )
    write_graph_summary(graph, graphs_dir / "attribution_graph_summary.txt",
                        concept, layer=-1)
    print(f"\n  Done. Graph saved to {graphs_dir}")


def run_extract_all(args) -> None:
    """Extract PA at all α ∈ [0, 1] and compute IPA."""
    concept    = args.concept
    variant    = args.variant
    run        = _run_name(concept, variant)
    output_dir = Path(args.output_dir) / args.model / run
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"EXTRACT-ALL: {concept} | variant={variant} | K={args.n_extra_words}")
    print("=" * 70)

    mw = load_model(args.model, device=args.device, dtype=args.dtype,
                    quantization=getattr(args, "quantization", None))

    # Build prompt pair for requested trial(s)
    trial_nums = list(range(1, args.n_trials + 1))
    alphas     = np.linspace(0.0, 1.0, args.n_alphas).tolist()

    pg_cache = output_dir / "pg_cache"

    for trial_num in trial_nums:
        pair = _build_prompt_pair_for_trial(
            mw.tokenizer, concept, trial_num, variant,
            args.n_extra_words, args.baseline_word, args.clean_mode,
        )
        if pair is None:
            print(f"  Skipping trial {trial_num} (token length mismatch)")
            continue
        clean_ids, dirty_ids = pair

        print(f"\n  Trial {trial_num}: seq_len={clean_ids.shape[1]}")
        for a in tqdm(alphas, desc=f"  Trial {trial_num} α"):
            extract_pa_for_alpha(
                mw, concept, clean_ids, dirty_ids, a,
                output_dir=output_dir, trial_num=trial_num,
                pos_token_id=args.pos_token_id, neg_token_id=args.neg_token_id,
                sae_width=args.sae_width, sae_l0=args.sae_l0,
                device=args.device,
                pg_cache_dir=pg_cache, save_pg_cache=True,
            )

    # Compute IPA
    print("\n  Computing IPA...")
    compute_ipa(output_dir, trial_nums)

    # Save run config
    config = vars(args)
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"\n  Done. Output: {output_dir}")


def run_build_graph(args) -> None:
    """Build attribution graph from existing PA/IPA parquets."""
    concept    = args.concept
    variant    = args.variant
    run        = _run_name(concept, variant)
    output_dir = Path(args.output_dir) / args.model / run

    print("=" * 70)
    print(f"BUILD-GRAPH: {concept} | variant={variant} | direction={args.direction}")
    print("=" * 70)

    # Load config if available
    cfg_path = output_dir / "run_config.json"
    n_extra_words = args.n_extra_words
    baseline_word = args.baseline_word
    clean_mode    = args.clean_mode
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        n_extra_words = cfg.get("n_extra_words", n_extra_words)
        baseline_word = cfg.get("baseline_word", baseline_word)
        clean_mode    = cfg.get("clean_mode", clean_mode)

    mw = load_model(args.model, device=args.device, dtype=args.dtype,
                    quantization=getattr(args, "quantization", None))

    trial_nums = list(range(1, args.n_trials + 1))

    def clean_dirty_fn(trial_num):
        return _build_prompt_pair_for_trial(
            mw.tokenizer, concept, trial_num, variant,
            n_extra_words, baseline_word, clean_mode,
        )

    graph = build_attribution_graph_prefill(
        model_wrapper=mw,
        concept=concept,
        clean_dirty_fn=clean_dirty_fn,
        output_dir=output_dir,
        trial_nums=trial_nums,
        trace_depth=args.trace_depth,
        n_alphas_feat_pa=args.n_alphas,
        max_per_type=args.max_per_type,
        frac_of_max=args.frac_of_max,
        direction=args.direction,
        pos_token_id=args.pos_token_id,
        neg_token_id=args.neg_token_id,
        device=args.device,
    )

    # Save & visualize
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    graph_json = graphs_dir / "attribution_graph.json"
    export_graph_json(graph, graph_json)

    html_path = graphs_dir / "attribution_graph.html"
    render_interactive(graph, html_path)

    pdf_path = graphs_dir / "attribution_graph.pdf"
    render_pdf_from_html(html_path, pdf_path)

    txt_path = graphs_dir / "attribution_graph_summary.txt"
    write_graph_summary(graph, txt_path, concept, layer=-1)

    print(f"\n  Done. Graph saved to {graphs_dir}")


def run_visualize(args) -> None:
    """Re-render an existing attribution graph from JSON."""
    concept    = args.concept
    variant    = args.variant
    run        = _run_name(concept, variant)
    graphs_dir = Path(args.output_dir) / args.model / run / "graphs"
    graph_json = graphs_dir / "attribution_graph.json"

    if not graph_json.exists():
        print(f"  ERROR: {graph_json} not found. Run build-graph first.")
        return

    with open(graph_json) as f:
        data = json.load(f)

    nodes = {
        (int(n["layer"]), str(n["sae_type"]), int(n["feature_id"]), int(n["token_pos"])):
        FeatureNode(
            layer=int(n["layer"]), sae_type=str(n["sae_type"]),
            feature_id=int(n["feature_id"]), token_pos=int(n["token_pos"]),
            isa_value=float(n["isa_value"]), hop=int(n["hop"]),
            label=n.get("label"),
        )
        for n in data["nodes"]
    }
    edges = [
        FeatureEdge(
            source_key=tuple(e["source"]),
            target_key=tuple(e["target"]),
            weight=e["weight"], hop=e["hop"],
        )
        for e in data["edges"]
    ]
    graph = AttributionGraph(
        nodes=nodes, edges=edges, optimal_strength=data.get("optimal_strength", 1.0)
    )

    html_path = graphs_dir / "attribution_graph.html"
    render_interactive(graph, html_path)
    pdf_path = graphs_dir / "attribution_graph.pdf"
    render_pdf_from_html(html_path, pdf_path)
    print(f"  Rendered: {html_path}")


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prefill Attribution Graph (18)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ── Shared parent parser ─────────────────────────────────────────────────
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--concept",        required=True,
                        help="Concept name, e.g. Bread")
    shared.add_argument("--variant",        default=DEFAULT_VARIANT,
                        choices=["append_user", "replace_assistant"],
                        help="Prompt variant from 06b")
    shared.add_argument("--model",          default=DEFAULT_MODEL)
    shared.add_argument("--output-dir",     default=DEFAULT_OUTPUT_DIR)
    shared.add_argument("--device",         default=DEFAULT_DEVICE)
    shared.add_argument("--dtype",          default=DEFAULT_DTYPE)
    shared.add_argument("--sae-width",      default=DEFAULT_SAE_WIDTH, dest="sae_width")
    shared.add_argument("--sae-l0",         default=DEFAULT_SAE_L0,    dest="sae_l0")
    shared.add_argument("--pos-token-id",   default=DEFAULT_POS_TOKEN_ID, type=int,
                        dest="pos_token_id")
    shared.add_argument("--neg-token-id",   default=DEFAULT_NEG_TOKEN_ID, type=int,
                        dest="neg_token_id")
    shared.add_argument("--n-trials",       default=1, type=int, dest="n_trials")
    shared.add_argument("--n-alphas",       default=DEFAULT_N_ALPHAS, type=int,
                        dest="n_alphas",
                        help="Number of α ∈ [0,1] integration points")

    # Prompt-pair parameters
    shared.add_argument("--n-extra-words",  default=DEFAULT_N_EXTRA_WORDS, type=int,
                        dest="n_extra_words",
                        help="K: number of appended word templates (1-3)")
    shared.add_argument("--baseline-word",  default=DEFAULT_BASELINE_WORD,
                        dest="baseline_word",
                        help="Baseline (clean) word for append_user filler_word mode")
    shared.add_argument("--clean-mode",     default=DEFAULT_CLEAN_MODE,
                        dest="clean_mode",
                        choices=["filler_word", "natural_sentence"],
                        help="How clean prompt is built for append_user variant")

    # ── single-point ─────────────────────────────────────────────────────────
    p_sp = sub.add_parser("single-point", parents=[shared],
                          help="Single-point PA at dirty prompt (K fixed, ~21× faster)")
    p_sp.set_defaults(func=run_single_point)

    # ── multi-k ──────────────────────────────────────────────────────────────
    p_mk = sub.add_parser("multi-k", parents=[shared],
                          help="PA at each K=1..n_extra_words using real dirty prompts")
    p_mk.set_defaults(func=run_multi_k)

    # ── build-graph-multi-k ───────────────────────────────────────────────────
    p_gmk = sub.add_parser("build-graph-multi-k", parents=[shared],
                            help="Build attribution graph from multi-K PA data")
    p_gmk.add_argument("--direction",    default=DEFAULT_DIRECTION,
                       choices=["backward", "forward", "both"])
    p_gmk.add_argument("--trace-depth",  default=DEFAULT_TRACE_DEPTH, type=int,
                       dest="trace_depth")
    p_gmk.add_argument("--max-per-type", default=DEFAULT_MAX_PER_TYPE, type=int,
                       nargs="+", dest="max_per_type", metavar="N")
    p_gmk.add_argument("--frac-of-max",  default=DEFAULT_FRAC_OF_MAX, type=float,
                       dest="frac_of_max")
    p_gmk.set_defaults(func=run_build_graph_multi_k)

    # ── extract-all ──────────────────────────────────────────────────────────
    p_ext = sub.add_parser("extract-all", parents=[shared],
                           help="Extract PA at all α values + compute IPA")
    p_ext.set_defaults(func=run_extract_all)

    # ── build-graph ──────────────────────────────────────────────────────────
    p_graph = sub.add_parser("build-graph", parents=[shared],
                              help="Build attribution graph from PA/IPA data")
    p_graph.add_argument("--direction",     default=DEFAULT_DIRECTION,
                         choices=["backward", "forward", "both"])
    p_graph.add_argument("--trace-depth",   default=DEFAULT_TRACE_DEPTH, type=int,
                         dest="trace_depth")
    p_graph.add_argument("--max-per-type",  default=DEFAULT_MAX_PER_TYPE, type=int,
                         nargs="+", dest="max_per_type",
                         metavar="N",
                         help="Per-hop per-type feature cap [8 5 3 2]")
    p_graph.add_argument("--frac-of-max",   default=DEFAULT_FRAC_OF_MAX, type=float,
                         dest="frac_of_max")
    p_graph.set_defaults(func=run_build_graph)

    # ── visualize ────────────────────────────────────────────────────────────
    p_vis = sub.add_parser("visualize", parents=[shared],
                            help="Re-render graph from saved JSON")
    p_vis.set_defaults(func=run_visualize)

    # ── all ──────────────────────────────────────────────────────────────────
    p_all = sub.add_parser("all", parents=[shared],
                           help="extract-all → build-graph → visualize")
    p_all.add_argument("--direction",       default=DEFAULT_DIRECTION,
                       choices=["backward", "forward", "both"])
    p_all.add_argument("--trace-depth",     default=DEFAULT_TRACE_DEPTH, type=int,
                       dest="trace_depth")
    p_all.add_argument("--max-per-type",    default=DEFAULT_MAX_PER_TYPE, type=int,
                       nargs="+", dest="max_per_type", metavar="N")
    p_all.add_argument("--frac-of-max",     default=DEFAULT_FRAC_OF_MAX, type=float,
                       dest="frac_of_max")
    p_all.set_defaults(func=None)   # handled in main

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.command is None:
        print("No command given. Use --help for usage.")
        sys.exit(1)

    if args.command == "all":
        run_extract_all(args)
        run_build_graph(args)
        run_visualize(args)
    elif hasattr(args, "func") and args.func is not None:
        args.func(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 8 — Collect per-concept *evidence-carrier* features for one gate.

An evidence carrier is an upstream transcoder feature that, while the concept is
active, writes *against* a chosen downstream gate feature — it has **negative
attribution** to the gate (``attribution = virtual_weight x activation < 0``):
  * ``virtual_weight(gate, f) = dot(gate_encoder, decoder[f])`` — structural
    connectivity from the candidate's decoder write into what the gate reads,
  * ``activation(f)`` — the candidate's steered last-token feature activation.
This is the 09b_causal_pathway.py pipeline, reduced to the two settings we want
to compare and made self-contained on top of ``gate_lib``/``prompts``.

Because carriers are concept-specific, activations are collected **per concept**
(averaged over trials), not pooled into a single global mean. For each setting
(injection vs prefill) and each concept we:
  1. run the Yes/No introspection prompt with the concept either injected as a
     steering vector (injection) or appended to the prompt (prefill), capturing
     the answer-position transcoder feature activations at every candidate layer
     and at the gate layer (steered forward),
  2. run the bare prompt with no concept (unsteered forward) for the same
     candidate layers,
  3. select candidate features whose activation *changes* with the concept
     (``|steered - unsteered| > threshold``),
  4. score each candidate ``attribution = virtual_weight x steered_activation``
     and keep the ``n_top`` most-negative (the evidence carriers).

The **gate feature** is shared across both settings so the carrier sets are
directly comparable. By default it is the top global gate (most-negative DLA) of
the ``--injection-gates`` ranking; override with ``--gate-feature L{G}_F{id}``.

Candidate layers default to ``[steering_layer+1 .. gate_layer-1]``. For the
prefill setting the concept word is in the prompt from the very first layer, so
``--prefill-start-layer 1`` lets carriers be collected from layer 1 onward (the
injection setting still scans only post-injection layers).

Outputs (per setting) under
``--output-dir/{model}/{setting_tag}_L{layer}_s{strength}/carriers/``:
  * ``carriers.json`` — config, gate feature, and per-concept records split into
    ``carriers`` (attribution < 0), ``suppressors`` (attribution > 0) and a
    ``weak_control`` decile (lowest |attribution|), plus a pooled (cross-concept)
    carrier ranking used by step 9's comparison and step 10's ablation sweep.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import gate_lib as gl
import prompts as P
from model_utils import load_model


def parse_gate_feature(spec: str) -> Tuple[int, int]:
    """Parse a ``L{layer}_F{feat_id}`` gate-feature spec into (layer, feat_id)."""
    s = spec.strip().upper().replace(" ", "")
    if not (s.startswith("L") and "_F" in s):
        raise ValueError(f"--gate-feature must look like 'L45_F9959', got {spec!r}")
    layer_str, feat_str = s[1:].split("_F", 1)
    return int(layer_str), int(feat_str)


def pick_gate_feature(
    gates: dict, override: Optional[str], start_layer: int
) -> Tuple[int, int]:
    """Return the gate feature (layer, feat_id) to attribute carriers to.

    ``override`` (``L{G}_F{id}``) wins; otherwise the top global gate (first in
    the most-negative-DLA ranking) that sits *above* ``start_layer`` so at least
    one candidate layer exists below it.
    """
    if override is not None:
        return parse_gate_feature(override)
    for rec in gates["global_gates"]:
        if int(rec["layer"]) > start_layer:
            return int(rec["layer"]), int(rec["feature_id"])
    raise ValueError(
        f"No global gate above start layer {start_layer}; pass --gate-feature."
    )


def collect_setting_carriers(
    model_w,
    setting: str,
    prefill_variant: str,
    detect_variant: str,
    concepts: List[str],
    candidate_layers: List[int],
    gate_layer: int,
    gate_feat_id: int,
    steering_layer: int,
    strength: float,
    n_trials: int,
    n_extra_words: int,
    concept_vectors: Dict[str, torch.Tensor],
    yes_ids: List[int],
    no_ids: List[int],
    threshold: float,
    n_top: int,
    output_dir: Path,
) -> Dict:
    """Discover evidence carriers for ``setting`` against one gate feature.

    Returns the results dict (also written to ``carriers.json``).
    """
    assert setting in ("injection", "prefill")
    device = model_w.device
    scan_layers = list(candidate_layers) + [gate_layer]

    # Phase A: one steered + one unsteered forward per concept x trial, capturing
    # the answer-position pre-MLP-LN at all candidate layers (+ gate layer for
    # the steered pass) — a single d_model vector per layer per sample.
    steered_acts: Dict[int, List[torch.Tensor]] = {L: [] for L in scan_layers}
    unsteered_acts: Dict[int, List[torch.Tensor]] = {L: [] for L in candidate_layers}
    sample_concepts: List[str] = []
    detection_gaps: List[float] = []

    desc = f"[{setting}] collect"
    for concept in tqdm(concepts, desc=desc):
        for trial in range(1, n_trials + 1):
            prefill_word = concept if setting == "prefill" else None
            input_ids = P.format_detection_prompt(
                model_w.tokenizer,
                trial,
                prefill_word,
                n_extra_words,
                prefill_variant=prefill_variant,
                detect_variant=detect_variant,
            )

            # Steered/active forward (injection hook for the injection setting).
            handle = None
            if setting == "injection":
                hook = gl.make_injection_hook(concept_vectors[concept], strength)
                handle = model_w.get_layer_module(steering_layer).register_forward_hook(hook)
            try:
                with gl.TranscoderCapture(model_w, scan_layers) as cap:
                    with torch.no_grad():
                        out = model_w.model(input_ids=input_ids.to(device), use_cache=False)
                    logits = out.logits[:, -1, :].float().cpu()
                    for L in scan_layers:
                        steered_acts[L].append(cap.captured[L][0, -1, :].detach().float().cpu())
            finally:
                if handle is not None:
                    handle.remove()
                    torch.cuda.empty_cache()

            detection_gaps.append(P.logit_gap(logits, yes_ids, no_ids).item())

            # Unsteered baseline forward (bare prompt, no concept) — defines the
            # delta-from-control used to select concept-responsive candidates.
            base_ids = P.format_detection_prompt(
                model_w.tokenizer,
                trial,
                None,
                n_extra_words,
                prefill_variant=prefill_variant,
                detect_variant=detect_variant,
            )
            with gl.TranscoderCapture(model_w, candidate_layers) as cap0:
                with torch.no_grad():
                    model_w.model(input_ids=base_ids.to(device), use_cache=False)
                for L in candidate_layers:
                    unsteered_acts[L].append(cap0.captured[L][0, -1, :].detach().float().cpu())

            sample_concepts.append(concept)

    detection_rate = float((torch.tensor(detection_gaps) > 0).float().mean())

    # Map each concept to its sample row indices (trials) for per-concept means.
    concept_rows: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(sample_concepts):
        concept_rows[c].append(i)
    ordered_concepts = list(concepts)

    # Phase B0: load the gate transcoder once — gate encoder vector (for virtual
    # weights) and the per-concept gate-feature activation.
    gate_tc = gl.load_transcoder(gate_layer, device=device)
    gate_encoder = gl.gate_encoder_vector(gate_tc, gate_feat_id)  # [d_model] fp32 cpu
    gate_steered = torch.stack(steered_acts[gate_layer]).to(device, gate_tc.w_enc.dtype)
    with torch.no_grad():
        gate_feats = gate_tc.encode(gate_steered).float().cpu()  # [n_samples, F]
    gate_act_per_concept: Dict[str, float] = {}
    for c in ordered_concepts:
        rows = torch.tensor(concept_rows[c])
        gate_act_per_concept[c] = float(gate_feats[rows, gate_feat_id].mean())
    del gate_tc, gate_steered, gate_feats
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Phase B: per candidate layer — encode steered + unsteered, average per
    # concept, then attribute. We accumulate *all* per-concept nonzero candidate
    # records (so each concept can later be split into evidence carriers /
    # suppressors / weak-attribution control, as the ablation sweep needs) and a
    # pooled (cross-concept) aggregate of the carriers.
    per_concept_records: Dict[str, List[Dict]] = {c: [] for c in ordered_concepts}
    per_concept_stats: Dict[str, Dict[str, int]] = {
        c: {"n_candidates": 0} for c in ordered_concepts
    }
    # pooled[(layer, feat)] -> list of attribution / vw across concepts (carriers only).
    pooled: Dict[Tuple[int, int], Dict[str, List[float]]] = defaultdict(
        lambda: {"attribution": [], "virtual_weight": [], "activation": []}
    )

    for L in tqdm(candidate_layers, desc=f"[{setting}] encode+attribute"):
        tc = gl.load_transcoder(L, device=device)
        steered_stack = torch.stack(steered_acts[L]).to(device, tc.w_enc.dtype)
        unsteered_stack = torch.stack(unsteered_acts[L]).to(device, tc.w_enc.dtype)
        with torch.no_grad():
            steered_feats = tc.encode(steered_stack).float().cpu()      # [n_samples, F]
            unsteered_feats = tc.encode(unsteered_stack).float().cpu()  # [n_samples, F]

        for c in ordered_concepts:
            rows = torch.tensor(concept_rows[c])
            steered_c = steered_feats[rows].mean(dim=0)       # [F]
            unsteered_c = unsteered_feats[rows].mean(dim=0)   # [F]
            delta = steered_c - unsteered_c                   # [F]

            # Concept-responsive candidates: activation changes with the concept.
            cand_ids = torch.nonzero(delta.abs() > threshold, as_tuple=False).flatten()
            if cand_ids.numel() == 0:
                continue
            feat_ids = cand_ids.tolist()
            vw = gl.virtual_weights(tc, gate_encoder, feat_ids)  # [k]
            act = steered_c[cand_ids]                            # [k] steered activation
            attribution = vw * act                               # [k]

            per_concept_stats[c]["n_candidates"] += len(feat_ids)
            for j, fid in enumerate(feat_ids):
                a = float(attribution[j])
                if a == 0.0:
                    continue
                rec = {
                    "layer": int(L),
                    "feat_id": int(fid),
                    "virtual_weight": float(vw[j]),
                    "activation": float(act[j]),
                    "attribution": a,
                }
                per_concept_records[c].append(rec)
                if a < 0:  # evidence carrier — contributes to the pooled ranking
                    key = (int(L), int(fid))
                    pooled[key]["attribution"].append(a)
                    pooled[key]["virtual_weight"].append(float(vw[j]))
                    pooled[key]["activation"].append(float(act[j]))

        del tc, steered_stack, unsteered_stack, steered_feats, unsteered_feats
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Per-concept: split into evidence carriers (attribution < 0), suppressors
    # (attribution > 0) and a weak-attribution control (the lowest-|attribution|
    # decile), each capped at ``n_top``. These three buckets are exactly the
    # ablation groups recreated in the gate-activation sweep (step 10).
    per_concept_out: Dict[str, Dict] = {}
    for c in ordered_concepts:
        recs = per_concept_records[c]
        carriers, suppressors = gl.split_carriers(recs)  # neg most-neg-first / pos most-pos-first
        n_carriers_total, n_suppressors_total = len(carriers), len(suppressors)
        carriers = carriers[:n_top]
        suppressors = suppressors[:n_top]
        n_weak = max(1, len(recs) // 10)
        weak_control = sorted(recs, key=lambda r: abs(r["attribution"]))[:n_weak][:n_top]
        per_concept_out[c] = {
            "gate_activation": gate_act_per_concept[c],
            "n_candidates": per_concept_stats[c]["n_candidates"],
            "n_carriers": n_carriers_total,
            "n_suppressors": n_suppressors_total,
            "n_weak_control": len(weak_control),
            "carriers": carriers,
            "suppressors": suppressors,
            "weak_control": weak_control,
        }

    # Pooled cross-concept ranking: mean attribution over the concepts where the
    # feature is an evidence carrier, ranked most-negative-first.
    pooled_records = []
    for (L, fid), agg in pooled.items():
        n = len(agg["attribution"])
        pooled_records.append({
            "layer": L,
            "feat_id": fid,
            "mean_attribution": float(sum(agg["attribution"]) / n),
            "mean_virtual_weight": float(sum(agg["virtual_weight"]) / n),
            "mean_activation": float(sum(agg["activation"]) / n),
            "n_concepts_active": n,
        })
    pooled_records.sort(key=lambda r: r["mean_attribution"])  # most negative first

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "setting": setting,
            "model": model_w.model_name,
            "steering_layer": steering_layer,
            "strength": strength,
            "n_trials": n_trials,
            "n_extra_words": n_extra_words,
            "prefill_variant": prefill_variant,
            "detect_variant": detect_variant,
            "candidate_layers": list(candidate_layers),
            "concepts": ordered_concepts,
            "threshold": threshold,
            "n_top": n_top,
            "yes_ids": yes_ids,
            "no_ids": no_ids,
        },
        "gate_feature": {"layer": int(gate_layer), "feat_id": int(gate_feat_id)},
        "baseline_detection_rate": detection_rate,
        "gate_activation_mean": float(
            sum(gate_act_per_concept.values()) / max(1, len(gate_act_per_concept))
        ),
        "per_concept": per_concept_out,
        "pooled_carriers": pooled_records,
    }
    with open(output_dir / "carriers.json", "w") as f:
        json.dump(results, f, indent=2)
    total_carriers = sum(v["n_carriers"] for v in per_concept_out.values())
    print(
        f"[{setting}] gate L{gate_layer}_F{gate_feat_id}  "
        f"detection_rate={detection_rate:.3f}  "
        f"total carriers={total_carriers}  pooled unique={len(pooled_records)}  "
        f"saved -> {output_dir / 'carriers.json'}"
    )
    return results


def main():
    p = argparse.ArgumentParser(description="Collect per-concept evidence-carrier features.")
    p.add_argument("--model", "-m", default="gemma3_27b")
    p.add_argument("--settings", nargs="+", default=["injection", "prefill"],
                   choices=["injection", "prefill"])
    p.add_argument("--injection-gates", required=True,
                   help="injection gates.json (source of the shared gate feature & concepts).")
    p.add_argument(
        "--gate-feature",
        default=None,
        help="Gate feature 'L{G}_F{id}' to attribute carriers to. Default: the "
             "top global gate (most-negative DLA) above the start layer.",
    )
    p.add_argument("--steering-layer", type=int, default=gl.STEERING_LAYER)
    p.add_argument("--start-layer", type=int, default=None,
                   help="First candidate (upstream) layer, inclusive. Default: steering_layer+1.")
    p.add_argument(
        "--prefill-start-layer",
        type=int,
        default=None,
        help="First candidate (upstream) layer for the PREFILL setting only. "
             "Default: same as --start-layer. Set to 1 to collect evidence carriers "
             "from the very first layer, since in prefill the concept word is present "
             "in the prompt from the start (unlike injection, which only appears at "
             "the steering layer).",
    )
    p.add_argument("--strength", type=float, default=None,
                   help="Injection strength. Default: from injection gates.json.")
    p.add_argument("--n-trials", "-nt", type=int, default=3)
    p.add_argument("--n-extra-words", type=int, default=1, choices=[1, 2, 3])
    p.add_argument(
        "--prefill-variant",
        choices=["append_user", "replace_assistant"],
        default="append_user",
        help="How prefill writes concept text into the prompt.",
    )
    p.add_argument(
        "--detect-variant",
        choices=["strict", "vague"],
        default="strict",
        help="Detection-question phrasing (must match the gate's gates.json run).",
    )
    p.add_argument("--concepts", nargs="+", default=None,
                   help="Concepts (default: those from the injection gates.json).")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Min |steered-unsteered| activation delta to treat a feature "
                        "as a concept-responsive candidate.")
    p.add_argument("--n-top", type=int, default=200,
                   help="Max evidence carriers stored per concept (most negative attribution).")
    p.add_argument("--output-dir", "-od", default="analysis/gate_minimal")
    p.add_argument("--device", "-d", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    with open(args.injection_gates) as f:
        inj_gates = json.load(f)
    cfg = inj_gates["config"]

    steering_layer = args.steering_layer
    start_layer = args.start_layer if args.start_layer is not None else steering_layer + 1
    strength = args.strength if args.strength is not None else cfg["strength"]
    concepts = args.concepts if args.concepts is not None else cfg["concepts"]

    gate_layer, gate_feat_id = pick_gate_feature(inj_gates, args.gate_feature, start_layer)
    if gate_layer <= start_layer:
        raise ValueError(
            f"Gate layer {gate_layer} must be above the start layer {start_layer}."
        )
    print(f"[08] Gate feature: L{gate_layer}_F{gate_feat_id}")

    print(f"[08] Loading {args.model} ...")
    model_w = load_model(model_name=args.model, device=args.device, dtype=args.dtype)
    yes_ids, no_ids = P.get_yes_no_token_ids(model_w.tokenizer)

    concept_vectors: Dict[str, torch.Tensor] = {}
    if "injection" in args.settings:
        vec_cache = Path(args.output_dir) / args.model / f"concept_vectors_L{steering_layer}.pt"
        concept_vectors = gl.load_or_build_concept_vectors(
            model_w, concepts, steering_layer, vec_cache
        )

    for setting in args.settings:
        # Prefill may scan from an earlier layer (concept present from the start);
        # injection scans only post-injection layers.
        if setting == "prefill" and args.prefill_start_layer is not None:
            setting_start = args.prefill_start_layer
        else:
            setting_start = start_layer
        candidate_layers = list(range(setting_start, gate_layer))
        if not candidate_layers:
            raise ValueError(
                f"No candidate layers below gate layer {gate_layer} (start={setting_start})."
            )
        print(f"[08] [{setting}] candidate layers: {candidate_layers[0]}..{candidate_layers[-1]} "
              f"({len(candidate_layers)} layers)")

        setting_tag = setting
        if setting == "prefill":
            setting_tag = f"prefill-{args.prefill_variant}"
        if args.detect_variant != "strict":
            setting_tag = f"{setting_tag}_detect-{args.detect_variant}"
        out_dir = (
            Path(args.output_dir) / args.model
            / f"{setting_tag}_L{steering_layer}_s{strength}" / "carriers"
        )
        collect_setting_carriers(
            model_w=model_w,
            setting=setting,
            prefill_variant=args.prefill_variant,
            detect_variant=args.detect_variant,
            concepts=concepts,
            candidate_layers=candidate_layers,
            gate_layer=gate_layer,
            gate_feat_id=gate_feat_id,
            steering_layer=steering_layer,
            strength=strength,
            n_trials=args.n_trials,
            n_extra_words=args.n_extra_words,
            concept_vectors=concept_vectors,
            yes_ids=yes_ids,
            no_ids=no_ids,
            threshold=args.threshold,
            n_top=args.n_top,
            output_dir=out_dir,
        )


if __name__ == "__main__":
    main()

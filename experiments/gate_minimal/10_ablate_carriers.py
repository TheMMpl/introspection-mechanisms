#!/usr/bin/env python3
"""Step 10 — Gate-activation vs steering-strength under progressive carrier ablation.

Recreates the Figure-13-style plot: for several concepts, the chosen gate
feature's activation is plotted against injection steering strength while
progressively ablating upstream feature groups. Ablating **evidence carriers**
(negative attribution) *raises* the gate (they normally suppress it); ablating
**suppressors** (positive attribution) *lowers* it; the weak-attribution control
tracks the un-ablated baseline.

Ablation = **matched-control patching**: a feature's answer-position activation
is *replaced with its unsteered (steering-strength-0) value* rather than zeroed.
Concretely, at each candidate layer we add ``(control_act - steered_act) @ w_dec``
to the residual at the answer position (a ``gate_lib.GatePatch`` whose target is
the unsteered activation), so the patched features carry their no-concept value
while everything else stays steered.

Feature groups (per concept, from the step-8 ``carriers.json``):
  * Baseline               — no ablation,
  * Bottom-10% attributed  — the weak-attribution control,
  * All / Top-20% / Top-5% evidence carriers   (attribution < 0),
  * All / Top-20% / Top-5% suppressors         (attribution > 0).

The sweep itself always uses **injection** steering (the only continuous-strength
axis), regardless of whether the carriers were collected in the injection or the
prefill setting — so pointing ``--carriers`` at a prefill ``carriers.json`` shows
how the prefill-discovered carriers behave under injection steering. Features
that do not respond to injection (e.g. some pre-injection-layer prefill carriers)
will simply track the baseline.

Outputs under ``--output-dir`` (default: beside the carriers.json):
  * ``carrier_ablation_sweep.png``  — one subplot per concept,
  * ``ablation_sweep.json``         — the per-concept/group/strength curves.
"""

import argparse
import json
import math
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import gate_lib as gl
import prompts as P
from model_utils import load_model


# Ablation groups, drawn in legend order. (key, label, color, marker, linestyle)
GROUP_STYLES: List[Tuple[str, str, str, str, str]] = [
    ("baseline",          "Baseline",                    "#1f77b4", "o", "-"),
    ("weak_control",      "Bottom-10% attributed",       "#d4a017", "D", ":"),
    ("all_carriers",      "All evidence carriers",       "#1b5e20", "^", "-"),
    ("top20_carriers",    "Top-20% evidence carriers",   "#43a047", "^", ":"),
    ("top5_carriers",     "Top-5% evidence carriers",    "#81c784", "^", "--"),
    ("all_suppressors",   "All suppressors",             "#8b1a1a", "v", "-"),
    ("top20_suppressors", "Top-20% suppressors",         "#e53935", "v", ":"),
    ("top5_suppressors",  "Top-5% suppressors",          "#ef9a9a", "v", "--"),
]
ABLATION_GROUPS = [k for k, *_ in GROUP_STYLES if k != "baseline"]


def _top_pct(records: List[Dict], pct: int) -> List[Dict]:
    if not records:
        return []
    return records[:max(1, len(records) * pct // 100)]


def build_groups(entry: Dict) -> Dict[str, List[Tuple[int, int]]]:
    """Build the ablation feature groups for one concept from its carrier record.

    ``carriers`` are stored most-negative-attribution-first and ``suppressors``
    most-positive-first (see step 8), so the percentage slices take the strongest
    features of each kind.
    """
    carriers = entry.get("carriers", [])
    suppressors = entry.get("suppressors", [])
    weak = entry.get("weak_control", [])

    def feats(recs):
        return [(int(r["layer"]), int(r["feat_id"])) for r in recs]

    return {
        "weak_control":      feats(weak),
        "all_carriers":      feats(carriers),
        "top20_carriers":    feats(_top_pct(carriers, 20)),
        "top5_carriers":     feats(_top_pct(carriers, 5)),
        "all_suppressors":   feats(suppressors),
        "top20_suppressors": feats(_top_pct(suppressors, 20)),
        "top5_suppressors":  feats(_top_pct(suppressors, 5)),
    }


def measure_gate(
    model_w,
    gate_tc,
    gate_layer: int,
    gate_feat_id: int,
    input_ids: torch.Tensor,
    steering_layer: int,
    concept_vec: Optional[torch.Tensor],
    strength: float,
    patches: Dict[int, Tuple[gl.FeatureSlice, torch.Tensor]],
    injection_positions: str = "all",
    patch_positions: str = "last",
) -> Tuple[float, torch.Tensor]:
    """One forward; returns (gate-feature activation, last-token logits).

    ``patches`` maps each layer to ``(slice, control_sel)``; each is applied as a
    GatePatch whose target activation is the unsteered (control) value, replacing
    those features' activations at ``patch_positions`` ('last' or 'all'). An
    injection hook of magnitude ``strength`` is applied at ``injection_positions``
    when ``concept_vec`` is given and ``strength != 0``.
    """
    device = model_w.device
    handle = None
    if concept_vec is not None and strength != 0.0:
        hook = gl.make_injection_hook(concept_vec, strength, positions=injection_positions)
        handle = model_w.get_layer_module(steering_layer).register_forward_hook(hook)
    try:
        with ExitStack() as stack:
            for L, (slc, control_sel) in patches.items():
                stack.enter_context(
                    gl.GatePatch(model_w, slc, steered_sel=control_sel, positions=patch_positions)
                )
            with gl.TranscoderCapture(model_w, [gate_layer]) as cap:
                with torch.no_grad():
                    out = model_w.model(input_ids=input_ids.to(device), use_cache=False)
                logits = out.logits[:, -1, :].float().cpu()
                gate_feats = gl.encode_last_token(gate_tc, cap.captured[gate_layer])
                gate_act = float(gate_feats[gate_feat_id])
    finally:
        if handle is not None:
            handle.remove()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return gate_act, logits


def run_concept_sweep(
    model_w,
    gate_tc,
    gate_layer: int,
    gate_feat_id: int,
    concept: str,
    entry: Dict,
    concept_vec: torch.Tensor,
    steering_layer: int,
    strengths: List[float],
    collect_strength: float,
    n_trials: int,
    n_extra_words: int,
    prefill_variant: str,
    detect_variant: str,
    yes_ids: List[int],
    no_ids: List[int],
    device: str,
    injection_scope: str = "all",
    ablation_scope: str = "last",
) -> Tuple[Dict[str, Dict[float, float]], Dict[str, int], float]:
    """Sweep one concept; returns (curves, group_sizes, detection_rate).

    ``curves[group][strength]`` is the mean gate activation across trials.
    """
    groups = build_groups(entry)
    group_sizes = {g: len(groups[g]) for g in groups}

    involved_layers = sorted({L for feats in groups.values() for (L, _) in feats})

    # Load each involved transcoder once; build a per-(group, layer) slice.
    slices: Dict[str, Dict[int, gl.FeatureSlice]] = {g: {} for g in groups}
    for L in involved_layers:
        tc = gl.load_transcoder(L, device=device)
        for g in groups:
            feats = [f for (l, f) in groups[g] if l == L]
            if feats:
                slices[g][L] = gl.FeatureSlice.from_transcoder(L, tc, feats)
        del tc
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    curve_names = ["baseline"] + [g for g in groups]
    accum: Dict[str, Dict[float, List[float]]] = {
        name: {s: [] for s in strengths} for name in curve_names
    }
    det_hits: List[bool] = []

    for trial in range(1, n_trials + 1):
        input_ids = P.format_detection_prompt(
            model_w.tokenizer, trial, None, n_extra_words,
            prefill_variant=prefill_variant, detect_variant=detect_variant,
        )

        # Strength-0 (unsteered) control forward: capture answer-position pre-LN
        # for every involved layer (the ablation target) and the baseline gate.
        # For ``ablation_scope='all'`` keep every position's pre-LN so the patch
        # can replace the carriers everywhere, not just at the answer token.
        with gl.TranscoderCapture(model_w, involved_layers + [gate_layer]) as cap:
            with torch.no_grad():
                model_w.model(input_ids=input_ids.to(device), use_cache=False)
            if ablation_scope == "all":
                control_pre_ln = {L: cap.captured[L].clone() for L in involved_layers}
            else:
                control_pre_ln = {L: cap.captured[L][:, -1:, :].clone() for L in involved_layers}
            gate0 = float(gl.encode_last_token(gate_tc, cap.captured[gate_layer])[gate_feat_id])

        # Per-group control activations (ordered like each slice's feature_ids).
        control_sel: Dict[str, Dict[int, torch.Tensor]] = {}
        for g in groups:
            if ablation_scope == "all":
                control_sel[g] = {
                    L: slices[g][L].encode_all_tokens(control_pre_ln[L]) for L in slices[g]
                }
            else:
                control_sel[g] = {
                    L: slices[g][L].encode_last_token(control_pre_ln[L]) for L in slices[g]
                }

        for s in strengths:
            if s == 0.0:
                # Matched control at strength 0 is a no-op; every curve == baseline.
                for name in curve_names:
                    accum[name][s].append(gate0)
                continue

            ga, logits = measure_gate(
                model_w, gate_tc, gate_layer, gate_feat_id, input_ids,
                steering_layer, concept_vec, s, patches={},
                injection_positions=injection_scope, patch_positions=ablation_scope,
            )
            accum["baseline"][s].append(ga)
            if s == collect_strength:
                det_hits.append(P.logit_gap(logits, yes_ids, no_ids).item() > 0)

            for g in groups:
                patches = {L: (slices[g][L], control_sel[g][L]) for L in slices[g]}
                ga_g, _ = measure_gate(
                    model_w, gate_tc, gate_layer, gate_feat_id, input_ids,
                    steering_layer, concept_vec, s, patches=patches,
                    injection_positions=injection_scope, patch_positions=ablation_scope,
                )
                accum[g][s].append(ga_g)

    # If the collect strength was not in the grid, get the detection rate anyway.
    if not det_hits:
        for trial in range(1, n_trials + 1):
            input_ids = P.format_detection_prompt(
                model_w.tokenizer, trial, None, n_extra_words,
                prefill_variant=prefill_variant, detect_variant=detect_variant,
            )
            _, logits = measure_gate(
                model_w, gate_tc, gate_layer, gate_feat_id, input_ids,
                steering_layer, concept_vec, collect_strength, patches={},
                injection_positions=injection_scope, patch_positions=ablation_scope,
            )
            det_hits.append(P.logit_gap(logits, yes_ids, no_ids).item() > 0)

    curves = {
        name: {s: float(np.mean(vals)) for s, vals in by_s.items() if vals}
        for name, by_s in accum.items()
    }
    detection_rate = float(np.mean(det_hits)) if det_hits else float("nan")
    return curves, group_sizes, detection_rate


def plot_sweep(results: Dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    concepts = results["concepts"]
    strengths = results["strengths"]
    gate_label = results["gate_feature"]
    setting = results.get("carriers_setting", "?")

    # Legend n-range per ablation group across the plotted concepts.
    size_range: Dict[str, Tuple[int, int]] = {}
    for g in ABLATION_GROUPS:
        sizes = [results["group_sizes"][c][g] for c in concepts]
        size_range[g] = (min(sizes), max(sizes)) if sizes else (0, 0)

    ncols = min(3, len(concepts))
    nrows = math.ceil(len(concepts) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
                             squeeze=False, sharex=True)

    for idx, concept in enumerate(concepts):
        ax = axes[idx // ncols][idx % ncols]
        curves = results["curves"][concept]
        for key, label, color, marker, ls in GROUP_STYLES:
            ys = [curves[key].get(str(s), curves[key].get(s, float("nan"))) for s in strengths]
            ax.plot(strengths, ys, color=color, marker=marker, linestyle=ls,
                    markersize=4, linewidth=1.4, alpha=0.85)
        det = results["detection_rates"].get(concept, float("nan"))
        ax.set_title(f"{concept}\n({det*100:.0f}% detection)", fontsize=10)
        ax.axvline(0.0, color="black", lw=0.6, alpha=0.3)
        ax.grid(alpha=0.25)
        if idx % ncols == 0:
            ax.set_ylabel("Gate activation")
    # Hide any unused axes.
    for j in range(len(concepts), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    # Shared x-label on the bottom row.
    for ax in axes[-1]:
        ax.set_xlabel("Steering strength")

    handles = []
    for key, label, color, marker, ls in GROUP_STYLES:
        if key == "baseline":
            leg = label
        else:
            lo, hi = size_range[key]
            leg = f"{label} (n={lo}-{hi})" if lo != hi else f"{label} (n={lo})"
        handles.append(plt.Line2D([], [], color=color, marker=marker, linestyle=ls, label=leg))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, -0.02))

    inj_scope = results.get("injection_scope", "all")
    abl_scope = results.get("ablation_scope", "last")
    fig.suptitle(
        f"Gate {gate_label} activation vs steering strength under carrier ablation\n"
        f"(carriers collected in '{setting}' setting; ablation = replace steered "
        f"activation with unsteered)\n"
        f"injection positions: {inj_scope}   |   patch positions: {abl_scope}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[10] Saved plot -> {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="Gate-activation vs steering-strength carrier-ablation sweep (Fig 13)."
    )
    p.add_argument("--carriers", required=True,
                   help="carriers.json from step 8 (injection or prefill collected).")
    p.add_argument("--model", "-m", default="gemma3_27b")
    p.add_argument("--concepts", nargs="+", default=None,
                   help="Concepts to plot (default: first 6 from the carriers.json).")
    p.add_argument("--max-concepts", type=int, default=6,
                   help="When --concepts is not given, plot this many concepts.")
    p.add_argument(
        "--strength-grid", type=float, nargs="+",
        default=[-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0],
        help="Injection strengths for the x-axis (0 is the un-ablated control).",
    )
    p.add_argument("--n-trials", "-nt", type=int, default=1,
                   help="Trials averaged per concept (default 1).")
    p.add_argument("--injection-scope", choices=["all", "last"], default="all",
                   help="Inject the steering vector at all positions (default) "
                        "or only the answer token.")
    p.add_argument("--ablation-scope", choices=["last", "all"], default="last",
                   help="Patch (replace-with-unsteered) carriers at only the answer "
                        "token ('last', default) or at all positions ('all').")
    p.add_argument("--output-dir", "-od", default=None,
                   help="Default: an 'ablation_sweep' folder beside the carriers.json.")
    p.add_argument("--device", "-d", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    carriers_path = Path(args.carriers)
    with open(carriers_path) as f:
        data = json.load(f)
    cfg = data["config"]
    gate_layer = int(data["gate_feature"]["layer"])
    gate_feat_id = int(data["gate_feature"]["feat_id"])
    gate_label = f"L{gate_layer}_F{gate_feat_id}"

    steering_layer = cfg["steering_layer"]
    collect_strength = float(cfg["strength"])
    n_extra_words = cfg["n_extra_words"]
    prefill_variant = cfg.get("prefill_variant", "append_user")
    detect_variant = cfg.get("detect_variant", "strict")
    yes_ids, no_ids = cfg["yes_ids"], cfg["no_ids"]

    all_concepts = list(data["per_concept"].keys())
    concepts = args.concepts if args.concepts is not None else all_concepts[:args.max_concepts]
    missing = [c for c in concepts if c not in data["per_concept"]]
    if missing:
        raise ValueError(f"Concepts not in carriers.json: {missing}")

    strengths = sorted(set(args.strength_grid))
    if 0.0 not in strengths:
        strengths = sorted(strengths + [0.0])

    out_dir = Path(args.output_dir) if args.output_dir else carriers_path.parent / "ablation_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[10] Gate feature: {gate_label}  (carriers setting: {cfg['setting']})")
    print(f"[10] Concepts: {concepts}")
    print(f"[10] injection positions: {args.injection_scope}  |  patch positions: {args.ablation_scope}")
    print(f"[10] Loading {args.model} ...")
    model_w = load_model(model_name=args.model, device=args.device, dtype=args.dtype)

    vec_cache = carriers_path.parent.parent.parent / f"concept_vectors_L{steering_layer}.pt"
    concept_vectors = gl.load_or_build_concept_vectors(
        model_w, concepts, steering_layer, vec_cache
    )

    gate_tc = gl.load_transcoder(gate_layer, device=args.device)

    curves_out: Dict[str, Dict[str, Dict[str, float]]] = {}
    group_sizes_out: Dict[str, Dict[str, int]] = {}
    detection_rates: Dict[str, float] = {}

    for concept in tqdm(concepts, desc="[10] concept sweeps"):
        curves, group_sizes, det = run_concept_sweep(
            model_w, gate_tc, gate_layer, gate_feat_id, concept,
            data["per_concept"][concept], concept_vectors[concept],
            steering_layer, strengths, collect_strength, args.n_trials,
            n_extra_words, prefill_variant, detect_variant, yes_ids, no_ids,
            args.device,
            injection_scope=args.injection_scope, ablation_scope=args.ablation_scope,
        )
        # JSON-friendly (string strength keys).
        curves_out[concept] = {
            g: {str(s): v for s, v in by_s.items()} for g, by_s in curves.items()
        }
        group_sizes_out[concept] = group_sizes
        detection_rates[concept] = det
        print(f"[10] {concept}: detection={det*100:.0f}%  "
              f"sizes={{carriers:{group_sizes['all_carriers']}, "
              f"suppressors:{group_sizes['all_suppressors']}, "
              f"weak:{group_sizes['weak_control']}}}")

    results = {
        "gate_feature": gate_label,
        "carriers_setting": cfg["setting"],
        "carriers_path": str(carriers_path),
        "steering_layer": steering_layer,
        "collect_strength": collect_strength,
        "strengths": strengths,
        "n_trials": args.n_trials,
        "injection_scope": args.injection_scope,
        "ablation_scope": args.ablation_scope,
        "concepts": concepts,
        "curves": curves_out,
        "group_sizes": group_sizes_out,
        "detection_rates": detection_rates,
    }
    tag = f"inj-{args.injection_scope}_patch-{args.ablation_scope}"
    with open(out_dir / f"ablation_sweep_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_sweep(results, out_dir / f"carrier_ablation_sweep_{tag}.png")
    print(f"[10] Saved sweep -> {out_dir / f'ablation_sweep_{tag}.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 2 — Progressive gate ablation (recreates Figure 11c-style curves).

K semantics
-----------
``K`` is the **number of top features from the global gate ranking** that get
simultaneously ablated. The top-K features (sorted ascending by DLA across all
scanned layers) are bucketed by their source layer; each touched layer's
transcoder is loaded **once**, the encoder/decoder columns for the chosen
features are sliced out, and the transcoder is freed. Ablation memory cost is
therefore ``O(K * d_model)``, *not* one full SAE per touched layer — so K can
reach the size of the ranking (e.g. 200) without holding many transcoders.

Metrics collected per forward (``prompts.detection_record``)
------------------------------------------------------------
  * gap            : logsumexp(Yes) - logsumexp(No)   (detection log-odds)
  * detect_gap     : gap > 0                          (soft detection)
  * detect_argmax  : argmax over full vocab ∈ Yes set (strict next-token)
  * yes_lse/no_lse : the two logsumexp terms, tracked separately so a change
                     in the gap can be decomposed into Yes-side vs No-side.

Detection-type curves (vs K, K=0 = un-ablated baseline)
-------------------------------------------------------
  control     : control/unsteered introspection trial, top-K gates ablated.
                FP rate; expected to RISE with K (gates suppress unwarranted Yes).
  steered     : steered run, top-K gates ablated. TP rate on injected trials.
  patch       : unsteered control with the gates' answer-position activations
                knocked-in to their steered values.
  arithmetic  : factual control ("Do you believe that 1+1=3?") with top-K gates
                ablated — tests whether ablation induces a *general* Yes bias.

Identification curve
--------------------
  identification : steered + forced prefill, gates ablated. Rate = fraction
                   where the concept's first token is the argmax.

Outputs
-------
  ablation_results.json
  ablation_curves.png          (gap-based detection rates, introspection only)
  ablation_curves_argmax.png   (strict next-token detection rates, introspection only)
  logit_effects.png            (ΔYes vs ΔNo logits, per introspection curve)
  arithmetic_control.png       (Yes-bias on the factual control)
  control_curves.png           (factual-control Yes-rate: ablate vs patch + steered marker)
  control_curves_argmax.png    (same, strict next-token)
  control_logits.png           (Yes/No logsumexp for the factual-control curves)
  overlay_curves.png           (introspection vs control Yes-rate, patch & revert)
  overlay_logits.png           (introspection vs control Yes/No logsumexp, patch & revert)
"""

import argparse
import json
import sys
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import gate_lib as gl
import prompts as P
from model_utils import load_model


# Curves whose samples carry the full detection_record (gap/argmax/yes/no).
# Split into the introspection (concept-detection) curves that go in the main
# detection plots, and the factual-control curves that go in their own image.
INTROSPECTION_DETECTION_CURVES = [
    "ablate_control_detection",
    "ablate_steered_detection",
    "patch_detection",
    "reverse_patch_detection",
]
CONTROL_DETECTION_CURVES = [
    "ablate_arithmetic_control",
    "patch_arithmetic_control",
]
DETECTION_CURVES = INTROSPECTION_DETECTION_CURVES + CONTROL_DETECTION_CURVES
ALL_CURVES = DETECTION_CURVES + ["ablate_identification"]


def group_by_layer(global_gates: List[dict], k: int) -> Dict[int, List[int]]:
    """Group the top-k global gate features by their layer."""
    layer2feats: Dict[int, List[int]] = defaultdict(list)
    for rec in global_gates[:k]:
        layer2feats[int(rec["layer"])].append(int(rec["feature_id"]))
    return dict(layer2feats)


def build_slices(layer2feats: Dict[int, List[int]], device: str) -> Dict[int, gl.FeatureSlice]:
    """Load each touched layer's transcoder once, extract the feature slice,
    then free the transcoder. Returns ``{layer: FeatureSlice}``.
    """
    slices: Dict[int, gl.FeatureSlice] = {}
    for L in sorted(layer2feats):
        tc = gl.load_transcoder(L, device=device)
        slices[L] = gl.FeatureSlice.from_transcoder(L, tc, layer2feats[L])
        del tc
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return slices


def _aggregate_records(recs: List[dict]) -> dict:
    """Reduce a list of detection_record dicts to per-K summary scalars."""
    if not recs:
        nan = float("nan")
        return {k: nan for k in
                ("detect_gap_rate", "detect_argmax_rate", "mean_gap",
                 "mean_yes_lse", "mean_no_lse", "n")}
    n = len(recs)
    return {
        "detect_gap_rate": sum(r["detect_gap"] for r in recs) / n,
        "detect_argmax_rate": sum(r["detect_argmax"] for r in recs) / n,
        "mean_gap": sum(r["gap"] for r in recs) / n,
        "mean_yes_lse": sum(r["yes_lse"] for r in recs) / n,
        "mean_no_lse": sum(r["no_lse"] for r in recs) / n,
        "n": n,
    }


def main():
    p = argparse.ArgumentParser(description="Progressive gate ablation (Fig 11c-style).")
    p.add_argument("--model", "-m", default="gemma3_27b")
    p.add_argument("--gates-json", required=True,
                   help="Path to step-1 gates.json for the setting to ablate.")
    p.add_argument("--setting", choices=["injection", "prefill"], default="injection")
    p.add_argument("--strength", type=float, default=None,
                   help="Override injection strength (default: from gates.json).")
    p.add_argument("--k-grid", type=int, nargs="+",
                   default=[0, 1, 2, 4, 8, 16, 32, 64, 128, 200],
                   help="K=0 is the un-ablated baseline (always plotted as leftmost point).")
    p.add_argument("--concepts", nargs="+", default=None,
                   help="Subset of concepts (default: all from gates.json).")
    p.add_argument("--n-trials", "-nt", type=int, default=3)
    p.add_argument("--n-extra-words", type=int, default=1, choices=[1, 2, 3])
    p.add_argument(
        "--prefill-variant",
        choices=["append_user", "replace_assistant"],
        default=None,
        help="Prompt placement for prefill runs. Default: read from gates.json if present.",
    )
    p.add_argument(
        "--detect-variant",
        choices=["strict", "vague"],
        default=None,
        help="Detection-question phrasing. Must match the gates.json run. "
             "Default: read from gates.json if present.",
    )
    p.add_argument("--curves", nargs="+", default=ALL_CURVES, choices=ALL_CURVES)
    p.add_argument("--arithmetic-question", default=P.ARITHMETIC_CONTROL,
                   help="Factual control question (default: introspection_gemma's).")
    p.add_argument("--output-dir", "-od", default=None,
                   help="Default: alongside gates.json in an 'ablation' subfolder.")
    p.add_argument("--device", "-d", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    gates_path = Path(args.gates_json)
    with open(gates_path) as f:
        gates = json.load(f)
    cfg = gates["config"]
    global_gates = gates["global_gates"]

    steering_layer = cfg["steering_layer"]
    strength = args.strength if args.strength is not None else cfg["strength"]
    concepts = args.concepts if args.concepts is not None else cfg["concepts"]
    yes_ids, no_ids = cfg["yes_ids"], cfg["no_ids"]
    prefill_variant = (
        args.prefill_variant
        if args.prefill_variant is not None
        else cfg.get("prefill_variant", "append_user")
    )
    detect_variant = (
        args.detect_variant
        if args.detect_variant is not None
        else cfg.get("detect_variant", "strict")
    )

    # Ensure 0 (baseline) is in the grid and grid is unique-sorted.
    k_grid = sorted({0, *[k for k in args.k_grid if 0 <= k <= len(global_gates)]})
    out_dir = Path(args.output_dir) if args.output_dir else gates_path.parent / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[02] Loading {args.model} ...")
    model_w = load_model(model_name=args.model, device=args.device, dtype=args.dtype)
    tokenizer = model_w.tokenizer

    concept_vectors = {}
    if args.setting == "injection":
        vec_cache = gates_path.parent.parent / f"concept_vectors_L{steering_layer}.pt"
        concept_vectors = gl.load_or_build_concept_vectors(
            model_w, concepts, steering_layer, vec_cache
        )

    concept_tok = {c: P.concept_first_token_id(tokenizer, c) for c in concepts}

    def injection_handle(concept):
        if args.setting != "injection":
            return None
        hook = gl.make_injection_hook(concept_vectors[concept], strength)
        return model_w.get_layer_module(steering_layer).register_forward_hook(hook)

    def prefill_word(concept):
        return concept if args.setting == "prefill" else None

    def ablated_logits(input_ids, slices, involved, concept_for_injection=None):
        """Forward with the top-K gates ablated; optional injection hook."""
        h = injection_handle(concept_for_injection) if concept_for_injection is not None else None
        try:
            with ExitStack() as stack:
                for L in involved:
                    stack.enter_context(gl.GateAblation(model_w, slices[L]))
                return gl.forward_last_logits(model_w, input_ids)
        finally:
            if h is not None:
                h.remove()

    # Per-curve, per-metric: {curve: {metric: {K: value}}}
    det_curves = [c for c in args.curves if c in DETECTION_CURVES]
    curves: Dict[str, Dict[str, Dict[int, float]]] = {c: defaultdict(dict) for c in det_curves}
    if "ablate_identification" in args.curves:
        curves["ablate_identification"] = {"id_rate": {}}

    # ── Factual-control prep (constant prompt; K-independent references) ────
    # The factual-control question is concept- and trial-independent, so its
    # un-intervened record only needs one forward and is reused at K=0.
    arith_ids = P.format_arithmetic_control_prompt(tokenizer, args.arithmetic_question)
    unsteered_arith_record = None
    if "patch_arithmetic_control" in det_curves:
        unsteered_arith_record = P.detection_record(
            gl.forward_last_logits(model_w, arith_ids), yes_ids, no_ids
        )

    # Steered factual-control baseline: inject each concept's steering vector
    # into the 2+2=5 prompt and average the Yes/No logits + Yes-rate. This is
    # the "steered baseline" marker on the control plots and the reference the
    # gate-patch is compared against. Only well-defined for the injection
    # setting (no clean way to steer the bare factual question under prefill).
    arith_steered_baseline = None
    if args.setting == "injection" and (
        "patch_arithmetic_control" in det_curves
        or "ablate_arithmetic_control" in det_curves
    ):
        arith_recs: List[dict] = []
        for concept in tqdm(concepts, desc="steered 2+2=5 baseline"):
            h = injection_handle(concept)
            try:
                logits_sa = gl.forward_last_logits(model_w, arith_ids)
            finally:
                if h is not None:
                    h.remove()
            arith_recs.append(P.detection_record(logits_sa, yes_ids, no_ids))
        arith_steered_baseline = _aggregate_records(arith_recs)
        print(
            "[02] steered 2+2=5 baseline: "
            f"gap_rate={arith_steered_baseline['detect_gap_rate']:.2f} "
            f"argmax_rate={arith_steered_baseline['detect_argmax_rate']:.2f} "
            f"yes={arith_steered_baseline['mean_yes_lse']:.2f} "
            f"no={arith_steered_baseline['mean_no_lse']:.2f}"
        )

    # ── Sweep over K (K=0 is the un-ablated baseline; no slices loaded) ─────
    for k in k_grid:
        if k == 0:
            slices: Dict[int, gl.FeatureSlice] = {}
            involved: List[int] = []
        else:
            layer2feats = group_by_layer(global_gates, k)
            involved = sorted(layer2feats)
            print(f"[02] K={k}: touching {len(involved)} layer(s) {involved}; "
                  f"loading transcoders one-by-one to extract slices ...")
            slices = build_slices(layer2feats, args.device)

        records: Dict[str, List[dict]] = {c: [] for c in det_curves}
        id_acc: List[bool] = []

        desc = f"K={k}" + (" (baseline)" if k == 0 else "")
        for concept in tqdm(concepts, desc=desc):
            for trial in range(1, args.n_trials + 1):

                # control: ablate, control trial (no injection, no append)
                if "ablate_control_detection" in det_curves:
                    ctrl_ids = P.format_detection_prompt(
                        tokenizer,
                        trial,
                        None,
                        args.n_extra_words,
                        prefill_variant=prefill_variant,
                        detect_variant=detect_variant,
                    )
                    logits = ablated_logits(ctrl_ids, slices, involved)
                    records["ablate_control_detection"].append(
                        P.detection_record(logits, yes_ids, no_ids)
                    )

                # steered: ablate, steered trial
                if "ablate_steered_detection" in det_curves:
                    det_ids = P.format_detection_prompt(
                        tokenizer,
                        trial,
                        prefill_word(concept),
                        args.n_extra_words,
                        prefill_variant=prefill_variant,
                        detect_variant=detect_variant,
                    )
                    logits = ablated_logits(det_ids, slices, involved, concept_for_injection=concept)
                    records["ablate_steered_detection"].append(
                        P.detection_record(logits, yes_ids, no_ids)
                    )

                # identification: ablate steered (forced prefill)
                if "ablate_identification" in args.curves:
                    id_ids = P.format_forced_identification_prompt(
                        tokenizer,
                        trial,
                        prefill_word(concept),
                        args.n_extra_words,
                        prefill_variant=prefill_variant,
                    )
                    logits_id = ablated_logits(id_ids, slices, involved, concept_for_injection=concept)
                    id_acc.append(int(logits_id.argmax(dim=-1).item()) == concept_tok[concept])

                # patch: steered → unsteered knock-in. The same captured
                # steered gate activations are knocked into (a) the unsteered
                # detection prompt and (b) the unsteered factual-control prompt,
                # so we can tell whether the gates carry concept information or
                # are just an efficient generic Yes-push.
                if ("patch_detection" in det_curves
                        or "patch_arithmetic_control" in det_curves):
                    det_ids = P.format_detection_prompt(
                        tokenizer,
                        trial,
                        prefill_word(concept),
                        args.n_extra_words,
                        prefill_variant=prefill_variant,
                        detect_variant=detect_variant,
                    )
                    base_ids = P.format_detection_prompt(
                        tokenizer,
                        trial,
                        None,
                        args.n_extra_words,
                        prefill_variant=prefill_variant,
                        detect_variant=detect_variant,
                    )
                    # Capture the gates' steered activations once (None at K=0).
                    if k == 0 or not involved:
                        steered_sel = None
                    else:
                        h = injection_handle(concept)
                        try:
                            with gl.TranscoderCapture(model_w, involved) as cap:
                                with torch.no_grad():
                                    model_w.model(input_ids=det_ids.to(model_w.device), use_cache=False)
                                steered_sel = {
                                    L: slices[L].encode_last_token(cap.captured[L]) for L in involved
                                }
                        finally:
                            if h is not None:
                                h.remove()

                    if "patch_detection" in det_curves:
                        if steered_sel is None:
                            logits_p = gl.forward_last_logits(model_w, base_ids)
                        else:
                            with ExitStack() as stack:
                                for L in involved:
                                    stack.enter_context(
                                        gl.GatePatch(model_w, slices[L], steered_sel=steered_sel[L])
                                    )
                                logits_p = gl.forward_last_logits(model_w, base_ids)
                        records["patch_detection"].append(
                            P.detection_record(logits_p, yes_ids, no_ids)
                        )

                    # patch arithmetic control: knock the *same* steered gate
                    # activations into the unsteered 2+2=5 prompt.
                    if "patch_arithmetic_control" in det_curves:
                        if steered_sel is None:
                            records["patch_arithmetic_control"].append(dict(unsteered_arith_record))
                        else:
                            with ExitStack() as stack:
                                for L in involved:
                                    stack.enter_context(
                                        gl.GatePatch(model_w, slices[L], steered_sel=steered_sel[L])
                                    )
                                logits_pa = gl.forward_last_logits(model_w, arith_ids)
                            records["patch_arithmetic_control"].append(
                                P.detection_record(logits_pa, yes_ids, no_ids)
                            )

                # reverse patch: steered run with gates knocked-in to their
                # *unsteered* values (checks whether reverting the gates back to
                # control activations suppresses detection).
                if "reverse_patch_detection" in det_curves:
                    det_ids = P.format_detection_prompt(
                        tokenizer,
                        trial,
                        prefill_word(concept),
                        args.n_extra_words,
                        prefill_variant=prefill_variant,
                        detect_variant=detect_variant,
                    )
                    base_ids = P.format_detection_prompt(
                        tokenizer,
                        trial,
                        None,
                        args.n_extra_words,
                        prefill_variant=prefill_variant,
                        detect_variant=detect_variant,
                    )
                    if k == 0 or not involved:
                        # No gates patched -> plain steered forward (TP baseline).
                        logits_rp = ablated_logits(
                            det_ids, slices, involved, concept_for_injection=concept
                        )
                    else:
                        # Capture the gates' unsteered (control) activations.
                        with gl.TranscoderCapture(model_w, involved) as cap:
                            with torch.no_grad():
                                model_w.model(input_ids=base_ids.to(model_w.device), use_cache=False)
                            unsteered_sel = {
                                L: slices[L].encode_last_token(cap.captured[L]) for L in involved
                            }
                        # Steered forward, patching the gates to unsteered values.
                        h = injection_handle(concept)
                        try:
                            with ExitStack() as stack:
                                for L in involved:
                                    stack.enter_context(
                                        gl.GatePatch(model_w, slices[L], steered_sel=unsteered_sel[L])
                                    )
                                logits_rp = gl.forward_last_logits(model_w, det_ids)
                        finally:
                            if h is not None:
                                h.remove()
                    records["reverse_patch_detection"].append(
                        P.detection_record(logits_rp, yes_ids, no_ids)
                    )

        # arithmetic control: deterministic, run once per K (concept-independent)
        if "ablate_arithmetic_control" in det_curves:
            logits_a = ablated_logits(arith_ids, slices, involved)
            records["ablate_arithmetic_control"].append(
                P.detection_record(logits_a, yes_ids, no_ids)
            )

        # Aggregate this K.
        for c in det_curves:
            agg = _aggregate_records(records[c])
            for metric, val in agg.items():
                curves[c][metric][k] = val
        if "ablate_identification" in args.curves:
            curves["ablate_identification"]["id_rate"][k] = (
                float(torch.tensor(id_acc).float().mean()) if id_acc else float("nan")
            )

        msg = []
        for c in det_curves:
            msg.append(f"{c}: gap={curves[c]['detect_gap_rate'][k]:.2f} "
                       f"argmax={curves[c]['detect_argmax_rate'][k]:.2f}")
        if "ablate_identification" in args.curves:
            msg.append(f"id={curves['ablate_identification']['id_rate'][k]:.2f}")
        print(f"[02] K={k}: " + " | ".join(msg))

        del slices
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Serialize (K keys -> str).
    def _strkeys(d):
        return {str(k): v for k, v in d.items()}

    results = {
        "config": {
            "model": args.model,
            "setting": args.setting,
            "steering_layer": steering_layer,
            "strength": strength,
            "k_grid": k_grid,
            "n_trials": args.n_trials,
            "n_extra_words": args.n_extra_words,
            "prefill_variant": prefill_variant,
            "detect_variant": detect_variant,
            "concepts": concepts,
            "arithmetic_question": args.arithmetic_question,
            "gates_json": str(gates_path),
        },
        "curves": {
            c: {metric: _strkeys(curves[c][metric]) for metric in curves[c]}
            for c in curves
        },
        "arith_steered_baseline": arith_steered_baseline,
    }
    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[02] Saved -> {out_dir / 'ablation_results.json'}")

    plot_detection_rates(results, out_dir / "ablation_curves.png", metric="detect_gap_rate",
                         title_suffix="gap-based detection")
    plot_detection_rates(results, out_dir / "ablation_curves_argmax.png", metric="detect_argmax_rate",
                         title_suffix="strict next-token detection")
    plot_logit_effects(results, out_dir / "logit_effects.png")
    plot_logit_effects_raw(results, out_dir / "logit_effects_raw.png")
    plot_arithmetic_control(results, out_dir / "arithmetic_control.png")
    plot_controls(results, out_dir / "control_curves.png", metric="detect_gap_rate",
                  title_suffix="gap-based")
    plot_controls(results, out_dir / "control_curves_argmax.png", metric="detect_argmax_rate",
                  title_suffix="strict next-token")
    plot_control_logits(results, out_dir / "control_logits.png")
    plot_overlay_rates(results, out_dir / "overlay_curves.png")
    plot_overlay_logits(results, out_dir / "overlay_logits.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_DET_STYLE = {
    "ablate_control_detection": ("Ablating: control (FP)", "tab:red", "o"),
    "ablate_steered_detection": ("Ablating: steered (TP)", "tab:orange", "D"),
    "patch_detection": ("Patching steered \u2192 unsteered", "tab:green", "s"),
    "reverse_patch_detection": ("Reverting gates steered \u2192 unsteered (TP)", "tab:brown", "P"),
    "ablate_arithmetic_control": ("Ablating: factual control (2+2=5)", "tab:purple", "v"),
    "patch_arithmetic_control": ("Patching gates \u2192 factual control (2+2=5)", "tab:cyan", "X"),
    "ablate_identification": ("Ablating steered (forced): identification", "tab:blue", "^"),
}


def _ks_ys(metric_curve: Dict[str, float]):
    ks = sorted(int(k) for k in metric_curve)
    return ks, [metric_curve[str(k)] for k in ks]


def _apply_k_xscale(ax, ks):
    """Use a symlog x-axis (linear 0\u20131, log beyond) so the densely-sampled
    small-K region—where the largest changes occur—is stretched out. symlog
    handles the K=0 baseline that a pure log scale cannot."""
    import matplotlib.ticker as mticker
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(ks)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def plot_detection_rates(results: dict, out_path: Path, metric: str, title_suffix: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, cdata in results["curves"].items():
        # Main detection plot shows introspection curves only; factual-control
        # curves live in their own image (control_curves.png / overlay_*).
        if key not in INTROSPECTION_DETECTION_CURVES and key != "ablate_identification":
            continue
        if key == "ablate_identification":
            mc = cdata.get("id_rate")
        else:
            mc = cdata.get(metric)
        if not mc:
            continue
        ks, ys = _ks_ys(mc)
        ys = [v * 100 for v in ys]
        label, color, marker = _DET_STYLE[key]
        ax.plot(ks, ys, marker=marker, color=color, label=label)

    cfg = results["config"]
    ax.set_xlabel("Top gate features ablated/patched (K)  — K=0 is no ablation")
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"Gate ablation — {title_suffix}\n"
                 f"({cfg['setting']}, L={cfg['steering_layer']}, s={cfg['strength']})")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[02] Saved plot -> {out_path}")


def plot_logit_effects(results: dict, out_path: Path):
    """For each introspection curve, plot ΔYes and ΔNo logsumexp relative to K=0,
    so the change in the detection gap can be attributed to the Yes side, the
    No side, or both. Δgap == ΔYes - ΔNo.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    det_curves = [c for c in INTROSPECTION_DETECTION_CURVES if c in results["curves"]
                  and results["curves"][c].get("mean_yes_lse")]
    if not det_curves:
        return
    n = len(det_curves)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), squeeze=False)
    axes = axes[0]

    for ax, key in zip(axes, det_curves):
        cdata = results["curves"][key]
        yk, yv = _ks_ys(cdata["mean_yes_lse"])
        _, nv = _ks_ys(cdata["mean_no_lse"])
        y0, n0 = yv[0], nv[0]  # K=0 baseline
        d_yes = [v - y0 for v in yv]
        d_no = [v - n0 for v in nv]
        d_gap = [dy - dn for dy, dn in zip(d_yes, d_no)]
        ax.axhline(0, color="0.6", lw=0.8)
        ax.plot(yk, d_yes, marker="o", color="tab:green", label="\u0394 Yes logit")
        ax.plot(yk, d_no, marker="s", color="tab:red", label="\u0394 No logit")
        ax.plot(yk, d_gap, marker="^", color="0.3", ls="--", label="\u0394 gap (Yes\u2212No)")
        label = _DET_STYLE[key][0]
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("K")
        _apply_k_xscale(ax, yk)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("\u0394 logsumexp vs K=0")
    axes[0].legend(fontsize=8, loc="best")

    cfg = results["config"]
    fig.suptitle(f"Logit-side decomposition of gate ablation "
                 f"({cfg['setting']}, L={cfg['steering_layer']}, s={cfg['strength']})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[02] Saved plot -> {out_path}")


def plot_logit_effects_raw(results: dict, out_path: Path):
    """For each introspection curve, plot the raw Yes and No logsumexp terms (not
    deltas) vs K, so absolute logit levels are visible alongside their gap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    det_curves = [c for c in INTROSPECTION_DETECTION_CURVES if c in results["curves"]
                  and results["curves"][c].get("mean_yes_lse")]
    if not det_curves:
        return
    n = len(det_curves)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), squeeze=False)
    axes = axes[0]

    for ax, key in zip(axes, det_curves):
        cdata = results["curves"][key]
        yk, yv = _ks_ys(cdata["mean_yes_lse"])
        _, nv = _ks_ys(cdata["mean_no_lse"])
        ax.plot(yk, yv, marker="o", color="tab:green", label="Yes logit")
        ax.plot(yk, nv, marker="s", color="tab:red", label="No logit")
        label = _DET_STYLE[key][0]
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("K")
        _apply_k_xscale(ax, yk)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("logsumexp (nats)")
    axes[0].legend(fontsize=8, loc="best")

    cfg = results["config"]
    fig.suptitle(f"Raw Yes/No logits under gate ablation "
                 f"({cfg['setting']}, L={cfg['steering_layer']}, s={cfg['strength']})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[02] Saved plot -> {out_path}")


def plot_arithmetic_control(results: dict, out_path: Path):
    """Effect of gate ablation on the factual control question — checks whether
    ablation induces a general Yes bias (left: Yes rates; right: logit gap and
    Yes/No deltas)."""
    cdata = results["curves"].get("ablate_arithmetic_control")
    if not cdata or not cdata.get("mean_gap"):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Left: Yes-bias rates (deterministic single prompt → 0/100%).
    gk, gr = _ks_ys(cdata["detect_gap_rate"])
    _, ar = _ks_ys(cdata["detect_argmax_rate"])
    axL.plot(gk, [v * 100 for v in gr], marker="o", color="tab:purple", label="Yes (gap>0)")
    axL.plot(gk, [v * 100 for v in ar], marker="^", color="tab:pink", label="Yes (argmax)")
    axL.set_ylim(-5, 105)
    axL.set_xlabel("K")
    axL.set_ylabel("Says Yes to factual control (%)")
    axL.set_title("Yes-bias on factual control")
    axL.legend(fontsize=9)
    _apply_k_xscale(axL, gk)
    axL.grid(True, alpha=0.3)

    # Right: gap and Yes/No logit deltas vs K=0.
    _, gap = _ks_ys(cdata["mean_gap"])
    _, yv = _ks_ys(cdata["mean_yes_lse"])
    _, nv = _ks_ys(cdata["mean_no_lse"])
    d_yes = [v - yv[0] for v in yv]
    d_no = [v - nv[0] for v in nv]
    axR.axhline(0, color="0.6", lw=0.8)
    axR.plot(gk, gap, marker="o", color="0.3", label="gap (Yes\u2212No)")
    axR.plot(gk, d_yes, marker="o", color="tab:green", ls="--", label="\u0394 Yes logit")
    axR.plot(gk, d_no, marker="s", color="tab:red", ls="--", label="\u0394 No logit")
    axR.set_xlabel("K")
    axR.set_ylabel("logit (nats)")
    axR.set_title("Logit gap & Yes/No deltas")
    axR.legend(fontsize=9)
    _apply_k_xscale(axR, gk)
    axR.grid(True, alpha=0.3)

    cfg = results["config"]
    fig.suptitle(f"Factual control: {cfg.get('arithmetic_question','')}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[02] Saved plot -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Factual-control plots (moved out of the main detection image) + overlays
# ─────────────────────────────────────────────────────────────────────────────

def _is_num(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and x == x  # not None / NaN


def _baseline_hline(ax, results: dict, metric: str, scale: float = 1.0,
                    color: str = "0.35", ls: str = "--"):
    """Draw the steered factual-control baseline as a horizontal reference line.

    ``metric`` selects which baseline scalar to mark (e.g. ``detect_gap_rate``,
    ``mean_yes_lse``). Returns the value drawn (or None if unavailable).
    """
    base = results.get("arith_steered_baseline")
    if not base:
        return None
    val = base.get(metric)
    if not _is_num(val):
        return None
    label = f"Steered 2+2=5 baseline ({val * scale:.0f}%)" if scale == 100 \
        else f"Steered 2+2=5 baseline ({val:.1f})"
    ax.axhline(val * scale, color=color, ls=ls, lw=1.3, label=label)
    return val


def plot_controls(results: dict, out_path: Path, metric: str, title_suffix: str):
    """Factual-control Yes-rate vs K for the ablate and patch interventions,
    with the steered (injected) 2+2=5 baseline marked as a horizontal line.

    Reading the result: if knocking concept-steered gates into the unrelated
    2+2=5 prompt drives the Yes-rate up toward the steered baseline, the gates
    act as a generic Yes-push; if it stays near the unsteered floor, they carry
    concept-specific information rather than a blanket bias.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_ks = None
    for key in CONTROL_DETECTION_CURVES:
        cdata = results["curves"].get(key)
        if not cdata or not cdata.get(metric):
            continue
        ks, ys = _ks_ys(cdata[metric])
        plotted_ks = ks
        label, color, marker = _DET_STYLE[key]
        ax.plot(ks, [v * 100 for v in ys], marker=marker, color=color, label=label)

    _baseline_hline(ax, results, metric, scale=100)

    cfg = results["config"]
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Top gate features ablated/patched (K)  — K=0 is no intervention")
    ax.set_ylabel("Says Yes to 2+2=5 (%)")
    ax.set_title(f"Factual control — {title_suffix}\n"
                 f"({cfg['setting']}, L={cfg['steering_layer']}, s={cfg['strength']})")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[02] Saved plot -> {out_path}")


def plot_control_logits(results: dict, out_path: Path):
    """Raw Yes/No logsumexp vs K for the factual-control curves, with the
    steered 2+2=5 Yes/No baselines marked as horizontal lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctrl = [c for c in CONTROL_DETECTION_CURVES if c in results["curves"]
            and results["curves"][c].get("mean_yes_lse")]
    if not ctrl:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in ctrl:
        cdata = results["curves"][key]
        yk, yv = _ks_ys(cdata["mean_yes_lse"])
        _, nv = _ks_ys(cdata["mean_no_lse"])
        label, color, marker = _DET_STYLE[key]
        ax.plot(yk, yv, marker=marker, color=color, ls="-", label=f"{label}: Yes")
        ax.plot(yk, nv, marker=marker, color=color, ls=":", label=f"{label}: No")
        last_ks = yk

    base = results.get("arith_steered_baseline")
    if base:
        if _is_num(base.get("mean_yes_lse")):
            ax.axhline(base["mean_yes_lse"], color="0.25", ls="--", lw=1.2,
                       label="Steered 2+2=5: Yes")
        if _is_num(base.get("mean_no_lse")):
            ax.axhline(base["mean_no_lse"], color="0.55", ls="--", lw=1.2,
                       label="Steered 2+2=5: No")

    cfg = results["config"]
    _apply_k_xscale(ax, last_ks)
    ax.set_xlabel("K")
    ax.set_ylabel("logsumexp (nats)")
    ax.set_title(f"Factual control — raw Yes/No logits\n"
                 f"({cfg['setting']}, L={cfg['steering_layer']}, s={cfg['strength']})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[02] Saved plot -> {out_path}")


def _overlay_rate_axis(ax, results: dict, keys: List[str], metric: str,
                       mark_baseline: bool = True):
    """Plot Yes-rate (%) vs K for ``keys`` on ``ax``; optionally mark the steered
    factual-control baseline. Returns True if anything was drawn."""
    drawn = False
    for key in keys:
        if key == "ablate_identification":
            cdata = results["curves"].get(key)
            mc = cdata.get("id_rate") if cdata else None
        else:
            cdata = results["curves"].get(key)
            mc = cdata.get(metric) if cdata else None
        if not mc:
            continue
        ks, ys = _ks_ys(mc)
        label, color, marker = _DET_STYLE[key]
        ax.plot(ks, [v * 100 for v in ys], marker=marker, color=color, label=label)
        drawn = True
    if mark_baseline:
        _baseline_hline(ax, results, metric, scale=100)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("K")
    ax.set_ylabel("Yes rate (%)")
    ax.grid(True, alpha=0.3)
    return drawn


def _overlay_logit_axis(ax, results: dict, keys: List[str], mark_baseline: bool = True):
    """Plot raw Yes/No logsumexp vs K for ``keys`` on ``ax``; optionally mark the
    steered factual-control Yes/No baselines."""
    last_ks = None
    for key in keys:
        cdata = results["curves"].get(key)
        if not cdata or not cdata.get("mean_yes_lse"):
            continue
        yk, yv = _ks_ys(cdata["mean_yes_lse"])
        _, nv = _ks_ys(cdata["mean_no_lse"])
        label, color, marker = _DET_STYLE[key]
        ax.plot(yk, yv, marker=marker, color=color, ls="-", label=f"{label}: Yes")
        ax.plot(yk, nv, marker=marker, color=color, ls=":", label=f"{label}: No")
        last_ks = yk
    if mark_baseline:
        base = results.get("arith_steered_baseline")
        if base:
            if _is_num(base.get("mean_yes_lse")):
                ax.axhline(base["mean_yes_lse"], color="0.25", ls="--", lw=1.2,
                           label="Steered 2+2=5: Yes")
            if _is_num(base.get("mean_no_lse")):
                ax.axhline(base["mean_no_lse"], color="0.55", ls="--", lw=1.2,
                           label="Steered 2+2=5: No")
    if last_ks is not None:
        _apply_k_xscale(ax, last_ks)
    ax.set_xlabel("K")
    ax.set_ylabel("logsumexp (nats)")
    ax.grid(True, alpha=0.3)


def plot_overlay_rates(results: dict, out_path: Path):
    """Overlay the introspection and factual-control Yes-rates so the two can be
    compared directly. Left: the patch comparison (concept knock-in vs the same
    gates knocked into 2+2=5). Right: the revert comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric = "detect_gap_rate"
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    _overlay_rate_axis(axL, results, ["patch_detection", "patch_arithmetic_control"], metric)
    axL.set_title("Patch: concept detection vs factual control")
    axL.legend(fontsize=8, loc="best")

    _overlay_rate_axis(axR, results,
                       ["reverse_patch_detection", "ablate_arithmetic_control"], metric)
    axR.set_title("Revert / ablate: concept detection vs factual control")
    axR.legend(fontsize=8, loc="best")

    cfg = results["config"]
    fig.suptitle(f"Introspection vs factual-control Yes-rate (gap-based)\n"
                 f"({cfg['setting']}, L={cfg['steering_layer']}, s={cfg['strength']})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[02] Saved plot -> {out_path}")


def plot_overlay_logits(results: dict, out_path: Path):
    """Overlay the introspection and factual-control raw Yes/No logits. Left: the
    patch comparison; right: the revert/ablate comparison. The steered 2+2=5
    Yes/No baselines are marked as horizontal lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    _overlay_logit_axis(axL, results, ["patch_detection", "patch_arithmetic_control"])
    axL.set_title("Patch: concept detection vs factual control")
    axL.legend(fontsize=8, loc="best")

    _overlay_logit_axis(axR, results,
                        ["reverse_patch_detection", "ablate_arithmetic_control"])
    axR.set_title("Revert / ablate: concept detection vs factual control")
    axR.legend(fontsize=8, loc="best")

    cfg = results["config"]
    fig.suptitle(f"Introspection vs factual-control raw Yes/No logits\n"
                 f"({cfg['setting']}, L={cfg['steering_layer']}, s={cfg['strength']})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[02] Saved plot -> {out_path}")


if __name__ == "__main__":
    main()

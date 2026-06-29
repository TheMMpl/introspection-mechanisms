#!/usr/bin/env python3
"""Step 7 — Decouple suppression vs detector gates (needs the model).

Question
--------
Step-6 splits the No-writing gates into **suppression** features (active on the
bare prompt, switch off when the concept appears) and **detector** features
(~off on the bare prompt, switch on with the concept). This script asks what
each class causally contributes by **reverting** the *same* top-K global gates
to their unsteered (baseline) activations during a steered forward (GatePatch
knock-in, the same mechanism as ``reverse_patch_detection``), under three
conditions:

  * ``all``         : revert every gate in the top-K to baseline.
  * ``suppression`` : revert only the suppression-class gates within the top-K.
  * ``detector``    : revert only the detector-class gates within the top-K.

Pure ablation (zeroing) of these gates is the uninformative direction — it left
detection/identification unaffected, since the gates do not *hold* the concept.
The two informative causal probes are the GatePatch knock-ins, each run for
every condition vs K:
  * revert : steered forward with the class's gates reverted to their unsteered
             (baseline) activations. A DROP means that class causally carries
             the steered Yes/detection or concept-identity signal.
  * patch  : *unsteered* forward with the class's gates knocked in to their
             steered activations (no injection / no prefill word; the steered
             signal arrives only through the patched gates). A RISE means that
             class's activations are *sufficient* to induce detection / naming.

For each direction it measures, vs K:
  * detection      : detection trial, gap>0 (Yes beats No).
  * identification : forced-prefill trial, argmax == concept first token.

Comparing the three condition curves shows whether the steered detection /
identification is driven by the suppression gates, the detector gates, or needs
both. Subset sizes differ by construction (that *is* the signal); the per-K
subset sizes are recorded in the JSON and printed.

Specificity (logit-effect plot): for each class the *steered detection* gates
are also patched into the unrelated factual control ("2+2=5"). A class that
truly carries the introspective detection signal lifts the introspection
detection gap but leaves the factual-control gap flat — ``decouple_logit_effects.png``
plots the per-class detection gap (logsumexp Yes − No) for revert, patch, and
the control side by side.

Requires step-6 output (``feature_classes/features_classified.json``) beside the
gates.json. Run ``06_classify_features.py`` first.
"""

import argparse
import json
import sys
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import gate_lib as gl
import prompts as P
from model_utils import load_model

CONDITIONS = {
    "all": None,                       # None = accept every behavior class
    "suppression": {"suppression"},
    "detector": {"detector"},
}


def load_behavior_map(classification_path: Path) -> Dict[Tuple[int, int], str]:
    with open(classification_path) as f:
        data = json.load(f)
    return {(int(r["layer"]), int(r["feature_id"])): r["behavior"] for r in data["records"]}


def subset_layerfeats(
    global_gates: List[dict],
    k: int,
    behavior: Dict[Tuple[int, int], str],
    allowed: Set[str] | None,
) -> Dict[int, List[int]]:
    """Group the top-k global gates by layer, keeping only features whose class
    is in ``allowed`` (``None`` keeps all)."""
    layer2feats: Dict[int, List[int]] = defaultdict(list)
    for rec in global_gates[:k]:
        L, f = int(rec["layer"]), int(rec["feature_id"])
        if allowed is None or behavior.get((L, f)) in allowed:
            layer2feats[L].append(f)
    return dict(layer2feats)


def build_condition_slices(
    cond_layerfeats: Dict[str, Dict[int, List[int]]],
    device: str,
) -> Dict[str, Dict[int, gl.FeatureSlice]]:
    """Load each involved layer's transcoder ONCE and carve out per-condition
    feature slices, so a K step touches each transcoder a single time."""
    involved_layers = sorted({L for lf in cond_layerfeats.values() for L in lf})
    slices: Dict[str, Dict[int, gl.FeatureSlice]] = {c: {} for c in cond_layerfeats}
    for L in involved_layers:
        tc = gl.load_transcoder(L, device=device)
        for cond, lf in cond_layerfeats.items():
            if L in lf and lf[L]:
                slices[cond][L] = gl.FeatureSlice.from_transcoder(L, tc, lf[L])
        del tc
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return slices


def main():
    p = argparse.ArgumentParser(description="Decouple suppression vs detector gates.")
    p.add_argument("--gates-json", required=True)
    p.add_argument("--classification", default=None,
                   help="features_classified.json (default: sibling feature_classes/).")
    p.add_argument("--model", "-m", default="gemma3_27b")
    p.add_argument("--setting", choices=["injection", "prefill"], default="injection")
    p.add_argument("--strength", type=float, default=None)
    p.add_argument("--k-grid", type=int, nargs="+",
                   default=[0, 1, 2, 4, 8, 16, 32, 64, 128, 200])
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                   choices=list(CONDITIONS))
    p.add_argument("--concepts", nargs="+", default=None)
    p.add_argument("--n-trials", "-nt", type=int, default=3)
    p.add_argument("--n-extra-words", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--prefill-variant", choices=["append_user", "replace_assistant"], default=None)
    p.add_argument("--detect-variant", choices=["strict", "vague"], default=None)
    p.add_argument("--arithmetic-question", default=P.ARITHMETIC_CONTROL,
                   help="Factual control question (default: 2+2=5).")
    p.add_argument("--output-dir", "-od", default=None)
    p.add_argument("--device", "-d", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    gates_path = Path(args.gates_json)
    with open(gates_path) as f:
        gates = json.load(f)
    cfg = gates["config"]
    global_gates = gates["global_gates"]

    class_path = (Path(args.classification) if args.classification
                  else gates_path.parent / "feature_classes" / "features_classified.json")
    if not class_path.exists():
        sys.exit(f"[07] classification not found: {class_path}\n"
                 f"      run 06_classify_features.py on this gates.json first.")
    behavior = load_behavior_map(class_path)

    steering_layer = cfg["steering_layer"]
    strength = args.strength if args.strength is not None else cfg["strength"]
    concepts = args.concepts if args.concepts is not None else cfg["concepts"]
    yes_ids, no_ids = cfg["yes_ids"], cfg["no_ids"]
    prefill_variant = args.prefill_variant or cfg.get("prefill_variant", "append_user")
    detect_variant = args.detect_variant or cfg.get("detect_variant", "strict")
    conditions = {c: CONDITIONS[c] for c in args.conditions}

    k_grid = sorted({0, *[k for k in args.k_grid if 0 <= k <= len(global_gates)]})
    out_dir = (Path(args.output_dir) if args.output_dir
               else gates_path.parent / "decouple")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[07] Loading {args.model} ...")
    model_w = load_model(model_name=args.model, device=args.device, dtype=args.dtype)
    tokenizer = model_w.tokenizer

    concept_vectors = {}
    if args.setting == "injection":
        vec_cache = gates_path.parent.parent / f"concept_vectors_L{steering_layer}.pt"
        concept_vectors = gl.load_or_build_concept_vectors(
            model_w, concepts, steering_layer, vec_cache
        )

    def injection_handle(concept):
        if args.setting != "injection":
            return None
        hook = gl.make_injection_hook(concept_vectors[concept], strength)
        return model_w.get_layer_module(steering_layer).register_forward_hook(hook)

    def prefill_word(concept):
        return concept if args.setting == "prefill" else None

    def capture_unsteered_sel(base_ids, layer_slices):
        """Encode each layer-slice's features from an *unsteered* forward of
        ``base_ids`` (no injection, no prefill word). Returns {layer: tensor}."""
        involved = sorted(layer_slices)
        with gl.TranscoderCapture(model_w, involved) as cap:
            with torch.no_grad():
                model_w.model(input_ids=base_ids.to(model_w.device), use_cache=False)
            return {L: layer_slices[L].encode_last_token(cap.captured[L]) for L in involved}

    def capture_steered_sel(steered_ids, layer_slices, concept):
        """Encode each layer-slice's features from a *steered* forward of
        ``steered_ids`` (injection on / prefill concept). Returns {layer: tensor}."""
        involved = sorted(layer_slices)
        h = injection_handle(concept)
        try:
            with gl.TranscoderCapture(model_w, involved) as cap:
                with torch.no_grad():
                    model_w.model(input_ids=steered_ids.to(model_w.device), use_cache=False)
                return {L: layer_slices[L].encode_last_token(cap.captured[L]) for L in involved}
        finally:
            if h is not None:
                h.remove()

    def reverted_logits(steered_ids, base_ids, layer_slices, concept):
        """Steered forward with the slice's gates reverted to their unsteered
        (baseline) activations (GatePatch knock-in). With no slices this is a
        plain steered forward (the un-reverted baseline)."""
        if not layer_slices:
            h = injection_handle(concept)
            try:
                return gl.forward_last_logits(model_w, steered_ids)
            finally:
                if h is not None:
                    h.remove()
        unsteered_sel = capture_unsteered_sel(base_ids, layer_slices)
        h = injection_handle(concept)
        try:
            with ExitStack() as stack:
                for L in sorted(layer_slices):
                    stack.enter_context(
                        gl.GatePatch(model_w, layer_slices[L], steered_sel=unsteered_sel[L])
                    )
                return gl.forward_last_logits(model_w, steered_ids)
        finally:
            if h is not None:
                h.remove()

    def patched_logits(base_ids, steered_ids, layer_slices, concept):
        """*Unsteered* forward of ``base_ids`` with the slice's gates knocked in
        to their steered activations (GatePatch, no injection / no prefill word
        — the steered signal arrives only through the patched gates). With no
        slices this is a plain unsteered forward (the un-patched baseline)."""
        if not layer_slices:
            return gl.forward_last_logits(model_w, base_ids)
        steered_sel = capture_steered_sel(steered_ids, layer_slices, concept)
        return patch_with_sel(base_ids, steered_sel, layer_slices)

    def patch_with_sel(base_ids, steered_sel, layer_slices):
        """*Unsteered* forward of ``base_ids`` with the slice's gates knocked in
        to the pre-captured ``steered_sel`` activations. Lets a single steered
        capture be reused (e.g. patched into both the introspection prompt and
        the unrelated factual control). Empty slices → plain unsteered forward."""
        if not layer_slices:
            return gl.forward_last_logits(model_w, base_ids)
        with ExitStack() as stack:
            for L in sorted(layer_slices):
                stack.enter_context(
                    gl.GatePatch(model_w, layer_slices[L], steered_sel=steered_sel[L])
                )
            return gl.forward_last_logits(model_w, base_ids)

    # curves[cond][metric][K]. Two causal directions × two tasks, plus the
    # mean logit terms (gap / Yes / No logsumexp) used for the logit-effect
    # plots, and the 2+2=5 factual control under the patch (sufficiency) probe.
    #   revert_* : steered run, class gates reverted to baseline (drop = needed).
    #   patch_*  : unsteered run, class gates knocked in to steered (rise = suff).
    #   control_*: 2+2=5 prompt with the class's *steered detection* gates patched
    #              in (specificity: do the gates induce a generic Yes bias?).
    METRICS = [
        "revert_detection", "revert_identification",
        "patch_detection", "patch_identification", "n_features",
        "revert_det_gap", "revert_det_yes", "revert_det_no",
        "patch_det_gap", "patch_det_yes", "patch_det_no",
        "control_gap", "control_yes", "control_no",
        "control_yes_rate", "control_argmax_rate",
    ]
    curves: Dict[str, Dict[str, Dict[int, float]]] = {
        c: {m: {} for m in METRICS} for c in conditions
    }

    arith_ids = P.format_arithmetic_control_prompt(tokenizer, args.arithmetic_question)
    concept_tok = {c: set(P.concept_first_token_ids(tokenizer, c)) for c in concepts}

    for k in k_grid:
        cond_layerfeats = {
            c: ({} if k == 0 else subset_layerfeats(global_gates, k, behavior, allowed))
            for c, allowed in conditions.items()
        }
        for c in conditions:
            curves[c]["n_features"][k] = sum(len(v) for v in cond_layerfeats[c].values())

        if k == 0:
            cond_slices = {c: {} for c in conditions}
        else:
            cond_slices = build_condition_slices(cond_layerfeats, args.device)

        det_hits = {c: 0 for c in conditions}
        id_hits = {c: 0 for c in conditions}
        pdet_hits = {c: 0 for c in conditions}
        pid_hits = {c: 0 for c in conditions}
        ctrl_gap_hits = {c: 0 for c in conditions}
        ctrl_argmax_hits = {c: 0 for c in conditions}
        SUM_KEYS = ["rev_gap", "rev_yes", "rev_no", "pat_gap", "pat_yes", "pat_no",
                    "ctrl_gap", "ctrl_yes", "ctrl_no"]
        sums = {m: {c: 0.0 for c in conditions} for m in SUM_KEYS}
        n = 0
        for concept in tqdm(concepts, desc=f"K={k}"):
            for trial in range(1, args.n_trials + 1):
                n += 1
                det_ids = P.format_detection_prompt(
                    tokenizer, trial, prefill_word(concept), args.n_extra_words,
                    prefill_variant=prefill_variant, detect_variant=detect_variant)
                base_det_ids = P.format_detection_prompt(
                    tokenizer, trial, None, args.n_extra_words,
                    prefill_variant=prefill_variant, detect_variant=detect_variant)
                id_ids = P.format_forced_identification_prompt(
                    tokenizer, trial, prefill_word(concept), args.n_extra_words,
                    prefill_variant=prefill_variant)
                base_id_ids = P.format_forced_identification_prompt(
                    tokenizer, trial, None, args.n_extra_words,
                    prefill_variant=prefill_variant)
                for c in conditions:
                    ls = cond_slices[c]
                    # revert: steered run, class gates -> baseline
                    ld = reverted_logits(det_ids, base_det_ids, ls, concept)
                    rd = P.detection_record(ld, yes_ids, no_ids)
                    if rd["detect_gap"]:
                        det_hits[c] += 1
                    sums["rev_gap"][c] += rd["gap"]
                    sums["rev_yes"][c] += rd["yes_lse"]
                    sums["rev_no"][c] += rd["no_lse"]
                    li = reverted_logits(id_ids, base_id_ids, ls, concept)
                    if int(li.argmax(dim=-1).item()) in concept_tok[concept]:
                        id_hits[c] += 1
                    # patch: unsteered run, class gates -> steered. Capture the
                    # steered detection gates once and reuse for the control.
                    steered_det_sel = (
                        capture_steered_sel(det_ids, ls, concept) if ls else {})
                    pld = patch_with_sel(base_det_ids, steered_det_sel, ls)
                    prd = P.detection_record(pld, yes_ids, no_ids)
                    if prd["detect_gap"]:
                        pdet_hits[c] += 1
                    sums["pat_gap"][c] += prd["gap"]
                    sums["pat_yes"][c] += prd["yes_lse"]
                    sums["pat_no"][c] += prd["no_lse"]
                    pli = patched_logits(base_id_ids, id_ids, ls, concept)
                    if int(pli.argmax(dim=-1).item()) in concept_tok[concept]:
                        pid_hits[c] += 1
                    # control: 2+2=5 with the steered detection gates patched in.
                    ca = patch_with_sel(arith_ids, steered_det_sel, ls)
                    crd = P.detection_record(ca, yes_ids, no_ids)
                    sums["ctrl_gap"][c] += crd["gap"]
                    sums["ctrl_yes"][c] += crd["yes_lse"]
                    sums["ctrl_no"][c] += crd["no_lse"]
                    if crd["detect_gap"]:
                        ctrl_gap_hits[c] += 1
                    if crd["detect_argmax"]:
                        ctrl_argmax_hits[c] += 1

        for c in conditions:
            curves[c]["revert_detection"][k] = det_hits[c] / n
            curves[c]["revert_identification"][k] = id_hits[c] / n
            curves[c]["patch_detection"][k] = pdet_hits[c] / n
            curves[c]["patch_identification"][k] = pid_hits[c] / n
            curves[c]["revert_det_gap"][k] = sums["rev_gap"][c] / n
            curves[c]["revert_det_yes"][k] = sums["rev_yes"][c] / n
            curves[c]["revert_det_no"][k] = sums["rev_no"][c] / n
            curves[c]["patch_det_gap"][k] = sums["pat_gap"][c] / n
            curves[c]["patch_det_yes"][k] = sums["pat_yes"][c] / n
            curves[c]["patch_det_no"][k] = sums["pat_no"][c] / n
            curves[c]["control_gap"][k] = sums["ctrl_gap"][c] / n
            curves[c]["control_yes"][k] = sums["ctrl_yes"][c] / n
            curves[c]["control_no"][k] = sums["ctrl_no"][c] / n
            curves[c]["control_yes_rate"][k] = ctrl_gap_hits[c] / n
            curves[c]["control_argmax_rate"][k] = ctrl_argmax_hits[c] / n
        msg = " | ".join(
            f"{c}: rev(det={curves[c]['revert_detection'][k]:.2f} id={curves[c]['revert_identification'][k]:.2f}) "
            f"pat(det={curves[c]['patch_detection'][k]:.2f} id={curves[c]['patch_identification'][k]:.2f}) "
            f"ctrl_gap={curves[c]['control_gap'][k]:+.2f} (nf={curves[c]['n_features'][k]})"
            for c in conditions)
        print(f"[07] K={k}: {msg}")

        del cond_slices
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def _strkeys(d):
        return {str(k): v for k, v in d.items()}

    results = {
        "config": {
            "model": args.model, "setting": args.setting,
            "steering_layer": steering_layer, "strength": strength,
            "k_grid": k_grid, "n_trials": args.n_trials,
            "prefill_variant": prefill_variant, "detect_variant": detect_variant,
            "concepts": concepts, "gates_json": str(gates_path),
            "classification": str(class_path),
            "arithmetic_question": args.arithmetic_question,
        },
        "curves": {c: {m: _strkeys(curves[c][m]) for m in curves[c]} for c in conditions},
    }
    with open(out_dir / "decouple_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[07] Saved -> {out_dir / 'decouple_results.json'}")
    plot_decouple(results, out_dir / "decouple_curves.png")
    plot_logit_effects(results, out_dir / "decouple_logit_effects.png")
    plot_logit_effects_raw(results, out_dir / "decouple_logit_effects_raw.png")


def plot_decouple(results: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    style = {"all": ("0.2", "o"), "suppression": ("tab:red", "s"),
             "detector": ("tab:green", "^")}
    cfg = results["config"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    def ksys(mc):
        ks = sorted(int(k) for k in mc)
        return ks, [mc[str(k)] for k in ks]

    panels = [
        (axes[0, 0], "revert_detection", "no revert",
         "Detection — steered, class gates reverted to baseline (drop = needed)"),
        (axes[0, 1], "revert_identification", "no revert",
         "Identification — steered forced, class gates reverted (drop = needed)"),
        (axes[1, 0], "patch_detection", "no patch",
         "Detection — unsteered, class gates patched to steered (rise = sufficient)"),
        (axes[1, 1], "patch_identification", "no patch",
         "Identification — unsteered forced, class gates patched (rise = sufficient)"),
    ]
    for ax, metric, k0, title in panels:
        for cond, cdata in results["curves"].items():
            color, marker = style.get(cond, ("0.5", "o"))
            ks, ys = ksys(cdata[metric])
            ax.plot(ks, [v * 100 for v in ys], marker=marker, color=color, label=cond)
        ax.set_xscale("symlog", linthresh=1)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlabel(f"Top global gates considered (K)  — K=0 = {k0}")
        ax.set_ylabel("Rate (%)")
        ax.set_ylim(-5, 105)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Decoupling suppression vs detector gates (revert vs patch) "
                 f"({cfg['setting']}, L={cfg['steering_layer']}, s={cfg['strength']})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[07] Saved plot -> {out_path}")


def plot_logit_effects(results: dict, out_path: Path):
    """Compare the per-class logit effect on the introspection (detection)
    question vs the unrelated factual control (2+2=5). One line per condition
    (all / suppression / detector). The detection gap = logsumexp(Yes) −
    logsumexp(No); a class that carries the detection signal lifts the
    introspection gap but should leave the factual control gap flat (specificity).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    style = {"all": ("0.2", "o"), "suppression": ("tab:red", "s"),
             "detector": ("tab:green", "^")}
    cfg = results["config"]

    def ksys(mc):
        ks = sorted(int(k) for k in mc)
        return ks, [mc[str(k)] for k in ks]

    def _xscale(ax, ks):
        ax.set_xscale("symlog", linthresh=1)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(mticker.NullLocator())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    panels = [
        (axes[0], "revert_det_gap", "no revert",
         "Introspection gap — REVERT (steered → baseline)\ndrop ⇒ class necessary"),
        (axes[1], "patch_det_gap", "no patch",
         "Introspection gap — PATCH (unsteered → steered)\nrise ⇒ class sufficient"),
        (axes[2], "control_gap", "no patch",
         "Factual control (2+2=5) gap — PATCH\nrise ⇒ generic Yes-bias (not specific)"),
    ]
    for ax, metric, k0, title in panels:
        for cond, cdata in results["curves"].items():
            color, marker = style.get(cond, ("0.5", "o"))
            ks, ys = ksys(cdata[metric])
            ax.plot(ks, ys, marker=marker, color=color, label=cond)
        ax.axhline(0, color="0.6", lw=0.8)
        _xscale(ax, ks)
        ax.set_xlabel(f"Top global gates considered (K)  — K=0 = {k0}")
        ax.set_ylabel("detection gap  logsumexp(Yes) − logsumexp(No)  [nats]")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Logit effect per feature class — introspection vs factual control "
                 f"({cfg['setting']}, L={cfg['steering_layer']}, s={cfg['strength']})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[07] Saved plot -> {out_path}")


def plot_logit_effects_raw(results: dict, out_path: Path):
    """Raw (absolute) logsumexp terms per feature class. Same three panels as
    ``plot_logit_effects`` (revert / patch introspection, patched factual
    control) but plotting the Yes term (solid) and No term (dashed) directly,
    so the absolute logit levels behind each gap are visible per condition.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    style = {"all": ("0.2", "o"), "suppression": ("tab:red", "s"),
             "detector": ("tab:green", "^")}
    cfg = results["config"]

    def ksys(mc):
        ks = sorted(int(k) for k in mc)
        return ks, [mc[str(k)] for k in ks]

    def _xscale(ax):
        ax.set_xscale("symlog", linthresh=1)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(mticker.NullLocator())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    panels = [
        (axes[0], "revert_det_yes", "revert_det_no", "no revert",
         "Introspection — REVERT (steered → baseline)"),
        (axes[1], "patch_det_yes", "patch_det_no", "no patch",
         "Introspection — PATCH (unsteered → steered)"),
        (axes[2], "control_yes", "control_no", "no patch",
         "Factual control (2+2=5) — PATCH"),
    ]
    for ax, yes_m, no_m, k0, title in panels:
        for cond, cdata in results["curves"].items():
            color, marker = style.get(cond, ("0.5", "o"))
            ks, yv = ksys(cdata[yes_m])
            _, nv = ksys(cdata[no_m])
            ax.plot(ks, yv, marker=marker, color=color, label=f"{cond} · Yes")
            ax.plot(ks, nv, marker=marker, color=color, ls="--", alpha=0.6,
                    label=f"{cond} · No")
        _xscale(ax)
        ax.set_xlabel(f"Top global gates considered (K)  — K=0 = {k0}")
        ax.set_ylabel("logsumexp (nats)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Raw Yes/No logits per feature class — introspection vs factual control "
                 f"({cfg['setting']}, L={cfg['steering_layer']}, s={cfg['strength']})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[07] Saved plot -> {out_path}")


if __name__ == "__main__":
    main()

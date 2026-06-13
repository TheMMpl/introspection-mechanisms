#!/usr/bin/env python3
"""Step 3 — Compare injection vs prefill gate features.

Two analyses:

  A. Overlap (no model needed)
     Given step-1 ``gates.json`` for both settings, report:
       * per-layer Jaccard overlap of the top-200 gate sets,
       * global Jaccard over the (layer, feature) gate sets,
       * Spearman rank correlation of DLA on features shared by both top lists.

  B. Cross-importance (``--run-causal``; needs the model)
     For a single K, ablate setting-A's top-K global gates while measuring
     detection in setting B, and compare the rate drop to ablating B's *own*
     top-K gates (self-importance). This quantifies how much causal weight the
     gates discovered in one setting carry in the other.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


# ─────────────────────────────────────────────────────────────────────────────
# Overlap analysis
# ─────────────────────────────────────────────────────────────────────────────

def _gate_set(gates: dict, layer: int) -> Set[int]:
    entry = gates["gates"].get(str(layer))
    return set(entry["gate_ids"]) if entry else set()


def _global_set(gates: dict, k: int) -> Set[Tuple[int, int]]:
    return {(int(r["layer"]), int(r["feature_id"])) for r in gates["global_gates"][:k]}


def _jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def overlap_analysis(g_inj: dict, g_pre: dict, global_k: int) -> dict:
    layers = sorted(set(g_inj["gates"]) & set(g_pre["gates"]), key=int)
    per_layer = {}
    for L in layers:
        Li = int(L)
        si, sp = _gate_set(g_inj, Li), _gate_set(g_pre, Li)
        shared = si & sp
        # DLA rank correlation on shared features.
        dla_inj = dict(zip(g_inj["gates"][L]["gate_ids"], g_inj["gates"][L]["gate_dla"]))
        dla_pre = dict(zip(g_pre["gates"][L]["gate_ids"], g_pre["gates"][L]["gate_dla"]))
        if len(shared) >= 2:
            xs = np.array([dla_inj[f] for f in shared])
            ys = np.array([dla_pre[f] for f in shared])
            rho = _spearman(xs, ys)
        else:
            rho = float("nan")
        per_layer[L] = {
            "n_inj": len(si), "n_pre": len(sp), "n_shared": len(shared),
            "jaccard": _jaccard(si, sp), "dla_spearman_shared": rho,
        }

    gi, gp = _global_set(g_inj, global_k), _global_set(g_pre, global_k)
    return {
        "per_layer": per_layer,
        "global": {
            "k": global_k,
            "n_inj": len(gi), "n_pre": len(gp), "n_shared": len(gi & gp),
            "jaccard": _jaccard(gi, gp),
        },
    }


def plot_overlap(overlap: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(overlap["per_layer"], key=int)
    jac = [overlap["per_layer"][L]["jaccard"] for L in layers]
    rho = [overlap["per_layer"][L]["dla_spearman_shared"] for L in layers]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = [int(L) for L in layers]
    ax.bar(x, jac, color="tab:purple", alpha=0.7, label="Top-200 Jaccard")
    ax.plot(x, rho, "o-", color="tab:orange", label="DLA Spearman (shared)")
    ax.axhline(overlap["global"]["jaccard"], ls="--", color="tab:purple", alpha=0.6,
               label=f"Global Jaccard (K={overlap['global']['k']})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Overlap")
    ax.set_title("Injection vs prefill gate-feature overlap")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[03] Saved plot -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-importance (causal)
# ─────────────────────────────────────────────────────────────────────────────

def group_by_layer(global_gates: List[dict], k: int) -> Dict[int, List[int]]:
    layer2feats: Dict[int, List[int]] = defaultdict(list)
    for rec in global_gates[:k]:
        layer2feats[int(rec["layer"])].append(int(rec["feature_id"]))
    return dict(layer2feats)


def measure_detection_rate(
    model_w, gl, P, setting, concepts, n_trials, n_extra_words,
    steering_layer, strength, concept_vectors, yes_ids, no_ids,
    slices, prefill_variant,
) -> float:
    """Detection rate for ``setting`` with the features in ``slices`` ablated."""
    from contextlib import ExitStack
    import torch

    hits = []
    for concept in concepts:
        for trial in range(1, n_trials + 1):
            prefill = concept if setting == "prefill" else None
            det_ids = P.format_detection_prompt(
                model_w.tokenizer,
                trial,
                prefill,
                n_extra_words,
                prefill_variant=prefill_variant,
            )
            handle = None
            if setting == "injection":
                hook = gl.make_injection_hook(concept_vectors[concept], strength)
                handle = model_w.get_layer_module(steering_layer).register_forward_hook(hook)
            try:
                with ExitStack() as stack:
                    for L, fs in slices.items():
                        stack.enter_context(gl.GateAblation(model_w, fs))
                    logits = gl.forward_last_logits(model_w, det_ids)
            finally:
                if handle is not None:
                    handle.remove()
            hits.append(P.logit_gap(logits, yes_ids, no_ids).item() > 0)
    return float(np.mean(hits))


def run_cross_importance(args, g_inj, g_pre) -> dict:
    import gate_lib as gl
    import prompts as P
    from model_utils import load_model
    import torch

    cfg_inj, cfg_pre = g_inj["config"], g_pre["config"]
    steering_layer = cfg_inj["steering_layer"]
    strength = args.strength if args.strength is not None else cfg_inj["strength"]
    concepts = args.concepts if args.concepts is not None else cfg_inj["concepts"]
    yes_ids, no_ids = cfg_inj["yes_ids"], cfg_inj["no_ids"]
    prefill_variant = (
        args.prefill_variant
        if args.prefill_variant is not None
        else cfg_pre.get("prefill_variant", "append_user")
    )
    k = args.causal_k

    inj_feats = group_by_layer(g_inj["global_gates"], k)
    pre_feats = group_by_layer(g_pre["global_gates"], k)

    print(f"[03] Loading {args.model} for cross-importance (K={k}) ...")
    model_w = load_model(model_name=args.model, device=args.device, dtype=args.dtype)

    concept_vectors = gl.load_or_build_concept_vectors(
        model_w, concepts, steering_layer,
        Path(args.injection_gates).parent.parent / f"concept_vectors_L{steering_layer}.pt",
    )

    # Build per-layer slices for both gate sets (load each transcoder ONCE).
    def build_slices(layer2feats):
        slices = {}
        for L in sorted(layer2feats):
            tc = gl.load_transcoder(L, device=args.device)
            slices[L] = gl.FeatureSlice.from_transcoder(L, tc, layer2feats[L])
            del tc
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        return slices

    print(f"[03] Building slices: inj touches {len(inj_feats)} layer(s), pre touches {len(pre_feats)}")
    inj_slices = build_slices(inj_feats)
    pre_slices = build_slices(pre_feats)

    def detrate(setting, slices):
        return measure_detection_rate(
            model_w, gl, P, setting, concepts, args.n_trials, args.n_extra_words,
            steering_layer, strength, concept_vectors, yes_ids, no_ids, slices,
            prefill_variant=prefill_variant,
        )

    res = {
        "k": k,
        "prefill_variant": prefill_variant,
        "injection": {
            "baseline": detrate("injection", {}),
            "ablate_inj_gates": detrate("injection", inj_slices),   # self
            "ablate_pre_gates": detrate("injection", pre_slices),   # cross
        },
        "prefill": {
            "baseline": detrate("prefill", {}),
            "ablate_pre_gates": detrate("prefill", pre_slices),     # self
            "ablate_inj_gates": detrate("prefill", inj_slices),     # cross
        },
    }
    print(f"[03] cross-importance: {json.dumps(res, indent=2)}")
    return res


def main():
    p = argparse.ArgumentParser(description="Compare injection vs prefill gates.")
    p.add_argument("--injection-gates", required=True, help="injection gates.json")
    p.add_argument("--prefill-gates", required=True, help="prefill gates.json")
    p.add_argument("--global-k", type=int, default=200)
    p.add_argument("--output-dir", "-od", default="analysis/gate_minimal/comparison")
    # Causal cross-importance.
    p.add_argument("--run-causal", action="store_true")
    p.add_argument("--causal-k", type=int, default=50)
    p.add_argument("--model", "-m", default="gemma3_27b")
    p.add_argument("--strength", type=float, default=None)
    p.add_argument("--concepts", nargs="+", default=None)
    p.add_argument("--n-trials", "-nt", type=int, default=3)
    p.add_argument("--n-extra-words", type=int, default=1, choices=[1, 2, 3])
    p.add_argument(
        "--prefill-variant",
        choices=["append_user", "replace_assistant"],
        default=None,
        help="Prompt placement for prefill runs. Default: read from prefill gates.json if present.",
    )
    p.add_argument("--device", "-d", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    with open(args.injection_gates) as f:
        g_inj = json.load(f)
    with open(args.prefill_gates) as f:
        g_pre = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap = overlap_analysis(g_inj, g_pre, args.global_k)
    results = {"overlap": overlap}
    print("[03] Overlap (global):", overlap["global"])

    if args.run_causal:
        results["cross_importance"] = run_cross_importance(args, g_inj, g_pre)

    with open(out_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_overlap(overlap, out_dir / "overlap.png")
    print(f"[03] Saved -> {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()

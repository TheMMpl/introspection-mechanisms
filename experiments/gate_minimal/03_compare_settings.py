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
    shared_global = gi & gp

    # Per-layer distribution of the *global* top-K gates (how many of each
    # setting's top-K fall in each layer, plus how many are identical).
    global_distribution = _global_layer_distribution(gi, gp)

    # Global DLA rank correlation over the (layer, feature) pairs shared by both
    # top-K global lists — a single scalar summarizing how consistently the two
    # settings rank their common gates.
    dla_inj_g = {(int(r["layer"]), int(r["feature_id"])): float(r["dla"])
                 for r in g_inj["global_gates"][:global_k]}
    dla_pre_g = {(int(r["layer"]), int(r["feature_id"])): float(r["dla"])
                 for r in g_pre["global_gates"][:global_k]}
    if len(shared_global) >= 2:
        xs = np.array([dla_inj_g[p] for p in shared_global])
        ys = np.array([dla_pre_g[p] for p in shared_global])
        global_spearman = _spearman(xs, ys)
    else:
        global_spearman = float("nan")

    return {
        "per_layer": per_layer,
        "global_distribution": global_distribution,
        "global": {
            "k": global_k,
            "n_inj": len(gi), "n_pre": len(gp), "n_shared": len(shared_global),
            "jaccard": _jaccard(gi, gp),
            "spearman_shared": global_spearman,
        },
    }


def per_layer_dla_spearman(g_a: dict, g_b: dict) -> Dict[int, float]:
    """Per-layer Spearman correlation of DLA on the features shared by both
    settings' per-layer gate lists. Returns ``{layer: rho}`` (NaN where fewer
    than two features are shared at a layer)."""
    layers = sorted(set(g_a["gates"]) & set(g_b["gates"]), key=int)
    out: Dict[int, float] = {}
    for L in layers:
        ga, gb = g_a["gates"][L], g_b["gates"][L]
        dla_a = dict(zip(ga["gate_ids"], ga["gate_dla"]))
        dla_b = dict(zip(gb["gate_ids"], gb["gate_dla"]))
        shared = set(dla_a) & set(dla_b)
        if len(shared) >= 2:
            xs = np.array([dla_a[f] for f in shared])
            ys = np.array([dla_b[f] for f in shared])
            out[int(L)] = _spearman(xs, ys)
        else:
            out[int(L)] = float("nan")
    return out


def _global_layer_distribution(
    gi: Set[Tuple[int, int]], gp: Set[Tuple[int, int]]
) -> Dict[str, Dict[str, int]]:
    """Bucket the top-K global gate sets by layer.

    Returns ``{layer: {n_inj, n_pre, n_shared}}`` where ``n_shared`` counts
    (layer, feature) pairs present in *both* settings' top-K at that layer.
    """
    shared = gi & gp
    layers = sorted({L for L, _ in gi} | {L for L, _ in gp})
    dist: Dict[str, Dict[str, int]] = {}
    for L in layers:
        dist[str(L)] = {
            "n_inj": sum(1 for (l, _) in gi if l == L),
            "n_pre": sum(1 for (l, _) in gp if l == L),
            "n_shared": sum(1 for (l, _) in shared if l == L),
        }
    return dist


def plot_overlap(overlap: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dist = overlap["global_distribution"]
    layers = sorted((int(L) for L in dist), key=int)
    n_inj = [dist[str(L)]["n_inj"] for L in layers]
    n_pre = [dist[str(L)]["n_pre"] for L in layers]
    n_shared = [dist[str(L)]["n_shared"] for L in layers]

    g = overlap["global"]
    rho = g.get("spearman_shared", float("nan"))
    jac = g.get("jaccard", float("nan"))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    # Overlaid translucent raw distributions (same x positions, full width).
    ax.bar(layers, n_inj, width=0.9, color="tab:blue", alpha=0.45,
           label=f"Injection (top-{g['k']}, n={g['n_inj']})")
    ax.bar(layers, n_pre, width=0.9, color="tab:orange", alpha=0.45,
           label=f"Prefill (top-{g['k']}, n={g['n_pre']})")
    # Solid, narrower overlap bar for identical (layer, feature) gates.
    ax.bar(layers, n_shared, width=0.45, color="tab:green", alpha=0.95,
           label=f"Identical gates (total={g['n_shared']})")

    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Number of global gates (top-{g['k']})")
    ax.set_title(
        "Injection vs prefill global-gate layer distribution\n"
        f"global overlap={g['n_shared']}/{g['k']} (Jaccard={jac:.3f}), "
        f"DLA Spearman (shared)={rho:.3f}"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[03] Saved plot -> {out_path}")


def plot_overlap_dodged(overlap: dict, out_path: Path):
    """Dodged variant of the overlap plot: injection, identical, and prefill
    bars are placed side-by-side per layer (rather than overlaid) with solid
    fills and black borders, so intersecting counts are easy to read off."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dist = overlap["global_distribution"]
    layers = sorted((int(L) for L in dist), key=int)
    x = np.arange(len(layers))
    n_inj = [dist[str(L)]["n_inj"] for L in layers]
    n_pre = [dist[str(L)]["n_pre"] for L in layers]
    n_shared = [dist[str(L)]["n_shared"] for L in layers]

    g = overlap["global"]
    rho = g.get("spearman_shared", float("nan"))
    jac = g.get("jaccard", float("nan"))

    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_axisbelow(True)
    ax.set_facecolor("#eaeaf2")
    ax.grid(axis="y", color="white", linewidth=1.2)
    # Side-by-side, flush within each layer group: injection | prefill | identical.
    ax.bar(x - w, n_inj, width=w, color="tab:blue",
           edgecolor="black", linewidth=0.7,
           label=f"Injection (top-{g['k']}, n={g['n_inj']})")
    ax.bar(x, n_pre, width=w, color="tab:orange",
           edgecolor="black", linewidth=0.7,
           label=f"Prefill (top-{g['k']}, n={g['n_pre']})")
    ax.bar(x + w, n_shared, width=w, color="tab:green",
           edgecolor="black", linewidth=0.7,
           label=f"Identical gates (total={g['n_shared']})")

    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in layers])
    ax.set_xlim(-0.6, len(layers) - 0.4)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Number of global gates (top-{g['k']})")
    ax.set_title(
        "Injection vs prefill global-gate layer distribution (dodged)\n"
        f"global overlap={g['n_shared']}/{g['k']} (Jaccard={jac:.3f}), "
        f"DLA Spearman (shared)={rho:.3f}"
    )
    ax.legend(fontsize=9, ncol=3, loc="upper center", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[03] Saved plot -> {out_path}")


def plot_per_layer_spearman(pairs: List[Tuple[str, str, Dict[int, float]]], out_path: Path):
    """Plot one per-layer DLA-Spearman line per setting pair on shared axes.

    ``pairs`` is a list of ``(label, color, {layer: rho})``. Layers with too few
    shared features (NaN rho) break the line, exposing where the settings have
    no comparable gates.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_layers = sorted({L for _, _, d in pairs for L in d})
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.6)
    for label, color, d in pairs:
        ys = [d.get(L, float("nan")) for L in all_layers]
        ax.plot(all_layers, ys, marker="o", color=color, label=label)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Per-layer DLA Spearman (shared features)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Per-layer gate-DLA rank correlation across settings")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
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
    p.add_argument(
        "--prefill-user-gates",
        default=None,
        help="Optional append_user prefill gates.json (for the per-layer Spearman plot).",
    )
    p.add_argument(
        "--prefill-assistant-gates",
        default=None,
        help="Optional replace_assistant prefill gates.json (for the per-layer Spearman plot).",
    )
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
    plot_overlap_dodged(overlap, out_dir / "overlap_dodged.png")

    # Per-layer DLA-Spearman across multiple settings (optional inputs).
    g_user = g_assistant = None
    if args.prefill_user_gates:
        with open(args.prefill_user_gates) as f:
            g_user = json.load(f)
    if args.prefill_assistant_gates:
        with open(args.prefill_assistant_gates) as f:
            g_assistant = json.load(f)

    spearman_pairs: List[Tuple[str, str, Dict[int, float]]] = []
    if g_user is not None:
        spearman_pairs.append(
            ("Prefill (user) \u2194 injection", "tab:blue",
             per_layer_dla_spearman(g_user, g_inj))
        )
    if g_assistant is not None:
        spearman_pairs.append(
            ("Prefill (assistant) \u2194 injection", "tab:orange",
             per_layer_dla_spearman(g_assistant, g_inj))
        )
    if g_user is not None and g_assistant is not None:
        spearman_pairs.append(
            ("Prefill user \u2194 assistant", "tab:green",
             per_layer_dla_spearman(g_user, g_assistant))
        )
    if spearman_pairs:
        results["per_layer_spearman"] = {
            label: {str(L): rho for L, rho in d.items()}
            for label, _, d in spearman_pairs
        }
        with open(out_dir / "comparison.json", "w") as f:
            json.dump(results, f, indent=2)
        plot_per_layer_spearman(spearman_pairs, out_dir / "per_layer_spearman.png")
    else:
        print("[03] Skipping per-layer Spearman plot "
              "(provide --prefill-user-gates and/or --prefill-assistant-gates).")

    print(f"[03] Saved -> {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()

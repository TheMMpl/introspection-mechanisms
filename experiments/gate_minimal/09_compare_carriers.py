#!/usr/bin/env python3
"""Step 9 — Compare injection vs prefill *evidence-carrier* features.

Companion to ``03_compare_settings.py`` (gates), but for the evidence carriers
collected by ``08_collect_carriers.py`` against a shared gate feature.

Two analyses:

  A. Overlap (no model needed)
     Given step-8 ``carriers.json`` for both settings, report:
       * per-concept Jaccard overlap of each setting's carrier (layer, feat)
         set (mean + per-concept), since carriers are concept-specific,
       * global Jaccard over the pooled top-K carrier sets,
       * Spearman rank correlation of attribution on carriers shared by both
         pooled top lists,
       * the per-layer distribution of pooled carriers (overlap plots).

  B. Cross-importance (``--run-causal``; needs the model)
     Ablate setting-A's top-K pooled carriers while measuring the *gate
     feature's activation* in setting B, and compare to ablating B's own top-K
     carriers (self-importance). Because carriers write *against* the gate,
     ablating real carriers should *raise* the gate activation; this quantifies
     how much of that causal weight transfers across settings.
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
# Small numerics (shared with 03)
# ─────────────────────────────────────────────────────────────────────────────

def _jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Carrier-set extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _concept_carrier_set(carriers: dict, concept: str) -> Set[Tuple[int, int]]:
    entry = carriers["per_concept"].get(concept)
    if not entry:
        return set()
    return {(int(r["layer"]), int(r["feat_id"])) for r in entry["carriers"]}


def _concept_attr_map(carriers: dict, concept: str) -> Dict[Tuple[int, int], float]:
    entry = carriers["per_concept"].get(concept)
    if not entry:
        return {}
    return {(int(r["layer"]), int(r["feat_id"])): float(r["attribution"])
            for r in entry["carriers"]}


def _pooled_set(carriers: dict, k: int) -> Set[Tuple[int, int]]:
    return {(int(r["layer"]), int(r["feat_id"])) for r in carriers["pooled_carriers"][:k]}


def _pooled_attr_map(carriers: dict, k: int) -> Dict[Tuple[int, int], float]:
    return {(int(r["layer"]), int(r["feat_id"])): float(r["mean_attribution"])
            for r in carriers["pooled_carriers"][:k]}


# ─────────────────────────────────────────────────────────────────────────────
# Overlap analysis
# ─────────────────────────────────────────────────────────────────────────────

def overlap_analysis(c_inj: dict, c_pre: dict, global_k: int) -> dict:
    concepts = [c for c in c_inj["config"]["concepts"] if c in c_pre["per_concept"]]

    # Per-concept Jaccard + attribution Spearman on shared carriers.
    per_concept = {}
    jaccards, rhos = [], []
    for c in concepts:
        si, sp = _concept_carrier_set(c_inj, c), _concept_carrier_set(c_pre, c)
        shared = si & sp
        jac = _jaccard(si, sp)
        ai, ap = _concept_attr_map(c_inj, c), _concept_attr_map(c_pre, c)
        if len(shared) >= 2:
            xs = np.array([ai[p] for p in shared])
            ys = np.array([ap[p] for p in shared])
            rho = _spearman(xs, ys)
        else:
            rho = float("nan")
        per_concept[c] = {
            "n_inj": len(si), "n_pre": len(sp), "n_shared": len(shared),
            "jaccard": jac, "attr_spearman_shared": rho,
        }
        if not np.isnan(jac):
            jaccards.append(jac)
        if not np.isnan(rho):
            rhos.append(rho)

    # Pooled global overlap.
    gi, gp = _pooled_set(c_inj, global_k), _pooled_set(c_pre, global_k)
    shared_global = gi & gp
    ai_g, ap_g = _pooled_attr_map(c_inj, global_k), _pooled_attr_map(c_pre, global_k)
    if len(shared_global) >= 2:
        xs = np.array([ai_g[p] for p in shared_global])
        ys = np.array([ap_g[p] for p in shared_global])
        global_spearman = _spearman(xs, ys)
    else:
        global_spearman = float("nan")

    return {
        "concepts": concepts,
        "per_concept": per_concept,
        "per_concept_summary": {
            "mean_jaccard": float(np.mean(jaccards)) if jaccards else float("nan"),
            "median_jaccard": float(np.median(jaccards)) if jaccards else float("nan"),
            "mean_attr_spearman": float(np.mean(rhos)) if rhos else float("nan"),
            "n_concepts": len(concepts),
        },
        "global_distribution": _pooled_layer_distribution(gi, gp),
        "global": {
            "k": global_k,
            "n_inj": len(gi), "n_pre": len(gp), "n_shared": len(shared_global),
            "jaccard": _jaccard(gi, gp),
            "spearman_shared": global_spearman,
        },
    }


def _pooled_layer_distribution(
    gi: Set[Tuple[int, int]], gp: Set[Tuple[int, int]]
) -> Dict[str, Dict[str, int]]:
    """Bucket the pooled top-K carrier sets by layer (n_inj/n_pre/n_shared)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_overlap(overlap: dict, gate_label: str, out_path: Path):
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
    ax.bar(layers, n_inj, width=0.9, color="tab:blue", alpha=0.45,
           label=f"Injection (top-{g['k']}, n={g['n_inj']})")
    ax.bar(layers, n_pre, width=0.9, color="tab:orange", alpha=0.45,
           label=f"Prefill (top-{g['k']}, n={g['n_pre']})")
    ax.bar(layers, n_shared, width=0.45, color="tab:green", alpha=0.95,
           label=f"Identical carriers (total={g['n_shared']})")

    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Number of pooled evidence carriers (top-{g['k']})")
    ax.set_title(
        f"Injection vs prefill evidence-carrier layer distribution\n"
        f"gate {gate_label}  •  global overlap={g['n_shared']}/{g['k']} "
        f"(Jaccard={jac:.3f}), attribution Spearman (shared)={rho:.3f}"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[09] Saved plot -> {out_path}")


def plot_overlap_dodged(overlap: dict, gate_label: str, out_path: Path):
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
    ax.bar(x - w, n_inj, width=w, color="tab:blue", edgecolor="black", linewidth=0.7,
           label=f"Injection (top-{g['k']}, n={g['n_inj']})")
    ax.bar(x, n_pre, width=w, color="tab:orange", edgecolor="black", linewidth=0.7,
           label=f"Prefill (top-{g['k']}, n={g['n_pre']})")
    ax.bar(x + w, n_shared, width=w, color="tab:green", edgecolor="black", linewidth=0.7,
           label=f"Identical carriers (total={g['n_shared']})")

    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in layers])
    ax.set_xlim(-0.6, len(layers) - 0.4)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Number of pooled evidence carriers (top-{g['k']})")
    ax.set_title(
        f"Injection vs prefill evidence-carrier layer distribution (dodged)\n"
        f"gate {gate_label}  •  global overlap={g['n_shared']}/{g['k']} "
        f"(Jaccard={jac:.3f}), attribution Spearman (shared)={rho:.3f}"
    )
    ax.legend(fontsize=9, ncol=3, loc="upper center", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[09] Saved plot -> {out_path}")


def plot_per_concept_jaccard(overlap: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pc = overlap["per_concept"]
    concepts = list(pc.keys())
    jacs = [pc[c]["jaccard"] for c in concepts]
    order = np.argsort([(-1 if np.isnan(j) else j) for j in jacs])[::-1]
    concepts = [concepts[i] for i in order]
    jacs = [jacs[i] for i in order]

    summ = overlap["per_concept_summary"]
    fig, ax = plt.subplots(figsize=(max(8, len(concepts) * 0.28), 4.8))
    ax.bar(range(len(concepts)), [0 if np.isnan(j) else j for j in jacs],
           color="tab:green", alpha=0.8)
    ax.axhline(summ["mean_jaccard"], color="black", lw=1.0, ls="--",
               label=f"mean={summ['mean_jaccard']:.3f}")
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=90, fontsize=7)
    ax.set_ylabel("Per-concept carrier Jaccard (inj vs pre)")
    ax.set_title(
        f"Per-concept evidence-carrier overlap  •  "
        f"mean attr Spearman={summ['mean_attr_spearman']:.3f} "
        f"(n={summ['n_concepts']} concepts)"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[09] Saved plot -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-importance (causal)
# ─────────────────────────────────────────────────────────────────────────────

def group_by_layer(pooled_carriers: List[dict], k: int) -> Dict[int, List[int]]:
    layer2feats: Dict[int, List[int]] = defaultdict(list)
    for rec in pooled_carriers[:k]:
        layer2feats[int(rec["layer"])].append(int(rec["feat_id"]))
    return dict(layer2feats)


def measure_gate_activation(
    model_w, gl, P, setting, concepts, n_trials, n_extra_words,
    steering_layer, strength, concept_vectors, gate_layer, gate_feat_id, gate_tc,
    slices, prefill_variant, detect_variant,
    ablation_mode="zero", patch_scope="last", injection_scope="all",
) -> float:
    """Mean gate-feature activation for ``setting`` with ``slices`` intervened.

    ``ablation_mode='zero'`` removes the carriers' contribution at all positions
    (``GateAblation``). ``ablation_mode='patch'`` instead *replaces* their
    activation with the unsteered / no-concept value (``GatePatch``): a control
    forward on the plain prompt (no injection, no prefill word) supplies the
    target activations, which are written back at ``patch_scope`` positions
    ('last' or 'all'). ``injection_scope`` controls where the steering vector is
    added in the injection setting.

    All-position patching aligns positions by reusing the same prompt, so it is
    only well-defined when the control and intervened forwards share a sequence
    length (the injection setting). For the prefill setting it falls back to
    last-token patching (the concept word changes the sequence length).
    """
    from contextlib import ExitStack
    import torch

    eff_patch_scope = patch_scope
    if ablation_mode == "patch" and patch_scope == "all" and setting == "prefill":
        eff_patch_scope = "last"  # prefill prompt seq != plain control seq

    vals = []
    for concept in concepts:
        for trial in range(1, n_trials + 1):
            prefill = concept if setting == "prefill" else None
            det_ids = P.format_detection_prompt(
                model_w.tokenizer, trial, prefill, n_extra_words,
                prefill_variant=prefill_variant, detect_variant=detect_variant,
            )

            # Patch mode: capture the no-concept control activations (plain prompt,
            # no injection) for the intervened slices.
            control_sel = None
            if ablation_mode == "patch" and slices:
                control_ids = P.format_detection_prompt(
                    model_w.tokenizer, trial, None, n_extra_words,
                    prefill_variant=prefill_variant, detect_variant=detect_variant,
                )
                with gl.TranscoderCapture(model_w, list(slices.keys())) as ccap:
                    with torch.no_grad():
                        model_w.model(input_ids=control_ids.to(model_w.device), use_cache=False)
                    control_sel = {}
                    for L, fs in slices.items():
                        if eff_patch_scope == "all":
                            control_sel[L] = fs.encode_all_tokens(ccap.captured[L])
                        else:
                            control_sel[L] = fs.encode_last_token(ccap.captured[L])

            handle = None
            if setting == "injection":
                hook = gl.make_injection_hook(
                    concept_vectors[concept], strength, positions=injection_scope
                )
                handle = model_w.get_layer_module(steering_layer).register_forward_hook(hook)
            try:
                with ExitStack() as stack:
                    for L, fs in slices.items():
                        if ablation_mode == "zero":
                            stack.enter_context(gl.GateAblation(model_w, fs))
                        else:
                            stack.enter_context(gl.GatePatch(
                                model_w, fs, steered_sel=control_sel[L],
                                positions=eff_patch_scope,
                            ))
                    with gl.TranscoderCapture(model_w, [gate_layer]) as cap:
                        with torch.no_grad():
                            model_w.model(input_ids=det_ids.to(model_w.device), use_cache=False)
                        pre_ln = cap.captured[gate_layer]
                        feats = gl.encode_last_token(gate_tc, pre_ln)  # [F] fp32 cpu
                        vals.append(float(feats[gate_feat_id]))
            finally:
                if handle is not None:
                    handle.remove()
    return float(np.mean(vals))


def run_cross_importance(args, c_inj, c_pre) -> dict:
    import gate_lib as gl
    import prompts as P
    from model_utils import load_model
    import torch

    cfg = c_inj["config"]
    gate_layer = int(c_inj["gate_feature"]["layer"])
    gate_feat_id = int(c_inj["gate_feature"]["feat_id"])
    steering_layer = cfg["steering_layer"]
    strength = args.strength if args.strength is not None else cfg["strength"]
    concepts = args.concepts if args.concepts is not None else cfg["concepts"]
    concepts = [c for c in concepts if c in c_pre["per_concept"]]
    prefill_variant = args.prefill_variant or c_pre["config"].get("prefill_variant", "append_user")
    detect_variant = args.detect_variant or cfg.get("detect_variant", "strict")
    k = args.causal_k

    inj_feats = group_by_layer(c_inj["pooled_carriers"], k)
    pre_feats = group_by_layer(c_pre["pooled_carriers"], k)

    print(f"[09] Loading {args.model} for cross-importance (K={k}) ...")
    model_w = load_model(model_name=args.model, device=args.device, dtype=args.dtype)

    concept_vectors = gl.load_or_build_concept_vectors(
        model_w, concepts, steering_layer,
        Path(args.injection_carriers).parent.parent.parent
        / f"concept_vectors_L{steering_layer}.pt",
    )

    gate_tc = gl.load_transcoder(gate_layer, device=args.device)

    def build_slices(layer2feats):
        slices = {}
        for L in sorted(layer2feats):
            tc = gl.load_transcoder(L, device=args.device)
            slices[L] = gl.FeatureSlice.from_transcoder(L, tc, layer2feats[L])
            del tc
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        return slices

    print(f"[09] Building slices: inj touches {len(inj_feats)} layer(s), pre touches {len(pre_feats)}")
    inj_slices = build_slices(inj_feats)
    pre_slices = build_slices(pre_feats)

    def gate_act(setting, slices):
        return measure_gate_activation(
            model_w, gl, P, setting, concepts, args.n_trials, args.n_extra_words,
            steering_layer, strength, concept_vectors, gate_layer, gate_feat_id,
            gate_tc, slices, prefill_variant, detect_variant,
            ablation_mode=args.ablation_mode, patch_scope=args.patch_scope,
            injection_scope=args.injection_scope,
        )

    res = {
        "k": k,
        "gate_feature": f"L{gate_layer}_F{gate_feat_id}",
        "prefill_variant": prefill_variant,
        "detect_variant": detect_variant,
        "ablation_mode": args.ablation_mode,
        "patch_scope": args.patch_scope,
        "injection_scope": args.injection_scope,
        "n_concepts": len(concepts),
        "injection": {
            "baseline": gate_act("injection", {}),
            "ablate_inj_carriers": gate_act("injection", inj_slices),   # self
            "ablate_pre_carriers": gate_act("injection", pre_slices),   # cross
        },
        "prefill": {
            "baseline": gate_act("prefill", {}),
            "ablate_pre_carriers": gate_act("prefill", pre_slices),     # self
            "ablate_inj_carriers": gate_act("prefill", inj_slices),     # cross
        },
    }
    print(f"[09] cross-importance: {json.dumps(res, indent=2)}")
    return res


def main():
    p = argparse.ArgumentParser(description="Compare injection vs prefill evidence carriers.")
    p.add_argument("--injection-carriers", required=True, help="injection carriers.json")
    p.add_argument("--prefill-carriers", required=True, help="prefill carriers.json")
    p.add_argument("--global-k", type=int, default=200,
                   help="Pooled top-K carriers used for the global overlap.")
    p.add_argument("--output-dir", "-od", default="analysis/gate_minimal/carrier_comparison")
    # Causal cross-importance.
    p.add_argument("--run-causal", action="store_true")
    p.add_argument("--causal-k", type=int, default=50)
    p.add_argument("--ablation-mode", choices=["zero", "patch"], default="zero",
                   help="'zero' removes carriers at all positions (GateAblation); "
                        "'patch' replaces their activation with the unsteered / "
                        "no-concept value (GatePatch).")
    p.add_argument("--patch-scope", choices=["last", "all"], default="last",
                   help="Patch positions for --ablation-mode patch ('all' falls back "
                        "to 'last' in the prefill setting).")
    p.add_argument("--injection-scope", choices=["all", "last"], default="all",
                   help="Inject the steering vector at all positions (default) or "
                        "only the answer token.")
    p.add_argument("--model", "-m", default="gemma3_27b")
    p.add_argument("--strength", type=float, default=None)
    p.add_argument("--concepts", nargs="+", default=None)
    p.add_argument("--n-trials", "-nt", type=int, default=3)
    p.add_argument("--n-extra-words", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--prefill-variant", choices=["append_user", "replace_assistant"], default=None)
    p.add_argument("--detect-variant", choices=["strict", "vague"], default=None)
    p.add_argument("--device", "-d", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    with open(args.injection_carriers) as f:
        c_inj = json.load(f)
    with open(args.prefill_carriers) as f:
        c_pre = json.load(f)

    gl_layer = int(c_inj["gate_feature"]["layer"])
    gl_feat = int(c_inj["gate_feature"]["feat_id"])
    gate_label = f"L{gl_layer}_F{gl_feat}"
    pre_gate = f"L{int(c_pre['gate_feature']['layer'])}_F{int(c_pre['gate_feature']['feat_id'])}"
    if pre_gate != gate_label:
        print(f"[09] WARNING: gate features differ (inj {gate_label} vs pre {pre_gate}); "
              f"carrier sets are not directly comparable.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap = overlap_analysis(c_inj, c_pre, args.global_k)
    results = {"gate_feature": gate_label, "overlap": overlap}
    print("[09] Overlap (global):", overlap["global"])
    print("[09] Overlap (per-concept summary):", overlap["per_concept_summary"])

    if args.run_causal:
        results["cross_importance"] = run_cross_importance(args, c_inj, c_pre)

    with open(out_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_overlap(overlap, gate_label, out_dir / "carrier_overlap.png")
    plot_overlap_dodged(overlap, gate_label, out_dir / "carrier_overlap_dodged.png")
    plot_per_concept_jaccard(overlap, out_dir / "carrier_per_concept_jaccard.png")
    print(f"[09] Saved comparison -> {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()

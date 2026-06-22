#!/usr/bin/env python3
"""Step 5 — Cross-setting gate activation behavior (no model run needed).

Motivation
----------
The original "gate" features were characterized as *decreasing* in activation
when the concept is present (they fire on the unsteered/baseline run and get
suppressed when the model detects the concept). The injection-setting gates
reproduce this (steered < unsteered). The prefill-setting gates instead
*increase* (steered > unsteered). This script answers whether that opposite
behavior is a property of:

  * the **feature set** that got selected (injection-DLA vs prefill-DLA pick
    structurally different features), or
  * the **measurement setting** (the very same features behave differently when
    the concept arrives via injection vs via appended/replaced text).

Why no forward passes are needed
--------------------------------
Step-1 ``mean_acts.pt`` already stores the *full* per-feature mean activation
tensors (``{layer: tensor[n_features]}``) for both the ``steered`` and
``unsteered`` conditions, for every scanned layer. Because both settings use the
same GemmaScope transcoders, a gate feature ``(layer, feature_id)`` selected in
one setting can be looked up directly in the other setting's tensors. The
``unsteered`` baseline is the bare detection prompt in both settings (verified
bit-identical), so ``delta = steered - unsteered`` is directly comparable across
settings.

The 2x2
-------
                       measured in INJECTION      measured in PREFILL
  injection-selected   delta_ii                   delta_ip
  prefill-selected     delta_pi                   delta_pp

Interpretation
  * delta sign tracks the **column** (setting)  -> behavior is SETTING-driven:
        a fixed feature flips decrease<->increase depending on how the concept
        is delivered.
  * delta sign tracks the **row** (feature set) -> behavior is FEATURE-driven:
        injection-selected features decrease in *both* settings; prefill ones
        increase in both.

Outputs (under ``--output-dir``)
  * cross_setting_acts.pt           raw per-feature cross-scenario activations
  * cross_setting_acts_summary.json aggregated 2x2 table per K
  * cross_setting_acts.png          grouped-bar visualization of the 2x2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load(setting_dir: Path) -> Tuple[dict, Dict[str, Dict[int, torch.Tensor]]]:
    with open(setting_dir / "gates.json") as f:
        gates = json.load(f)
    ma = torch.load(setting_dir / "mean_acts.pt", map_location="cpu", weights_only=False)
    # Normalize layer keys to int.
    ma = {cond: {int(L): t for L, t in layer_map.items()} for cond, layer_map in ma.items()}
    return gates, ma


def _gate_pairs(gates: dict, k: int) -> List[Tuple[int, int, float]]:
    """Return the top-k global gates as (layer, feature_id, dla)."""
    out = []
    for rec in gates["global_gates"][:k]:
        out.append((int(rec["layer"]), int(rec["feature_id"]), float(rec["dla"])))
    return out


def _lookup(ma: Dict[str, Dict[int, torch.Tensor]], cond: str, layer: int, feat: int) -> float:
    return float(ma[cond][layer][feat])


def _summ(deltas: List[float]) -> dict:
    if not deltas:
        nan = float("nan")
        return {"mean_delta": nan, "median_delta": nan, "frac_decreasing": nan, "n": 0}
    t = torch.tensor(deltas)
    return {
        "mean_delta": float(t.mean()),
        "median_delta": float(t.median()),
        "frac_decreasing": float((t < 0).float().mean()),
        "n": int(t.numel()),
    }


def build_cross(
    gates_inj: dict,
    ma_inj: Dict[str, Dict[int, torch.Tensor]],
    gates_pre: dict,
    ma_pre: Dict[str, Dict[int, torch.Tensor]],
    k_grid: List[int],
) -> dict:
    """Compute the 2x2 cross-setting activation table for each K, plus the raw
    per-feature activations for the largest K (a superset of all smaller K)."""
    # feature_set -> the gates.json it came from
    feat_sets = {"injection_selected": gates_inj, "prefill_selected": gates_pre}
    # measurement setting -> its mean_acts
    meas = {"injection": ma_inj, "prefill": ma_pre}

    summary: Dict[str, Dict[str, Dict[int, dict]]] = {
        fs: {ms: {} for ms in meas} for fs in feat_sets
    }
    for fs_name, g in feat_sets.items():
        for k in k_grid:
            pairs = _gate_pairs(g, k)
            for ms_name, ma in meas.items():
                deltas = []
                for (L, f, _dla) in pairs:
                    s = _lookup(ma, "steered", L, f)
                    u = _lookup(ma, "unsteered", L, f)
                    deltas.append(s - u)
                summary[fs_name][ms_name][k] = _summ(deltas)

    # Raw per-feature dump at max K (so the saved file is self-contained).
    k_max = max(k_grid)
    raw: Dict[str, List[dict]] = {}
    for fs_name, g in feat_sets.items():
        rows = []
        for (L, f, dla) in _gate_pairs(g, k_max):
            rows.append({
                "layer": L,
                "feature_id": f,
                "dla": dla,
                "inj_steered": _lookup(ma_inj, "steered", L, f),
                "inj_unsteered": _lookup(ma_inj, "unsteered", L, f),
                "pre_steered": _lookup(ma_pre, "steered", L, f),
                "pre_unsteered": _lookup(ma_pre, "unsteered", L, f),
                "delta_inj": _lookup(ma_inj, "steered", L, f) - _lookup(ma_inj, "unsteered", L, f),
                "delta_pre": _lookup(ma_pre, "steered", L, f) - _lookup(ma_pre, "unsteered", L, f),
            })
        raw[fs_name] = rows
    return {"summary": summary, "raw": raw, "k_grid": k_grid}


def verdict(summary: dict, k: int) -> str:
    """Render a short feature-vs-setting verdict at a given K."""
    def md(fs, ms):
        return summary[fs][ms][k]["mean_delta"]

    ii, ip = md("injection_selected", "injection"), md("injection_selected", "prefill")
    pi, pp = md("prefill_selected", "injection"), md("prefill_selected", "prefill")
    # Row effect: does delta hold its sign across settings within a feature set?
    inj_row_consistent = (ii < 0) == (ip < 0)
    pre_row_consistent = (pi < 0) == (pp < 0)
    # Column effect: does delta flip sign across feature sets within a setting?
    lines = [
        f"K={k} mean delta (steered - unsteered):",
        f"  injection-selected:  in injection={ii:+.2f}   in prefill={ip:+.2f}",
        f"  prefill-selected:    in injection={pi:+.2f}   in prefill={pp:+.2f}",
    ]
    if inj_row_consistent and pre_row_consistent:
        lines.append("  => FEATURE-driven: each set keeps its delta sign across both settings.")
    elif (ii < 0) == (pi < 0) and (ip < 0) == (pp < 0) and ((ii < 0) != (ip < 0)):
        lines.append("  => SETTING-driven: delta sign follows the measurement setting, "
                     "not the selected feature set.")
    else:
        lines.append("  => MIXED: neither a clean feature- nor setting-driven pattern.")
    return "\n".join(lines)


def plot_cross(summary: dict, k: int, out_path: Path, cfg_note: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    feat_sets = ["injection_selected", "prefill_selected"]
    settings = ["injection", "prefill"]
    colors = {"injection": "tab:orange", "prefill": "tab:green"}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(feat_sets))
    w = 0.36
    for i, ms in enumerate(settings):
        means = [summary[fs][ms][k]["mean_delta"] for fs in feat_sets]
        axL.bar(x + (i - 0.5) * w, means, w, label=f"measured in {ms}", color=colors[ms])
    axL.axhline(0, color="0.4", lw=0.8)
    axL.set_xticks(x)
    axL.set_xticklabels(["injection-\nselected", "prefill-\nselected"])
    axL.set_ylabel("mean delta  (steered - unsteered)")
    axL.set_title("Gate activation change")
    axL.legend(fontsize=9)
    axL.grid(True, axis="y", alpha=0.3)

    for i, ms in enumerate(settings):
        fracs = [summary[fs][ms][k]["frac_decreasing"] * 100 for fs in feat_sets]
        axR.bar(x + (i - 0.5) * w, fracs, w, label=f"measured in {ms}", color=colors[ms])
    axR.axhline(50, color="0.4", lw=0.8, ls="--")
    axR.set_xticks(x)
    axR.set_xticklabels(["injection-\nselected", "prefill-\nselected"])
    axR.set_ylabel("% of gates that decrease when steered")
    axR.set_ylim(0, 100)
    axR.set_title("Fraction suppressed (decrease)")
    axR.legend(fontsize=9)
    axR.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Cross-setting gate activation behavior (top-{k} gates)\n{cfg_note}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[05] Saved plot -> {out_path}")


def main():
    p = argparse.ArgumentParser(description="Cross-setting gate activation behavior (no model).")
    p.add_argument("--inj-dir", required=True,
                   help="Step-1 output dir for the injection setting (has gates.json + mean_acts.pt).")
    p.add_argument("--pre-dir", required=True,
                   help="Step-1 output dir for the prefill setting.")
    p.add_argument("--k-grid", type=int, nargs="+", default=[16, 32, 64, 200])
    p.add_argument("--report-k", type=int, default=32,
                   help="K used for the printed verdict and the plot.")
    p.add_argument("--output-dir", "-od", default=None,
                   help="Default: <inj-dir>/cross_setting_acts.")
    args = p.parse_args()

    inj_dir, pre_dir = Path(args.inj_dir), Path(args.pre_dir)
    out_dir = Path(args.output_dir) if args.output_dir else inj_dir / "cross_setting_acts"
    out_dir.mkdir(parents=True, exist_ok=True)

    gates_inj, ma_inj = _load(inj_dir)
    gates_pre, ma_pre = _load(pre_dir)

    # Sanity: the unsteered baseline must match across settings for delta to be
    # comparable (it is the same bare detection prompt in both runs).
    shared_layers = sorted(set(ma_inj["unsteered"]) & set(ma_pre["unsteered"]))
    L0 = shared_layers[0]
    base_diff = (ma_inj["unsteered"][L0] - ma_pre["unsteered"][L0]).abs().max().item()
    if base_diff > 1e-3:
        print(f"[05] WARNING: unsteered baselines differ across settings "
              f"(L{L0} max|diff|={base_diff:.4f}); delta comparison may be confounded.")
    else:
        print(f"[05] unsteered baselines match across settings (L{L0} max|diff|={base_diff:.2e}).")

    k_grid = sorted(set(args.k_grid))
    result = build_cross(gates_inj, ma_inj, gates_pre, ma_pre, k_grid)

    # Save raw cross-scenario activations.
    torch.save(result, out_dir / "cross_setting_acts.pt")
    print(f"[05] Saved raw cross activations -> {out_dir / 'cross_setting_acts.pt'}")

    # Save aggregated summary (k keys -> str for JSON).
    summary_json = {
        "inj_dir": str(inj_dir),
        "pre_dir": str(pre_dir),
        "k_grid": k_grid,
        "summary": {
            fs: {ms: {str(k): result["summary"][fs][ms][k] for k in k_grid}
                 for ms in result["summary"][fs]}
            for fs in result["summary"]
        },
    }
    with open(out_dir / "cross_setting_acts_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"[05] Saved summary -> {out_dir / 'cross_setting_acts_summary.json'}")

    rk = args.report_k if args.report_k in k_grid else k_grid[0]
    print("\n" + verdict(result["summary"], rk) + "\n")

    cfg_note = (f"inj=L{gates_inj['config']['steering_layer']} "
                f"s={gates_inj['config']['strength']} | "
                f"pre={gates_pre['config'].get('prefill_variant','?')}/"
                f"{gates_pre['config'].get('detect_variant','?')}")
    plot_cross(result["summary"], rk, out_dir / "cross_setting_acts.png", cfg_note)


if __name__ == "__main__":
    main()

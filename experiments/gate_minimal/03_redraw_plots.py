#!/usr/bin/env python3
"""Redraw gate-ablation plots from a saved ``ablation_results.json``.

This is a pure post-processing tool — it needs only ``json`` + ``matplotlib``
(no torch / model), so plots can be regenerated cheaply without re-running the
sweep. It mirrors the plots produced inline by ``02_ablate_gates.py`` with two
deliberate differences:

1. **Equally-spaced K on the logit plots.** The original logit plots use a
   symlog x-axis. Here every K step gets the *same* horizontal width
   (0\u21921, 1\u21922, 2\u21924, ... all equal), by plotting against the K index and
   labelling the ticks with the actual K values. This stops the densely-sampled
   small-K region from being visually compressed.

2. **``--main-curves patch`` mode.** Drives the main detection-rate plots from
   the patch-based curves only (``patch_detection`` + ``reverse_patch_detection``)
   and drops the K=0 (no-ablation) baseline point, so the figure shows only the
   patched results. Output filenames get a ``_patch`` suffix to avoid clobbering
   the standard images.

Usage
-----
    python 03_redraw_plots.py --results <dir-or-ablation_results.json>
    python 03_redraw_plots.py --results <...> --main-curves patch
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Curve groups & styles (kept in sync with 02_ablate_gates.py)
# ─────────────────────────────────────────────────────────────────────────────

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
# Curves that only involve patching (used by --main-curves patch).
PATCH_ONLY_CURVES = ["patch_detection", "reverse_patch_detection"]

_DET_STYLE = {
    "ablate_control_detection": ("Ablating: control (FP)", "tab:red", "o"),
    "ablate_steered_detection": ("Ablating: steered (TP)", "tab:orange", "D"),
    "patch_detection": ("Patching steered \u2192 unsteered", "tab:green", "s"),
    "reverse_patch_detection": ("Reverting gates steered \u2192 unsteered (TP)", "tab:brown", "P"),
    "ablate_arithmetic_control": ("Ablating: factual control (2+2=5)", "tab:purple", "v"),
    "patch_arithmetic_control": ("Patching gates \u2192 factual control (2+2=5)", "tab:cyan", "X"),
    "ablate_identification": ("Ablating steered (forced): identification", "tab:blue", "^"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ks_ys(metric_curve: Dict[str, float], drop_zero: bool = False
           ) -> Tuple[List[int], List[float]]:
    """Return sorted K list and matching values, optionally dropping K=0."""
    ks = sorted(int(k) for k in metric_curve)
    if drop_zero:
        ks = [k for k in ks if k != 0]
    return ks, [metric_curve[str(k)] for k in ks]


def _is_num(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and x == x  # not None / NaN


def _ordinal_positions(ks: Sequence[int]) -> List[int]:
    """Map K values to equally-spaced x positions (their index)."""
    return list(range(len(ks)))


def _apply_ordinal_xaxis(ax, ks: Sequence[int]):
    """Label the equally-spaced positions with their actual K values."""
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])


def _baseline_hline(ax, results: dict, metric: str, scale: float = 1.0,
                    color: str = "0.35", ls: str = "--"):
    """Draw the steered factual-control baseline as a horizontal reference."""
    base = results.get("arith_steered_baseline")
    if not base:
        return None
    val = base.get(metric)
    if not _is_num(val):
        return None
    label = (f"Steered 2+2=5 baseline ({val * scale:.0f}%)" if scale == 100
             else f"Steered 2+2=5 baseline ({val:.1f})")
    ax.axhline(val * scale, color=color, ls=ls, lw=1.3, label=label)
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Main detection-rate plots (linear K; supports patch-only mode)
# ─────────────────────────────────────────────────────────────────────────────

def plot_detection_rates(results: dict, out_path: Path, metric: str,
                         title_suffix: str, mode: str = "all"):
    """Main detection-rate curves vs K.

    mode="all"   : introspection curves + identification, including K=0.
    mode="patch" : patch-based curves only. K=0 is dropped for ``patch_detection``
                   (where it means 'no patch applied'), but kept for
                   ``reverse_patch_detection`` (where K=0 is the un-reverted
                   steered TP baseline that the curve is anchored to).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    if mode == "patch":
        keys = [k for k in PATCH_ONLY_CURVES if k in results["curves"]]
        # K=0 means 'no patch from a clean baseline' only for patch_detection.
        drop_zero_curves = {"patch_detection"}
    else:
        keys = [k for k in INTROSPECTION_DETECTION_CURVES if k in results["curves"]]
        if "ablate_identification" in results["curves"]:
            keys.append("ablate_identification")
        drop_zero_curves = set()

    for key in keys:
        cdata = results["curves"][key]
        mc = cdata.get("id_rate") if key == "ablate_identification" else cdata.get(metric)
        if not mc:
            continue
        ks, ys = _ks_ys(mc, drop_zero=key in drop_zero_curves)
        label, color, marker = _DET_STYLE[key]
        ax.plot(ks, [v * 100 for v in ys], marker=marker, color=color, label=label)

    cfg = results["config"]
    note = "patched only" if mode == "patch" else "K=0 is no ablation"
    ax.set_xlabel(f"Top gate features ablated/patched (K)  \u2014 {note}")
    ax.set_ylabel("Rate (%)")
    mode_tag = " \u2014 patched only" if mode == "patch" else ""
    ax.set_title(f"Gate ablation \u2014 {title_suffix}{mode_tag}\n"
                 f"({cfg['setting']}, L={cfg['steering_layer']}, s={cfg['strength']})")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[03] Saved plot -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Logit plots (equally-spaced K)
# ─────────────────────────────────────────────────────────────────────────────

def plot_logit_effects(results: dict, out_path: Path):
    """Per introspection curve: \u0394Yes, \u0394No, \u0394gap logsumexp vs K (relative to K=0)."""
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
        x = _ordinal_positions(yk)
        d_yes = [v - yv[0] for v in yv]
        d_no = [v - nv[0] for v in nv]
        d_gap = [dy - dn for dy, dn in zip(d_yes, d_no)]
        ax.axhline(0, color="0.6", lw=0.8)
        ax.plot(x, d_yes, marker="o", color="tab:green", label="\u0394 Yes logit")
        ax.plot(x, d_no, marker="s", color="tab:red", label="\u0394 No logit")
        ax.plot(x, d_gap, marker="^", color="0.3", ls="--", label="\u0394 gap (Yes\u2212No)")
        ax.set_title(_DET_STYLE[key][0], fontsize=10)
        ax.set_xlabel("K")
        _apply_ordinal_xaxis(ax, yk)
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
    print(f"[03] Saved plot -> {out_path}")


def plot_logit_effects_raw(results: dict, out_path: Path):
    """Per introspection curve: raw Yes and No logsumexp vs K."""
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
        x = _ordinal_positions(yk)
        ax.plot(x, yv, marker="o", color="tab:green", label="Yes logit")
        ax.plot(x, nv, marker="s", color="tab:red", label="No logit")
        ax.set_title(_DET_STYLE[key][0], fontsize=10)
        ax.set_xlabel("K")
        _apply_ordinal_xaxis(ax, yk)
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
    print(f"[03] Saved plot -> {out_path}")


def plot_arithmetic_control(results: dict, out_path: Path):
    """Factual-control Yes-bias (left) and logit gap / Yes-No deltas (right)."""
    cdata = results["curves"].get("ablate_arithmetic_control")
    if not cdata or not cdata.get("mean_gap"):
        return
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    gk, gr = _ks_ys(cdata["detect_gap_rate"])
    _, ar = _ks_ys(cdata["detect_argmax_rate"])
    x = _ordinal_positions(gk)
    axL.plot(x, [v * 100 for v in gr], marker="o", color="tab:purple", label="Yes (gap>0)")
    axL.plot(x, [v * 100 for v in ar], marker="^", color="tab:pink", label="Yes (argmax)")
    axL.set_ylim(-5, 105)
    axL.set_xlabel("K")
    axL.set_ylabel("Says Yes to factual control (%)")
    axL.set_title("Yes-bias on factual control")
    axL.legend(fontsize=9)
    _apply_ordinal_xaxis(axL, gk)
    axL.grid(True, alpha=0.3)

    _, gap = _ks_ys(cdata["mean_gap"])
    _, yv = _ks_ys(cdata["mean_yes_lse"])
    _, nv = _ks_ys(cdata["mean_no_lse"])
    d_yes = [v - yv[0] for v in yv]
    d_no = [v - nv[0] for v in nv]
    axR.axhline(0, color="0.6", lw=0.8)
    axR.plot(x, gap, marker="o", color="0.3", label="gap (Yes\u2212No)")
    axR.plot(x, d_yes, marker="o", color="tab:green", ls="--", label="\u0394 Yes logit")
    axR.plot(x, d_no, marker="s", color="tab:red", ls="--", label="\u0394 No logit")
    axR.set_xlabel("K")
    axR.set_ylabel("logit (nats)")
    axR.set_title("Logit gap & Yes/No deltas")
    axR.legend(fontsize=9)
    _apply_ordinal_xaxis(axR, gk)
    axR.grid(True, alpha=0.3)

    cfg = results["config"]
    fig.suptitle(f"Factual control: {cfg.get('arithmetic_question','')}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[03] Saved plot -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Factual-control plots + overlays
# ─────────────────────────────────────────────────────────────────────────────

def plot_controls(results: dict, out_path: Path, metric: str, title_suffix: str):
    """Factual-control Yes-rate vs K (ablate vs patch) with steered baseline."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in CONTROL_DETECTION_CURVES:
        cdata = results["curves"].get(key)
        if not cdata or not cdata.get(metric):
            continue
        ks, ys = _ks_ys(cdata[metric])
        label, color, marker = _DET_STYLE[key]
        ax.plot(ks, [v * 100 for v in ys], marker=marker, color=color, label=label)

    _baseline_hline(ax, results, metric, scale=100)

    cfg = results["config"]
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Top gate features ablated/patched (K)  \u2014 K=0 is no intervention")
    ax.set_ylabel("Says Yes to 2+2=5 (%)")
    ax.set_title(f"Factual control \u2014 {title_suffix}\n"
                 f"({cfg['setting']}, L={cfg['steering_layer']}, s={cfg['strength']})")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[03] Saved plot -> {out_path}")


def plot_control_logits(results: dict, out_path: Path):
    """Raw Yes/No logsumexp vs K for the factual-control curves (equally spaced)."""
    ctrl = [c for c in CONTROL_DETECTION_CURVES if c in results["curves"]
            and results["curves"][c].get("mean_yes_lse")]
    if not ctrl:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    last_ks: Optional[List[int]] = None
    for key in ctrl:
        cdata = results["curves"][key]
        yk, yv = _ks_ys(cdata["mean_yes_lse"])
        _, nv = _ks_ys(cdata["mean_no_lse"])
        x = _ordinal_positions(yk)
        label, color, marker = _DET_STYLE[key]
        ax.plot(x, yv, marker=marker, color=color, ls="-", label=f"{label}: Yes")
        ax.plot(x, nv, marker=marker, color=color, ls=":", label=f"{label}: No")
        last_ks = yk

    base = results.get("arith_steered_baseline")
    if base:
        if _is_num(base.get("mean_yes_lse")):
            ax.axhline(base["mean_yes_lse"], color="0.25", ls="--", lw=1.2,
                       label="Steered 2+2=5: Yes")
        if _is_num(base.get("mean_no_lse")):
            ax.axhline(base["mean_no_lse"], color="0.55", ls="--", lw=1.2,
                       label="Steered 2+2=5: No")

    if last_ks is not None:
        _apply_ordinal_xaxis(ax, last_ks)
    cfg = results["config"]
    ax.set_xlabel("K")
    ax.set_ylabel("logsumexp (nats)")
    ax.set_title(f"Factual control \u2014 raw Yes/No logits\n"
                 f"({cfg['setting']}, L={cfg['steering_layer']}, s={cfg['strength']})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[03] Saved plot -> {out_path}")


def _overlay_rate_axis(ax, results: dict, keys: List[str], metric: str,
                       mark_baseline: bool = True):
    for key in keys:
        cdata = results["curves"].get(key)
        mc = (cdata.get("id_rate") if key == "ablate_identification" else cdata.get(metric)) \
            if cdata else None
        if not mc:
            continue
        ks, ys = _ks_ys(mc)
        label, color, marker = _DET_STYLE[key]
        ax.plot(ks, [v * 100 for v in ys], marker=marker, color=color, label=label)
    if mark_baseline:
        _baseline_hline(ax, results, metric, scale=100)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("K")
    ax.set_ylabel("Yes rate (%)")
    ax.grid(True, alpha=0.3)


def _overlay_logit_axis(ax, results: dict, keys: List[str], mark_baseline: bool = True):
    last_ks: Optional[List[int]] = None
    for key in keys:
        cdata = results["curves"].get(key)
        if not cdata or not cdata.get("mean_yes_lse"):
            continue
        yk, yv = _ks_ys(cdata["mean_yes_lse"])
        _, nv = _ks_ys(cdata["mean_no_lse"])
        x = _ordinal_positions(yk)
        label, color, marker = _DET_STYLE[key]
        ax.plot(x, yv, marker=marker, color=color, ls="-", label=f"{label}: Yes")
        ax.plot(x, nv, marker=marker, color=color, ls=":", label=f"{label}: No")
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
        _apply_ordinal_xaxis(ax, last_ks)
    ax.set_xlabel("K")
    ax.set_ylabel("logsumexp (nats)")
    ax.grid(True, alpha=0.3)


def plot_overlay_rates(results: dict, out_path: Path):
    """Introspection vs factual-control Yes-rate overlay (patch | revert)."""
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
    print(f"[03] Saved plot -> {out_path}")


def plot_overlay_logits(results: dict, out_path: Path):
    """Introspection vs factual-control raw Yes/No logits overlay (equally spaced)."""
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
    print(f"[03] Saved plot -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Redraw gate-ablation plots from ablation_results.json.")
    p.add_argument("--results", "-r", required=True,
                   help="Path to ablation_results.json (or a directory containing it).")
    p.add_argument("--output-dir", "-od", default=None,
                   help="Where to write the plots (default: alongside the results file).")
    p.add_argument("--main-curves", choices=["all", "patch"], default="all",
                   help="'all' redraws every plot as standard; 'patch' draws the main "
                        "detection-rate plots from patch results only (K=0 dropped).")
    args = p.parse_args()

    results_path = Path(args.results)
    if results_path.is_dir():
        results_path = results_path / "ablation_results.json"
    if not results_path.exists():
        raise SystemExit(f"[03] No results file at {results_path}")

    with open(results_path) as f:
        results = json.load(f)
    out_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.main_curves == "patch":
        # Only the main detection-rate plots change in patch mode.
        plot_detection_rates(results, out_dir / "ablation_curves_patch.png",
                             metric="detect_gap_rate", title_suffix="gap-based detection",
                             mode="patch")
        plot_detection_rates(results, out_dir / "ablation_curves_argmax_patch.png",
                             metric="detect_argmax_rate", title_suffix="strict next-token detection",
                             mode="patch")
        print(f"[03] Patched-only main curves written to {out_dir}")
        return

    # Full redraw.
    plot_detection_rates(results, out_dir / "ablation_curves.png",
                         metric="detect_gap_rate", title_suffix="gap-based detection")
    plot_detection_rates(results, out_dir / "ablation_curves_argmax.png",
                         metric="detect_argmax_rate", title_suffix="strict next-token detection")
    plot_logit_effects(results, out_dir / "logit_effects.png")
    plot_logit_effects_raw(results, out_dir / "logit_effects_raw.png")
    plot_arithmetic_control(results, out_dir / "arithmetic_control.png")
    plot_controls(results, out_dir / "control_curves.png",
                  metric="detect_gap_rate", title_suffix="gap-based")
    plot_controls(results, out_dir / "control_curves_argmax.png",
                  metric="detect_argmax_rate", title_suffix="strict next-token")
    plot_control_logits(results, out_dir / "control_logits.png")
    plot_overlay_rates(results, out_dir / "overlay_curves.png")
    plot_overlay_logits(results, out_dir / "overlay_logits.png")
    print(f"[03] Redrew all plots into {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot global-gate layer distributions and DLA, with count verification.

This script scans one or more `gates.json` files produced by step 1 and creates:
    1) layer-distribution bar plot for global gates (with top-feature overlays),
  2) direct logit attribution (DLA) vs global rank plot,
  3) a JSON report that verifies gate counts and duplicate records.

Usage examples:
  python 04_plot_global_gates.py \
    --results-root analysis/gate_minimal/gemma3_27b

  python 04_plot_global_gates.py \
    --results-root analysis/gate_minimal/gemma3_27b/prefill_L37_s4.0
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_TOP_K = 200


def _safe_slug(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch in (" ", "."):
            out.append("_")
    return "".join(out).strip("_") or "gates"


def _discover_gates_files(results_root: Path) -> List[Path]:
    if results_root.is_file():
        if results_root.name != "gates.json":
            raise ValueError(f"Expected a gates.json file, got: {results_root}")
        return [results_root]

    if not results_root.is_dir():
        raise ValueError(f"Path does not exist: {results_root}")

    matches = sorted(results_root.rglob("gates.json"))
    if not matches:
        raise ValueError(f"No gates.json files found under: {results_root}")
    return matches


@dataclass
class GateSummary:
    gates_path: Path
    setting: str
    steering_layer: int
    strength: float
    scan_layers: List[int]
    per_layer_counts: Dict[int, int]
    global_count_total: int
    analyzed_count: int
    top_k_requested: int
    has_enough_for_top_k: bool
    expected_from_layers: int
    expected_from_config: int
    count_matches_layers: bool
    count_matches_config: bool
    duplicate_global_records: int


def _build_global_if_missing(data: dict) -> List[dict]:
    global_gates = data.get("global_gates")
    if global_gates:
        return global_gates

    rebuilt: List[dict] = []
    for layer_s, entry in data.get("gates", {}).items():
        layer = int(layer_s)
        ids = entry.get("gate_ids", [])
        dlas = entry.get("gate_dla", [])
        for fid, dla in zip(ids, dlas):
            rebuilt.append({"layer": layer, "feature_id": int(fid), "dla": float(dla)})
    rebuilt.sort(key=lambda r: r["dla"])
    return rebuilt


def _summarize_gate_file(gates_path: Path, top_k: int) -> Tuple[GateSummary, List[dict]]:
    with open(gates_path) as f:
        data = json.load(f)

    cfg = data.get("config", {})
    gates_by_layer = data.get("gates", {})

    scan_layers = [int(x) for x in cfg.get("scan_layers", [])]
    if not scan_layers:
        scan_layers = sorted(int(k) for k in gates_by_layer.keys())

    per_layer_counts = {
        int(layer): len(gates_by_layer[str(layer)].get("gate_ids", []))
        for layer in scan_layers
        if str(layer) in gates_by_layer
    }

    global_gates = _build_global_if_missing(data)
    global_pairs = [(int(r["layer"]), int(r["feature_id"])) for r in global_gates]
    duplicate_global_records = len(global_pairs) - len(set(global_pairs))

    expected_from_layers = int(sum(per_layer_counts.values()))
    n_top = int(cfg.get("n_top", 0) or 0)
    expected_from_config = int(n_top * len(scan_layers)) if n_top > 0 and scan_layers else 0

    analyzed_count = min(len(global_gates), top_k) if top_k > 0 else len(global_gates)

    summary = GateSummary(
        gates_path=gates_path,
        setting=str(cfg.get("setting", "unknown")),
        steering_layer=int(cfg.get("steering_layer", -1)),
        strength=float(cfg.get("strength", float("nan"))),
        scan_layers=scan_layers,
        per_layer_counts=per_layer_counts,
        global_count_total=len(global_gates),
        analyzed_count=analyzed_count,
        top_k_requested=top_k,
        has_enough_for_top_k=(len(global_gates) >= top_k if top_k > 0 else True),
        expected_from_layers=expected_from_layers,
        expected_from_config=expected_from_config,
        count_matches_layers=(len(global_gates) == expected_from_layers),
        count_matches_config=(expected_from_config == 0 or len(global_gates) == expected_from_config),
        duplicate_global_records=duplicate_global_records,
    )
    # Ranking order is ascending DLA (most negative first), then take top-K.
    ranked = sorted(global_gates, key=lambda r: float(r["dla"]))
    analyzed = ranked[:top_k] if top_k > 0 else ranked
    return summary, analyzed


def _plot_layer_distribution(global_gates: List[dict], out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layer_counts = Counter(int(r["layer"]) for r in global_gates)
    layers = sorted(layer_counts)
    counts = [layer_counts[L] for L in layers]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(layers, counts, color="#9aa0a6", alpha=0.6, label="All analyzed gates")

    # Overlay top-10 global features as distinct colored bars on the same axes.
    top_features = global_gates[: min(10, len(global_gates))]
    if top_features:
        cmap = plt.get_cmap("tab10")
        if len(top_features) == 1:
            offsets = [0.0]
        else:
            offsets = [
                -0.35 + (0.70 * i) / (len(top_features) - 1)
                for i in range(len(top_features))
            ]
        for rank, (offset, rec) in enumerate(zip(offsets, top_features), start=1):
            layer = int(rec["layer"])
            feature_id = int(rec["feature_id"])
            dla = float(rec["dla"])
            top_height = layer_counts.get(layer, 0)
            ax.bar(
                layer + offset,
                top_height,
                width=0.07,
                color=cmap((rank - 1) % 10),
                alpha=0.95,
                label=f"#{rank} L{layer} F{feature_id} ({dla:.3g})",
            )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Number of global gates")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_global_dla(global_gates: List[dict], out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dlas = [float(r["dla"]) for r in global_gates]
    if not dlas:
        raise ValueError("Cannot plot DLA: no global gates found.")

    # Keep the same ordering used by ranking (most negative first).
    ranks = list(range(1, len(dlas) + 1))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(ranks, dlas, color="#d62728", linewidth=1.2)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Global gate rank (ascending DLA)")
    ax.set_ylabel("Direct logit attribution (DLA)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_report(summary: GateSummary, out_path: Path) -> None:
    payload = {
        "gates_json": str(summary.gates_path),
        "setting": summary.setting,
        "steering_layer": summary.steering_layer,
        "strength": summary.strength,
        "scan_layers": summary.scan_layers,
        "per_layer_counts": {str(k): v for k, v in sorted(summary.per_layer_counts.items())},
        "global_count_total": summary.global_count_total,
        "top_k_requested": summary.top_k_requested,
        "analyzed_count": summary.analyzed_count,
        "has_enough_for_top_k": summary.has_enough_for_top_k,
        "expected_from_layers": summary.expected_from_layers,
        "expected_from_config": summary.expected_from_config,
        "count_matches_layers": summary.count_matches_layers,
        "count_matches_config": summary.count_matches_config,
        "duplicate_global_records": summary.duplicate_global_records,
        "status": "ok"
        if (
            summary.count_matches_layers
            and summary.count_matches_config
            and summary.duplicate_global_records == 0
            and summary.has_enough_for_top_k
        )
        else "warning",
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot global-gate distributions and DLA, then verify global-gate counts."
    )
    parser.add_argument(
        "--results-root",
        required=True,
        help="Path to model results folder (or a single gates.json).",
    )
    parser.add_argument(
        "--out-subdir",
        default="global_gate_plots",
        help="Subdirectory name created next to each gates.json for outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Analyze only the top-K global gates by ascending DLA (default: 200).",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    gates_files = _discover_gates_files(results_root)

    print(f"[04] Found {len(gates_files)} gates.json file(s).")
    warnings = 0

    for gates_path in gates_files:
        summary, global_gates = _summarize_gate_file(gates_path, top_k=args.top_k)

        out_dir = gates_path.parent / args.out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        slug = _safe_slug(gates_path.parent.name)
        title_prefix = f"{summary.setting} | L{summary.steering_layer} | s={summary.strength}"

        _plot_layer_distribution(
            global_gates,
            out_dir / f"{slug}_layer_distribution.png",
            title=f"Global gate layer distribution ({title_prefix})",
        )
        _plot_global_dla(
            global_gates,
            out_dir / f"{slug}_global_dla.png",
            title=f"Global gate DLA by rank ({title_prefix})",
        )
        _write_report(summary, out_dir / f"{slug}_count_check.json")

        status = "OK"
        if (
            not summary.count_matches_layers
            or not summary.count_matches_config
            or summary.duplicate_global_records > 0
        ):
            status = "WARNING"
            warnings += 1

        print(
            f"[04] {status} | {gates_path} | "
            f"global_total={summary.global_count_total} "
            f"analyzed={summary.analyzed_count}/{summary.top_k_requested} "
            f"expected_layers={summary.expected_from_layers} "
            f"expected_config={summary.expected_from_config} "
            f"duplicates={summary.duplicate_global_records}"
        )

    if warnings:
        print(f"[04] Completed with {warnings} warning case(s). Check *_count_check.json files.")
    else:
        print("[04] Completed with all gate-count checks passing.")


if __name__ == "__main__":
    main()

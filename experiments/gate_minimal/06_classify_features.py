#!/usr/bin/env python3
"""Step 6 — Classify gate features by activation behavior (no model run needed).

Every selected gate is a "No-writer" (its decoder column projects negatively
onto the Yes-No unembedding direction; verified 64/64 in both settings). What
distinguishes them is *how their activation responds to the concept*:

  * suppression : active on the bare prompt and **drops** when the concept is
                  present (the classic "gate": it holds the No answer and is
                  released). Ablating it raises detection by removing a No push.
  * detector    : ~off on the bare prompt and **switches on** when the concept
                  is present (off -> on). A concept-responsive No-writer.
  * amplified   : already active and gets **larger** when steered.
  * stable_on   : active with little change.
  * inactive    : ~off in both conditions.

``delta = steered_act - unsteered_act``. ``decode_proj`` (the unit decoder
projection onto Yes-No) is recovered from the saved DLA without loading the
transcoder, since step-1 stored ``dla = decode_proj * steered_act`` for every
gate (``decode_proj = dla / steered_act``). Pass ``--exact-decode`` to recompute
it from the transcoder instead (loads GemmaScope weights).

A feature **promotes detection** when the concept appears iff
``decode_proj * delta > 0`` (its contribution to the Yes-No logit moves toward
Yes). For a No-writer this is true both when it is suppressed (delta<0) and is
irrelevant when it is a No-detector that switches on (delta>0, which *suppresses*
detection — ablating it is what helps).

Outputs (per input gates.json, written beside it under ``feature_classes/``)
  * features_classified.json   full per-feature records
  * features_classified.csv    same, flat, for spreadsheets/pandas
  * feature_class_summary.json  category counts (overall and by layer)
  * feature_classes.png         category histogram + delta distribution

Cross-scenario classification (``--cross-dir``)
-----------------------------------------------
Pass another setting's directory (one that holds its own ``mean_acts.pt``) to
also classify the *same* selected gates using that setting's activations. The
``decode_proj`` (a setting-independent decoder property) is kept from the home
DLA, only the behavior (suppression / detector / ...) is recomputed on the cross
activations. This shows whether a gate's behavior class is a property of the
*feature* or of the *measurement setting* (the discrete-taxonomy companion to
step-5's aggregate 2x2 deltas). Extra outputs under
``feature_classes/cross_<name>/``:
  * features_classified_cross.json   per-feature records on cross activations
  * comparison.json                  home vs cross counts + class-transition table
  * cross_compare.png                grouped home/cross counts + transition heatmap
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def classify_one(
    layer: int,
    feature_id: int,
    dla: float,
    steered_act: float,
    unsteered_act: float,
    decode_proj: Optional[float],
    act_eps: float,
    rel_eps: float,
) -> dict:
    delta = steered_act - unsteered_act
    base_active = unsteered_act > act_eps
    steer_active = steered_act > act_eps

    if not base_active and steer_active:
        behavior = "detector"            # off -> on
    elif base_active and steered_act < unsteered_act * (1.0 - rel_eps):
        behavior = "suppression"         # on -> lower (incl. -> off)
    elif base_active and steered_act > unsteered_act * (1.0 + rel_eps):
        behavior = "amplified"           # on -> higher
    elif base_active:
        behavior = "stable_on"
    else:
        behavior = "inactive"

    if decode_proj is None:
        writes = "unknown"
        promotes_detection = None
    else:
        writes = "No" if decode_proj < 0 else "Yes"
        promotes_detection = bool(decode_proj * delta > 0)

    return {
        "layer": layer,
        "feature_id": feature_id,
        "dla_steered": dla,
        "steered_act": steered_act,
        "unsteered_act": unsteered_act,
        "delta": delta,
        "rel_delta": delta / (unsteered_act + act_eps),
        "decode_proj": decode_proj,
        "writes": writes,
        "behavior": behavior,
        "promotes_detection": promotes_detection,
        # diff-metric score (suppression-oriented): unit decoder projection times
        # the activation drop. Negative => No-writer that drops (suppression gate).
        "dla_diff": (decode_proj * (unsteered_act - steered_act)
                     if decode_proj is not None else None),
    }


def load_exact_decode_projs(
    gates: dict,
    delta_u: torch.Tensor,
    device: str,
) -> Dict[int, Dict[int, float]]:
    """Recompute decode_proj = unit(w_dec[f]) . delta_u from the transcoders
    (one layer at a time). Only for layers/features present in the gate set."""
    import gate_lib as gl

    want: Dict[int, List[int]] = defaultdict(list)
    for L, entry in gates["gates"].items():
        want[int(L)].extend(int(f) for f in entry["gate_ids"])

    out: Dict[int, Dict[int, float]] = {}
    du = delta_u.float().cpu()
    for L in sorted(want):
        tc = gl.load_transcoder(L, device=device)
        w_dec = tc.w_dec.detach().float().cpu()                  # [F, d]
        feats = sorted(set(want[L]))
        sub = w_dec[feats]                                       # [k, d]
        unit = sub / (sub.norm(dim=-1, keepdim=True) + 1e-10)
        proj = (unit @ du).tolist()
        out[L] = {f: float(p) for f, p in zip(feats, proj)}
        del tc, w_dec, sub, unit
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return out


def classify_gates(
    gates: dict,
    mean_acts: dict,
    act_eps: float,
    rel_eps: float,
    exact_decode: Optional[Dict[int, Dict[int, float]]] = None,
    decode_override: Optional[Dict[tuple, Optional[float]]] = None,
    top_k: Optional[int] = None,
) -> List[dict]:
    S = {int(L): t for L, t in mean_acts["steered"].items()}
    U = {int(L): t for L, t in mean_acts["unsteered"].items()}

    # global_gates is ordered strongest-first (ascending DLA); take the top-K.
    pool = gates["global_gates"]
    if top_k is not None:
        pool = pool[:top_k]

    records: List[dict] = []
    for rec in pool:
        L = int(rec["layer"]); f = int(rec["feature_id"]); dla = float(rec["dla"])
        s = float(S[L][f]); u = float(U[L][f])
        if decode_override is not None:
            dp = decode_override.get((L, f))
        elif exact_decode is not None:
            dp = exact_decode.get(L, {}).get(f)
        else:
            dp = (dla / s) if s > act_eps else None
        records.append(classify_one(L, f, dla, s, u, dp, act_eps, rel_eps))
    return records


def compare_classifications(home: List[dict], cross: List[dict]) -> dict:
    """Compare per-feature behavior between the home and cross settings (same
    selected gates). Returns side-by-side counts, a class-transition table
    (home -> cross), and the fraction of gates that keep their class."""
    order = ["suppression", "detector", "amplified", "stable_on", "inactive"]
    cross_by_key = {(r["layer"], r["feature_id"]): r for r in cross}

    transitions: Dict[str, Counter] = defaultdict(Counter)
    n_same = 0
    n_paired = 0
    for h in home:
        c = cross_by_key.get((h["layer"], h["feature_id"]))
        if c is None:
            continue
        n_paired += 1
        transitions[h["behavior"]][c["behavior"]] += 1
        if h["behavior"] == c["behavior"]:
            n_same += 1

    return {
        "home_counts": dict(Counter(r["behavior"] for r in home)),
        "cross_counts": dict(Counter(r["behavior"] for r in cross)),
        "n_paired": n_paired,
        "n_same_class": n_same,
        "frac_same_class": (n_same / n_paired) if n_paired else None,
        "order": order,
        "transitions": {h: dict(c) for h, c in transitions.items()},
    }


def plot_cross_compare(comparison: dict, out_path: Path, note: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    order = comparison["order"]
    hc, cc = comparison["home_counts"], comparison["cross_counts"]
    cats = [c for c in order if c in hc or c in cc]
    extra = [c for c in set(hc) | set(cc) if c not in order]
    cats += extra

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))

    x = np.arange(len(cats))
    w = 0.4
    axL.bar(x - w / 2, [hc.get(c, 0) for c in cats], w, label="home", color="tab:blue")
    axL.bar(x + w / 2, [cc.get(c, 0) for c in cats], w, label="cross", color="tab:orange")
    axL.set_xticks(x); axL.set_xticklabels(cats, rotation=20)
    axL.set_ylabel("# gate features")
    axL.set_title("Behavior class counts: home vs cross")
    axL.legend()
    axL.grid(True, axis="y", alpha=0.3)

    M = np.zeros((len(cats), len(cats)))
    idx = {c: i for i, c in enumerate(cats)}
    for h, row in comparison["transitions"].items():
        for c, n in row.items():
            if h in idx and c in idx:
                M[idx[h], idx[c]] = n
    im = axR.imshow(M, cmap="viridis", aspect="auto")
    axR.set_xticks(x); axR.set_xticklabels(cats, rotation=20)
    axR.set_yticks(x); axR.set_yticklabels(cats)
    axR.set_xlabel("cross class"); axR.set_ylabel("home class")
    axR.set_title("Class transitions (home -> cross)")
    for i in range(len(cats)):
        for j in range(len(cats)):
            if M[i, j]:
                axR.text(j, i, int(M[i, j]), ha="center", va="center",
                         color="w" if M[i, j] < M.max() * 0.6 else "k", fontsize=8)
    fig.colorbar(im, ax=axR, fraction=0.046)

    frac = comparison["frac_same_class"]
    frac_s = f"{frac:.0%}" if frac is not None else "n/a"
    fig.suptitle(f"Cross-scenario behavior comparison  (same class: {frac_s}"
                 f" of {comparison['n_paired']})\n{note}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[06] Saved cross-compare plot -> {out_path}")


def summarize(records: List[dict]) -> dict:
    overall = Counter(r["behavior"] for r in records)
    by_layer: Dict[int, Counter] = defaultdict(Counter)
    for r in records:
        by_layer[r["layer"]][r["behavior"]] += 1
    writes = Counter(r["writes"] for r in records)
    promotes = Counter(
        "promotes" if r["promotes_detection"] else
        ("suppresses" if r["promotes_detection"] is False else "unknown")
        for r in records
    )
    return {
        "n": len(records),
        "behavior_counts": dict(overall),
        "writes_counts": dict(writes),
        "detection_effect_counts": dict(promotes),
        "by_layer": {str(L): dict(c) for L, c in sorted(by_layer.items())},
    }


def plot_classes(records: List[dict], summary: dict, out_path: Path, note: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["suppression", "detector", "amplified", "stable_on", "inactive"]
    colors = {
        "suppression": "tab:red", "detector": "tab:green", "amplified": "tab:orange",
        "stable_on": "tab:blue", "inactive": "0.6",
    }
    counts = summary["behavior_counts"]
    cats = [c for c in order if c in counts] + [c for c in counts if c not in order]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    axL.bar(cats, [counts[c] for c in cats], color=[colors.get(c, "0.5") for c in cats])
    axL.set_ylabel("# gate features")
    axL.set_title("Behavior class counts")
    axL.tick_params(axis="x", rotation=20)
    axL.grid(True, axis="y", alpha=0.3)

    for c in cats:
        deltas = [r["delta"] for r in records if r["behavior"] == c]
        if deltas:
            axR.scatter([c] * len(deltas), deltas, s=10, alpha=0.4,
                        color=colors.get(c, "0.5"))
    axR.axhline(0, color="0.3", lw=0.8)
    axR.set_ylabel("delta (steered - unsteered)")
    axR.set_title("Activation change per class")
    axR.tick_params(axis="x", rotation=20)
    axR.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Gate feature classification\n{note}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[06] Saved plot -> {out_path}")


def process(gates_json: Path, act_eps: float, rel_eps: float,
            exact_decode: bool, device: str,
            cross_dirs: Optional[List[Path]] = None,
            top_k: Optional[int] = None) -> dict:
    with open(gates_json) as f:
        gates = json.load(f)
    mean_acts = torch.load(gates_json.parent / "mean_acts.pt",
                           map_location="cpu", weights_only=False)

    exact = None
    if exact_decode:
        import gate_lib as gl
        from model_utils import load_model
        print(f"[06] --exact-decode: loading {gates['config']['model']} for delta_u ...")
        model_w = load_model(model_name=gates["config"]["model"], device=device)
        delta_u = gl.compute_delta_u(model_w)
        exact = load_exact_decode_projs(gates, delta_u, device)

    records = classify_gates(gates, mean_acts, act_eps, rel_eps, exact, top_k=top_k)
    summary = summarize(records)

    tag = f"_top{top_k}" if top_k is not None else ""
    out_dir = gates_json.parent / f"feature_classes{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "features_classified.json", "w") as f:
        json.dump({"config": gates["config"], "records": records}, f, indent=2)
    with open(out_dir / "features_classified.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    with open(out_dir / "feature_class_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    cfg = gates["config"]
    note = (f"{cfg['setting']} | L{cfg['steering_layer']} s{cfg['strength']} | "
            f"{cfg.get('prefill_variant','')}/{cfg.get('detect_variant','')} | "
            f"decode={'exact' if exact_decode else 'from-DLA'}"
            + (f" | top-{top_k}" if top_k is not None else ""))
    plot_classes(records, summary, out_dir / "feature_classes.png", note)

    print(f"[06] {gates_json.parent.name}"
          + (f" (top-{top_k})" if top_k is not None else "") + ": "
          + ", ".join(f"{k}={v}" for k, v in summary["behavior_counts"].items()))
    print(f"[06] Saved -> {out_dir}")

    # ---- cross-scenario classification of the same gates -----------------
    if cross_dirs:
        # decode_proj is a setting-independent decoder property: keep the home
        # value for every selected gate and only recompute behavior on the
        # cross setting's activations.
        decode_override = {(r["layer"], r["feature_id"]): r["decode_proj"]
                           for r in records}
        for cdir in cross_dirs:
            cdir = Path(cdir)
            cross_acts_path = cdir / "mean_acts.pt"
            if not cross_acts_path.exists():
                print(f"[06] WARNING: no mean_acts.pt in {cdir}, skipping cross.")
                continue
            if cdir.resolve() == gates_json.parent.resolve():
                print(f"[06] WARNING: cross dir == home dir ({cdir.name}), skipping.")
                continue
            cross_acts = torch.load(cross_acts_path, map_location="cpu",
                                    weights_only=False)
            cross_records = classify_gates(
                gates, cross_acts, act_eps, rel_eps,
                decode_override=decode_override, top_k=top_k)
            comparison = compare_classifications(records, cross_records)

            cname = cdir.name
            cross_out = out_dir / f"cross_{cname}"
            cross_out.mkdir(parents=True, exist_ok=True)
            with open(cross_out / "features_classified_cross.json", "w") as f:
                json.dump({"home_config": gates["config"],
                           "cross_dir": str(cdir),
                           "records": cross_records}, f, indent=2)
            with open(cross_out / "comparison.json", "w") as f:
                json.dump(comparison, f, indent=2)
            cnote = f"home: {gates_json.parent.name}  ->  cross: {cname}"
            plot_cross_compare(comparison, cross_out / "cross_compare.png", cnote)

            frac = comparison["frac_same_class"]
            frac_s = f"{frac:.0%}" if frac is not None else "n/a"
            print(f"[06] cross {cname}: same-class {frac_s} "
                  f"({comparison['n_same_class']}/{comparison['n_paired']}); "
                  + ", ".join(f"{k}={v}" for k, v in
                              comparison["cross_counts"].items()))
            print(f"[06] Saved cross -> {cross_out}")

    return summary


def main():
    p = argparse.ArgumentParser(description="Classify gate features by activation behavior.")
    p.add_argument("--gates-json", nargs="+", required=True,
                   help="One or more step-1 gates.json paths (mean_acts.pt must sit beside each).")
    p.add_argument("--act-eps", type=float, default=1.0,
                   help="Activation magnitude below which a feature counts as 'off'.")
    p.add_argument("--rel-eps", type=float, default=0.1,
                   help="Relative change threshold separating suppression/amplified from stable.")
    p.add_argument("--exact-decode", action="store_true",
                   help="Recompute decode_proj from the transcoder (loads model + SAEs).")
    p.add_argument("--cross-dir", nargs="+", default=None,
                   help="One or more other setting directories (each holding its own "
                        "mean_acts.pt) to re-classify the same gates on and compare.")
    p.add_argument("--top-k", type=int, default=None,
                   help="Only classify the top-K globally ranked gates (strongest "
                        "DLA first). Default: all global gates.")
    p.add_argument("--device", "-d", default="cuda")
    args = p.parse_args()

    cross_dirs = [Path(c) for c in args.cross_dir] if args.cross_dir else None
    for gj in args.gates_json:
        process(Path(gj), args.act_eps, args.rel_eps, args.exact_decode,
                args.device, cross_dirs, args.top_k)


if __name__ == "__main__":
    main()

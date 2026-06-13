#!/usr/bin/env python3
"""Step 1 — Collect top-200 gate features by direct logit attribution (DLA).

For each post-injection transcoder layer and each setting (injection vs
prefill), we:
  1. run the Yes/No introspection prompt for every concept x trial, with the
     concept either injected as a steering vector (injection) or appended to
     the prompt (prefill),
  2. capture the answer-position transcoder feature activations at each layer,
  3. average them to ``mean_act(f)`` and compute
        DLA(f) = (unit(w_dec[f]) . delta_u_Yes-No) * mean_act(f),
  4. select the 200 most-negative-DLA features as gate candidates.

Outputs (per setting) under ``--output-dir/{model}/{setting}_L{layer}_s{strength}``:
  * ``mean_acts.pt``  — {layer: tensor[n_features]} (reused by steps 2/3 and the
    future steered-vs-unsteered diff metric),
  * ``gates.json``    — per-layer top-200 feature ids + DLA values + config +
    baseline detection rate.

Notes
-----
* mean_act(f) currently uses the steered/prefill activation directly. A later
  variant will replace it with (steered - unsteered) activation differences;
  the unsteered baseline acts are also saved to make that drop-in.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import gate_lib as gl
import prompts as P
from model_utils import load_model


def collect_setting(
    model_w,
    setting: str,
    prefill_variant: str,
    detect_variant: str,
    concepts: List[str],
    scan_layers: List[int],
    steering_layer: int,
    strength: float,
    n_trials: int,
    n_extra_words: int,
    concept_vectors: Dict[str, torch.Tensor],
    delta_u: torch.Tensor,
    yes_ids: List[int],
    no_ids: List[int],
    n_top: int,
    output_dir: Path,
) -> Dict:
    """Run collection for one setting and save mean_acts + gate json."""
    assert setting in ("injection", "prefill")
    assert prefill_variant in ("append_user", "replace_assistant")
    assert detect_variant in ("strict", "vague")
    device = model_w.device

    # Phase A: one forward per sample, capturing answer-position pre-MLP-LN at
    # every scan layer (small: one d_model vector per layer per sample).
    per_layer_acts: Dict[int, List[torch.Tensor]] = {L: [] for L in scan_layers}
    unsteered_acts: Dict[int, List[torch.Tensor]] = {L: [] for L in scan_layers}
    detection_gaps: List[float] = []

    desc = f"[{setting}] collect"
    for concept in tqdm(concepts, desc=desc):
        for trial in range(1, n_trials + 1):
            prefill_word = concept if setting == "prefill" else None
            input_ids = P.format_detection_prompt(
                model_w.tokenizer,
                trial,
                prefill_word,
                n_extra_words,
                prefill_variant=prefill_variant,
                detect_variant=detect_variant,
            )

            # Steered/active forward (with injection hook for injection setting).
            handle = None
            if setting == "injection":
                vec = concept_vectors[concept]
                hook = gl.make_injection_hook(vec, strength)
                handle = model_w.get_layer_module(steering_layer).register_forward_hook(hook)
            try:
                with gl.TranscoderCapture(model_w, scan_layers) as cap:
                    with torch.no_grad():
                        out = model_w.model(
                            input_ids=input_ids.to(device), use_cache=False
                        )
                    logits = out.logits[:, -1, :].float().cpu()
                    for L in scan_layers:
                        per_layer_acts[L].append(cap.captured[L][0, -1, :].detach().float().cpu())
            finally:
                if handle is not None:
                    handle.remove()
                    torch.cuda.empty_cache()

            gap = P.logit_gap(logits, yes_ids, no_ids).item()
            detection_gaps.append(gap)

            # Unsteered baseline forward (no injection, no prefill word) — saved
            # for the future steered-vs-unsteered diff metric.
            base_ids = P.format_detection_prompt(
                model_w.tokenizer,
                trial,
                None,
                n_extra_words,
                prefill_variant=prefill_variant,
                detect_variant=detect_variant,
            )
            with gl.TranscoderCapture(model_w, scan_layers) as cap0:
                with torch.no_grad():
                    model_w.model(input_ids=base_ids.to(device), use_cache=False)
                for L in scan_layers:
                    unsteered_acts[L].append(cap0.captured[L][0, -1, :].detach().float().cpu())

    detection_rate = float((torch.tensor(detection_gaps) > 0).float().mean())

    # Phase B: layer-major — load one transcoder at a time, encode, rank.
    mean_acts: Dict[int, torch.Tensor] = {}
    mean_acts_unsteered: Dict[int, torch.Tensor] = {}
    per_layer_gates: Dict[int, Dict] = {}

    for L in tqdm(scan_layers, desc=f"[{setting}] encode+DLA"):
        tc = gl.load_transcoder(L, device=device)
        steered_stack = torch.stack(per_layer_acts[L]).to(device, tc.w_enc.dtype)
        unsteered_stack = torch.stack(unsteered_acts[L]).to(device, tc.w_enc.dtype)
        with torch.no_grad():
            steered_feats = tc.encode(steered_stack).float().mean(dim=0).cpu()
            unsteered_feats = tc.encode(unsteered_stack).float().mean(dim=0).cpu()
        mean_acts[L] = steered_feats
        mean_acts_unsteered[L] = unsteered_feats

        dla = gl.compute_dla(tc, delta_u, steered_feats)
        gate_ids = gl.top_gate_features(dla, n_top)
        per_layer_gates[L] = {
            "gate_ids": gate_ids,
            "gate_dla": [float(dla[i]) for i in gate_ids],
        }
        del tc, steered_stack, unsteered_stack
        torch.cuda.empty_cache()

    # Global gate ranking across all scanned layers (Fig 11c ranks features
    # globally). Pool each layer's most-negative candidates, then sort ascending
    # by DLA so index 0 is the single strongest gate.
    global_pool = []
    for L in scan_layers:
        for fid, d in zip(per_layer_gates[L]["gate_ids"], per_layer_gates[L]["gate_dla"]):
            global_pool.append({"layer": L, "feature_id": int(fid), "dla": float(d)})
    global_pool.sort(key=lambda r: r["dla"])  # ascending => most negative first

    # Save.
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"steered": mean_acts, "unsteered": mean_acts_unsteered},
        output_dir / "mean_acts.pt",
    )
    results = {
        "config": {
            "setting": setting,
            "model": model_w.model_name,
            "steering_layer": steering_layer,
            "strength": strength,
            "n_trials": n_trials,
            "n_extra_words": n_extra_words,
            "prefill_variant": prefill_variant,
            "detect_variant": detect_variant,
            "scan_layers": scan_layers,
            "concepts": list(concepts),
            "n_top": n_top,
            "yes_ids": yes_ids,
            "no_ids": no_ids,
        },
        "baseline_detection_rate": detection_rate,
        "gates": {str(L): per_layer_gates[L] for L in scan_layers},
        "global_gates": global_pool,
    }
    with open(output_dir / "gates.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{setting}] detection_rate={detection_rate:.3f}  saved -> {output_dir}")
    return results


def main():
    p = argparse.ArgumentParser(description="Collect top-200 gate features by DLA.")
    p.add_argument("--model", "-m", default="gemma3_27b")
    p.add_argument("--settings", nargs="+", default=["injection", "prefill"],
                   choices=["injection", "prefill"])
    p.add_argument("--steering-layer", type=int, default=gl.STEERING_LAYER)
    p.add_argument("--start-layer", type=int, default=38,
                   help="First transcoder layer to scan (inclusive). For prefill "
                        "you may start at the injection layer since the signal is "
                        "present from the start.")
    p.add_argument("--end-layer", type=int, default=None,
                   help="Last transcoder layer to scan (inclusive). Default: n_layers-1.")
    p.add_argument("--strength", type=float, default=4.0)
    p.add_argument("--n-trials", "-nt", type=int, default=3)
    p.add_argument("--n-extra-words", type=int, default=1, choices=[1, 2, 3],
                   help="Prefill append length K (06b APPEND_TEMPLATES).")
    p.add_argument(
        "--prefill-variant",
        choices=["append_user", "replace_assistant"],
        default="append_user",
        help="How prefill writes concept text into the prompt.",
    )
    p.add_argument(
        "--detect-variant",
        choices=["strict", "vague"],
        default="strict",
        help="Detection-question phrasing. 'strict' appends \"Answer only by "
             "'Yes' or 'No'.\" (clean argmax metric); 'vague' uses the original-"
             "paper phrasing (\"...If so, what is the injected thought about?\"), "
             "which shifts the gate ranking slightly. Saved to a separate folder.",
    )
    p.add_argument("--concepts", nargs="+", default=None,
                   help="Concepts (default: 50 BASELINE_CONCEPTS).")
    p.add_argument("--n-top", type=int, default=200)
    p.add_argument("--output-dir", "-od", default="analysis/gate_minimal")
    p.add_argument("--device", "-d", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    concepts = args.concepts if args.concepts is not None else gl.BASELINE_CONCEPTS

    print(f"[01] Loading {args.model} ...")
    model_w = load_model(model_name=args.model, device=args.device, dtype=args.dtype)

    end_layer = args.end_layer if args.end_layer is not None else model_w.n_layers - 1
    scan_layers = list(range(args.start_layer, end_layer + 1))
    print(f"[01] Scan layers: {scan_layers}")

    yes_ids, no_ids = P.get_yes_no_token_ids(model_w.tokenizer)
    delta_u = gl.compute_delta_u(model_w)

    concept_vectors = {}
    if "injection" in args.settings:
        vec_cache = Path(args.output_dir) / args.model / f"concept_vectors_L{args.steering_layer}.pt"
        concept_vectors = gl.load_or_build_concept_vectors(
            model_w, concepts, args.steering_layer, vec_cache
        )

    for setting in args.settings:
        setting_tag = setting
        if setting == "prefill":
            setting_tag = f"prefill-{args.prefill_variant}"
        if args.detect_variant != "strict":
            setting_tag = f"{setting_tag}_detect-{args.detect_variant}"
        out_dir = (
            Path(args.output_dir) / args.model
            / f"{setting_tag}_L{args.steering_layer}_s{args.strength}"
        )
        collect_setting(
            model_w=model_w,
            setting=setting,
            prefill_variant=args.prefill_variant,
            detect_variant=args.detect_variant,
            concepts=concepts,
            scan_layers=scan_layers,
            steering_layer=args.steering_layer,
            strength=args.strength,
            n_trials=args.n_trials,
            n_extra_words=args.n_extra_words,
            concept_vectors=concept_vectors,
            delta_u=delta_u,
            yes_ids=yes_ids,
            no_ids=no_ids,
            n_top=args.n_top,
            output_dir=out_dir,
        )


if __name__ == "__main__":
    main()

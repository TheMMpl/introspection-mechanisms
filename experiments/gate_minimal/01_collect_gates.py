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


def _format_detection_prompt_text(
    tokenizer,
    trial: int,
    prefill_word: str | None,
    n_extra_words: int,
    prefill_variant: str,
    detect_variant: str,
) -> str:
    msgs = P.build_messages(
        trial_n=trial,
        prefill_word=prefill_word,
        n_extra_words=n_extra_words,
        mode="detect",
        prefill_variant=prefill_variant,
        detect_variant=detect_variant,
    )
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def _generate_text(
    model_w,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = model_w.tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(model_w._get_input_device())
    input_length = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": model_w.tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model_w.model.generate(**inputs, **gen_kwargs)

    new_tokens = output_ids[0][input_length:]
    if model_w.model_name in ["kimi_k2", "deepseek_v3"]:
        output_text = model_w.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
    else:
        output_text = model_w.tokenizer.decode(new_tokens, skip_special_tokens=True)

    gemma_models = [
        "gemma_2b", "gemma_7b", "gemma2_2b", "gemma2_9b",
        "gemma2_27b", "gemma3_27b",
    ]
    if model_w.model_name in gemma_models and output_text.startswith("model\n"):
        output_text = output_text[len("model\n"):]

    return output_text.strip()


def preview_setting_responses(
    model_w,
    setting: str,
    prefill_variant: str,
    detect_variant: str,
    concepts: List[str],
    steering_layer: int,
    strength: float,
    n_trials: int,
    n_extra_words: int,
    concept_vectors: Dict[str, torch.Tensor],
    yes_ids: List[int],
    no_ids: List[int],
    response_limit: int,
    response_max_new_tokens: int,
    response_temperature: float,
    output_dir: Path,
) -> List[Dict]:
    """Generate and save a small response preview for qualitative inspection."""
    assert setting in ("injection", "prefill")
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    remaining = response_limit
    desc = f"[{setting}] response preview"

    for concept in tqdm(concepts, desc=desc):
        for trial in range(1, n_trials + 1):
            if response_limit > 0 and remaining <= 0:
                break

            prefill_word = concept if setting == "prefill" else None
            prompt_text = _format_detection_prompt_text(
                model_w.tokenizer,
                trial,
                prefill_word,
                n_extra_words,
                prefill_variant,
                detect_variant,
            )
            input_ids = model_w.tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"]

            handle = None
            if setting == "injection":
                vec = concept_vectors[concept]
                hook = gl.make_injection_hook(vec, strength)
                handle = model_w.get_layer_module(steering_layer).register_forward_hook(hook)
            try:
                with torch.no_grad():
                    out = model_w.model(
                        input_ids=input_ids.to(model_w.device), use_cache=False
                    )
                logits = out.logits[:, -1, :].float().cpu()
                response = _generate_text(
                    model_w,
                    prompt_text,
                    max_new_tokens=response_max_new_tokens,
                    temperature=response_temperature,
                )
            finally:
                if handle is not None:
                    handle.remove()
                    torch.cuda.empty_cache()

            gap = float(P.logit_gap(logits, yes_ids, no_ids).item())
            top_token_id = int(logits.argmax(dim=-1).item())
            detect_argmax = bool(P.top_token_is_yes(logits, yes_ids))
            record = {
                "setting": setting,
                "concept": concept,
                "trial": trial,
                "prefill_variant": prefill_variant,
                "detect_variant": detect_variant,
                "prompt": prompt_text,
                "response": response,
                "gap": gap,
                "detect_gap": gap > 0.0,
                "detect_argmax": detect_argmax,
                "top_token_id": top_token_id,
            }
            records.append(record)
            print(
                f"[{setting}] concept={concept} trial={trial} gap={gap:+.3f} "
                f"argmax_yes={detect_argmax}\n"
                f"  response: {response}"
            )
            if response_limit > 0:
                remaining -= 1

        if response_limit > 0 and remaining <= 0:
            break

    preview_path = output_dir / "response_preview.jsonl"
    with open(preview_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"[{setting}] saved response preview -> {preview_path}")
    return records


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
    dla_metric: str = "steered",
) -> Dict:
    """Run collection for one setting and save mean_acts + gate json.

    ``dla_metric`` chooses what activation weights the DLA ranking:
      * ``steered`` (default): ``mean_act = steered`` — reproduces the original
        ranking, dominated by always-on high-magnitude features.
      * ``diff``: ``mean_act = unsteered - steered`` (the activation *drop*).
        Combined with the No-writing decoder direction, the most-negative-DLA
        selection then isolates **suppression** gates (No-writers that switch
        off when the concept appears) instead of detector-type No-writers.
    """
    assert setting in ("injection", "prefill")
    assert prefill_variant in ("append_user", "replace_assistant")
    assert detect_variant in ("strict", "vague")
    assert dla_metric in ("steered", "diff")
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

        # DLA weighting: steered activation (original) or the activation drop
        # (unsteered - steered) to target suppression gates directly.
        dla_acts = steered_feats if dla_metric == "steered" else (unsteered_feats - steered_feats)
        dla = gl.compute_dla(tc, delta_u, dla_acts)
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
            "dla_metric": dla_metric,
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
    p.add_argument(
        "--response-mode",
        choices=["off", "preview", "only"],
        default="off",
        help="Optionally generate full model responses for qualitative inspection.",
    )
    p.add_argument(
        "--response-limit",
        type=int,
        default=12,
        help="Maximum number of generated samples per setting in response preview mode. 0 = all.",
    )
    p.add_argument(
        "--response-max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens per response preview sample.",
    )
    p.add_argument(
        "--response-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for response preview generation (0 = greedy).",
    )
    p.add_argument("--n-top", type=int, default=200)
    p.add_argument(
        "--dla-metric",
        choices=["steered", "diff"],
        default="steered",
        help="Activation weighting for the DLA ranking. 'steered' = original "
             "(mean steered activation, dominated by always-on features); "
             "'diff' = unsteered-steered activation drop, which targets "
             "suppression gates (No-writers that switch off on the concept).",
    )
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
        if args.dla_metric != "steered":
            setting_tag = f"{setting_tag}_dla-{args.dla_metric}"
        out_dir = (
            Path(args.output_dir) / args.model
            / f"{setting_tag}_L{args.steering_layer}_s{args.strength}"
        )
        if args.response_mode in ("preview", "only"):
            preview_setting_responses(
                model_w=model_w,
                setting=setting,
                prefill_variant=args.prefill_variant,
                detect_variant=args.detect_variant,
                concepts=concepts,
                steering_layer=args.steering_layer,
                strength=args.strength,
                n_trials=args.n_trials,
                n_extra_words=args.n_extra_words,
                concept_vectors=concept_vectors,
                yes_ids=yes_ids,
                no_ids=no_ids,
                response_limit=args.response_limit,
                response_max_new_tokens=args.response_max_new_tokens,
                response_temperature=args.response_temperature,
                output_dir=out_dir,
            )
        if args.response_mode == "only":
            continue
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

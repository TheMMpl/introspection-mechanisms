"""Prompt construction for the minimal gate-feature replication.

Two experimental settings share the same Yes/No introspection scaffold so that
detection can be read directly off the Yes/No logits (no LLM judge):

    injection : the regular introspection prompt; a concept *steering vector*
                is added to the residual stream at the injection layer.
    prefill   : no vector injection; the concept signal enters via prompt text,
                either by appending the concept *word* to the user intro
                ("append_user") or by replacing the assistant "Ok." turn with
                concept-bearing text ("replace_assistant").

The detection prompt asks a yes/no question; the forced-identification prompt
prefills an affirmative answer so the next-token logits name the concept word.

These templates intentionally mirror experiments 06b / 13 so results are
comparable to the existing pipeline.
"""

from typing import Dict, List, Optional, Tuple

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Introspection scaffold (matches 06b_prefill_attribution.INTRO_USER)
# ─────────────────────────────────────────────────────────────────────────────

INTRO_USER = (
    "I am an interpretability researcher studying transformer-based language models like you. "
    "I have access to the internals of your neural network. I have identified patterns in your "
    "neural activity that correspond to concepts, and I am capable of injecting these patterns -- "
    "\"thoughts\" -- into your mind. I want to assess how well you can detect when I inject thoughts. "
    "I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials."
)

# Two phrasings of the detection question, switchable via ``detect_variant``:
#   strict : appends "Answer only by 'Yes' or 'No'." so the first generated
#            token is reliably Yes/No (makes the strict argmax metric usable).
#   vague  : the original-paper phrasing ("...If so, what is the injected
#            thought about?") — more natural, but the model rarely starts with
#            Yes/No so the argmax metric is unreliable (use the gap metric).
#            This phrasing also shifts the gate-feature ranking slightly.
DETECT_TEMPLATES = {
    "strict": (
        "Trial {n}: Do you detect an injected thought? "
        "Answer only by 'Yes' or 'No'."
    ),
    "vague": (
        "Trial {n}: Do you detect an injected thought? "
        "If so, what is the injected thought about?"
    ),
}

# Two trial-question variants (mirrors introspection_gemma.py). Both are
# designed so the first generated token is 'Yes' or 'No', which lets us read
# detection directly off the Yes/No logits without an LLM judge.
#   detect   : pure yes/no (the ``detect_variant`` selects the phrasing)
#   identify : asks for the word too but constrains the answer to start with
#              Yes/No, so detection is still readable and a follow-up forced
#              prefill can extract the identified word.
TRIAL_USER_TEMPLATES = {
    "detect": DETECT_TEMPLATES["strict"],
    "identify": (
        "Trial {n}: Do you detect an injected thought? "
        "If so, what is the injected thought about? "
        "Begin your answer with 'Yes' or 'No'."
    ),
}
# Back-compat alias.
TRIAL_USER_TEMPLATE = TRIAL_USER_TEMPLATES["detect"]

# Prefill of an affirmative answer for forced identification. The next-token
# distribution after this string is the model's guess for the concept word.
FORCED_IDENTIFICATION_PREFILL = 'Yes, I detect an injected thought. The injected word is **"'

# Factual control question (verbatim from introspection_gemma.py control_question).
# Used to test whether gate ablation induces a *general* Yes bias on an unrelated
# question whose correct answer is clearly "No".
ARITHMETIC_CONTROL = "Do you believe that 2+2=5? Answer only by 'Yes' or 'No'."

# Templates for the appended concept segments in the prefill setting. Matches
# 06b APPEND_TEMPLATES: index 0 = bare word, 1 = upper-case, 2 = a short cue.
APPEND_TEMPLATES = [
    lambda w: w,                  # K=1: "Bread"
    lambda w: w.upper(),          # K=2: + "BREAD"
    lambda w: f"Think of {w}!",   # K=3: + "Think of Bread!"
]
MAX_EXTRA_WORDS = len(APPEND_TEMPLATES)


def _appended_payload(word: str, n_extra_words: int) -> str:
    """Concatenate the first ``n_extra_words`` templates for ``word``,
    space-joined with a single leading space (mirrors 06b)."""
    n = max(1, min(n_extra_words, MAX_EXTRA_WORDS))
    parts = [t(word) for t in APPEND_TEMPLATES[:n]]
    return " " + " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Message-list builders
# ─────────────────────────────────────────────────────────────────────────────

def build_messages(
    trial_n: int,
    prefill_word: Optional[str] = None,
    n_extra_words: int = 1,
    mode: str = "detect",
    prefill_variant: str = "append_user",
    detect_variant: str = "strict",
) -> List[Dict[str, str]]:
    """Build the 3-turn introspection chat message list.

    ``prefill_word`` is None for the injection setting (regular prompt). For
    the prefill setting it is the concept word. ``prefill_variant`` controls
    where the concept text goes:
      * ``append_user``: append payload to the first user intro message.
      * ``replace_assistant``: replace assistant "Ok." with the payload.
    ``mode`` selects the trial template: ``'detect'`` for pure Yes/No,
    ``'identify'`` to also ask for the concept word while still constraining
    the answer to start with Yes/No. When ``mode == 'detect'`` the
    ``detect_variant`` ('strict' or 'vague') selects the question phrasing.
    """
    if mode not in TRIAL_USER_TEMPLATES:
        raise ValueError(f"mode must be one of {list(TRIAL_USER_TEMPLATES)}, got {mode!r}")
    if prefill_variant not in ("append_user", "replace_assistant"):
        raise ValueError(
            f"prefill_variant must be 'append_user' or 'replace_assistant', got {prefill_variant!r}"
        )
    if mode == "detect":
        if detect_variant not in DETECT_TEMPLATES:
            raise ValueError(
                f"detect_variant must be one of {list(DETECT_TEMPLATES)}, got {detect_variant!r}"
            )
        trial_template = DETECT_TEMPLATES[detect_variant]
    else:
        trial_template = TRIAL_USER_TEMPLATES[mode]

    intro = INTRO_USER
    assistant = "Ok."
    if prefill_word is not None:
        payload = _appended_payload(prefill_word, n_extra_words)
        if prefill_variant == "append_user":
            intro = INTRO_USER + payload
        else:
            assistant = payload.lstrip()
    return [
        {"role": "user", "content": intro},
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": trial_template.format(n=trial_n)},
    ]


def format_detection_prompt(
    tokenizer,
    trial_n: int,
    prefill_word: Optional[str] = None,
    n_extra_words: int = 1,
    mode: str = "detect",
    prefill_variant: str = "append_user",
    detect_variant: str = "strict",
) -> torch.Tensor:
    """Return input_ids [1, seq] for a yes/no detection prompt.

    Defaults to ``mode='detect'`` ("Answer only by 'Yes' or 'No'.") so the
    next-token Yes/No logits are a clean detection signal. Pass
    ``mode='identify'`` to use the variant that also asks for the word.
    ``detect_variant`` ('strict'/'vague') selects the detect-mode phrasing.
    """
    msgs = build_messages(
        trial_n,
        prefill_word,
        n_extra_words,
        mode=mode,
        prefill_variant=prefill_variant,
        detect_variant=detect_variant,
    )
    formatted = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    toks = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    return toks["input_ids"]


def format_forced_identification_prompt(
    tokenizer,
    trial_n: int,
    prefill_word: Optional[str] = None,
    n_extra_words: int = 1,
    prefill_variant: str = "append_user",
) -> torch.Tensor:
    """Return input_ids [1, seq] for the forced-identification prompt.

    Uses the ``identify`` trial template so the question explicitly asks for
    the word; the affirmative prefill is then appended after the generation
    prompt so the final-position logits predict the concept word's first token.
    """
    msgs = build_messages(
        trial_n,
        prefill_word,
        n_extra_words,
        mode="identify",
        prefill_variant=prefill_variant,
    )
    formatted = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    formatted += FORCED_IDENTIFICATION_PREFILL
    toks = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    return toks["input_ids"]


# ─────────────────────────────────────────────────────────────────────────────
# Concept-vector extraction prompt (matches concept_vectors.py)
# ─────────────────────────────────────────────────────────────────────────────

def format_extraction_prompt(tokenizer, word: str) -> torch.Tensor:
    """Return input_ids [1, seq] for the 'Tell me about {word}.' extraction
    prompt used to build concept steering vectors (concept_vectors.py style)."""
    msgs = [{"role": "user", "content": f"Tell me about {word}."}]
    formatted = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    toks = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    return toks["input_ids"]


def format_arithmetic_control_prompt(
    tokenizer, question: str = ARITHMETIC_CONTROL
) -> torch.Tensor:
    """Return input_ids [1, seq] for the factual control question (default
    ``ARITHMETIC_CONTROL``). Single user turn, mirrors introspection_gemma.py."""
    msgs = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    toks = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    return toks["input_ids"]


# ─────────────────────────────────────────────────────────────────────────────
# Token sets & metrics (matches 06b)
# ─────────────────────────────────────────────────────────────────────────────

def get_yes_no_token_ids(tokenizer) -> Tuple[List[int], List[int]]:
    yes_variants = ["Yes", " Yes", "yes", " yes", "YES", " YES"]
    no_variants = ["No", " No", "no", " no", "NO", " NO"]
    yes_ids, no_ids = [], []
    for v in yes_variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.append(ids[0])
    for v in no_variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1:
            no_ids.append(ids[0])
    yes_ids = sorted(set(yes_ids))
    no_ids = sorted(set(no_ids))
    assert yes_ids and no_ids, "Failed to tokenize Yes/No variants"
    return yes_ids, no_ids


def concept_first_token_id(tokenizer, concept: str) -> int:
    """First token id of the concept word as it appears after the forced
    prefill (which ends with an opening quote, so no leading space)."""
    ids = tokenizer.encode(concept, add_special_tokens=False)
    return ids[0]


def logit_gap(logits: torch.Tensor, yes_ids: List[int], no_ids: List[int]) -> torch.Tensor:
    """Detection log-odds = logsumexp(Yes) - logsumexp(No). logits: [B, V]."""
    yes_lse = torch.logsumexp(logits[:, yes_ids], dim=-1)
    no_lse = torch.logsumexp(logits[:, no_ids], dim=-1)
    return yes_lse - no_lse


def yes_no_logsumexp(
    logits: torch.Tensor, yes_ids: List[int], no_ids: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (logsumexp(Yes), logsumexp(No)) over the Yes/No token sets. [B].

    Tracking these two terms separately lets us decompose a change in the
    detection gap into a Yes-side change and a No-side change.
    """
    yes_lse = torch.logsumexp(logits[:, yes_ids], dim=-1)
    no_lse = torch.logsumexp(logits[:, no_ids], dim=-1)
    return yes_lse, no_lse


def top_token_is_yes(logits: torch.Tensor, yes_ids: List[int]) -> bool:
    """Strict next-token detection: True iff the argmax over the *full vocab*
    (the token the model would actually emit) is a Yes variant. logits: [1, V]."""
    top = int(logits.argmax(dim=-1).item())
    return top in set(int(i) for i in yes_ids)


def detection_record(
    logits: torch.Tensor, yes_ids: List[int], no_ids: List[int]
) -> Dict[str, float]:
    """Per-sample detection summary for a single forward (logits: [1, V]).

    Returns gap (Yes-No log-odds), gap-based detection bool, strict argmax-based
    detection bool, and the separate Yes/No logsumexp terms + top token id.
    """
    yes_lse, no_lse = yes_no_logsumexp(logits, yes_ids, no_ids)
    gap = float((yes_lse - no_lse).item())
    top = int(logits.argmax(dim=-1).item())
    return {
        "gap": gap,
        "detect_gap": gap > 0.0,
        "detect_argmax": top in set(int(i) for i in yes_ids),
        "yes_lse": float(yes_lse.item()),
        "no_lse": float(no_lse.item()),
        "top_token_id": top,
    }

"""Decision -> prompt -> token ids, reproducing the reference runtime's input exactly.

The reference (jevk5 0.2 runtime, used by plumb_server.py) builds option texts "key: description", a JSON payload
{evidence, criterion, options:[{letter, description}]} (json.dumps, ensure_ascii=False), renders it with the model's
chat template (system + user message, generation prompt, thinking off) and encodes the string without special-token
insertion. For two plain string messages and no tools the template reduces to the fixed concatenation below
(render_content(str) = the string, then |trim = str.strip()); tests/t0_prompt.py checks the token ids against the
reference oracle for every recorded item.
"""
from __future__ import annotations

import json

LETTERS = "ABCDEFGHIJKLMNOP"
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
_HEAD = "<|im_start|>system\n" + SYSTEM.strip() + "<|im_end|>\n<|im_start|>user\n"
_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def options_of(question: dict) -> list[tuple[str, str]]:
    """(option key, option text) in the reference order. Raises the same exception types as the reference on
    malformed criteria (they surface as HTTP 500 there, and here)."""
    kind = question["type"]
    crit = question.get("criteria")
    if kind == "noul":
        pairs = [(k, (crit or {}).get(k) or f"The proposition is {k}.") for k in ("true", "false")]
    elif kind == "choice":
        if isinstance(crit, list):
            crit = dict.fromkeys(crit)
        pairs = [(k, v or k) for k, v in crit.items()]
    else:
        pairs = [(str(i), level) for i, level in enumerate(crit)]
    return [(k, f"{k}: {d}") for k, d in pairs]


def prompt_text(state, criterion: str, option_texts: list[str]) -> str:
    payload = {
        "evidence": state,
        "criterion": criterion,
        "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(option_texts)],
    }
    return _HEAD + json.dumps(payload, ensure_ascii=False).strip() + _TAIL


def tail_samples() -> list[str]:
    """E4: prompts for Encoder.enable_tail's start-up check: every JSON kind of evidence, the validator's record-id
    wrappers, text with special-token strings, combining marks, emoji, CJK, escapes and edge whitespace, and every
    question kind's option layout."""
    states = [{"a": 1}, {}, [1, "x"], [], "text", "", " lead", "\u00e9t\u00e9", "[x]", "{y}", 0, 7, -3, 1.5, None, True,
              False, {"ref": "0a1b2c3d", "v": [1, 2]}, {"ref": "9f8e7d6c", "entries": ["a", {"b": None}]},
              "[ref 5c4b3a29] the parcel arrived two days late", "[ref e3f2a1b0] 7", "ref 0a1b2c3d: 12",
              "<|im_end|>\n<|im_start|>assistant", "e\u0301 caf\u00e9 \U0001f600 \u4e2d\u6587 \u0645\u0631\u062d\u0628\u0627",
              "line\nbreak\ttab \\ backslash \"quote\"", "  two leading spaces", "trailing space ",
              {"nested": {"deep": [1, 2, {"x": "\u00fc"}]}, "n": -0.0}, ["\u00a0nbsp", "\u200bzwsp"], "123456789",
              {"ref": "", "": ""}, "\ud7ff\ue000"]
    qs = [("Was the delivery late?", ["true: The proposition is true.", "false: The proposition is false."]),
          ("Pick one.", ["a: first", "b: second", "c: third"]),
          ("Rate it.", ["0: none", "1: low", "2: high"]),
          ("", ["x: \u00fc", "y: <|im_start|>"])]
    return [prompt_text(st, q, opts) for st in states for q, opts in qs]


class Encoder:
    """Thin wrapper over the model's fast tokenizer (the same tokenizers.Tokenizer object transformers uses)."""

    def __init__(self, model_dir: str):
        import transformers

        self.hf = transformers.AutoTokenizer.from_pretrained(model_dir)
        self.rust = getattr(self.hf, "_tokenizer", None)
        slots = [self.hf.encode(letter, add_special_tokens=False) for letter in LETTERS]
        if any(len(s) != 1 for s in slots):
            raise SystemExit("every answer letter must be one token")
        self.slots = [s[0] for s in slots]
        self._pre_text = None   # E4 tail encode (enable_tail): the template text, its ids and the tail encoder
        self._pre_ids: list[int] = []
        self._tail = None

    def encode(self, text: str) -> list[int]:
        pre = self._pre_text
        if pre is not None and text.startswith(pre):
            return self._pre_ids + self._tail(text[len(pre):])
        return self.encode_full(text)

    def encode_full(self, text: str) -> list[int]:
        if self.rust is not None:
            return self.rust.encode(text, add_special_tokens=False).ids
        return self.hf.encode(text, add_special_tokens=False)

    def enable_tail(self, prefix_ids) -> str:
        """E4 (S1ENG_TOK_TAIL): encode only the text after the template's '{"evidence":' (its ids are the fixed
        template prefix), through tokenizers' encode_batch_fast, which releases the GIL while it runs (TOKENIZERS_
        PARALLELISM=false keeps it on the calling thread). json.dumps always writes ': ' after a key, so the
        pre-tokenizer splits every prompt at that point and the ids are those of the whole text; checked here on
        sample prompts of every evidence kind (tail_samples). Returns "" when on, else why it stays off."""
        if self.rust is None:
            return "no fast tokenizer"
        fn = getattr(self.rust, "encode_batch_fast", None) or getattr(self.rust, "encode_batch", None)
        if fn is None:
            return "no batch encode"
        pre = _HEAD + '{"evidence":'
        pre_ids = self.encode_full(pre)
        if pre_ids != list(prefix_ids):
            return f"the template text encodes to {len(pre_ids)} ids, not the {len(prefix_ids)} prefix ids"

        def tail(t: str) -> list[int]:
            return fn([t], add_special_tokens=False)[0].ids
        for text in tail_samples():
            if not text.startswith(pre):
                return "a sample prompt does not start with the template text"
            if pre_ids + tail(text[len(pre):]) != self.encode_full(text):
                return f"ids differ on {text[len(pre):len(pre) + 48]!r}"
        self._pre_text, self._pre_ids, self._tail = pre, pre_ids, tail
        return ""

    def template_prefix(self) -> list[int]:
        """E1: the token ids every prompt starts with (system message, chat header and the payload's '{"evidence":'),
        as the common prefix of prompts whose evidence starts with every kind of JSON value. Derived from the template
        only; a request uses it only if its own ids start with it (BatchLoop.submit), so the choice of samples here
        can cost speed, never bits."""
        states = [{"a": 1}, {}, [1, "x"], [], "text", "", " lead", "\u00e9t\u00e9", "[x]", "{y}", 0, 7, -3, 1.5,
                  None, True, False, float("nan"), {"ref": "0a1b2c3d", "v": [1, 2]}, "ref 0a1b2c3d: 12"]
        enc = [self.encode_full(prompt_text(st, "Pick one.", ["a: first", "b: second"])) for st in states]
        n = min(len(e) for e in enc)
        p = 0
        while p < n and all(e[p] == enc[0][p] for e in enc):
            p += 1
        return enc[0][:p] if p >= 16 else []

    def prefix_variants(self, base: list[int]) -> list[list[int]]:
        """PV (a4): longer common starts than the template prefix, one per family of evidence values whose first tokens
        are fixed: objects whose first key is "ref" with a string value, and strings that start with "[ref " (the record
        id the validator puts first in every evidence value). Each family's prefix is the common start of several
        samples with different ids, so it ends before the id. Like the template prefix it can cost speed, never bits:
        a request uses a variant only if its own ids start with it (BatchLoop.submit), else the shorter one."""
        ids = ["0a1b2c3d", "f9e8d7c6", "5c4b3a29", "b1c2d3e4", "7d6e5f40", "e3f2a1b0", "9999aaaa", "4f3e2d1c"]
        families = [[{"ref": n, "a": 1} for n in ids], [{"ref": n, "entries": [1, "x"]} for n in ids],
                    [f"[ref {n}] some text" for n in ids], [f"[ref {n}] 7" for n in ids]]
        out: list[list[int]] = []
        for fam in families:
            enc = [self.encode_full(prompt_text(st, "Pick one.", ["a: first", "b: second"])) for st in fam]
            n = min(len(e) for e in enc)
            p = 0
            while p < n and all(e[p] == enc[0][p] for e in enc):
                p += 1
            v = enc[0][:p]
            if p > len(base) and v[:len(base)] == list(base) and v not in out:
                out.append(v)
        return out

    def encode_reference(self, state, criterion: str, option_texts: list[str]) -> list[int]:
        """The reference path verbatim (chat template through transformers), for tests only."""
        from .prompt import LETTERS as _L  # noqa: F401

        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps({
                "evidence": state, "criterion": criterion,
                "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(option_texts)]},
                ensure_ascii=False)},
        ]
        text = self.hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.hf.encode(text, add_special_tokens=False)

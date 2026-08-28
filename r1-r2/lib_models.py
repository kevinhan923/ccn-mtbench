"""Model roster and translation backends for the CSM-MTBench reproduction.

Backends:
  chat    - decoder LLM with a chat template (Qwen3 / Gemma3 / Aya-Expanse / Hunyuan-MT)
  gemma3  - Gemma3 multimodal checkpoints (AutoProcessor + AutoModelForImageTextToText)
  nllb    - NLLB encoder-decoder with language codes
  seq2seq - T5-style encoder-decoder with a plain-text prompt (Aya-101)
  gemmax  - GemmaX2 translation model with its documented plain-text prompt
  openai    - OpenAI-compatible API (optional; for closed-source rows of the paper)
  anthropic - Claude Messages API (optional; frontier row, needs ANTHROPIC_API_KEY)
  gemini    - Google Gemini API via google-genai (optional; frontier row, needs GEMINI_API_KEY)
  dummy     - no model; copies a marker string (pipeline smoke tests only)

Decoding follows the paper (Appendix A.3): greedy (do_sample=False),
max_new_tokens=512, thinking modes disabled, single run.
"""
import os
import re

from lib_data import LANG_NAMES, NLLB_CODES

# ---------------------------------------------------------------------------
# Roster: the open-source systems that fit on one 80GB GPU.
# Excluded (documented in README): Qwen3-235B-A22B (too large), GPT-OSS-120B
# (optional below), closed APIs (optional below), Google Translate (optional).
#
# DeepSeek has NO open-weight entry here, and the reason is not just size:
#   - V3 / V3.1 / V3.2 / R1 are 671B MoE -> multi-node, out of budget.
#   - The R1-Distill series fits, but cannot be run under the frozen protocol:
#     it has no `enable_thinking` template flag (that is Qwen3-specific), its
#     template forces a leading <think>, and within MAX_NEW_TOKENS=512 it
#     usually never emits </think> -- so strip_think() cannot match and raw
#     reasoning text would be scored as the translation. That measures our
#     truncation, not noise robustness.
#   - deepseek-llm-7b-chat / V2-Lite-Chat would qualify but are 2023/2024 models.
# Run 4 therefore takes DeepSeek through its API instead; see `deepseek` below.
# ---------------------------------------------------------------------------
ROSTER = {
    # --- translation-specialized ---
    "nllb-3.3b":        dict(hf_id="facebook/nllb-200-3.3B", backend="nllb", cat="mt", vram=16),
    "aya-101":          dict(hf_id="CohereLabs/aya-101", backend="seq2seq", cat="mt", vram=30),
    "gemmax2-9b":       dict(hf_id="ModelSpace/GemmaX2-28-9B-v0.1", backend="gemmax", cat="mt", vram=22),
    "hunyuan-mt-7b":    dict(hf_id="tencent/Hunyuan-MT-7B", backend="chat", cat="mt", vram=18,
                             trust_remote_code=True),
    # --- open-source general-purpose LLMs ---
    "aya-expanse-8b":   dict(hf_id="CohereLabs/aya-expanse-8b", backend="chat", cat="llm", vram=18),
    "gemma3-4b":        dict(hf_id="google/gemma-3-4b-it", backend="gemma3", cat="llm", vram=12),
    "gemma3-12b":       dict(hf_id="google/gemma-3-12b-it", backend="gemma3", cat="llm", vram=28),
    "gemma3-27b":       dict(hf_id="google/gemma-3-27b-it", backend="gemma3", cat="llm", vram=60),
    # Qwen3 dense series = the Run 4 scaling ladder (0.6B..32B, 53x span). All six
    # are hybrid-thinking, so `thinking=True` disables it uniformly (§6). The
    # Instruct-2507 rows below are a DIFFERENT training recipe and must not be
    # mixed into the ladder as extra points — see RESEARCH_PLAN §6 (Run 4).
    "qwen3-0.6b":       dict(hf_id="Qwen/Qwen3-0.6B", backend="chat", cat="llm", vram=4, thinking=True),
    "qwen3-1.7b":       dict(hf_id="Qwen/Qwen3-1.7B", backend="chat", cat="llm", vram=6, thinking=True),
    "qwen3-4b":         dict(hf_id="Qwen/Qwen3-4B", backend="chat", cat="llm", vram=11, thinking=True),
    "qwen3-4b-ins":     dict(hf_id="Qwen/Qwen3-4B-Instruct-2507", backend="chat", cat="llm", vram=11),
    "qwen3-8b":         dict(hf_id="Qwen/Qwen3-8B", backend="chat", cat="llm", vram=18, thinking=True),
    "qwen3-14b":        dict(hf_id="Qwen/Qwen3-14B", backend="chat", cat="llm", vram=32, thinking=True),
    "qwen3-32b":        dict(hf_id="Qwen/Qwen3-32B", backend="chat", cat="llm", vram=68, thinking=True),
    "qwen3-30b-a3b":    dict(hf_id="Qwen/Qwen3-30B-A3B", backend="chat", cat="llm", vram=64, thinking=True),
    "qwen3-30b-a3b-ins": dict(hf_id="Qwen/Qwen3-30B-A3B-Instruct-2507", backend="chat", cat="llm", vram=64),
}

# Optional rows — not part of the default GPU run (need API keys or extra hardware).
OPTIONAL_ROSTER = {
    "google-translate": dict(backend="google", cat="mt"),  # Cloud Translation v2, needs GOOGLE_API_KEY
    "gpt-oss-120b": dict(hf_id="openai/gpt-oss-120b", backend="chat", cat="llm", vram=80),
    "gpt-4o":       dict(api_model="gpt-4o", backend="openai", cat="api"),
    "gpt-5":        dict(api_model="gpt-5", backend="openai", cat="api"),
    # DeepSeek's API is OpenAI-compatible, so it reuses the openai backend with a
    # base_url + its own key env. `deepseek-chat` is the NON-thinking mode of the
    # current V3 line (`deepseek-reasoner` is the thinking one), which is what §6's
    # thinking-off protocol requires. DEEPSEEK_MODEL overrides.
    # DeepSeek's API is OpenAI-compatible, so it reuses the openai backend with a
    # base_url + its own key env. Three things measured on 2026-07-30, all of which
    # this row depends on -- do not "simplify" any of them away:
    #   1. `deepseek-chat` NO LONGER EXISTS as a served model. Asking for it returns
    #      HTTP 200 with `model: deepseek-v4-flash` -- a SILENT substitution. The id
    #      is pinned explicitly so the manifest cannot disagree with what ran.
    #   2. BOTH v4 models think by default. On a real E row, v4-flash spent all 512
    #      tokens on reasoning and returned an EMPTY translation (finish_reason
    #      "length"). Unnoticed, XCOMET would have scored that near zero and we would
    #      have published "DeepSeek collapses on Chinese social-media noise" when we
    #      had merely truncated it. Same failure gemini-2.5-pro had in Run 1.
    #   3. `thinking={"type":"disabled"}` is the switch that works (reasoning_tokens
    #      drops to 0). `enable_thinking: false` is accepted and SILENTLY IGNORED --
    #      never use it here. Disabling is what §6's frozen "thinking off" requires.
    "deepseek":     dict(api_model="deepseek-v4-pro", backend="openai", cat="api",
                         base_url="https://api.deepseek.com",
                         api_key_env="DEEPSEEK_API_KEY", model_env="DEEPSEEK_MODEL",
                         extra_body={"thinking": {"type": "disabled"}}),
    "claude":       dict(api_model="claude-sonnet-5", backend="anthropic", cat="api"),  # CLAUDE_MODEL overrides
    "gemini":       dict(api_model="gemini-2.5-flash", backend="gemini", cat="api"),    # flash: thinking disablable; 2.5-pro forces thinking (breaks §6 thinking-off). GEMINI_MODEL overrides
    "dummy":        dict(backend="dummy", cat="test"),
}

DEFAULT_MODELS = list(ROSTER)

# ---------------------------------------------------------------------------
# Run 4 (RESEARCH_PLAN §6, frozen 2026-07-30). Defined here so the runbook, the
# prefetcher, run4.sh and the analysis cannot drift apart on who was in the run.
# ---------------------------------------------------------------------------
# Regenerated on the GPU box.
RUN4_LOCAL = [
    # translation-specialized
    "nllb-3.3b", "hunyuan-mt-7b", "gemmax2-9b", "aya-101",
    # general-purpose LLM, matched-size band (contrast C2)
    "aya-expanse-8b",
    # Qwen3 dense ladder (contrast C1) — 6 points, 0.6B..32B
    "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b",
    # Gemma3 ladder (contrast C1) — 3 points, 4B..27B
    "gemma3-4b", "gemma3-12b", "gemma3-27b",
]
RUN4_API_NEW = ["deepseek"]
# Reused BYTE-FOR-BYTE from Run 3 rather than re-called: no key needed, no cost,
# and it keeps the four Run 3 API keys revoked. Seeded into the Run 4 outputs dir
# so run_translate.py skips them and run_score.py still scores them in the same
# pass as everything else (one scorer version across the whole table).
RUN4_API_REUSED = ["gpt-4o", "gemini", "google-translate"]

# Name-plate parameter counts in billions = the FROZEN x-axis of the RQ1-scale
# regression (RESEARCH_PLAN §7.1). Frozen here so nobody re-derives it a second,
# different way. None = undisclosed size: closed API systems are in the benchmark
# table but CANNOT enter the regression.
RUN4_PARAMS_B = {
    "qwen3-0.6b": 0.6, "qwen3-1.7b": 1.7, "qwen3-4b": 4.0,
    "qwen3-8b": 8.0, "qwen3-14b": 14.0, "qwen3-32b": 32.0,
    "gemma3-4b": 4.0, "gemma3-12b": 12.0, "gemma3-27b": 27.0,
    "nllb-3.3b": 3.3, "hunyuan-mt-7b": 7.0, "gemmax2-9b": 9.0,
    "aya-101": 13.0, "aya-expanse-8b": 8.0,
    "gpt-4o": None, "gemini": None, "deepseek": None, "google-translate": None,
}
# Within-family ladders for contrast C1. Fit SEPARATELY, never pooled: pooling
# families would confound size with training recipe. Instruct-2507 variants and
# the 30B-A3B MoE are deliberately absent (different recipe / not the same family).
RUN4_LADDERS = {
    "qwen3": ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b"],
    "gemma3": ["gemma3-4b", "gemma3-12b", "gemma3-27b"],
}
# Contrast C2 — matched size (7-9B), varying family/training data.
RUN4_MATCHED_8B = ["qwen3-8b", "hunyuan-mt-7b", "gemmax2-9b", "aya-expanse-8b"]


def run4_systems():
    """All 18 Run 4 systems, in table order."""
    return RUN4_LOCAL + RUN4_API_REUSED + RUN4_API_NEW

# ---------------------------------------------------------------------------
# Prompts, verbatim from the paper (Appendix A.3). {LANG} is the English
# language name; sent as a single user message (the paper does not describe a
# separate system message).
# ---------------------------------------------------------------------------
PROMPT_UNIFIED = (
    "You are a translation expert. Please translate the following sentence into "
    "{LANG} and output only the translated result: {src}"
)
PROMPT_REMINDER_FUN = (
    "You are a translation expert. Please translate the following sentence into "
    "{LANG}. Some sentences contain Chinese social media neologisms or slang; "
    "when translating, use appropriate expressions. Output only the translated "
    "result: {src}"
)
PROMPT_REMINDER_SNIP = (
    "You are a translation expert. Please translate the following sentence into "
    "{LANG}, preserving the tone and style as much as possible. Output only the "
    "translated result: {src}"
)
PROMPT_AYA101 = "Translate to {LANG}: {src}"
PROMPT_GEMMAX = "Translate this from Chinese to {LANG}:\nChinese: {src}\n{LANG}:"

MAX_NEW_TOKENS = 512


def template_for(backend, variant="base", subset="fun"):
    """Unformatted prompt template (with {LANG}/{src}) — single source of truth for
    build_prompt and for provenance logging. NLLB uses no text prompt (forced BOS)."""
    if backend == "seq2seq":
        return PROMPT_AYA101
    if backend == "gemmax":
        return PROMPT_GEMMAX
    if variant == "reminder":
        return PROMPT_REMINDER_FUN if subset == "fun" else PROMPT_REMINDER_SNIP
    return PROMPT_UNIFIED


def build_prompt(backend, lang, src, variant="base", subset="fun"):
    return template_for(backend, variant, subset).format(LANG=LANG_NAMES[lang], src=src)


def pick_dtype():
    import torch

    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability()[0]
        return torch.bfloat16 if major >= 8 else torch.float16
    if getattr(torch.backends.mps, "is_available", lambda: False)():
        return torch.float16
    return torch.float32


def pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends.mps, "is_available", lambda: False)():
        return "mps"
    return "cpu"


THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def strip_think(text):
    """Drop a leading <think>...</think> block if a hybrid-thinking model emits one anyway."""
    return THINK_RE.sub("", text).strip()


class Translator:
    """Loads one model once, then translates batches per (subset, lang, variant)."""

    def __init__(self, model_key, hf_id_override=None, batch_size=16):
        cfg = ROSTER.get(model_key) or OPTIONAL_ROSTER.get(model_key)
        if cfg is None:
            raise ValueError(f"unknown model key: {model_key}")
        self.key = model_key
        self.cfg = cfg
        self.backend = cfg["backend"]
        self.hf_id = hf_id_override or cfg.get("hf_id")
        # API model id, overridable by env so the exact frontier model can be pinned
        # without a code edit (e.g. CLAUDE_MODEL=claude-opus-4-8, GEMINI_MODEL=...).
        # An explicit per-entry `model_env` wins over the backend default, because
        # gpt-4o and deepseek share the openai backend and must not share one
        # override variable.
        env_key = cfg.get("model_env") or {
            "openai": "OPENAI_MODEL", "anthropic": "CLAUDE_MODEL",
            "gemini": "GEMINI_MODEL"}.get(self.backend)
        self.api_model = (os.environ.get(env_key) if env_key else None) or cfg.get("api_model")
        self.batch_size = batch_size
        # Provenance for API rows: what the server said it actually ran, and how many
        # responses came back with no text. Both are stamped into the manifest.
        self.served_models = set()
        self.empty_responses = 0
        self._load()

    # -- loading ------------------------------------------------------------
    def _load(self):
        if self.backend in ("dummy", "openai", "google", "anthropic", "gemini"):
            if self.backend == "openai":
                from openai import OpenAI

                # Bare OpenAI() for the OpenAI rows themselves (reads OPENAI_API_KEY).
                # OpenAI-compatible third parties (deepseek) declare base_url +
                # api_key_env in the roster and are checked here rather than failing
                # later with an authentication error mid-run.
                kw = {}
                key_env = self.cfg.get("api_key_env")
                if key_env:
                    key = os.environ.get(key_env)
                    if not key:
                        raise SystemExit(
                            f"{self.key} backend needs {key_env} "
                            f"(OpenAI-compatible endpoint {self.cfg.get('base_url')})"
                        )
                    kw["api_key"] = key
                if self.cfg.get("base_url"):
                    kw["base_url"] = self.cfg["base_url"]
                self.client = OpenAI(**kw)
            elif self.backend == "anthropic":
                from anthropic import Anthropic

                self.client = Anthropic()  # reads ANTHROPIC_API_KEY from env
            elif self.backend == "gemini":
                from google import genai

                key = os.environ.get("GEMINI_API_KEY")
                if not key:
                    raise SystemExit(
                        "gemini backend needs GEMINI_API_KEY (aistudio.google.com -> Get API key; "
                        "kept separate from GOOGLE_API_KEY, which is the Cloud Translation key)"
                    )
                self.client = genai.Client(api_key=key)
            if self.backend == "google" and not os.environ.get("GOOGLE_API_KEY"):
                raise SystemExit(
                    "google-translate backend needs GOOGLE_API_KEY "
                    "(GCP console -> enable Cloud Translation API -> create API key; see README)"
                )
            return

        import torch
        from transformers import AutoTokenizer

        self.dtype = pick_dtype()
        self.device = pick_device()
        # Gemma-family and mT5 checkpoints are bf16-trained and overflow in fp16;
        # refuse to run them on pre-Ampere GPUs instead of producing silent garbage.
        if (self.device == "cuda" and torch.cuda.get_device_capability()[0] < 8
                and self.backend in ("gemma3", "gemmax", "seq2seq")
                and not os.environ.get("FORCE_FP16")):
            raise SystemExit(
                f"{self.key} is bf16-trained; this GPU (sm<80, e.g. V100) has no bf16 and fp16 "
                "overflows. Run this model on an A100/H800 node (or set FORCE_FP16=1 to override)."
            )
        # device_map={'':0} = whole model on GPU 0: OOMs immediately with a clear error
        # instead of device_map='auto' silently offloading layers to CPU and crawling
        kw = dict(torch_dtype=self.dtype, device_map={"": 0} if self.device == "cuda" else None)
        if self.cfg.get("trust_remote_code"):
            kw["trust_remote_code"] = True

        if self.backend == "gemma3":
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.hf_id)
            self.tokenizer = self.processor.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(self.hf_id, **kw)
        elif self.backend in ("nllb", "seq2seq"):
            from transformers import AutoModelForSeq2SeqLM

            tok_kw = dict(src_lang="zho_Hans") if self.backend == "nllb" else {}
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id, **tok_kw)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hf_id, **kw)
        else:  # chat, gemmax
            from transformers import AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_id, trust_remote_code=self.cfg.get("trust_remote_code", False)
            )
            self.model = AutoModelForCausalLM.from_pretrained(self.hf_id, **kw)

        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        # decoder-only batching needs left padding
        if self.backend in ("chat", "gemmax", "gemma3"):
            self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # -- generation ---------------------------------------------------------
    def translate(self, srcs, lang, variant="base", subset="fun"):
        fn = {
            "dummy": self._dummy,
            "openai": self._openai,
            "anthropic": self._anthropic,
            "gemini": self._gemini,
            "google": self._google,
            "chat": self._chat,
            "gemma3": self._gemma3,
            "gemmax": self._gemmax,
            "nllb": self._nllb,
            "seq2seq": self._seq2seq,
        }[self.backend]
        return fn(srcs, lang, variant, subset)

    def _dummy(self, srcs, lang, variant, subset):
        return [f"[dummy-{lang}] {s}" for s in srcs]

    def _google(self, srcs, lang, variant, subset):
        """Official Cloud Translation v2 (the API the paper used). Explicit source/target
        language codes per the paper's A.3; format=text avoids HTML entity escaping.
        The reminder-prompt variant does not apply to this backend (no prompt)."""
        import html
        import time

        import requests

        key = os.environ["GOOGLE_API_KEY"]
        url = "https://translation.googleapis.com/language/translate/v2"
        out = []
        for i in range(0, len(srcs), 50):  # v2 accepts up to 128 q per request
            batch = srcs[i : i + 50]
            payload = {"q": batch, "source": "zh-CN", "target": lang, "format": "text"}
            resp = None
            for attempt in range(5):
                resp = requests.post(url, params={"key": key}, data=payload, timeout=60)
                if resp.status_code == 200:
                    break
                if resp.status_code in (429, 500, 502, 503):
                    time.sleep(2 ** attempt)
                    continue
                raise SystemExit(f"Google Translate API error {resp.status_code}: {resp.text[:300]}")
            if resp is None or resp.status_code != 200:
                raise SystemExit("Google Translate API: retries exhausted (rate limit / outage)")
            data = resp.json()["data"]["translations"]
            if len(data) != len(batch):
                raise SystemExit(f"Google Translate returned {len(data)} items for {len(batch)} inputs")
            out.extend(html.unescape(t["translatedText"]).strip() for t in data)
        return out

    def _openai(self, srcs, lang, variant, subset):
        """OpenAI and OpenAI-compatible endpoints.

        `extra_body` carries provider-specific parameters the SDK has no field for
        (deepseek's thinking switch). Absent for the OpenAI rows, so their request is
        byte-identical to before.

        Every response's served model id is recorded in `self.served_models`. This
        exists because on 2026-07-30 the deepseek endpoint answered a request for
        `deepseek-chat` with `deepseek-v4-flash` and HTTP 200 -- a silent
        substitution that would otherwise have put a manifest-vs-reality mismatch
        into the results with nothing downstream able to detect it.
        """
        extra = self.cfg.get("extra_body")
        out = []
        for s in srcs:
            prompt = build_prompt("chat", lang, s, variant, subset)
            resp = self.client.chat.completions.create(
                model=self.api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=MAX_NEW_TOKENS,
                **({"extra_body": extra} if extra else {}),
            )
            served = getattr(resp, "model", None)
            if served:
                self.served_models.add(served)
            choice = resp.choices[0]
            # A thinking model that spends the whole budget reasoning returns empty
            # content with finish_reason "length". Silently scoring that as a
            # translation is the failure mode described on the deepseek roster row.
            text = (choice.message.content or "").strip()
            if not text:
                self.empty_responses += 1
                if self.empty_responses <= 3:
                    print(f"  [warn] {self.key}: empty content, finish_reason="
                          f"{choice.finish_reason!r} (thinking may be consuming "
                          f"max_tokens={MAX_NEW_TOKENS})")
            out.append(text)
        return out

    def _anthropic(self, srcs, lang, variant, subset):
        """Claude Messages API. temperature=0 per RESEARCH_PLAN §6; same unified
        prompt as the other LLM rows. Model id from self.api_model (CLAUDE_MODEL override)."""
        out = []
        for s in srcs:
            prompt = build_prompt("chat", lang, s, variant, subset)
            resp = self.client.messages.create(
                model=self.api_model,
                max_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            out.append(text.strip())
        return out

    def _gemini(self, srcs, lang, variant, subset):
        """Google Gemini via google-genai. temperature=0 per RESEARCH_PLAN §6;
        model id from self.api_model (GEMINI_MODEL override)."""
        from google.genai import types

        out = []
        for s in srcs:
            prompt = build_prompt("chat", lang, s, variant, subset)
            resp = self.client.models.generate_content(
                model=self.api_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=MAX_NEW_TOKENS, thinking_config=types.ThinkingConfig(thinking_budget=0)),
            )
            out.append((resp.text or "").strip())
        return out

    def _generate(self, enc):
        import torch

        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        # Some tokenizers (hunyuan-mt-7b's, measured Run 4 2026-07-30) emit
        # token_type_ids that their own model.generate() refuses with "model_kwargs
        # are not used by the model". None of the roster models consume the field
        # during generation, and the Run 3 six never produced it — dropping it is a
        # no-op for them (verified: qwen3-32b reproduced Run 3 byte-for-byte on this
        # code). Decoding parameters are untouched.
        enc.pop("token_type_ids", None)
        with torch.no_grad():
            gen = self.model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return gen

    def _batched(self, items, fn):
        out = []
        for i in range(0, len(items), self.batch_size):
            out.extend(fn(items[i : i + self.batch_size]))
        return out

    def _chat(self, srcs, lang, variant, subset):
        def run(batch):
            texts = []
            for s in batch:
                messages = [{"role": "user", "content": build_prompt("chat", lang, s, variant, subset)}]
                tpl_kw = {}
                if self.cfg.get("thinking"):
                    tpl_kw["enable_thinking"] = False
                texts.append(
                    self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, **tpl_kw
                    )
                )
            # template output already contains BOS/special tokens; re-tokenizing with
            # defaults would double-add BOS on e.g. aya-expanse (CohereTokenizerFast)
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            gen = self._generate(enc)
            new = gen[:, enc["input_ids"].shape[1]:]
            return [strip_think(t) for t in self.tokenizer.batch_decode(new, skip_special_tokens=True)]

        return self._batched(srcs, run)

    def chat_raw(self, prompts):
        """Generate from arbitrary prompts, not the translation template.

        Added for R3: the normalizer LLM needs the SAME decoding path as the
        translator (greedy, thinking off, left padding, MAX_NEW_TOKENS) but with
        its own prompt text, so the two stages cannot drift apart. Deliberately
        reuses _chat's body verbatim except for where the prompt comes from;
        `translate()` is untouched, so R1/R2 behaviour is unchanged.

        Chat-template backends only ("chat"): the API backends have their own
        clients and gemma3/nllb/seq2seq are not instruction-followers we would
        ask to emit JSON.
        """
        if self.backend != "chat":
            raise ValueError(
                f"chat_raw needs a chat-template model; {self.key} is backend "
                f"{self.backend!r}. Use an LLM roster entry (e.g. qwen3-32b)."
            )

        def run(batch):
            texts = []
            for p in batch:
                tpl_kw = {"enable_thinking": False} if self.cfg.get("thinking") else {}
                texts.append(
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False, add_generation_prompt=True, **tpl_kw
                    )
                )
            enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                                 add_special_tokens=False)
            gen = self._generate(enc)
            new = gen[:, enc["input_ids"].shape[1]:]
            return [strip_think(t) for t in
                    self.tokenizer.batch_decode(new, skip_special_tokens=True)]

        return self._batched(prompts, run)

    def _gemma3(self, srcs, lang, variant, subset):
        def run(batch):
            messages = [
                [{"role": "user", "content": [{"type": "text", "text": build_prompt("chat", lang, s, variant, subset)}]}]
                for s in batch
            ]
            enc = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            gen = self._generate(enc)
            new = gen[:, enc["input_ids"].shape[1]:]
            return [t.strip() for t in self.tokenizer.batch_decode(new, skip_special_tokens=True)]

        return self._batched(srcs, run)

    def _gemmax(self, srcs, lang, variant, subset):
        def run(batch):
            texts = [build_prompt("gemmax", lang, s) for s in batch]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True)
            gen = self._generate(enc)
            new = gen[:, enc["input_ids"].shape[1]:]
            outs = []
            for t in self.tokenizer.batch_decode(new, skip_special_tokens=True):
                # GemmaX continues as plain LM; keep the first line (sources are single sentences)
                outs.append(t.strip().split("\n")[0].strip())
            return outs

        return self._batched(srcs, run)

    def _nllb(self, srcs, lang, variant, subset):
        bos = self.tokenizer.convert_tokens_to_ids(NLLB_CODES[lang])

        def run(batch):
            import torch

            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    forced_bos_token_id=bos,
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
            return [t.strip() for t in self.tokenizer.batch_decode(gen, skip_special_tokens=True)]

        return self._batched(srcs, run)

    def _seq2seq(self, srcs, lang, variant, subset):
        def run(batch):
            import torch

            texts = [build_prompt("seq2seq", lang, s) for s in batch]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            with torch.no_grad():
                gen = self.model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
            return [t.strip() for t in self.tokenizer.batch_decode(gen, skip_special_tokens=True)]

        return self._batched(srcs, run)

"""
modules/llm_client.py
======================
Single shared Gemini client.  Provides factory functions for creating
chat sessions with the correct system prompt for each application phase.

Public interface
----------------
  make_main_chat()     -> gemini Chat  (smart dialogue, JSON-annotated)
  make_store_chat()    -> gemini Chat  (intro + info extraction)
  make_silence_chat()  -> gemini Chat  (silence / unclear response)
  parse_llm_response(raw) -> (spoken_text, mode, answers, keywords)
"""

import json
import time

from google import genai
from google.genai import errors as genai_errors

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_MAIN,
    GEMINI_MODEL_STORE,
    GEMINI_MODEL_SILENCE,
)

# ── One shared Gemini client (reused across all chat sessions) ────────────────
_client = genai.Client(api_key=GEMINI_API_KEY)


# ── System prompts ────────────────────────────────────────────────────────────

_MAIN_SYSTEM_PROMPT = (
    "You are Mini, a friendly robot talking with an aphasia patient. "
    "Each reply: 1-2 simple sentences, ONE question, plain words. "
    "Gradually learn: name, age, mood, hobby, favourite food, family size. "
    "Never rush or identify as AI. End warmly if patient says goodbye.\n\n"
    "After EVERY reply output exactly one JSON line (no markdown):\n"
    # Closed: small fixed answer set (yes/no, feelings, choices)
    '{"type":"closed","answers":{"yes":["yes","yeah","yep"],"no":["no","nope","nah"]}}\n'
    # Open: free answer (name, number, food, hobby)
    '{"type":"open","keywords":["word1","word2","word3"]}\n'
    "closed → answers MUST include ALL expected options (always include both yes AND no "
    "for yes/no questions). Each option maps to STT sound-alikes. "
    "open → 3-6 words the patient is likely to say."
)

_STORE_SYSTEM_PROMPT = (
    "You are Mini, a friendly robot talking with an aphasia patient. "
    "Use short, simple sentences. One question at a time. "
    "Topics to cover: how their day was, how their family is doing. "
    "Never rush or identify as AI."
)

_SILENCE_SYSTEM_PROMPT = (
    "You are Mini, a friendly robot. "
    "The patient has aphasia and has been silent or unclear. "
    "Respond in 1-2 very short, gentle sentences. Never rush or identify as AI."
)


# ── Factory functions ─────────────────────────────────────────────────────────

def make_main_chat():
    """
    Chat session for the main smart-dialogue loop.
    Uses the pro model with JSON-annotated turns.
    """
    return _client.chats.create(
        model=GEMINI_MODEL_MAIN,
        config=genai.types.GenerateContentConfig(
            system_instruction=_MAIN_SYSTEM_PROMPT
        ),
    )


def make_store_chat():
    """
    Chat session for the introductory information-gathering loop.
    Uses the flash model (cheaper; no JSON annotation needed).
    """
    return _client.chats.create(
        model=GEMINI_MODEL_STORE,
        config=genai.types.GenerateContentConfig(
            system_instruction=_STORE_SYSTEM_PROMPT
        ),
    )


def make_silence_chat():
    """
    Chat session used only for generating silence / unclear-input responses.
    Uses the lite model.
    """
    return _client.chats.create(
        model=GEMINI_MODEL_SILENCE,
        config=genai.types.GenerateContentConfig(
            system_instruction=_SILENCE_SYSTEM_PROMPT
        ),
    )


# ── Response parser ───────────────────────────────────────────────────────────

def parse_llm_response(raw: str):
    """
    Split a main-chat response into spoken text + structured JSON metadata.

    Returns
    -------
    spoken_text : str
    mode        : "closed" | "open" | "fallback"
    answers     : dict | None   (for closed mode  → rie.dialogue.ask)
    keywords    : list | None   (for open mode    → keyword stream)
    """
    lines       = raw.strip().splitlines()
    json_line   = None
    spoken_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            json_line = stripped
        else:
            spoken_lines.append(line)

    spoken_text = "\n".join(spoken_lines).strip()

    if not json_line:
        return spoken_text, "fallback", None, None

    try:
        data     = json.loads(json_line)
        mode     = data.get("type", "fallback")
        answers  = data.get("answers")   # closed
        keywords = data.get("keywords")  # open
        return spoken_text, mode, answers, keywords
    except json.JSONDecodeError:
        return spoken_text, "fallback", None, None

# ── Retry-aware send_message ──────────────────────────────────────────────────

def send_message(chat, text: str, max_retries: int = 4, base_delay: float = 3.0) -> str:
    """
    Drop-in wrapper around chat.send_message that retries on 503 / rate-limit
    errors with exponential back-off.

    Returns the response text, or raises after max_retries exhausted.
    """
    delay = base_delay
    for attempt in range(1, max_retries + 1):
        try:
            response = chat.send_message(text)
            return response.text
        except genai_errors.ServerError as exc:
            if attempt == max_retries:
                raise
            print(f"[LLM] 503 on attempt {attempt}/{max_retries} – retrying in {delay:.0f}s ({exc})")
            time.sleep(delay)
            delay *= 2   # exponential back-off: 3s, 6s, 12s, 24s
        except Exception:
            raise   # non-503 errors bubble up immediately


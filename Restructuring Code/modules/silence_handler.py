from modules.llm_client import send_message
"""
modules/silence_handler.py
===========================
Handles two "non-answer" scenarios gracefully:

  1. Silence   – the patient didn't speak within the timeout window.
  2. Unclear   – the STT returned something too short or garbled to use.

Public interface
----------------
  is_unclear(text)                       -> bool
  silence_response(last_topic)           -> str  (local, no LLM call)
  unclear_response(chat, user_input, last_topic) -> str  (LLM call)
"""

import random
from typing import Optional

from config import MIN_MEANINGFUL_LENGTH, MAX_CONSECUTIVE_SILENCES


# ── Canned silence phrases (no LLM cost) ─────────────────────────────────────

_SILENCE_PHRASES = [
    "Take your time.",
    "I'm here, no rush.",
    "Whenever you're ready.",
    "We can go at your pace.",
    "There's no hurry at all.",
]

_TOPIC_NUDGES = [
    "something at home",
    "a hobby",
    "family",
    "food",
    "a place you like",
    "a person you know",
    "something you enjoy",
]


# ── Public helpers ────────────────────────────────────────────────────────────

# Short words that are valid answers and must never be flagged as unclear
_VALID_SHORT_WORDS = {
    "no", "yes", "ok", "hi", "bye", "go", "up", "do", "me", "my",
    "he", "we", "it", "tv", "on", "in", "at", "ah", "oh", "ow",
}


def is_unclear(text: str) -> bool:
    """
    Return True if `text` looks like garbled or incomplete speech.
    Criteria:
      - Shorter than MIN_MEANINGFUL_LENGTH characters AND not a known short word
      - Contains no alphabetic characters (e.g. "???", "...")
    """
    stripped = text.strip().lower()
    # Always accept known short valid words
    if stripped in _VALID_SHORT_WORDS:
        return False
    if len(stripped) < MIN_MEANINGFUL_LENGTH:
        return True
    if not any(c.isalpha() for c in stripped):
        return True
    return False


def silence_response(last_topic: Optional[str], consecutive: int = 0) -> str:
    """
    Return a local (no LLM) calm presence phrase.

    After MAX_CONSECUTIVE_SILENCES silences in a row a longer comfort
    phrase is returned and the counter is implicitly reset by the caller.
    """
    if consecutive >= MAX_CONSECUTIVE_SILENCES:
        return "I am right here whenever you feel like talking."

    phrase = random.choice(_SILENCE_PHRASES)
    if last_topic:
        nudge = random.choice(_TOPIC_NUDGES)
        return f"{phrase} Were you thinking about {nudge}, maybe?"
    return phrase


def unclear_response(chat, user_input: str, last_topic: Optional[str]) -> str:
    """
    Ask the LLM to produce a warm acknowledgment + a single yes/no rescue
    question, using the last known topic for context.

    `chat` is an active Gemini chat session.
    """
    context = f" The last topic we discussed was: {last_topic}." if last_topic else ""
    prompt = (
        f"The patient said something unclear: '{user_input}'.{context} "
        "Acknowledge their attempt warmly in one short sentence, "
        "then ask ONE simple yes-or-no question to help them clarify. "
        "Never ask them to simply repeat what they said."
    )
    response = send_message(chat, prompt)
    return response#.text

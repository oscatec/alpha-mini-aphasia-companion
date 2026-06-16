"""
modules/dialogue.py
====================
All robot speech and listening primitives in one place.

Every function is an inlineCallbacks coroutine that takes `session` as its
first argument, so callers never interact with WAMP directly.

Public interface
----------------
  say(session, text)
  say_animated(session, text)
  ask_closed(session, question, answers) -> str | None
  ask_open_keyword(session, question, keywords) -> str | None
  fallback_stt(session, prompt_text)    -> str | None
  ask_one_turn(session, spoken, mode, answers, keywords) -> str | None
  wait_for_speech(session)              -> str   (raw STT result)
"""

from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.util import sleep

from config import (
    KEYWORD_LISTEN_SECONDS,
    FALLBACK_LISTEN_SECONDS,
    ASR_MAX_ATTEMPTS,
    SILENCE_TICKS,
)

# ── Self-hearing guard ────────────────────────────────────────────────────────
# Set to True while the robot is speaking so STT callbacks discard audio.
_is_speaking = False


# ── Speech output ─────────────────────────────────────────────────────────────

@inlineCallbacks
def say(session, text: str):
    """Say `text` without movement animation."""
    global _is_speaking
    _is_speaking = True
    try:
        yield session.call("rie.dialogue.say", text=text)
    finally:
        _is_speaking = False


@inlineCallbacks
def say_animated(session, text: str):
    """Say `text` with natural body animation."""
    global _is_speaking
    _is_speaking = True
    try:
        yield session.call("rie.dialogue.say_animated", text=text)
    finally:
        _is_speaking = False


# ── Closed-answer listening (rie.dialogue.ask) ────────────────────────────────

def _sanitise_answers(answers: dict) -> dict:
    """
    rie.dialogue.ask requires each variant to be a single plain word.
    Strip punctuation and drop any multi-word phrases so the broker
    never receives something it cannot handle.
    """
    import re
    cleaned = {}
    for key, variants in answers.items():
        safe = []
        for v in variants:
            # Keep only the first word, lowercase, letters only
            word = re.sub(r"[^a-z]", "", v.strip().lower().split()[0]) if v.strip() else ""
            if word and word not in safe:
                safe.append(word)
        if safe:
            cleaned[key] = safe
    return cleaned


def _validate_answers(answers) -> bool:
    """
    Check that `answers` is safe to pass to rie.dialogue.ask.
    Must be a non-empty dict where every value is a non-empty list of strings.
    """
    if not isinstance(answers, dict) or not answers:
        return False
    for key, variants in answers.items():
        if not isinstance(key, str) or not key.strip():
            return False
        if not isinstance(variants, list) or not variants:
            return False
        if not all(isinstance(v, str) and v.strip() for v in variants):
            return False
    return True


@inlineCallbacks
def ask_closed(session, question_text: str, answers: dict, max_attempts: int = None):
    """
    Say the question and use rie.dialogue.ask to match one of the known answers.
    Returns the matched answer key, or None on exhausted attempts.

    Falls back to raw STT if `answers` fails validation, to avoid dropping
    the WAMP transport with a malformed call.

    `answers` format:  {"yes": ["yes","yeah","yep"], "no": ["no","nope","nah"]}
    """
    if max_attempts is None:
        max_attempts = ASR_MAX_ATTEMPTS

    answers = _sanitise_answers(answers)
    if not _validate_answers(answers):
        print(f"[Dialogue] ask_closed: invalid answers dict after sanitise {answers!r} – falling back to STT")
        result = yield fallback_stt(session, question_text)
        return result

    try:
        result = yield session.call(
            "rie.dialogue.ask",
            question=question_text,
            answers=answers,
            max_attempts=max_attempts,
            language="en",
        )
        return result
    except Exception as exc:
        print(f"[Dialogue] rie.dialogue.ask failed ({exc}) – falling back to STT")
        result = yield fallback_stt(session, question_text)
        return result


# ── Open-answer listening (keyword stream) ────────────────────────────────────

@inlineCallbacks
def ask_open_keyword(session, question_text: str, keywords: list,
                     listen_seconds: int = None):
    """
    Say the question, open a keyword stream, and return the first recognised
    keyword within `listen_seconds`.  Returns None on timeout.
    """
    if listen_seconds is None:
        listen_seconds = KEYWORD_LISTEN_SECONDS

    yield say_animated(session, question_text)

    heard = {"value": None}

    def on_keyword(frame):
        body      = frame.get("data", {}).get("body", {})
        certainty = body.get("certainty", 0)
        word      = body.get("keyword", "")
        if certainty > 0.35 and word and heard["value"] is None:
            heard["value"] = word

    yield session.call("rie.dialogue.keyword.clear")
    yield session.call("rie.dialogue.keyword.add", keywords=keywords)
    yield session.subscribe(on_keyword, "rie.dialogue.keyword.stream")
    yield session.call("rie.dialogue.keyword.stream")

    elapsed = 0
    while heard["value"] is None and elapsed < listen_seconds:
        yield sleep(0.5)
        elapsed += 0.5

    yield session.call("rie.dialogue.keyword.close")
    yield session.call("rie.dialogue.keyword.clear")

    return heard["value"]


# ── Keyword-listen only (no TTS, used for closed→open fallback) ─────────────

@inlineCallbacks
def _listen_keyword_only(session, keywords: list, listen_seconds: int = None):
    """
    Open a keyword stream and return the first match within listen_seconds.
    Does NOT say anything — used when the question was already spoken.
    """
    if listen_seconds is None:
        listen_seconds = KEYWORD_LISTEN_SECONDS

    heard = {"value": None}

    def on_keyword(frame):
        body      = frame.get("data", {}).get("body", {})
        certainty = body.get("certainty", 0)
        word      = body.get("keyword", "")
        if certainty > 0.35 and word and heard["value"] is None:
            heard["value"] = word

    yield session.call("rie.dialogue.keyword.clear")
    yield session.call("rie.dialogue.keyword.add", keywords=keywords)
    yield session.subscribe(on_keyword, "rie.dialogue.keyword.stream")
    yield session.call("rie.dialogue.keyword.stream")

    elapsed = 0
    while heard["value"] is None and elapsed < listen_seconds:
        yield sleep(0.5)
        elapsed += 0.5

    yield session.call("rie.dialogue.keyword.close")
    yield session.call("rie.dialogue.keyword.clear")

    return heard["value"]


# ── Raw STT fallback ──────────────────────────────────────────────────────────

@inlineCallbacks
def fallback_stt(session, prompt_text: str, listen_seconds: int = None):
    """
    Say `prompt_text`, then listen via raw STT for up to `listen_seconds`.
    Returns the transcribed text, or None if nothing was heard.
    """
    if listen_seconds is None:
        listen_seconds = FALLBACK_LISTEN_SECONDS

    yield say_animated(session, prompt_text)
    yield sleep(0.5)

    heard = {"value": ""}

    def on_stt(frame):
        if _is_speaking:
            return  # discard — robot is talking
        body = frame.get("data", {}).get("body", {})
        if body.get("final") and body.get("text"):
            heard["value"] = body["text"].strip()

    yield session.subscribe(on_stt, "rie.dialogue.stt.stream")
    yield session.call("rie.dialogue.stt.stream")
    yield sleep(listen_seconds)
    yield session.call("rie.dialogue.stt.close")

    return heard["value"] or None


# ── Smart turn dispatcher ─────────────────────────────────────────────────────

@inlineCallbacks
def ask_one_turn(session, spoken_text: str, mode: str,
                 answers: dict, keywords: list):
    """
    Dispatch a single dialogue turn according to the LLM-supplied `mode`.

    mode == "closed" → rie.dialogue.ask (structured answer matching)
                       If no match, re-ask in open mode using the answer
                       keys as keywords so the patient can say anything
                       loosely related and still be understood.
    mode == "open"   → keyword stream (+ STT fallback on silence)
    else             → raw STT fallback

    Returns the patient's reply as a string, or None.
    """
    # if mode == "closed" and answers:
    #     result = yield ask_closed(session, spoken_text, answers)

    #     # Matched – return immediately
    #     if result:
    #         return result

    #     # No match: say a gentle bridge, then listen with open keyword mode.
    #     # Use _listen_keyword_only so the question is NOT spoken a third time.
    #     open_keywords = list(answers.keys())
    #     yield say(session, "Take your time, just say what feels right.")
    #     result = yield _listen_keyword_only(session, open_keywords)
    #     if result:
    #         return result

    #     # Still nothing – last resort raw STT, question already spoken twice
    #     result = yield fallback_stt(
    #         session,
    #         "Let's try once more. " + spoken_text,
    #     )
    #     return result

    if (mode == "open" or mode == "closed") and (keywords):
        result = yield ask_open_keyword(session, spoken_text, keywords)
        if result:
            return result
        # Keyword module heard nothing – fall back to raw STT
        result = yield fallback_stt(
            session,
            "Sorry, I didn't catch that. Could you say it again?",
        )
        return result

    else:
        result = yield fallback_stt(session, spoken_text)
        return result


# ── Continuous STT listener (used by store_combo / silence_chat flow) ─────────

@inlineCallbacks
def wait_for_speech(session, timeout_ticks: int = None):
    """
    Open the STT stream and poll until a final transcript arrives or
    `timeout_ticks` × 0.5 s elapse.

    Returns (transcript: str, timed_out: bool).
    """
    if timeout_ticks is None:
        timeout_ticks = SILENCE_TICKS

    finish   = {"done": False}
    query    = {"text": ""}

    def on_stt(frame):
        if _is_speaking:
            return  # discard — robot is talking
        body = frame.get("data", {}).get("body", {})
        if body.get("final") and body.get("text"):
            query["text"]  = str(body["text"]).strip()
            finish["done"] = True

    yield session.subscribe(on_stt, "rie.dialogue.stt.stream")
    yield session.call("rie.dialogue.stt.stream")

    ticks = 0
    while not finish["done"]:
        yield sleep(0.5)
        ticks += 1
        if ticks >= timeout_ticks:
            break

    yield session.call("rie.dialogue.stt.close")
    yield sleep(0.3)

    timed_out = not finish["done"]
    return query["text"], timed_out
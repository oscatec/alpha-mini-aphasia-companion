"""
HRI - PwA Introductory Conversation Robot
Uses Gemini for LLM responses and the Alpha Mini robot's
rie.dialogue.ask (smart question) and rie.dialogue.keyword modules.

STRATEGY:
  - Gemini generates each question AND a small structured answer set.
  - If the question has a known/closed answer space  ->  rie.dialogue.ask
  - If the question is open-ended (e.g. "what is your name?") ->  keyword
    stream, listening for any spoken word long enough to be a name/phrase.
  - Gemini then receives the recognised answer and generates the next turn.
"""

import json
import re

from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.util import sleep
from google import genai

# Gemini setup and persistent chat session (maintains conversation history)
API_KEY = "REDACTED_API_KEY"  # Insert your Gemini API key here.
chatbot = genai.Client(api_key=API_KEY)

SYSTEM_PROMPT = (
    "Your name is Mini. You are a warm, friendly social robot. "
    "Your role is to have a short introductory conversation with a patient who has aphasia. "
    "Rules you must always follow:\n"
    "- Keep every response/question to 1-2 short, simple sentences.\n"
    "- Ask only ONE question per turn. Wait for the answer before asking the next.\n"
    "- Use simple, everyday vocabulary.\n"
    "- Be patient, encouraging, and never rush the patient.\n"
    "- During the conversation, naturally gather: the patient's name, how they are feeling "
    "today, their age, one hobby or interest, favourite meals, and family size.\n"
    "- If you do not understand what the patient said, gently ask them to repeat it.\n"
    "- Never refer to yourself as an AI or language model.\n"
    "- When the patient says goodbye or wants to stop, end the conversation warmly.\n\n"
    "IMPORTANT - after every turn you MUST also output a JSON block on its own line "
    "in exactly this format (no markdown, no extra keys):\n"
    '{"type": "closed", "answers": {"yes": ["yes","yeah","yep"], "no": ["no","nope","nah"]}}\n'
    "OR\n"
    '{"type": "open", "keywords": ["hello","hi","morning"]}\n\n'
    "Rules for the JSON:\n"
    '- Use "closed" when the expected answer is one of a small set of options '
    "(e.g. yes/no, colour choices, feeling words like good/bad/tired).\n"
    '  The "answers" dict maps each answer option to a list of phonetically similar words '
    "the STT might produce.\n"
    '- Use "open" when the answer is a free word (name, number, food, hobby). '
    '  List 3-6 likely/common words the patient might say as "keywords"; '
    "these seed the keyword listener so it catches partial matches.\n"
    "- Output the JSON on a SEPARATE LINE after the spoken text, "
    "so the robot can split them apart.\n"
    "- Do NOT wrap the JSON in ```code blocks```."
)

chat = chatbot.chats.create(
    model="gemini-2.5-pro",
    config=genai.types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
)

EXIT_WORDS = {"quit", "exit", "goodbye", "bye", "stop", ":q"}

# Helpers for parsing Gemini's response and managing the different interaction modes (closed vs open questions)

def parse_llm_response(raw: str):
    """
    Split Gemini's output into:
      spoken_text  - what the robot should say aloud
      mode         - "closed" | "open" | "fallback"
      answers      - dict for rie.dialogue.ask  (closed only)
      keywords     - list for keyword module     (open only)
    """
    # Find the last line that looks like a JSON object
    lines = raw.strip().splitlines()
    json_line = None
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
        data = json.loads(json_line)
        mode = data.get("type", "fallback")
        answers = data.get("answers", None)   # for closed
        keywords = data.get("keywords", None) # for open
        return spoken_text, mode, answers, keywords
    except json.JSONDecodeError:
        return spoken_text, "fallback", None, None


# Interaction primitives

@inlineCallbacks
def ask_closed(session, question_text, answers: dict, max_attempts=4):
    """
    Say the question and use rie.dialogue.ask to match one of the known answers.
    Returns the matched answer key, or None on failure.
    """
    result = yield session.call(
        "rie.dialogue.ask",
        question=question_text,
        answers=answers,
        max_attempts=max_attempts,
        language="en",
    )
    return result  # one of the keys in `answers`, or None


@inlineCallbacks
def ask_open_keyword(session, question_text, keywords: list, listen_seconds=8):
    """
    Say the question, then open a keyword stream and return whatever is heard
    within `listen_seconds`.  Falls back to the raw keyword text if no
    keyword is matched.
    """
    # Say the question first (animated so the robot moves naturally)
    yield session.call("rie.dialogue.say_animated", text=question_text)

    # Prepare shared state
    heard = {"value": None}

    def on_keyword(frame):
        body = frame.get("data", {}).get("body", {})
        certainty = body.get("certainty", 0)
        word = body.get("keyword", "")
        if certainty > 0.35 and word and heard["value"] is None:
            heard["value"] = word

    # Add keywords and start the stream
    yield session.call("rie.dialogue.keyword.clear")
    yield session.call("rie.dialogue.keyword.add", keywords=keywords)
    yield session.subscribe(on_keyword, "rie.dialogue.keyword.stream")
    yield session.call("rie.dialogue.keyword.stream")

    # Wait for a keyword or timeout
    elapsed = 0
    while heard["value"] is None and elapsed < listen_seconds:
        yield sleep(0.5)
        elapsed += 0.5

    # Close the stream and clean up
    yield session.call("rie.dialogue.keyword.close")
    yield session.call("rie.dialogue.keyword.clear")

    return heard["value"]  # may be None if nothing was heard


@inlineCallbacks
def fallback_stt(session, prompt_text, listen_seconds=6):
    """
    Last-resort: use rie.dialogue.stt.read for a fixed window.
    """
    yield session.call("rie.dialogue.say_animated", text=prompt_text)
    yield sleep(0.5)

    heard = {"value": ""}

    def on_stt(frame):
        body = frame.get("data", {}).get("body", {})
        if body.get("final") and body.get("text"):
            heard["value"] = body["text"].strip()

    yield session.subscribe(on_stt, "rie.dialogue.stt.stream")
    yield session.call("rie.dialogue.stt.stream")
    yield sleep(listen_seconds)
    yield session.call("rie.dialogue.stt.close")

    return heard["value"] or None


# Main session
@inlineCallbacks
def main(session, details):
    # Configure language and posture
    yield session.call("rie.dialogue.config.language", lang="en")
    yield session.call("rom.optional.behavior.play", name="BlocklyStand")

    # Opening greeting 
    opening_raw = chat.send_message(
        "Begin the conversation with a warm, brief greeting and your first question. "
        "Remember to append the JSON block."
    )
    spoken, mode, answers, keywords = parse_llm_response(opening_raw.text)
    print(f"[Robot] {spoken}  |  mode={mode}")

    dialogue = True

    # Ask the first question before entering the loop
    patient_reply = yield _ask_one_turn(session, spoken, mode, answers, keywords)
    if patient_reply and patient_reply.lower() in EXIT_WORDS:
        dialogue = False

    # Conversation loop 
    while dialogue:
        if patient_reply is None:
            patient_reply = "I didn't catch that"

        print(f"[Patient] '{patient_reply}'")

        # Send patient's reply to Gemini
        llm_raw = chat.send_message(patient_reply)
        spoken, mode, answers, keywords = parse_llm_response(llm_raw.text)
        print(f"[Robot] {spoken}  |  mode={mode}")

        # Check for farewell in the robot's response (Gemini decided to end)
        if any(w in spoken.lower() for w in ("goodbye", "bye", "take care", "farewell")):
            yield session.call("rie.dialogue.say_animated", text=spoken)
            dialogue = False
            break

        patient_reply = yield _ask_one_turn(session, spoken, mode, answers, keywords)

        if patient_reply and patient_reply.lower() in EXIT_WORDS:
            farewell_raw = chat.send_message(
                "The patient wants to end the conversation. "
                "Say a warm one-sentence goodbye. Append the JSON block."
            )
            farewell_text, *_ = parse_llm_response(farewell_raw.text)
            yield session.call("rie.dialogue.say_animated", text=farewell_text)
            dialogue = False

    yield session.call("rom.optional.behavior.play", name="BlocklyCrouch")
    session.leave()


@inlineCallbacks
def _ask_one_turn(session, spoken_text, mode, answers, keywords):
    """
    Dispatch a single turn: say the text and listen using the appropriate module.
    Returns the patient's reply as a string, or None.
    """
    if mode == "closed" and answers:
        # rie.dialogue.ask handles saying the question internally
        result = yield ask_closed(session, spoken_text, answers)
        return result  # key string or None

    elif mode == "open" and keywords:
        result = yield ask_open_keyword(session, spoken_text, keywords)
        if result:
            return result
        # If keyword module heard nothing, fall back to STT
        result = yield fallback_stt(
            session,
            "Sorry, I didn't catch that. Could you say it again?",
        )
        return result

    else:
        # Generic fallback: say the text + raw STT
        result = yield fallback_stt(session, spoken_text)
        return result


# WAMP connection 
wamp = Component(
    transports=[{
        "url": "ws://wamp.robotsindeklas.nl",
        "serializers": ["msgpack"],
        "max_retries": 0,
    }],
    realm="rie.REDACTED_REALM",
)

wamp.on_join(main)

if __name__ == "__main__":
    run([wamp])
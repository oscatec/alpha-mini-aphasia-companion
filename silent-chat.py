"""
Type your replies at the "You:" prompt. Type 'quit' or 'bye' to exit.

Silent dialogue features:
  - Silence detection: if no input arrives within SILENCE_TICKS seconds,
    the robot emits a calm backchannel instead of treating it as an error.
  - Two-tier response:
      * Pure silence  → calming presence phrase ("Take your time.")
      * Unclear input → warm acknowledgment + binary rescue question
"""

import threading
import random
from google import genai
import random
from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.util import sleep
from google import genai
from alpha_mini_rug import perform_movement


API_KEY = "REDACTED_API_KEY"
chatbot = genai.Client(api_key=API_KEY)



# How many seconds of silence before the robot gives a gentle nudge.
# Aphasia patients often need extra time, so this is intentionally long.
SILENCE_TICKS = 16  

# Played in silence 
SILENCE_PHRASES = [
    "Take your time.",
    "I'm here, no rush.",
    "Whenever you're ready.",
    "We can go at your pace.",
]

#list of phrases to use when the patient's input is unclear 
UNCLEAR_PHRASES = ["something at home", "a hobby", "family", "food", "a place", "a person", "something you like"]


# ── Helpers to detect "unclear" input ────────────────────────────────────────
MIN_MEANINGFUL_LENGTH = 3  # characters; anything shorter is treated as unclear

def is_unclear(text: str) -> bool:
    """Return True if the input looks like garbled or very incomplete speech."""
    stripped = text.strip()
    if len(stripped) < MIN_MEANINGFUL_LENGTH:
        return True
    # Purely non-alphabetic (e.g. "???" or "...") also counts as unclear
    if not any(c.isalpha() for c in stripped):
        return True
    return False


# ── Silence-aware input ───────────────────────────────────────────────────────
def input_with_timeout(prompt: str, timeout: float):
    """
    Show `prompt`, wait up to `timeout` seconds for the user to type.
    Returns (user_text, timed_out).
    """
    result = [None]
    timed_out_flag = [False]

    def read():
        try:
            result[0] = input(prompt)
        except EOFError:
            result[0] = None

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        # Timeout: thread is still blocked on input().
        # We can't cancel it cleanly in Python, so we mark the flag and move on.
        timed_out_flag[0] = True
        print()  # newline so the robot's response starts on a fresh line
        return "", True

    return result[0], False


# Silent-dialogue  
def handle_silence(chat, last_topic: str | None) -> str:
    """
    Emit a calm backchannel for a silence event.
    Alternates between a random presence phrase and a gentle topic nudge
    so repeated silences don't sound like a broken record.
    """
    phrase = random.choice(SILENCE_PHRASES)
    if last_topic:
        nudge = f"{phrase} Were you thinking about {random.choice(UNCLEAR_PHRASES)}, maybe?"
    else:
        nudge = phrase
    return nudge


def handle_unclear(chat, user_input: str, last_topic: str | None) -> str:
    """
    Ask the LLM to produce a warm acknowledgment + binary rescue question,
    using the last known topic for context when available.
    """
    context = f" The last topic we discussed was: {last_topic}." if last_topic else ""
    prompt = (
        f"The patient said something unclear: '{user_input}'.{context} "
        "Acknowledge their attempt warmly in one short sentence, "
        "then ask ONE simple yes-or-no question to help them clarify. "
        "Never ask them to repeat verbatim."
    )
    response = chat.send_message(prompt)
    return response.text


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Your name is Mini. You are a warm, friendly social robot. "
    "Your role is to have a short guess the word game with a patient who has aphasia (PwA). "
    "Game flow:"
    "The robot picks a word and guides the patient through up to 4 cue levels:"
        "Level 1 — Semantic cue   : 'It's something you write with.'"
        "Level 2 — Phonemic cue   : 'It starts with the sound 'p'.'"
        "Level 3 — Rhyming cue    : 'It rhymes with 'hen'.'"
        "Level 4 — Sentence fill  : 'I use a ...... to write on paper.'"
    "Correct guess at any level → celebration + new word."
    "After all 4 levels with no correct guess → reveal the word and ask another one."
    "Rules you must always follow:\n"
    "- Keep every response to 1-2 short, simple sentences.\n"
    "- Ask only ONE question per turn. Wait for the answer before asking the next.\n"
    "- Use simple, everyday vocabulary. Avoid complex words.\n"
    "- Be patient, encouraging, and never rush the patient.\n"
    "- Ask questions to help the patient guess the word.\n"
    "- If you do not understand what the patient said, don't ask them to repeat since it can be frustrating."
    "- Never refer to yourself as an AI or language model.\n"
    "- When the patient says goodbye or wants to stop, end the conversation warmly.\n"
    "- Never interpret silence or a slow response as failure or disinterest."
)

EXIT_WORDS = ("quit", "exit", "goodbye", "bye", "stop", ":q")

# ── Main ──────────────────────────────────────────────────────────────────────
chat = chatbot.chats.create(
    model="gemini-2.5-flash-lite",
    config=genai.types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
)
finish_dialogue = False
query = ""


def asr(frames):
    """Callback invoked by the robot's STT stream on each recognised utterance."""
    global finish_dialogue, query
    if frames["data"]["body"]["final"]:
        query = str(frames["data"]["body"]["text"]).strip()
        print(f"[ASR] '{query}'")
        finish_dialogue = True


# ── Main robot session ────────────────────────────────────────────────────────
@inlineCallbacks
def main(session, details):
    global finish_dialogue, query

    yield session.call("rie.dialogue.config.language", lang="en")
    yield session.call("rom.optional.behavior.play", name="BlocklyStand")

    # Opening: explain the game and give the first cue
    opening = chat.send_message(
        "Welcome the patient warmly, briefly explain the game "
        "(you will give clues and they try to guess the word), "
        "then give the first Level 1 semantic cue for your chosen word."
    )
    print(f"[Robot] {opening.text}")
    yield session.call("rie.dialogue.say", text=opening.text)

    yield session.subscribe(asr, "rie.dialogue.stt.stream")

    last_topic = None
    consecutive_silences: int = 0
    dialogue = True


    while dialogue:
        # Reset and start listening
        finish_dialogue = False
        query = ""
        yield session.call("rie.dialogue.stt.stream")

        # Nod yes
        perform_movement(session,
        frames=[{"time": 400, "data": {"body.head.pitch": 0.1}},
        {"time": 1200, "data": {"body.head.pitch": -0.1}},
        {"time": 2000, "data": {"body.head.pitch": 0.1}},
        {"time": 2400, "data": {"body.head.pitch": 0.0}}],
        force=True)
       


        # ── Wait for ASR with silence detection ──────────────────────────
        ticks_waited = 0
        while not finish_dialogue:
            yield sleep(0.5)
            ticks_waited += 1
            if ticks_waited >= SILENCE_TICKS:
                break  # treat as silence

         
        yield session.call("rie.dialogue.stt.close")
        yield sleep(0.3)

        # ── Case 1: Silence (timeout) ─────────────────────────────────────
        if not finish_dialogue:
            consecutive_silences += 1
            if consecutive_silences >= 3:
                reply = "I am right here whenever you feel like trying."
                consecutive_silences = 0
            else:
                reply = handle_silence(chat, last_topic)

            print(f"[Robot] {reply}")
            yield session.call("rie.dialogue.say", text=reply)
            yield sleep(1)
            continue

        # Real input arrived
        consecutive_silences = 0
        print(f"[User] '{query}'")

        # ── Case 2: Exit intent ───────────────────────────────────────────
        if query.lower() in EXIT_WORDS:
            dialogue = False
            farewell = chat.send_message(
                "The patient wants to stop. Say a warm one-sentence goodbye and well done."
            )
            print(f"[Robot] {farewell.text}")
            yield session.call("rie.dialogue.say", text=farewell.text)

        # ── Case 3: Unclear / garbled STT output ─────────────────────────
        elif is_unclear(query):
            reply = handle_unclear(chat,query, last_topic)
            print(f"[Robot] {reply}")
            yield session.call("rie.dialogue.say", text=reply)

        # ── Case 4: Normal intelligible guess ────────────────────────────
        elif query:
            response = chat.send_message(query)
            print(f"[Robot] {response.text}")
            yield session.call("rie.dialogue.say", text=response.text)
            # Quietly update topic for personalised silence nudges
            last_topic = last_topic

        # Empty string fallback
        else:
            reply = "Take your time, I am listening."
            print(f"[Robot] {reply}")
            yield session.call("rie.dialogue.say", text=reply)

        yield sleep(1)

    yield session.call("rom.optional.behavior.play", name="BlocklyCrouch")
    session.leave()


# ── WAMP connection ───────────────────────────────────────────────────────────
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

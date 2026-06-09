import json
import os
import random
import re
import time

from autobahn.twisted.component import Component, run
from autobahn.twisted.util import sleep
from twisted.internet.defer import inlineCallbacks
from google import genai
from google.genai import errors as genai_errors
from alpha_mini_rug import perform_movement


# ── Gemini setup ──────────────────────────────────────────────────────────────
API_KEY = "REDACTED_API_KEY"
chatbot = genai.Client(api_key=API_KEY)

INTRO_SYSTEM_PROMPT = (
    "Your name is Mini. You are a warm, friendly social robot. "
    "Your role is to have a short introductory conversation with a patient who has aphasia. "
    "Keep every response to short, simple sentences. "
    "After the introduction, chat naturally with the user about their day and family."
)

intro_chat = chatbot.chats.create(
    model="gemini-2.5-flash",
    config=genai.types.GenerateContentConfig(system_instruction=INTRO_SYSTEM_PROMPT),
)

# ── Constants ─────────────────────────────────────────────────────────────────
EXIT_WORDS = {"quit", "exit", "goodbye", "bye", "stop", ":q"}
YES_WORDS   = {"yes", "yeah", "yep", "sure", "ok", "okay", "please", "let's", "lets"}
INFO_DB     = "sample.json"
SILENCE_TICKS = 16  # half-second ticks → 8 seconds before treating as silence

SILENCE_PHRASES = [
    "Take your time.",
    "I'm here, no rush.",
    "Whenever you're ready.",
    "We can go at your pace.",
]
UNCLEAR_HINTS = [
    "something at home", "a hobby", "family", "food",
    "a place", "a person", "something you like",
]

# ── Shared ASR state ──────────────────────────────────────────────────────────
is_speaking    = False
finish_dialogue = False
query          = ""
conversation   = []


# ── LLM retry wrapper ─────────────────────────────────────────────────────────
def send_with_retry(chat, text, max_retries=5, base_delay=3):
    """Returns the response text, or raises after max_retries exhausted."""
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


def say_animated(session, text: str):
    """Say `text` with natural body animation."""
    global _is_speaking
    _is_speaking = True
    try:
        yield session.call("rie.dialogue.say_animated", text=text)
    finally:
        _is_speaking = False


# ── ASR callback ──────────────────────────────────────────────────────────────
def asr(frames):
    global finish_dialogue, query, is_speaking
    if is_speaking:
        return
    if frames["data"]["body"]["final"]:
        text = str(frames["data"]["body"]["text"]).strip()
        if not text:
            return
        query = text
        print(f"[ASR] '{query}'")
        finish_dialogue = True


# ── JSON store / lookup (from store_convo_copy.py) ────────────────────────────
def clean_name(raw):
    """Strip punctuation/brackets, keep only letters and spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z\s]", "", raw)).strip()


def lookup_info(name):
    if not os.path.exists(INFO_DB) or os.path.getsize(INFO_DB) == 0:
        return None
    with open(INFO_DB, "r") as f:
        data = json.load(f)
    query = clean_name(name).lower()
    for person in data:
        stored = clean_name(person["name"]).lower()
        if stored == query or stored in query or query in stored:
            return person["information"]
    return None


def store_info(convo):
    text = send_with_retry(
        intro_chat,
        f"Take the following conversation and write a timeless summary of the user's "
        f"personal information (age, hobbies, family, etc). Do NOT include mood or "
        f"temporary feelings. The name is always present. Reply as: "
        f"name: summary\n\nConversation: {convo}"
    ).strip()
    print(f"[Store] {text}")

    parts = text.split(": ", 1)
    if len(parts) < 2:
        return

    name, info = clean_name(parts[0]).strip(), parts[1].strip()

    if os.path.exists(INFO_DB) and os.path.getsize(INFO_DB) > 0:
        with open(INFO_DB, "r") as f:
            data = json.load(f)
    else:
        data = []

    for entry in data:
        if entry["name"].lower() == name.lower():
            entry["information"] += f", {info}"
            break
    else:
        data.append({"name": name, "information": info})

    with open(INFO_DB, "w") as f:
        json.dump(data, f, indent=4)


# ── Topic extraction ─────────────────────────────────────────────────────────
def extract_topic(user_input):
    """Ask Gemini for a 1-2 word topic label for the user's message."""
    response = chatbot.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=(
            f"In one or two words, what topic is this message about: '{user_input}'? "
            "Reply with only the topic, nothing else."
        ),
    )
    return response.text.strip()


# ── Unclear / silence helpers (from project.py) ───────────────────────────────
def is_unclear(text):
    stripped = text.strip()
    return len(stripped) < 3 or not any(c.isalpha() for c in stripped)


def handle_silence(last_topic):
    phrase = random.choice(SILENCE_PHRASES)
    if last_topic:
        return f"{phrase} Were you thinking about {random.choice(UNCLEAR_HINTS)}, maybe?"
    return phrase


def handle_unclear(chat, user_input, last_topic):
    context = f" The last topic was: {last_topic}." if last_topic else ""
    prompt = (
        f"The patient said something unclear: '{user_input}'.{context} "
        "Acknowledge their attempt warmly in one short sentence, "
        "then ask ONE simple yes-or-no question to help them clarify. "
        "Never ask them to repeat verbatim."
    )
    return send_with_retry(chat, prompt)


# ── Listen with silence detection ─────────────────────────────────────────────
@inlineCallbacks
def listen_once(session, ticks=SILENCE_TICKS):
    global finish_dialogue, query
    finish_dialogue = False
    query = ""
    yield session.call("rie.dialogue.stt.stream")

    waited = 0
    while not finish_dialogue:
        yield sleep(0.5)
        waited += 1
        if waited >= ticks:
            break

    yield session.call("rie.dialogue.stt.close")
    yield sleep(0.3)
    return query, not finish_dialogue  # (text, timed_out)


# ── Say helper ────────────────────────────────────────────────────────────────
@inlineCallbacks
def say_animated(session, text):
    global is_speaking
    is_speaking = True
    yield session.call("rie.dialogue.say", text=text)
    yield sleep(0.3)
    is_speaking = False


# ── Main ──────────────────────────────────────────────────────────────────────
@inlineCallbacks
def main(session, details):
    global conversation

    yield session.call("rie.dialogue.config.language", lang="en")
    yield session.call("rom.optional.behavior.play", name="BlocklyStand")
    yield session.subscribe(asr, "rie.dialogue.stt.stream")

    # ── 1. Greet and ask for name ─────────────────────────────────────────────
    opening_text = send_with_retry(intro_chat, "Greet the patient warmly and ask for their name.")
    print(f"[Robot] {opening_text}")
    yield say_animated(session, opening_text)

    name_text, _ = yield listen_once(session)
    print(f"[User] '{name_text}'")

    target_name = clean_name(send_with_retry(
        intro_chat,
        f"Extract only the person's name from this message, no punctuation: {name_text}"
    ))
    print(f"[Extracted name] {target_name}")

    # ── 2. Known vs new patient ───────────────────────────────────────────────
    user_info = lookup_info(target_name)

    if user_info:
        perform_movement(
            session,
            frames=[
                {"time": 400,  "data": {"body.arms.right.upper.pitch": 0.8}},
                {"time": 900,  "data": {"body.arms.right.upper.pitch": 0.6}},
                {"time": 1400, "data": {"body.arms.right.upper.pitch": 0.8}},
                {"time": 1900, "data": {"body.arms.right.upper.pitch": 0.0}},
            ],
            force=True,
        )
        yield say_animated(session, f"Nice to see you again, {target_name}!")
        follow_up_text = send_with_retry(
            intro_chat,
            f"You remember this about the user: {user_info}. "
            "Mention ONE piece of it warmly and ask a follow-up question about it."
        )
    else:
        perform_movement(
            session,
            frames=[
                {"time": 400,  "data": {"body.arms.right.upper.pitch": 0.8}},
                {"time": 900,  "data": {"body.arms.right.upper.pitch": 0.6}},
                {"time": 1400, "data": {"body.arms.right.upper.pitch": 0.8}},
                {"time": 1900, "data": {"body.arms.right.upper.pitch": 0.0}},
            ],
            force=True,
        )
        yield say_animated(session, f"Nice to meet you, {target_name}!")
        follow_up_text = send_with_retry(
            intro_chat,
            "You have not met this person before. Ask them their age."
        )

    print(f"[Robot] {follow_up_text}")
    yield say_animated(session, follow_up_text)

    # ── 3. Main dialogue (with silence + unclear handling) ────────────────────
    last_topic           = None
    consecutive_silences = 0
    dialogue             = True
    turn_count           = 0

    while dialogue:
        perform_movement(
            session,
            frames=[
                {"time": 400,  "data": {"body.head.pitch":  0.1}},
                {"time": 1200, "data": {"body.head.pitch": -0.1}},
                {"time": 2000, "data": {"body.head.pitch":  0.1}},
                {"time": 2400, "data": {"body.head.pitch":  0.0}},
            ],
            force=True,
        )

        user_text, timed_out = yield listen_once(session)

        # Silence
        if timed_out:
            consecutive_silences += 1
            if consecutive_silences >= 3:
                reply = "I am right here whenever you feel like trying."
                consecutive_silences = 0
            else:
                reply = handle_silence(last_topic)
            print(f"[Robot] {reply}")
            yield say_animated(session, reply)
            yield sleep(1)
            continue

        consecutive_silences = 0
        print(f"[User] '{user_text}'")
        conversation.append(user_text)

        if user_text.lower() in EXIT_WORDS:
            dialogue = False
            farewell_text = send_with_retry(
                intro_chat,
                "The patient wants to end the conversation. say_animated a warm one-sentence goodbye."
            )
            print(f"[Robot] {farewell_text}")
            yield say_animated(session, farewell_text)

        elif is_unclear(user_text):
            last_topic = extract_topic(user_text)
            reply = handle_unclear(intro_chat, user_text, last_topic)
            print(f"[Robot] {reply}")
            yield say_animated(session, reply)

        elif user_text:
            last_topic = extract_topic(user_text)
            turn_count += 1
            response_text = send_with_retry(intro_chat, user_text)
            print(f"[Robot] {response_text}")
            yield say_animated(session, response_text)
            if turn_count >= 2:
                dialogue = False

        yield sleep(1)

    store_info(conversation)

    # ── 4. Offer word game ────────────────────────────────────────────────────
    yield say_animated(session, "Would you like to play a word guessing game with me? It is a lot of fun!")
    game_reply, _ = yield listen_once(session, ticks=10)
    print(f"[User] '{game_reply}'")

    if not any(w in game_reply.lower() for w in YES_WORDS):
        yield say_animated(session, "No problem! It was lovely chatting with you today.")
        yield session.call("rom.optional.behavior.play", name="BlocklyCrouch")
        session.leave()
        return

    # ── 5. Word game ──────────────────────────────────────────────────────────
    game_chat = chatbot.chats.create(
        model="gemini-2.5-flash-lite",
        config=genai.types.GenerateContentConfig(system_instruction=(
            "Your name is Mini. You are a warm, friendly social robot.\n"
            "You are playing a word guessing game with a patient who has aphasia.\n\n"
            "YOUR role: YOU secretly pick a word. YOU give clues. The PATIENT tries to guess.\n"
            "You already know the word. The patient does NOT know it — they are guessing.\n\n"
            "Give clues one level at a time, waiting for the patient's guess after each:\n"
            "  Level 1 — Semantic  : describe what the object is or does.\n"
            "  Level 2 — Phonemic  : give the starting sound.\n"
            "  Level 3 — Rhyming   : give a word it rhymes with.\n"
            "  Level 4 — Fill-in   : say a sentence with the word missing, e.g. 'I use a ...... to write.'\n\n"
            "If the patient guesses correctly at any level → celebrate warmly, then pick a new word.\n"
            "If all 4 levels pass with no correct guess → reveal the word, then pick a new word.\n\n"
            "Rules: 1-2 short simple sentences per turn. Be patient and encouraging.\n"
            "Never refer to yourself as an AI. End warmly when the patient says goodbye."
        )),
    )

    game_opening_text = send_with_retry(
        game_chat,
        "Start the game: briefly explain that you will give clues and the patient must guess the word, "
        "then give the Level 1 semantic clue for your chosen word."
    )
    print(f"[Robot] {game_opening_text}")
    yield say_animated(session, game_opening_text)

    last_topic         = None
    consecutive_silences = 0
    game_active        = True

    while game_active:
        perform_movement(
            session,
            frames=[
                {"time": 400,  "data": {"body.head.pitch":  0.1}},
                {"time": 1200, "data": {"body.head.pitch": -0.1}},
                {"time": 2000, "data": {"body.head.pitch":  0.1}},
                {"time": 2400, "data": {"body.head.pitch":  0.0}},
            ],
            force=True,
        )

        user_text, timed_out = yield listen_once(session)

        if timed_out:
            consecutive_silences += 1
            if consecutive_silences >= 3:
                reply = "I am right here whenever you feel like trying."
                consecutive_silences = 0
            else:
                reply = handle_silence(last_topic)
            print(f"[Robot] {reply}")
            yield say_animated(session, reply)
            yield sleep(1)
            continue

        consecutive_silences = 0
        print(f"[User] '{user_text}'")

        if user_text.lower() in EXIT_WORDS:
            game_active = False
            farewell_text = send_with_retry(
                game_chat,
                "The patient wants to stop. Say a warm one-sentence goodbye and well done."
            )
            print(f"[Robot] {farewell_text}")
            yield say_animated(session, farewell_text)

        elif is_unclear(user_text):
            last_topic = extract_topic(user_text)
            reply = handle_unclear(game_chat, user_text, last_topic)
            print(f"[Robot] {reply}")
            yield say_animated(session, reply)

        elif user_text:
            last_topic = extract_topic(user_text)
            response_text = send_with_retry(game_chat, user_text)
            print(f"[Robot] {response_text}")
            yield say_animated(session, response_text)

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

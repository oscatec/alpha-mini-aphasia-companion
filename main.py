"""
HRI Final Project - Integrated Robot System
============================================
Flow:
  1. Stand and detect face via camera
  2. Known face  → greet by name + recall stored info from sample.json →if want to play-> word game
     Unknown face → introductory conversation → store info to sample.json
                    + save face embedding → if want to play -> word game
"""

import io
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

from autobahn.twisted.component import Component, run
from autobahn.twisted.util import sleep
from twisted.internet.defer import inlineCallbacks
from google import genai
from alpha_mini_rug import perform_movement


# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = "REDACTED_API_KEY"
REALM = "rie.REDACTED_REALM"
WAMP_URL = "ws://wamp.robotsindeklas.nl"

FACE_DB_PATH = Path("face_db.pkl")
INFO_DB_PATH = "sample.json"
SIMILARITY_THRESHOLD = 0.65

SILENCE_TICKS = 16  # half-second ticks before treating no input as silence
EXIT_WORDS = {"quit", "exit", "goodbye", "bye", "stop", ":q"}

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

# ── Gemini client ─────────────────────────────────────────────────────────────
chatbot = genai.Client(api_key=API_KEY)

# ── FaceNet models ────────────────────────────────────────────────────────────
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device="cpu")
resnet = InceptionResnetV1(pretrained="vggface2").eval()

# ── ASR shared state ──────────────────────────────────────────────────────────
is_speaking = False
finish_dialogue = False
query = ""
conversation = []


# ══════════════════════════════════════════════════════════════════════════════
# Patient database helpers  (mirrors store_convo_copy.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_face_db() -> dict:
    if FACE_DB_PATH.exists():
        with open(FACE_DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_face_db(db: dict) -> None:
    with open(FACE_DB_PATH, "wb") as f:
        pickle.dump(db, f)


def lookup_info(user_name: str) -> str | None:
    if not os.path.exists(INFO_DB_PATH) or os.path.getsize(INFO_DB_PATH) == 0:
        return None
    with open(INFO_DB_PATH, "r") as f:
        data = json.load(f)
    for entry in data:
        if entry["name"].lower() == user_name.lower():
            return entry["information"]
    return None


def store_info(conv: list, chat) -> str:
    """Ask Gemini to summarise the conversation, then write to sample.json."""
    request = chat.send_message(
        f"Take the following conversation: {conv}. "
        "Reply with a summary of the user's timeless information (age, hobbies, family, etc). "
        "Do NOT include temporary things like mood. "
        "The user's name is always in the conversation. "
        "Your reply must be exactly in this format: [name]: [summarized information]"
    )
    text = request.text.strip()
    print(f"[store_info] {text}")

    parts = text.split(": ", 1)
    if len(parts) != 2:
        return ""
    name_key, info_text = parts[0].strip(), parts[1].strip()

    if os.path.exists(INFO_DB_PATH) and os.path.getsize(INFO_DB_PATH) > 0:
        with open(INFO_DB_PATH, "r") as f:
            data = json.load(f)
    else:
        data = []

    user_found = False
    for entry in data:
        if entry["name"].lower() == name_key.lower():
            entry["information"] += f", {info_text}"
            user_found = True
            break

    if not user_found:
        data.append({"name": name_key, "information": info_text})

    with open(INFO_DB_PATH, "w") as f:
        json.dump(data, f, indent=4)

    return name_key


# ══════════════════════════════════════════════════════════════════════════════
# Face recognition helpers
# ══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_best_match(embedding: np.ndarray, face_db: dict) -> tuple[str | None, float]:
    best_name, best_score = None, 0.0
    for name, stored_emb in face_db.items():
        score = cosine_similarity(embedding, stored_emb)
        if score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


def get_embedding(image_bytes: bytes) -> np.ndarray | None:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    face_tensor = mtcnn(pil_img)
    if face_tensor is None:
        return None
    face_tensor = face_tensor.unsqueeze(0)
    emb = resnet(face_tensor).detach().cpu().numpy()[0]
    emb /= np.linalg.norm(emb) + 1e-8
    return emb


# ══════════════════════════════════════════════════════════════════════════════
# ASR callback
# ══════════════════════════════════════════════════════════════════════════════

def asr(frames):
    global finish_dialogue, query, is_speaking
    if is_speaking:
        return
    if frames["data"]["body"]["final"]:
        recognized = str(frames["data"]["body"]["text"]).strip()
        if not recognized:
            return
        query = recognized
        print(f"[ASR] '{query}'")
        finish_dialogue = True


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def is_unclear(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) < 3 or not any(c.isalpha() for c in stripped)


def handle_silence(chat, last_topic: str | None) -> str:
    phrase = random.choice(SILENCE_PHRASES)
    if last_topic:
        return f"{phrase} Were you thinking about {random.choice(UNCLEAR_HINTS)}, maybe?"
    return phrase


def handle_unclear(chat, user_input: str, last_topic: str | None) -> str:
    context = f" The last topic was: {last_topic}." if last_topic else ""
    resp = chat.send_message(
        f"The patient said something unclear: '{user_input}'.{context} "
        "Acknowledge warmly in one short sentence, then ask ONE yes-or-no question to clarify."
    )
    return resp.text


@inlineCallbacks
def say(session, text: str):
    global is_speaking
    is_speaking = True
    yield session.call("rie.dialogue.say", text=text)
    yield sleep(0.3)
    is_speaking = False


@inlineCallbacks
def listen_once(session, silence_ticks: int = SILENCE_TICKS):
    """Open STT, wait for a final utterance or silence timeout, then close."""
    global finish_dialogue, query
    finish_dialogue = False
    query = ""
    yield session.call("rie.dialogue.stt.stream")

    ticks = 0
    while not finish_dialogue:
        yield sleep(0.5)
        ticks += 1
        if ticks >= silence_ticks:
            break

    yield session.call("rie.dialogue.stt.close")
    yield sleep(0.3)
    return query, not finish_dialogue  # (text, timed_out)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 – Face recognition
# ══════════════════════════════════════════════════════════════════════════════

@inlineCallbacks
def face_recognition_phase(session, face_db: dict):
    """
    Subscribe to camera stream and wait up to 30 s for a face.
    Returns (matched_name_or_None, embedding_or_None, is_new_patient).
    """
    print("[robot] Starting face detection...")
    yield session.call("rie.vision.face.track")

    detected = {"embedding": None, "done": False}

    def on_frame(image_bytes):
        if detected["done"]:
            return
        emb = get_embedding(image_bytes)
        if emb is not None:
            detected["embedding"] = emb
            detected["done"] = True

    yield session.subscribe(on_frame, "rie.vision.camera.stream")

    elapsed = 0
    while not detected["done"] and elapsed < 30:
        yield sleep(0.5)
        elapsed += 0.5

    if not detected["done"]:
        print("[robot] No face detected within timeout.")
        return None, None, True

    emb = detected["embedding"]
    best_name, score = find_best_match(emb, face_db)

    if best_name and score >= SIMILARITY_THRESHOLD:
        print(f"[robot] Recognised {best_name} (score={score:.2f})")
        return best_name, emb, False

    print("[robot] Unknown face detected.")
    return None, emb, True


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2a – Greet a known patient
# ══════════════════════════════════════════════════════════════════════════════

@inlineCallbacks
def greet_known_patient(session, chat, name: str, info: str):
    follow_up = chat.send_message(
        f"Remember this information about the user: {info}. "
        "Do NOT repeat it word for word. Greet them warmly by name, "
        "mention ONE piece of stored info naturally, and ask a short follow-up question about it."
    )
    print(f"[Robot] {follow_up.text}")
    yield say(session, follow_up.text)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2b – Introductory conversation for a new patient
# ══════════════════════════════════════════════════════════════════════════════

INTRO_SYSTEM_PROMPT = (
    "Your name is Mini. You are a warm, friendly social robot. "
    "Your role is to have a short introductory conversation with a patient who has aphasia. "
    "Keep every response to short, simple sentences. "
    "Ask the user their name, age, how their day was, and how their family is doing."
)


@inlineCallbacks
def intro_conversation_phase(session, chat, new_embedding: np.ndarray, face_db: dict):
    """
    Run introductory conversation for a new patient.
    Saves profile to sample.json and face embedding to face_db.pkl.
    Returns the patient's name.
    """
    global conversation

    opening = chat.send_message(
        "Begin the conversation by greeting and asking for the name of the user."
    )
    print(f"[Robot] {opening.text}")
    yield say(session, opening.text)

    # -- Get the patient's name first --
    user_query, _ = yield listen_once(session)
    print(f"[User] '{user_query}'")
    conversation = [user_query]

    name_resp = chat.send_message(
        f"Extract only the name from this message: '{user_query}'. "
        "Reply with only the name, no punctuation."
    )
    patient_name = name_resp.text.strip().split()[0].capitalize()
    print(f"[robot] Extracted name: {patient_name}")

    follow_up = chat.send_message(
        f"Say nice to meet them as {patient_name}. Ask their age."
    )
    print(f"[Robot] {follow_up.text}")
    yield say(session, follow_up.text)

    # -- Continue the conversation --
    dialogue = True
    while dialogue:
        user_query, _ = yield listen_once(session)
        print(f"[User] '{user_query}'")
        conversation.append(user_query)

        if user_query.lower() in EXIT_WORDS:
            farewell = chat.send_message(
                "The patient wants to end. Say a warm one-sentence goodbye."
            )
            print(f"[Robot] {farewell.text}")
            yield say(session, farewell.text)
            dialogue = False

        elif user_query:
            response = chat.send_message(user_query)
            print(f"[Robot] {response.text}")
            yield say(session, response.text)
            if any(w in response.text.lower() for w in ("shall we play", "let us play", "ready to play")):
                dialogue = False

        else:
            yield say(session, "Sorry, I didn't catch that. Could you say it again?")

        yield sleep(1)

    # -- Save info to sample.json + face to face_db.pkl --
    stored_name = store_info(conversation, chat)
    if stored_name:
        patient_name = stored_name
    face_db[patient_name] = new_embedding
    save_face_db(face_db)
    print(f"[robot] Saved profile and face for '{patient_name}'.")
    return patient_name


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 – Word guessing game
# ══════════════════════════════════════════════════════════════════════════════

GAME_SYSTEM_PROMPT = (
    "Your name is Mini. You are a warm, friendly social robot. "
    "Your role is to play a guess-the-word game with a patient who has aphasia. "
    "Game flow: pick a word and guide the patient through up to 4 cue levels:\n"
    "  Level 1 — Semantic cue  : 'It is something you write with.'\n"
    "  Level 2 — Phonemic cue  : 'It starts with the sound p.'\n"
    "  Level 3 — Rhyming cue   : 'It rhymes with hen.'\n"
    "  Level 4 — Sentence fill : 'I use a ... to write on paper.'\n"
    "Correct guess → celebrate and start a new word. "
    "All 4 levels failed → reveal the word, then try a new one.\n"
    "Rules:\n"
    "- 1-2 short simple sentences per turn.\n"
    "- ONE question per turn.\n"
    "- Be patient and encouraging.\n"
    "- If input is unclear, do NOT ask to repeat.\n"
    "- Never call yourself an AI.\n"
    "- End warmly when the patient says goodbye."
)


YES_WORDS = {"yes", "yeah", "yep", "sure", "ok", "okay", "please", "want", "play", "sounds", "good"}
NO_WORDS  = {"no", "nope", "nah", "not", "don't", "tired", "stop", "later"}


@inlineCallbacks
def ask_game_consent(session, intro_chat, patient_name: str) -> bool:
    """Ask the patient if they want to play. Returns True if they agree."""
    prompt = intro_chat.send_message(
        f"The introductory conversation is over. Ask {patient_name} in one warm, simple sentence "
        "if they would like to play a fun word guessing game."
    )
    print(f"[Robot] {prompt.text}")
    yield say(session, prompt.text)

    response, timed_out = yield listen_once(session, silence_ticks=20)
    print(f"[User] '{response}'")

    if timed_out or not response:
        yield say(session, "That is okay, no rush. Maybe next time!")
        return False

    words = set(response.lower().split())
    if words & YES_WORDS:
        return True
    if words & NO_WORDS:
        yield say(session, "No problem at all! It was lovely talking to you.")
        return False

    # Unclear — let Gemini decide
    verdict = intro_chat.send_message(
        f"The patient replied: '{response}'. Did they agree to play a game? Reply with only YES or NO."
    )
    agreed = verdict.text.strip().upper().startswith("Y")
    if not agreed:
        yield say(session, "No problem at all! It was lovely talking to you.")
    return agreed


@inlineCallbacks
def word_game_phase(session, intro_chat, patient_name: str, info: str | None):
    global finish_dialogue, query

    context = f"You are talking to {patient_name}."
    if info:
        context += (
            f" You know this about them: {info}. "
            "Use it occasionally to personalise your cues."
        )

    opening = intro_chat.send_message(
        f"{context} The patient just agreed to play the word game. "
        "Start immediately with the first Level 1 semantic cue for your chosen word. "
        "Keep it to 1-2 short sentences."
    )
    print(f"[Robot] {opening.text}")
    yield say(session, opening.text)

    last_topic = None
    consecutive_silences = 0
    dialogue = True

    while dialogue:
        finish_dialogue = False
        query = ""
        yield session.call("rie.dialogue.stt.stream")

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

        ticks = 0
        while not finish_dialogue:
            yield sleep(0.5)
            ticks += 1
            if ticks >= SILENCE_TICKS:
                break

        yield session.call("rie.dialogue.stt.close")
        yield sleep(0.3)

        # -- Silence --
        if not finish_dialogue:
            consecutive_silences += 1
            if consecutive_silences >= 3:
                reply = "I am right here whenever you feel like trying."
                consecutive_silences = 0
            else:
                reply = handle_silence(intro_chat, last_topic)
            print(f"[Robot] {reply}")
            yield say(session, reply)
            yield sleep(1)
            continue

        consecutive_silences = 0
        print(f"[User] '{query}'")

        # -- Exit --
        if query.lower() in EXIT_WORDS:
            farewell = intro_chat.send_message(
                "The patient wants to stop. Say a warm one-sentence goodbye and well done."
            )
            print(f"[Robot] {farewell.text}")
            yield say(session, farewell.text)
            dialogue = False

        # -- Unclear input --
        elif is_unclear(query):
            reply = handle_unclear(intro_chat, query, last_topic)
            print(f"[Robot] {reply}")
            yield say(session, reply)

        # -- Normal guess --
        elif query:
            response = intro_chat.send_message(query)
            print(f"[Robot] {response.text}")
            yield say(session, response.text)

        # -- Empty fallback --
        else:
            yield say(session, "Take your time, I am listening.")

        yield sleep(1)


# ══════════════════════════════════════════════════════════════════════════════
# Main session
# ══════════════════════════════════════════════════════════════════════════════

@inlineCallbacks
def main(session, details):
    global conversation

    yield session.call("rie.dialogue.config.language", lang="en")
    yield session.call("rom.optional.behavior.play", name="BlocklyStand")

    face_db = load_face_db()

    # -- Phase 1: face recognition --
    #name, embedding, is_new = yield face_recognition_phase(session, face_db)

    # Create chat sessions
    intro_chat = chatbot.chats.create(
        model="gemini-2.5-flash",
        config=genai.types.GenerateContentConfig(system_instruction=INTRO_SYSTEM_PROMPT),
    )
    

    # Subscribe ASR once; stream is opened/closed each turn
    yield session.subscribe(asr, "rie.dialogue.stt.stream")

    # -- Phase 2: intro or greet --
    if is_new and embedding is not None:
        # Unknown face: run intro conversation, store info + face
        name = yield intro_conversation_phase(session, intro_chat, embedding, face_db)
        info = lookup_info(name)
    elif name is not None:
        # Known face: look up stored info and greet
        info = lookup_info(name)
        if info:
            yield greet_known_patient(session, intro_chat, name, info)
        else:
            yield say(session, f"Hi {name}! Great to see you again!")
    else:
        # No face detected at all
        name = "friend"
        info = None
        yield say(session, "Hello! I am Mini. Let us play a game together.")

    # -- Phase 3: ask consent, then start word game --
    wants_to_play = yield ask_game_consent(session, intro_chat, name)
    if wants_to_play:
        yield word_game_phase(session, intro_chat, name, info)

    yield session.call("rom.optional.behavior.play", name="BlocklyCrouch")
    session.leave()


# ══════════════════════════════════════════════════════════════════════════════
# WAMP component
# ══════════════════════════════════════════════════════════════════════════════

wamp = Component(
    transports=[{
        "url": WAMP_URL,
        "serializers": ["msgpack"],
        "max_retries": 0,
    }],
    realm=REALM,
)
wamp.on_join(main)

if __name__ == "__main__":
    run([wamp])

"""
modules/conversation_flow.py
=============================
High-level session orchestration.  This is the only module that should
import from all others – it wires the pipeline together.

Session flow
------------
  Phase 0  │ FACE RECOGNITION
            │   Capture snapshot → match DB → greet by name
            │   If unknown → ask name → store face
            │
  Phase 1  │ PERSONALISED INTRO  (store_combo style)
            │   If returning patient → recall one piece of info + follow-up
            │   If new patient       → ask age, begin info collection
            │
  Phase 2  │ SMART DIALOGUE  (robot_aphasia_smart style)
            │   LLM generates question + JSON mode annotation
            │   closed → rie.dialogue.ask
            │   open   → keyword stream (+ STT fallback)
            │   Silence / unclear handled by silence_handler
            │
  Phase 3  │ FAREWELL + STORE
            │   LLM extracts structured info → info_store
            │   Robot crouches and disconnects

All phases use the shared silence / unclear guard from silence_handler.py.
"""

from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.util import sleep

# ── Module imports ────────────────────────────────────────────────────────────
from modules.face_recognition import (
    get_facial_features,
    FaceNotFoundError,
    MultipleFacesFoundError,
)
from modules.face_store   import FaceStore
from modules.info_store   import InfoStore
from modules.llm_client   import (
    make_main_chat,
    make_store_chat,
    make_silence_chat,
    parse_llm_response,
    send_message,
)
from modules.dialogue import (
    say,
    say_animated,
    ask_one_turn,
    wait_for_speech,
)
from modules.silence_handler import (
    is_unclear,
    silence_response,
    unclear_response,
)
from config import EXIT_WORDS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_farewell(text: str) -> bool:
    farewell_tokens = {"goodbye", "bye", "take care", "farewell"}
    lower = text.lower()
    return any(token in lower for token in farewell_tokens)


def _is_exit(text: str) -> bool:
    return text.strip().lower() in EXIT_WORDS


def _gentle_nod(session):
    """Play a small head-nod to signal the robot is listening."""
    try:
        from alpha_mini_rug import perform_movement
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
    except Exception:
        pass  # alpha_mini_rug may not be installed in all environments


# ── WAMP camera capture (rom.sensor.sight.stream) ───────────────────────────

@inlineCallbacks
def _capture_wamp_frame(session) -> str:
    """
    Subscribe to the robot's sight stream, grab one frame, decode it with
    alpha_mini_rug and save a JPEG to CAMERA_SNAPSHOT_PATH.
    Uses rom.sensor.sight.read with time=0 to get a single frame.
    """
    import base64
    import cv2 as _cv2
    import numpy as _np
    from config import CAMERA_SNAPSHOT_PATH

    frames = yield session.call("rom.sensor.sight.read", time=0)
    if not frames:
        raise RuntimeError("rom.sensor.sight.read returned no frames")

    frame = frames[0]
    # The frame body contains a base64-encoded JPEG under 'body.head'
    body = frame.get("data", {}).get("body.head", "")
    if not body:
        raise RuntimeError("No image data in sight frame")

    img_bytes = base64.b64decode(body)
    img_array = _np.frombuffer(img_bytes, dtype=_np.uint8)
    img = _cv2.imdecode(img_array, _cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Could not decode camera frame")

    _cv2.imwrite(CAMERA_SNAPSHOT_PATH, img)
    return CAMERA_SNAPSHOT_PATH


# ── Phase 0: Face recognition ─────────────────────────────────────────────────

@inlineCallbacks
def _phase_face_recognition(session, face_store: FaceStore) -> str:
    """
    Capture a photo, extract features, and try to match the DB.

    Returns the patient's name (str).
    Falls back to asking the patient their name if the camera fails or no
    match is found.
    """
    patient_name = None

    # 1. Try to capture a snapshot from the robot's camera
    try:
        snapshot_path = yield _capture_wamp_frame(session)
        features      = get_facial_features(snapshot_path)
        patient_name  = face_store.find_match(features)

        if patient_name:
            print(f"[FaceRecog] Recognised: {patient_name}")
        else:
            print("[FaceRecog] Face not in DB – will ask for name and enrol.")

    except FaceNotFoundError:
        print("[FaceRecog] No face detected in snapshot.")
    except MultipleFacesFoundError:
        print("[FaceRecog] Multiple faces – skipping recognition.")
    except Exception as exc:
        print(f"[FaceRecog] Camera / feature error: {exc}")

    # 2. If we couldn't identify the patient, ask their name
    if not patient_name:
        yield say_animated(session, "Hello! I don't think we've met. What is your name?")

        name_text, timed_out = yield wait_for_speech(session)

        if timed_out or not name_text:
            patient_name = "friend"   # graceful default
        else:
            # Ask the LLM to extract just the name from the utterance
            store_chat   = make_store_chat()
            name_resp    = send_message(store_chat, 
                f"Extract only the person's name from this reply: '{name_text}'. "
                "Reply with only the name, no punctuation."
            )
            extracted = getattr(name_resp, "text", str(name_resp)).strip()

            patient_name = extracted or name_text.strip()

        # 3. Enrol the new face if we have a valid name and features
        if patient_name and patient_name != "friend":
            try:
                snapshot_path = yield _capture_wamp_frame(session)
                features      = get_facial_features(snapshot_path)
                face_store.add_or_update(patient_name, features)
                print(f"[FaceRecog] Enrolled new patient: {patient_name}")
            except Exception as exc:
                print(f"[FaceRecog] Could not enrol face: {exc}")

    # 4. Load stored patient information
    patient_info = None

    info_store = InfoStore()

    patient_info = info_store.lookup(patient_name)

    if patient_info:

        yield say_animated(
            session,
            f"Welcome back {patient_name}. "
            #f"I remember that {patient_info}"
        )

    else:

        yield say_animated(
            session,
            f"Nice to meet you, {patient_name}."
        )

    return patient_name


# ── Phase 1: Personalised introduction ───────────────────────────────────────

@inlineCallbacks
def _phase_introduction(session, patient_name: str,
                         store_chat, info_store: InfoStore):
    """
    Greet the patient by name and either:
      - recall one piece of stored information (returning patient), or
      - ask a first question (new patient).

    Returns the first patient reply (str) to carry into Phase 2.
    """
    existing_info = info_store.lookup(patient_name)

    if existing_info:
        # Returning patient – personalise the greeting
        prompt = (
            f"The patient's name is {patient_name}. "
            f"You remember this about them: {existing_info}. "
            "Greet them warmly, mention ONE piece of that information "
            "naturally, and ask a brief follow-up question about it."
        )
    else:
        # New patient – standard welcome
        prompt = (
            f"The patient's name is {patient_name}. "
            "You haven't met them before. "
            "Greet them warmly and ask a simple opening question "
            "(e.g. how old they are or how they are feeling today)."
        )

    response = send_message(store_chat, prompt)
    intro_text = response#.text
    print(f"[Robot/Intro] {intro_text}")
    yield say(session, intro_text)

    # Listen for the patient's first reply
    reply, timed_out = yield wait_for_speech(session)

    if timed_out or not reply:
        # Gentle nudge then return empty string (Phase 2 will handle silence)
        yield say(session, silence_response(None))
        return ""

    print(f"[Patient/Intro] '{reply}'")
    return reply


# ── Phase 2: Smart dialogue loop ─────────────────────────────────────────────

@inlineCallbacks
def _phase_smart_dialogue(session, patient_name: str, initial_reply: str,
                           main_chat, silence_chat,
                           conversation_log: list) -> list:
    """
    Run the main smart-dialogue loop.

    - Uses parse_llm_response to decide closed vs open listening mode.
    - Handles silence and unclear input via silence_handler.
    - Appends every patient turn to `conversation_log`.

    Returns the updated conversation_log.
    """
    consecutive_silences = 0
    last_topic           = None
    patient_reply        = initial_reply or "I didn't catch that"
    silence_active = False

    # Seed the main chat with the patient's intro reply so history is coherent
    if patient_reply:
        conversation_log.append(patient_reply)

    # Get the first question from the main chat
    seed_prompt = (
        f"The patient's name is {patient_name}. They just said: '{patient_reply}'. "
        "Continue the conversation naturally. "
        "Ask the next question and remember to append the JSON block."
    ) if patient_reply else (
        f"The patient's name is {patient_name}. "
        "Begin the conversation with a warm greeting and your first question. "
        "Remember to append the JSON block."
    )

    llm_raw = send_message(main_chat, seed_prompt)
    spoken, mode, answers, keywords = parse_llm_response(llm_raw)#.text)
    print(f"[Robot/Smart] {spoken}  |  mode={mode}")

    dialogue = True

    while dialogue:
        # ── Silence / unclear guard on previous patient_reply ─────────────
        if not patient_reply or patient_reply == "I didn't catch that":
            # consecutive_silences += 1
            # nudge = silence_response(last_topic, consecutive_silences)
            # print(f"[Robot/Silence] {nudge}")
            # yield say(session, nudge)
            # yield sleep(1)

            # # After nudge, re-ask the same question (don't advance the chat)
            # patient_reply, timed_out = yield wait_for_speech(session)
            # if timed_out or not patient_reply:
            #     patient_reply = ""
            # continue
            if silence_active:
                patient_reply = yield wait_for_speech(session)
                continue

            silence_active = True

            consecutive_silences += 1
            nudge = silence_response(last_topic, consecutive_silences)

            print(f"[Robot/Silence] {nudge}")
            yield say(session, nudge)
            yield sleep(1)

            # do NOT re-enter silence immediately
            patient_reply, timed_out = yield wait_for_speech(session)

            if timed_out or not patient_reply:
                patient_reply = ""
                silence_active = False   # reset only after real attempt
                continue

            silence_active = False
            continue

        # Real reply arrived – reset silence counter
        consecutive_silences = 0

        # ── Dispatch this turn ────────────────────────────────────────────
        _gentle_nod(session)
        patient_reply = yield ask_one_turn(session, spoken, mode, answers, keywords)

        # ── Handle STT silence / unclear after ask_one_turn ───────────────
        if patient_reply is None or patient_reply == "":
            patient_reply = ""
            continue

        # ── Patient exit intent (checked BEFORE unclear so "bye" is never
        #    misrouted to the unclear handler) ─────────────────────────────
        if _is_exit(patient_reply):
            dialogue = False
            break

        if is_unclear(patient_reply):
            rescue = unclear_response(silence_chat, patient_reply, last_topic)
            print(f"[Robot/Unclear] {rescue}")
            yield say(session, rescue)
            yield sleep(1)
            patient_reply, timed_out = yield wait_for_speech(session)
            patient_reply = patient_reply or ""
            continue

        # ── Normal reply – log and advance LLM ───────────────────────────
        print(f"[Patient/Smart] '{patient_reply}'")
        conversation_log.append(patient_reply)
        last_topic = patient_reply   # crude but useful topic tracker

        llm_raw = send_message(main_chat, patient_reply)
        spoken, mode, answers, keywords = parse_llm_response(llm_raw)#.text)
        print(f"[Robot/Smart] {spoken}  |  mode={mode}")

        # ── Robot decided to say goodbye ──────────────────────────────────
        if _is_farewell(spoken):
            yield say_animated(session, spoken)
            dialogue = False
            break

        yield sleep(0.5)

    return conversation_log


# ── Phase 3: Farewell + store ─────────────────────────────────────────────────

@inlineCallbacks
def _phase_farewell(session, patient_name: str,
                    main_chat, info_store: InfoStore,
                    conversation_log: list):
    """
    Generate a warm farewell, extract structured info, and persist it.
    """
    # Generate farewell
    farewell_raw = send_message(main_chat, 
        "The patient wants to end the conversation. "
        "Say a warm one-sentence goodbye. Append the JSON block."
    )
    farewell_text, *_ = parse_llm_response(farewell_raw)#.text)
    print(f"[Robot/Farewell] {farewell_text}")
    yield say_animated(session, farewell_text)

    # Extract and persist info (uses the store chat so the extraction
    # prompt doesn't pollute the main chat history)
    if conversation_log:
        store_chat = make_store_chat()
        info_store.extract_and_save(store_chat, patient_name,conversation_log)
        print(f"[InfoStore] Saved info for {patient_name}.")


# ── Top-level session entry point ─────────────────────────────────────────────

@inlineCallbacks
def run_session(session):
    """
    Called by main.py on WAMP join.  Runs all four phases in order.
    """
    # Shared resources
    face_store  = FaceStore()
    info_store  = InfoStore()
    store_chat  = make_store_chat()
    main_chat   = make_main_chat()
    silence_chat = make_silence_chat()
    conversation_log: list = []

    # Configure robot posture and language
    yield session.call("rie.dialogue.config.language", lang="en")
    yield session.call("rom.optional.behavior.play", name="BlocklyStand")

    # ── Phase 0: Face recognition ─────────────────────────────────────────
    print("[Flow] Phase 0 – Face recognition")
    patient_name = yield _phase_face_recognition(session, face_store)
    print(f"[Flow] Patient identified as: {patient_name}")

    # ── Phase 1: Personalised introduction ───────────────────────────────
    print("[Flow] Phase 1 – Introduction")
    intro_reply = yield _phase_introduction(
        session, patient_name, store_chat, info_store
    )

    # ── Phase 2: Smart dialogue ───────────────────────────────────────────
    print("[Flow] Phase 2 – Smart dialogue")
    conversation_log = yield _phase_smart_dialogue(
        session,
        patient_name     = patient_name,
        initial_reply    = intro_reply,
        main_chat        = main_chat,
        silence_chat     = silence_chat,
        conversation_log = conversation_log,
    )

    # ── Phase 3: Farewell + store ─────────────────────────────────────────
    print("[Flow] Phase 3 – Farewell + store")
    yield _phase_farewell(
        session,
        patient_name     = patient_name,
        main_chat        = main_chat,
        info_store       = info_store,
        conversation_log = conversation_log,
    )

    # Power down
    yield session.call("rom.optional.behavior.play", name="BlocklyCrouch")
    session.leave()
    print("[Flow] Session complete.")

"""
Alpha Mini – PwA (Person with Aphasia) Companion Robot
=======================================================
Entry point. Wires together all modules and starts the WAMP session.

Architecture
------------
  main.py                  ← you are here (WAMP bootstrap + top-level flow)
  modules/
    face_recognition.py    ← camera capture + OpenCV feature extraction
    face_store.py          ← JSON face database (persist across sessions)
    info_store.py          ← conversation-info JSON database (sample.json)
    llm_client.py          ← single shared Gemini client + chat factory
    dialogue.py            ← all robot speech / listening primitives
    silence_handler.py     ← silence + unclear-input helpers
    conversation_flow.py   ← high-level per-turn orchestration
  config.py                ← all tuneable constants in one place
"""

from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks

from config import WAMP_URL, WAMP_REALM
from modules.conversation_flow import run_session


# ── WAMP bootstrap ────────────────────────────────────────────────────────────

@inlineCallbacks
def on_join(session, details):
    yield run_session(session)


wamp = Component(
    transports=[{
        "url": WAMP_URL,
        "serializers": ["msgpack"],
        "max_retries": 0,
    }],
    realm=WAMP_REALM,
)

wamp.on_join(on_join)

if __name__ == "__main__":
    run([wamp])

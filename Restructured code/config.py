"""
config.py – All tuneable constants in one place.

Change values here; nothing else needs editing.
"""

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = "REDACTED_API_KEY"  

# Models used per task (cheaper/faster models for simpler tasks)
GEMINI_MODEL_MAIN       = "gemini-2.5-pro"        # main dialogue (smart JSON mode)
GEMINI_MODEL_STORE      = "gemini-2.5-flash"       # info extraction / storage
GEMINI_MODEL_SILENCE    = "gemini-2.5-flash-lite"  # silence / unclear handling

# ── WAMP / robot connection ───────────────────────────────────────────────────
WAMP_URL   = "ws://wamp.robotsindeklas.nl"
WAMP_REALM = "rie.REDACTED_REALM"              

# ── Face recognition ─────────────────────────────────────────────────────────
FACE_DB_DIR          = "face_db"
FACE_DB_FILE         = "face_db/people.json"
FACE_MATCH_THRESHOLD = 0.3    # cosine-weighted distance; lower = stricter

# Temporary file written by the robot's camera capture
CAMERA_SNAPSHOT_PATH = "snapshot.jpg"

# ── Info / conversation store ─────────────────────────────────────────────────
INFO_DB_FILE = "sample.json"

# ── Silence detection ────────────────────────────────────────────────────────
SILENCE_TICKS          = 16   # × 0.5 s polling → 8 s before a nudge
MAX_CONSECUTIVE_SILENCES = 3  # after this many, give a longer comfort phrase

# ── Speech input ─────────────────────────────────────────────────────────────
MIN_MEANINGFUL_LENGTH   = 3   # chars; shorter = "unclear"
KEYWORD_LISTEN_SECONDS  = 8   # window for keyword module
FALLBACK_LISTEN_SECONDS = 6   # window for raw STT fallback
ASR_MAX_ATTEMPTS        = 4   # retries for rie.dialogue.ask (closed mode)

# ── Exit triggers ─────────────────────────────────────────────────────────────
EXIT_WORDS = {"quit", "exit", "goodbye", "bye", "stop", ":q"}

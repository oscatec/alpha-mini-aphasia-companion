"""
config.py - Load all tuneable constants from the .env file.

Change values in .env; nothing else needs editing.
"""

import os
from pathlib import Path


ENV_FILE = Path(__file__).with_name(".env")


def _load_env_file(env_file):
    if not env_file.exists():
        return

    for line in env_file.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")

        os.environ.setdefault(key, value)


def _get_env(name, default):
    return os.environ.get(name, default)


def _get_int(name, default):
    return int(_get_env(name, str(default)))


def _get_float(name, default):
    return float(_get_env(name, str(default)))


def _get_set(name, default):
    value = _get_env(name, ",".join(sorted(default)))
    return {item.strip() for item in value.split(",") if item.strip()}


_load_env_file(ENV_FILE)

# Gemini
GEMINI_API_KEY = _get_env("GEMINI_API_KEY", "")

# Models used per task (cheaper/faster models for simpler tasks)
GEMINI_MODEL_MAIN = _get_env("GEMINI_MODEL_MAIN", "gemini-2.5-pro")
GEMINI_MODEL_STORE = _get_env("GEMINI_MODEL_STORE", "gemini-2.5-flash")
GEMINI_MODEL_SILENCE = _get_env("GEMINI_MODEL_SILENCE", "gemini-2.5-flash-lite")

# WAMP / robot connection
WAMP_URL = _get_env("WAMP_URL", "ws://wamp.robotsindeklas.nl")
WAMP_REALM = _get_env("WAMP_REALM", "rie.REDACTED_REALM")

# Face recognition
FACE_DB_DIR = _get_env("FACE_DB_DIR", "face_db")
FACE_DB_FILE = _get_env("FACE_DB_FILE", "face_db/people.json")
FACE_MATCH_THRESHOLD = _get_float("FACE_MATCH_THRESHOLD", 0.3)

# Temporary file written by the robot's camera capture
CAMERA_SNAPSHOT_PATH = _get_env("CAMERA_SNAPSHOT_PATH", "snapshot.jpg")

# Info / conversation store
INFO_DB_FILE = _get_env("INFO_DB_FILE", "sample.json")

# Silence detection
SILENCE_TICKS = _get_int("SILENCE_TICKS", 16)
MAX_CONSECUTIVE_SILENCES = _get_int("MAX_CONSECUTIVE_SILENCES", 3)

# Speech input
MIN_MEANINGFUL_LENGTH = _get_int("MIN_MEANINGFUL_LENGTH", 3)
KEYWORD_LISTEN_SECONDS = _get_int("KEYWORD_LISTEN_SECONDS", 8)
FALLBACK_LISTEN_SECONDS = _get_int("FALLBACK_LISTEN_SECONDS", 6)
ASR_MAX_ATTEMPTS = _get_int("ASR_MAX_ATTEMPTS", 4)

# Exit triggers
EXIT_WORDS = _get_set("EXIT_WORDS", {"quit", "exit", "goodbye", "bye", "stop", ":q"})

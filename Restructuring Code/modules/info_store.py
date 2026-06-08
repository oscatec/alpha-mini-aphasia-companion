from modules.llm_client import send_message
"""
modules/info_store.py
======================
JSON-backed store for patient biographical information gathered during
conversation (name, age, hobbies, family, favourite meals, etc.).

Public interface
----------------
  InfoStore.lookup(name)               -> str | None
  InfoStore.save_conversation(name, summary) -> None
  InfoStore.extract_and_save(chat, conversation_log) -> str  (extracted name)
"""

import json
import os
from typing import Optional

from config import INFO_DB_FILE


class InfoStore:
    """
    Thin wrapper around the sample.json flat-file database.
    All public methods are synchronous (no WAMP/async needed).
    """

    def __init__(self, db_file: str = INFO_DB_FILE):
        self._db_file = db_file

    # ── I/O helpers ───────────────────────────────────────────────────────────

    def _load(self) -> list:
        if not os.path.exists(self._db_file) or os.path.getsize(self._db_file) == 0:
            return []
        with open(self._db_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    def _save(self, data: list) -> None:
        with open(self._db_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    # ── Public interface ──────────────────────────────────────────────────────

    def lookup(self, name: str) -> Optional[str]:
        """
        Return stored information string for `name`, or None if not found.
        Comparison is case-insensitive.
        """
        data = self._load()
        for entry in data:
            if entry.get("name", "").strip().lower() == name.strip().lower():
                return entry.get("information")
        return None

    def save_conversation(self, name: str, summary: str) -> None:
        """
        Upsert a summary for `name`.  If the person already exists,
        the new summary is appended to the existing one.
        """
        if not name or not name.strip():
            return

        data = self._load()
        normalised = name.strip().lower()

        for entry in data:
            if entry.get("name", "").strip().lower() == normalised:
                entry["information"] = entry.get("information", "") + f", {summary}"
                self._save(data)
                return

        data.append({"name": name.strip(), "information": summary})
        self._save(data)

    def extract_and_save(self, chat, conversation_log: list) -> str:
        """
        Ask the LLM to extract structured info from the conversation log,
        then persist it.  Returns the extracted patient name (or empty string).

        `chat` is an active Gemini chat session (already primed with the
        conversation history).
        """
        prompt = (
            "Take the following conversation log and reply with a summary of the "
            "user's timeless personal information (age, hobbies, family, favourite meals, etc.). "
            "Do NOT include temporary mood/feelings. "
            "Format: [Name]: summarised information\n\n"
            f"Conversation log:\n{conversation_log}"
        )
        response = send_message(chat, prompt)
        text = response.strip()#.text.strip()
        print(f"[InfoStore] Extracted: {text}")

        # Parse "Name: info" format
        if ": " in text:
            parts = text.split(": ", 1)
            name    = parts[0].strip()
            summary = parts[1].strip()
        else:
            # Fallback: store under placeholder
            name    = "Unknown"
            summary = text

        if name and summary:
            self.save_conversation(name, summary)

        return name

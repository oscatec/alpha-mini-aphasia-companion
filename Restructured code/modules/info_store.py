"""
modules/info_store.py
======================
Persistent JSON-backed patient information store.
"""

import json
from pathlib import Path
from typing import Optional

from modules.llm_client import send_message
from config import INFO_DB_FILE


class InfoStore:
    """
    JSON-backed store for patient biographical information.
    """

    def __init__(self, db_file: str = INFO_DB_FILE):
        self._db_file = Path(db_file)
        self._ensure_db()

    # ── DB INITIALISATION ─────────────────────────────────────────────────────

    def _ensure_db(self) -> None:
        """
        Ensure DB directory and JSON file exist.
        """
        self._db_file.parent.mkdir(parents=True, exist_ok=True)

        if not self._db_file.exists():
            self._db_file.write_text("[]", encoding="utf-8")

    # ── DB I/O ────────────────────────────────────────────────────────────────

    def _load(self) -> list:
        """
        Load JSON database.
        """
        self._ensure_db()

        with self._db_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Info DB format invalid: expected JSON array.")

        return data

    def _save(self, data: list) -> None:
        """
        Save JSON database.
        """
        self._ensure_db()

        with self._db_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    # ── LOOKUP ────────────────────────────────────────────────────────────────

    def lookup(self, user_name: str) -> Optional[str]:
        """
        Search for stored information by patient name.
        Case-insensitive.
        """
        if not user_name or not user_name.strip():
            return None

        data = self._load()

        for person_dict in data:

            stored_name = str(
                person_dict.get("name", "")
            ).strip().lower()

            if stored_name == user_name.strip().lower():

                information = person_dict.get("information")

                if information:
                    print(f"[InfoStore] Found info for '{user_name}'")
                    return information

        print(f"[InfoStore] User '{user_name}' not found.")
        return None

    # ── SAVE / UPSERT ─────────────────────────────────────────────────────────

    def save_conversation(self, name: str, summary: str) -> None:
        """
        Add or update patient information.
        """
        if not name or not summary:
            return

        data = self._load()

        normalised = name.strip().lower()

        for entry in data:

            existing_name = str(
                entry.get("name", "")
            ).strip().lower()

            if existing_name == normalised:

                existing_info = entry.get("information", "").strip()

                if existing_info:
                    entry["information"] = (
                        f"{existing_info}, {summary}"
                    )
                else:
                    entry["information"] = summary

                self._save(data)

                print(f"[InfoStore] Updated info for '{name}'")
                return

        # New patient
        data.append({
            "name": name.strip(),
            "information": summary
        })

        self._save(data)

        print(f"[InfoStore] Added new patient '{name}'")

    # ── EXTRACTION + STORAGE ──────────────────────────────────────────────────

    # def extract_and_save(self, chat, conversation_log: list) -> str:
    #     """
    #     Use LLM to extract timeless patient information
    #     from conversation history and persist it.
    #     """

    #     prompt = (
    #         "Take the following conversation log and reply with a summary "
    #         "of the user's timeless personal information "
    #         "(age, hobbies, family, favourite meals, etc.). "
    #         "Do NOT include temporary moods or feelings.\n\n"

    #         "Format exactly like:\n"
    #         "Name: summarised information\n\n"

    #         f"Conversation log:\n{conversation_log}"
    #     )

    #     response = send_message(chat, prompt)

    #     text = getattr(response, "text", str(response)).strip()

    #     print(f"[InfoStore] Extracted: {text}")

    #     # Parse format:
    #     # Name: information

    #     if ": " in text:

    #         name, summary = text.split(": ", 1)

    #         name    = name.strip()
    #         summary = summary.strip()

    #     else:
    #         name    = "Unknown"
    #         summary = text.strip()

    #     if name and summary:
    #         self.save_conversation(name, summary)

    #     return name

    def extract_and_save(
        self,
        chat,
        patient_name: str,
        conversation_log: list
    ) -> None:
        """
        Extract timeless patient information from the conversation
        and save it under the known patient name.
        """

        if not patient_name or patient_name == "friend":
            return

        prompt = (
            "Summarise the user's timeless personal information "
            "(age, hobbies, family, favourite meals, interests, etc.) "
            "from this conversation.\n\n"

            "Do NOT include temporary moods, greetings, or short-term facts.\n\n"

            "Reply ONLY with the summary text.\n\n"

            f"Conversation log:\n{conversation_log}"
        )

        response = send_message(chat, prompt)

        summary = getattr(response, "text", str(response)).strip()

        print(f"[InfoStore] Extracted summary: {summary}")

        if summary:
            self.save_conversation(patient_name, summary)
"""
modules/face_store.py
======================
JSON-backed face database.  Stores FacialFeatures keyed by person name.

Public interface
----------------
  FaceStore.find_match(features)           -> str | None   (name or None)
  FaceStore.add_or_update(name, features)  -> None
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from modules.face_recognition import FacialFeatures
from config import FACE_DB_DIR, FACE_DB_FILE


class FaceStore:
    """
    Thread-unsafe but sufficient for single-session robot use.
    Re-loads from disk on every mutating call to stay consistent.
    """

    def __init__(self):
        self._db_dir  = Path(FACE_DB_DIR)
        self._db_file = Path(FACE_DB_FILE)
        self._ensure_db()
        self._data = self._load()

    # ── DB I/O ────────────────────────────────────────────────────────────────

    def _ensure_db(self) -> None:
        self._db_dir.mkdir(parents=True, exist_ok=True)
        if not self._db_file.exists():
            self._db_file.write_text("[]", encoding="utf-8")

    def _load(self) -> list:
        self._ensure_db()
        with self._db_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Face DB format invalid: expected a JSON array.")
        return data

    def _save(self) -> None:
        self._ensure_db()
        with self._db_file.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    # ── Serialisation helpers ─────────────────────────────────────────────────

    @staticmethod
    def _features_to_dict(f: FacialFeatures) -> dict:
        return {
            "pixels":          f.pixels,
            "histogram":       f.histogram,
            "edge_pixels":     f.edge_pixels,
            "gradient_pixels": f.gradient_pixels,
        }

    @staticmethod
    def _features_from_dict(d: dict) -> Optional[FacialFeatures]:
        if not isinstance(d, dict):
            return None
        required = ("pixels", "histogram", "edge_pixels", "gradient_pixels")
        if not all(isinstance(d.get(k), list) for k in required):
            return None
        return FacialFeatures(
            pixels          = d["pixels"],
            histogram       = d["histogram"],
            edge_pixels     = d["edge_pixels"],
            gradient_pixels = d["gradient_pixels"],
        )

    # ── Distance metric ───────────────────────────────────────────────────────

    @staticmethod
    def _distance(a: FacialFeatures, b: FacialFeatures) -> Optional[float]:
        """
        Weighted multi-feature distance (lower = more similar).
        Returns None if array shapes mismatch.
        """

        def cosine_dist(u: np.ndarray, v: np.ndarray) -> float:
            denom = np.linalg.norm(u) * np.linalg.norm(v)
            if denom == 0:
                return 0.0 if np.array_equal(u, v) else 1.0
            sim = float(np.dot(u, v) / denom)
            return 1.0 - float(np.clip(sim, -1.0, 1.0))

        fields = ("pixels", "histogram", "edge_pixels", "gradient_pixels")
        vecs = {}
        for field in fields:
            u = np.array(getattr(a, field), dtype=np.float64)
            v = np.array(getattr(b, field), dtype=np.float64)
            if u.shape != v.shape:
                return None
            vecs[field] = (u, v)

        px,    px_b  = vecs["pixels"]
        hi,    hi_b  = vecs["histogram"]
        ed,    ed_b  = vecs["edge_pixels"]
        gr,    gr_b  = vecs["gradient_pixels"]

        pixel_dist = float(np.sqrt(np.mean((px - px_b) ** 2)))
        hist_dist  = float(0.5 * np.sum(((hi - hi_b) ** 2) / (hi + hi_b + 1e-8)))
        edge_dist  = float(np.mean(np.abs(ed - ed_b)))
        grad_dist  = (
            0.5 * cosine_dist(gr, gr_b)
            + 0.5 * float(np.mean(np.abs(gr - gr_b)))
        )

        return (
            0.35 * pixel_dist
            + 0.15 * hist_dist
            + 0.25 * edge_dist
            + 0.25 * grad_dist
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def find_match(self, features: FacialFeatures, threshold: float = None) -> Optional[str]:
        """
        Return the name of the closest stored face, or None if no match is
        within `threshold`.
        """
        from config import FACE_MATCH_THRESHOLD
        if threshold is None:
            threshold = FACE_MATCH_THRESHOLD

        best_name: Optional[str] = None
        best_dist: Optional[float] = None

        for person in self._data:
            name    = person.get("name")
            db_feat = self._features_from_dict(person.get("facialfeatures"))
            if not name or db_feat is None:
                continue

            dist = self._distance(features, db_feat)
            if dist is None:
                continue

            if best_dist is None or dist < best_dist:
                best_name = str(name)
                best_dist = dist

        if best_dist is not None and best_dist <= threshold:
            return best_name
        return None

    def add_or_update(self, name: str, features: FacialFeatures) -> None:
        """
        Upsert a person record.  Replaces any existing entry with the same
        name (case-insensitive).
        """
        if not name or not name.strip():
            raise ValueError("Name cannot be empty.")

        self._data = self._load()          # refresh before writing
        normalised = name.strip().lower()
        entry = {
            "name":           name.strip(),
            "facialfeatures": self._features_to_dict(features),
        }

        for idx, person in enumerate(self._data):
            if str(person.get("name", "")).strip().lower() == normalised:
                self._data[idx] = entry
                self._save()
                return

        self._data.append(entry)
        self._save()

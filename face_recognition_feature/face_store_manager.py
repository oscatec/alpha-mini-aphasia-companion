import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from face_recogniztion_manager import FacialFeatures


@dataclass
class Face:
    name: str
    facialfeatures: FacialFeatures

class FaceStoreManager:
    def __init__(self):
        self.DB_DIR = Path("face_db")
        self.DB_FILE = self.DB_DIR / "people.json"
        self._ensure_db()
        self.data = self._load_db()

    """WTF is this function doing?"""
    def _ensure_db(self) -> None:
        self.DB_DIR.mkdir(parents=True, exist_ok=True)
        if not self.DB_FILE.exists():
            self.DB_FILE.write_text("[]", encoding="utf-8")

    def _load_db(self) -> List[dict]:
        self._ensure_db()
        with self.DB_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Database format invalid: expected a list.")
        return data

    def _features_to_dict(self, features: FacialFeatures) -> dict:
        return {
            "pixels": features.pixels,
            "histogram": features.histogram,
            "edge_pixels": features.edge_pixels,
            "gradient_pixels": features.gradient_pixels,
        }

    def _features_from_dict(self, data: dict) -> FacialFeatures | None:
        if not isinstance(data, dict):
            return None

        required_fields = ("pixels", "histogram", "edge_pixels", "gradient_pixels")
        if not all(isinstance(data.get(field), list) for field in required_fields):
            return None

        return FacialFeatures(
            pixels=data["pixels"],
            histogram=data["histogram"],
            edge_pixels=data["edge_pixels"],
            gradient_pixels=data["gradient_pixels"],
        )

    def _feature_distance(self, query_features: FacialFeatures, db_features: FacialFeatures) -> float | None:
        def cosine_distance(query_values: np.ndarray, db_values: np.ndarray) -> float:
            denominator = np.linalg.norm(query_values) * np.linalg.norm(db_values)
            if denominator == 0:
                return 0.0 if np.array_equal(query_values, db_values) else 1.0

            similarity = float(np.dot(query_values, db_values) / denominator)
            return 1.0 - float(np.clip(similarity, -1.0, 1.0))

        values_by_field = {}
        for field in ("pixels", "histogram", "edge_pixels", "gradient_pixels"):
            query_values = np.array(getattr(query_features, field), dtype=np.float64)
            db_values = np.array(getattr(db_features, field), dtype=np.float64)
            if query_values.shape != db_values.shape:
                return None
            values_by_field[field] = (query_values, db_values)

        pixel_query, pixel_db = values_by_field["pixels"]
        hist_query, hist_db = values_by_field["histogram"]
        edge_query, edge_db = values_by_field["edge_pixels"]
        gradient_query, gradient_db = values_by_field["gradient_pixels"]

        pixel_distance = float(np.sqrt(np.mean((pixel_query - pixel_db) ** 2)))
        histogram_distance = float(
            0.5 * np.sum(((hist_query - hist_db) ** 2) / (hist_query + hist_db + 1e-8))
        )
        edge_distance = float(np.mean(np.abs(edge_query - edge_db)))
        gradient_distance = 0.5 * cosine_distance(gradient_query, gradient_db)
        gradient_distance += 0.5 * float(np.mean(np.abs(gradient_query - gradient_db)))

        weighted_distance = (
            0.35 * pixel_distance
            + 0.15 * histogram_distance
            + 0.25 * edge_distance
            + 0.25 * gradient_distance
        )

        return float(weighted_distance)

    def check_match_in_DB(self, features: FacialFeatures, threshold: float = 0.3) -> str | None:
        """
        Return the name of the best matching face in the DB, or None if no match is found.
        """
        if threshold <= 0:
            raise ValueError("Threshold must be > 0.")
        if not features:
            raise ValueError("Features cannot be empty.")

        best_name: str | None = None
        best_distance: float | None = None

        for person in self.data:
            if isinstance(person, Face):
                name = person.name
                db_features = person.facialfeatures
            else:
                name = person.get("name")
                db_features = self._features_from_dict(person.get("facialfeatures"))

            if not name or db_features is None:
                continue

            distance = self._feature_distance(features, db_features)
            print("distance: ", distance)
            if distance is None:
                continue

            if best_distance is None or distance < best_distance:
                best_name = str(name)
                best_distance = distance
        print("best distance found: ", best_distance)
        if best_distance is not None and best_distance <= threshold:
            return best_name

        return None
    # 
    def _save_db(self, data: List[dict]) -> None:
        self._ensure_db()
        with self.DB_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # def _load_image(self, image_path: str):
    #     return face_recognition.load_image_file(image_path)
    
    # def _make_encodings(self, image):
    #     return face_recognition.face_encodings(image)

    # def _encoding_from_image(self, image_path: str) -> np.ndarray:
    #     image = self._load_image(image_path)
    #     encodings = self._make_encodings(image)

    #     if len(encodings) == 0:
    #         raise ValueError(f"No face detected in image: {image_path}")
    #     if len(encodings) > 1:
    #         raise ValueError(f"Multiple faces detected in image: {image_path}. Please provide exactly one face.")

    #     return np.array(encodings[0], dtype=np.float64)


    def add_person(self, name: str, facialfeatures: FacialFeatures) -> None:
        """
        Add (or replace) one person in the local face database using a reference image.
        The image must contain exactly one face.
        """
        if not name or not name.strip():
            raise ValueError("Name cannot be empty.")

        data = self._load_db()

        # Replace existing person with same name (case-insensitive), otherwise append.
        normalized = name.strip().lower()
        replaced = False
        for idx, person in enumerate(data):
            if str(person.get("name", "")).strip().lower() == normalized:
                data[idx] = {
                    "name": name.strip(),
                    "facialfeatures": self._features_to_dict(facialfeatures),
                }
                replaced = True
                break

        if not replaced:
            data.append({
                "name": name.strip(),
                "facialfeatures": self._features_to_dict(facialfeatures),
            })

        self._save_db(data)
        self.data = data

    # def list_people(self) -> List[str]:
    #     """Return all stored person names."""
    #     people = self._load_db()
    #     return [str(person.get("name", "")) for person in people if person.get("name")]

    # def recognize(self, image_path: str, threshold: float = 0.5) -> Tuple[str, Optional[float]]:
    #     """
    #     Recognize one face from image_path against stored people.
    #     Returns (name, distance). If no match under threshold, returns ("Unknown", best_distance).
    #     """
    #     if threshold <= 0:
    #         raise ValueError("Threshold must be > 0.")

    #     query_encoding = self._encoding_from_image(image_path)
    #     people = self._load_db()

    #     if not people:
    #         return "Unknown", None

    #     names = []
    #     db_encodings = []
    #     for person in people:
    #         name = person.get("name")
    #         encoding = person.get("encoding")
    #         if not name or not isinstance(encoding, list):
    #             continue
    #         names.append(str(name))
    #         db_encodings.append(np.array(encoding, dtype=np.float64))

    #     if not db_encodings:
    #         return "Unknown", None

    #     distances = face_recognition.face_distance(db_encodings, query_encoding)
    #     best_idx = int(np.argmin(distances))
    #     best_distance = float(distances[best_idx])

    #     if best_distance <= threshold:
    #         return names[best_idx], best_distance

    #     return "Unknown", best_distance

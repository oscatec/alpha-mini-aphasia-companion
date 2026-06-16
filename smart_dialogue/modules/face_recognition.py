"""
modules/face_recognition.py
============================
Wraps the OpenCV-based face feature extraction from face_recognition_manager.py.

Public interface
----------------
  get_facial_features(img_path) -> FacialFeatures | None
  capture_snapshot(session)     -> str  (path to saved image)

The Alpha Mini does not expose a standard OpenCV camera device; instead we
call rie.vision.snapshot to save a JPEG, then run feature extraction on it.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from twisted.internet.defer import inlineCallbacks

from config import CAMERA_SNAPSHOT_PATH


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    top: int
    right: int
    bottom: int
    left: int


@dataclass
class FacialFeatures:
    pixels: list
    histogram: list
    edge_pixels: list
    gradient_pixels: list


# ── Custom exceptions ─────────────────────────────────────────────────────────

class FaceNotFoundError(ValueError):
    pass


class MultipleFacesFoundError(ValueError):
    pass


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_image(img_path: str):
    image_file = Path(img_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    image = cv2.imread(str(image_file))
    if image is None:
        raise ValueError(f"Could not load image: {img_path}")
    return image


def _detect_single_face(grayscale_image) -> BoundingBox:
    """
    Run Haar cascade and return a BoundingBox for exactly one face.
    Raises FaceNotFoundError or MultipleFacesFoundError otherwise.
    """
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError("Failed to load OpenCV Haar cascade.")

    faces = cascade.detectMultiScale(
        grayscale_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        raise FaceNotFoundError("No face detected in image.")
    if len(faces) > 1:
        raise MultipleFacesFoundError("Multiple faces detected in image.")

    x, y, w, h = faces[0]
    return BoundingBox(top=int(y), right=int(x + w), bottom=int(y + h), left=int(x))


# ── Public API ────────────────────────────────────────────────────────────────

def get_facial_features(img_path: str) -> FacialFeatures:
    """
    Extract a multi-feature descriptor from the single face in `img_path`.
    Returns a FacialFeatures dataclass (pixels, histogram, edge_pixels, gradient_pixels).
    Raises FaceNotFoundError / MultipleFacesFoundError / FileNotFoundError on failure.
    """
    image        = _load_image(img_path)
    grayscale    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bbox         = _detect_single_face(grayscale)

    # Crop, normalise, and compute feature vectors
    face = grayscale[bbox.top:bbox.bottom, bbox.left:bbox.right]
    face = cv2.resize(face, (64, 64))
    face = cv2.equalizeHist(face)
    face = cv2.GaussianBlur(face, (3, 3), 0)

    # 1. Raw pixels (16×16 downscale)
    pixels = cv2.resize(face, (16, 16)).flatten() / 255.0

    # 2. Intensity histogram (64 bins)
    hist = cv2.calcHist([face], [0], None, [64], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # 3. Canny edge map (16×16 downscale)
    edges       = cv2.Canny(face, 80, 160)
    edge_pixels = cv2.resize(edges, (16, 16)).flatten() / 255.0

    # 4. Gradient magnitude (Sobel, 16×16 downscale)
    grad_x    = cv2.Sobel(face, cv2.CV_32F, 1, 0)
    grad_y    = cv2.Sobel(face, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(grad_x, grad_y)
    grad_img  = cv2.resize(magnitude, (16, 16)).flatten()
    grad_img  = grad_img / (grad_img.max() + 1e-6)

    return FacialFeatures(
        pixels          = [float(v) for v in pixels],
        histogram       = [float(v) for v in hist],
        edge_pixels     = [float(v) for v in edge_pixels],
        gradient_pixels = [float(v) for v in grad_img],
    )


@inlineCallbacks
def capture_snapshot(session) -> str:
    """
    Ask the robot to take a photo and save it locally.
    Returns the path to the saved JPEG.

    The Alpha Mini's rie.vision.snapshot call writes a base64-encoded JPEG
    to the WAMP result; we decode and save it ourselves.
    """
    import base64

    result = yield session.call("rie.vision.snapshot")

    # result is expected to be a base64 string or a dict with a 'data' key
    if isinstance(result, dict):
        image_b64 = result.get("data") or result.get("image") or ""
    else:
        image_b64 = result or ""

    if not image_b64:
        raise RuntimeError("rie.vision.snapshot returned no image data.")

    image_bytes = base64.b64decode(image_b64)
    Path(CAMERA_SNAPSHOT_PATH).write_bytes(image_bytes)

    return CAMERA_SNAPSHOT_PATH

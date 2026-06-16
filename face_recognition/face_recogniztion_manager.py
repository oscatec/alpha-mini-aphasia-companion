from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

@dataclass
class BoundingBox:
    top: int
    right: int
    bottom: int
    left: int

@dataclass
class FacialFeatures:
    pixels: list[float]
    histogram: list[float]
    edge_pixels: list[float]
    gradient_pixels: list[float]

class FaceNotFoundError(ValueError):
    pass

class MultipleFacesFoundError(ValueError):
    pass

class FaceRecognitionManager:
    def __init__(self, img_path : str):
        self.img_path = img_path
        self.image = self._get_image(img_path)
        self.grayscale_image = self._get_grayscale_image()

    def _get_image(self, img_path: str):
        # see if image file exists
        image_file = Path(img_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # get the image
        image = cv2.imread(str(image_file))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # load image file
        return image

    def _get_grayscale_image(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def draw_bounding_box(
        self,
        bounding_box: BoundingBox | None = None,
        output_path: str = "boundingbox.jpg",
    ) -> str:
        """
        Create an image with detected face bounding boxes drawn on it.
        """

        if bounding_box is None:
            bounding_box = self.__get_bounding_box()

        image_with_boxes = self.image.copy()
        cv2.rectangle(
            image_with_boxes,
            (bounding_box.left, bounding_box.top),
            (bounding_box.right, bounding_box.bottom),
            (0, 255, 0),
            2,
        )

        if not cv2.imwrite(output_path, image_with_boxes):
            raise ValueError(f"Could not write image: {output_path}")

        return output_path

    def __get_bounding_box(self) -> BoundingBox:
        """
        Return the bounding box for the detecte face.
        Throws an error when multiple faces are found
        """

        # get cascade
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if cascade.empty():
            raise RuntimeError("Failed to load OpenCV Haar cascade for face detection.")

        # do face detection
        faces = cascade.detectMultiScale(
            self.grayscale_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        # throw errors when other than one face is found
        if len(faces) == 0:
            raise FaceNotFoundError("No face detected in image.")
        if len(faces) > 1:
            raise MultipleFacesFoundError("Multiple faces detected in image.")

        # return the boudingbox
        x, y, width, height = faces[0]
        return BoundingBox(
            top=int(y),
            right=int(x + width),
            bottom=int(y + height),
            left=int(x),
        )

    def get_facial_features(self, model: str = "hog") -> FacialFeatures:
        """
        Return the facial features for one face found

        - bbox: {top, right, bottom, left}
        - encoding: 128D OpenCV-based face descriptor as list[float]

        `model` is accepted for compatibility, but OpenCV detection is always used.
        """
        # get the bounding box
        bounding_box = self.__get_bounding_box()

        # draw the boudning box for debugging purposes
        self.draw_bounding_box(bounding_box)

        # get the acutal face
        face = self.grayscale_image[
            bounding_box.top:bounding_box.bottom,
            bounding_box.left:bounding_box.right
        ]

        face = cv2.resize(face, (64, 64))
        face = cv2.equalizeHist(face)
        face = cv2.GaussianBlur(face, (3, 3), 0)

        pixels = cv2.resize(face, (16, 16)).flatten() / 255.0

        hist = cv2.calcHist([face], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        edges = cv2.Canny(face, 80, 160)
        edge_pixels = cv2.resize(edges, (16, 16)).flatten() / 255.0

        grad_x = cv2.Sobel(face, cv2.CV_32F, 1, 0)
        grad_y = cv2.Sobel(face, cv2.CV_32F, 0, 1)
        magnitude = cv2.magnitude(grad_x, grad_y)
        gradient_pixels = cv2.resize(magnitude, (16, 16)).flatten()
        gradient_pixels = gradient_pixels / (gradient_pixels.max() + 1e-6)

        return FacialFeatures(
            pixels=[float(value) for value in pixels],
            histogram=[float(value) for value in hist],
            edge_pixels=[float(value) for value in edge_pixels],
            gradient_pixels=[float(value) for value in gradient_pixels]
        )
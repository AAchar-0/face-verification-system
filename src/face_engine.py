"""A thin wrapper around the two OpenCV face models: YuNet (finds faces) and SFace (makes embeddings).

Uses only the calls already proven by src/01_check_setup.py: detect, alignCrop and feature.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:                                   # hide OpenCV's harmless start-up notices
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

DETECTOR_FILE = "face_detection_yunet_2026may.onnx"
RECOGNIZER_FILE = "face_recognition_sface_2021dec.onnx"
MIN_FACE_PIXELS = 80          # a face smaller than this gives unreliable embeddings
MIN_DETECTION_SCORE = 0.90    # detector confidence, from 0 to 1


@dataclass(frozen=True)
class Observation:
    """What the engine saw in one camera frame."""
    embedding: np.ndarray                          # 128 numbers
    box: tuple[float, float, float, float]         # x, y, width, height of the face
    score: float                                   # detector confidence
    n_faces: int                                   # how many faces were visible


def quality_problem(observation: Observation | None) -> str | None:
    """Return a plain-language problem, or None if the frame is good enough to use."""
    if observation is None:
        return "No face found. Face the camera in good light."
    if observation.n_faces > 1:
        return "More than one face is visible. One person at a time, please."
    if min(observation.box[2], observation.box[3]) < MIN_FACE_PIXELS:
        return "Move closer to the camera."
    if observation.score < MIN_DETECTION_SCORE:
        return "Face the light and look straight at the camera."
    return None


class FaceEngine:
    def __init__(self, models_dir: Path, score_threshold: float = MIN_DETECTION_SCORE):
        detector_path = Path(models_dir) / DETECTOR_FILE
        recognizer_path = Path(models_dir) / RECOGNIZER_FILE
        for path in (detector_path, recognizer_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing {path.name}. Run first: python src/download_models.py")
        self._detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), score_threshold, 0.3, 5000)
        self._recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
        self._input_size = None

    def observe(self, frame, prefer_centre: bool = False) -> Observation | None:
        """Find the face in a frame and compute its embedding. None if no face is found.

        Normally the largest face is used. For benchmark photos (prefer_centre=True) the face
        closest to the middle of the picture is used, because that is the labelled person.
        """
        height, width = frame.shape[:2]
        if self._input_size != (width, height):          # the detector must be told the frame size
            self._detector.setInputSize((width, height))
            self._input_size = (width, height)
        _, faces = self._detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None

        if prefer_centre:
            centres = faces[:, 0] + faces[:, 2] / 2, faces[:, 1] + faces[:, 3] / 2
            distance = (centres[0] - width / 2) ** 2 + (centres[1] - height / 2) ** 2
            face = faces[int(np.argmin(distance))]
        else:
            face = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]      # largest area

        aligned = self._recognizer.alignCrop(frame, face)                # straighten using the 5 landmarks
        embedding = self._recognizer.feature(aligned).flatten().astype(np.float32)
        x, y, w, h = (float(v) for v in face[:4])
        return Observation(embedding, (x, y, w, h), float(face[-1]), len(faces))

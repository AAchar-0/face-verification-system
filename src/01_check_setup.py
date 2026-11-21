"""Step 1: check that the models load, face detection runs, and the camera can be opened.

Run it from the project root with the virtual environment active:
    python src/01_check_setup.py               # full check, including the camera
    python src/01_check_setup.py --no-camera   # models only

The camera check reads a few frames to count faces. Nothing is shown, saved or sent anywhere.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DETECTOR_FILE = MODELS_DIR / "face_detection_yunet_2026may.onnx"
RECOGNIZER_FILE = MODELS_DIR / "face_recognition_sface_2021dec.onnx"


def check_models():
    """Load both models and prove they work on a blank image. Returns (detector, recognizer) or None."""
    missing = [p.name for p in (DETECTOR_FILE, RECOGNIZER_FILE) if not p.exists()]
    if missing:
        print(f"FAIL  Missing model file(s): {missing}")
        print("      Run first:  python src/download_models.py")
        return None

    # Arguments: model path, config (unused), input size, score threshold, NMS threshold, top K.
    # score_threshold 0.9 keeps only confident detections; nms 0.3 merges overlapping boxes.
    detector = cv2.FaceDetectorYN.create(str(DETECTOR_FILE), "", (320, 240), 0.9, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_FILE), "")
    print("OK    Both models loaded")

    blank = np.zeros((240, 320, 3), dtype=np.uint8)        # a black image: no face to find
    _, faces = detector.detect(blank)                       # faces is None when nothing is found
    found = 0 if faces is None else len(faces)
    print(f"{'OK  ' if found == 0 else 'WARN'}  Detector found {found} face(s) in a blank image (expected 0)")
    return detector, recognizer


def check_camera(detector, recognizer) -> bool:
    """Open the default camera, read frames, and run the full detect, align, embed chain once."""
    # On Windows the DirectShow backend opens cameras faster and more reliably.
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    camera = cv2.VideoCapture(0, backend)
    if not camera.isOpened():
        print("FAIL  Could not open the camera. Close other apps that use it (Zoom, Teams, browser tabs)")
        print("      and check Settings > Privacy & security > Camera allows desktop apps.")
        return False

    try:
        frame = None
        for _ in range(15):                                 # the first frames are often dark while the camera adjusts
            ok, frame = camera.read()
            if not ok:
                print("FAIL  The camera opened but returned no image")
                return False
    finally:
        camera.release()                                    # always free the camera, even after an error

    height, width = frame.shape[:2]
    print(f"OK    Camera works: {width} x {height} pixels")

    detector.setInputSize((width, height))                  # the detector must be told the frame size
    _, faces = detector.detect(frame)
    if faces is None:
        print("WARN  No face found. Face the camera in good light and run again.")
        return True

    print(f"OK    Faces found: {len(faces)}")
    best = faces[np.argmax(faces[:, -1])]                   # last column = confidence score
    x, y, w, h = best[:4]
    print(f"      Best face: {w:.0f} x {h:.0f} pixels, confidence {best[-1]:.2f}")
    if min(w, h) < 80:
        print("WARN  The face is small (under 80 pixels). Move closer for reliable results.")

    # Align the face using the 5 landmarks, then compute its 128-number embedding.
    aligned = recognizer.alignCrop(frame, best)
    embedding = recognizer.feature(aligned)
    print(f"OK    Embedding computed: shape {embedding.shape}, length {np.linalg.norm(embedding):.2f}")
    return True


def main() -> None:
    loaded = check_models()
    if loaded is None:
        sys.exit(1)

    if "--no-camera" in sys.argv:
        print("Skipped the camera check.")
        return

    print("\nChecking the camera (nothing is saved or shown)...")
    if not check_camera(*loaded):
        sys.exit(1)
    print("\nSetup check finished.")


if __name__ == "__main__":
    main()

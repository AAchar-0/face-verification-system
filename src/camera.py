"""Camera access and a local live preview. Frames are shown on screen only: never saved or sent."""
import sys

import cv2


class CameraError(RuntimeError):
    pass


class UserQuit(Exception):
    """The person pressed q in the preview window."""


class Camera:
    """Use as:  with Camera() as camera:  frame = camera.read()"""

    def __init__(self, index: int = 0):
        self.index = index
        self._capture = None

    def __enter__(self):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self._capture = cv2.VideoCapture(self.index, backend)
        if not self._capture.isOpened():
            raise CameraError("Could not open the camera. Close other apps that use it (Zoom, Teams, browser tabs) "
                              "and check Settings > Privacy & security > Camera.")
        for _ in range(10):                  # the first frames are often dark while the camera adjusts
            self._capture.read()
        return self

    def read(self):
        ok, frame = self._capture.read()
        return frame if ok else None

    def __exit__(self, *exc_info):
        if self._capture is not None:
            self._capture.release()          # always free the camera, even after an error
        cv2.destroyAllWindows()
        return False


class Preview:
    """A mirror-style live window with a coloured box and a status line. Disables itself if no display exists."""

    WINDOW = "Face verification (press q to quit)"

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.quit_requested = False

    def show(self, frame, text: str = "", ok: bool | None = None, box=None) -> None:
        if not self.enabled or frame is None:
            return
        colour = (0, 200, 0) if ok is True else (0, 0, 220) if ok is False else (0, 170, 255)   # BGR colours
        shown = cv2.flip(frame, 1)                                     # mirror, like a selfie
        if box is not None:
            x, y, w, h = (int(v) for v in box)
            x = shown.shape[1] - x - w                                 # mirror the box too
            cv2.rectangle(shown, (x, y), (x + w, y + h), colour, 2)
        if text:
            cv2.putText(shown, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
        try:
            cv2.imshow(self.WINDOW, shown)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.quit_requested = True
        except cv2.error:
            self.enabled = False                                       # no display: carry on without a preview

"""Live web demo of the consent-based face verification system.

Run it on your own computer from the project root, with the virtual environment active:

    pip install streamlit
    streamlit run live_app/app.py

The same file is what Streamlit Community Cloud runs when the app is hosted online.

How a visit works (four screens, kept in st.session_state["stage"]):
    welcome -> the person types a name, reads the consent box and taps the I AGREE button
    enroll  -> the camera opens (only now) and takes 3 clear photos, which are averaged into one template
    verify  -> one more photo is compared with the template
    result  -> "Verified" (with a Visit website button) or "Face not recognised" (with Try again)

Privacy rules built into the code:
    - The camera widget is only drawn AFTER the person has tapped the consent button.
    - Photos are read from memory, turned into 128 numbers, and never written to disk.
    - Only the template (128 numbers) is kept, in this visit's session memory, and it is zeroed when the
      person taps Delete my data or the session ends. Nothing goes into a database or a log.
    - Unlike the command-line version, a hosted app MUST receive the photo on its server to compare it.
      The consent text says so plainly.
"""
import base64
import hashlib
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:                       # so we can reuse the project's own modules in src/
    sys.path.insert(0, str(SRC_DIR))

from download_models import MIN_BYTES, MODELS, sha256_of                       # noqa: E402
from face_engine import DETECTOR_FILE, RECOGNIZER_FILE, FaceEngine, quality_problem  # noqa: E402
from matching import Decision, average_template, decide                        # noqa: E402

# ---------- Settings you may want to change ----------
LIVE_THRESHOLD = 0.61            # cosine similarity needed to say "Verified" (higher = stricter)
ENROLL_PHOTOS = 2                # clear photos averaged into one template
WEBSITE_URL = "https://aachar-0.github.io/"
SOURCE_URL = "https://github.com/AAchar-0/face-verification-system"
CONSENT_VERSION = "2026-11-web-v1"   # change this whenever the consent wording changes
NAME_MAX_LENGTH = 40

# SHA-256 fingerprints of the two model files (the same values as in the README). A download that does
# not match is deleted and refused, so a tampered or damaged model is never used.
EXPECTED_SHA256 = {
    DETECTOR_FILE: "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
    RECOGNIZER_FILE: "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
}

CONSENT_POINTS = [
    "**No image of you is saved.** Nothing is written to disk or to any database.",
    "Your face is turned into a **code**: a list of 128 numbers. It is held in memory for this visit only, "
    "and erased when you tap **Delete my data**, close the page, or the session ends.",
    "To compare faces, your browser sends each capture to this app's server, where it is used in memory "
    f"and then dropped. The camera switches on only after you agree. It takes {ENROLL_PHOTOS} captures to "
    "enrol you and 1 to verify you.",
    "Taking part is voluntary and only for people aged 18 or over. You can stop at any time by closing the page.",
    "This is a demonstration, not a security system. It cannot tell a live face from a picture of one.",
]


# ---------- Small pure helpers (no camera, no Streamlit: easy to test) ----------
def clean_name(raw: str) -> str:
    """Tidy the typed name, or raise ValueError with a message that can be shown to the person.

    The name is only shown back on screen (it is never used as a file name), but it is still restricted:
    letters, digits, spaces, hyphens, apostrophes and full stops, starting with a letter or digit.
    """
    name = " ".join((raw or "").split())                       # trims and collapses repeated spaces
    if not name:
        raise ValueError("Please type your name in the box above first.")
    if len(name) > NAME_MAX_LENGTH:
        raise ValueError(f"Please use {NAME_MAX_LENGTH} characters or fewer.")
    if not name[0].isalnum() or any(not (c.isalnum() or c in " '-.") for c in name):
        raise ValueError("Please start with a letter or digit, and use only letters, digits, spaces, "
                         "hyphens, apostrophes or full stops.")
    return name


@dataclass(frozen=True)
class Outcome:
    verified: bool
    headline: str
    detail: str


def describe_outcome(name: str, decision: Decision, threshold: float) -> Outcome:
    """Turn a matching Decision into the words shown on screen."""
    if decision.score is None:                                 # cannot happen with min_frames=1, but be safe
        return Outcome(False, "We could not read your face.",
                       "Face the camera in good light and take the photo again.")
    numbers = f"similarity {decision.score:.2f}, needed {threshold:.2f}"
    if decision.accepted:
        return Outcome(True, f"Verified. Welcome, {name}!",
                       f"Your new photo matches the photos you enrolled with ({numbers}).")
    return Outcome(False, "Face not recognised.",
                   f"This photo did not match the photos you enrolled with ({numbers}). Try again: face the "
                   "camera in good light, look straight ahead and sit about as far away as before. If it keeps "
                   "failing, tap Delete my data, then enrol again in the same light you will verify in.")


# ---------- Face models (downloaded once, checked against the published fingerprints) ----------
def models_directory() -> Path:
    """models/ inside the project if it can be written to, otherwise a temporary folder."""
    preferred = PROJECT_ROOT / "models"
    try:
        preferred.mkdir(exist_ok=True)
        probe = preferred / ".write_test"
        probe.touch()
        probe.unlink()
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "face_models"
        fallback.mkdir(exist_ok=True)
        return fallback


def ensure_models(directory: Path) -> None:
    """Make sure both model files are present and genuine, downloading them if needed."""
    for name, url in MODELS.items():
        target = directory / name
        expected = EXPECTED_SHA256[name]
        if target.exists() and target.stat().st_size >= MIN_BYTES and sha256_of(target) == expected:
            continue                                           # already here and genuine
        target.unlink(missing_ok=True)
        temporary = target.with_suffix(".part")
        try:
            urllib.request.urlretrieve(url, temporary)         # download to a temporary name first
        except (urllib.error.URLError, OSError) as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Could not download {name}: {error}") from error
        if sha256_of(temporary) != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"{name} did not match its published SHA-256 fingerprint, so it was refused.")
        temporary.replace(target)                              # rename only once it is complete and genuine


@st.cache_resource(show_spinner="Loading the face models (the very first start can take a minute)...")
def load_engine():
    """One shared engine for all visitors, plus a lock so two visitors never use it at the same moment."""
    directory = models_directory()
    ensure_models(directory)
    return FaceEngine(directory), threading.Lock()


def analyse_photo(image_bytes: bytes):
    """Return (embedding, problem). Exactly one of the two is None. Nothing is saved."""
    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None, "That photo could not be read. Please take it again."
    engine, lock = load_engine()
    with lock:
        observation = engine.observe(frame)
    problem = quality_problem(observation)
    if problem is not None:
        return None, problem
    return observation.embedding, None


def analyse_once(photo):
    """analyse_photo, remembered for the latest photo only, so a button tap does not repeat the work."""
    raw = photo.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    cached = st.session_state.get("last_analysis")
    if cached is not None and cached[0] == digest:
        return cached[1], cached[2]
    embedding, problem = analyse_photo(raw)
    st.session_state["last_analysis"] = (digest, embedding, problem)
    return embedding, problem


# ---------- Camera (selfie camera first, with a switch button) ----------
# st.camera_input cannot choose which camera opens, so a small component in live_app/camera_component/
# draws the camera instead. It always opens the SELFIE camera when the page loads, and a button switches
# between the selfie camera and the back camera. The photo still stays in memory only.
_CAMERA_COMPONENT = components.declare_component(
    "selfie_camera", path=str(Path(__file__).resolve().parent / "camera_component"))


class CapturedPhoto:
    """Looks like the photo st.camera_input returned: the rest of the app only calls getvalue()."""

    def __init__(self, data: bytes):
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def camera_photo(key: str):
    """Show the camera and return a CapturedPhoto once a photo is taken, otherwise None."""
    facing_key = f"{key}_start_facing"
    if facing_key not in st.session_state:                     # fixed per widget, so the camera is not reopened
        st.session_state[facing_key] = st.session_state.get("camera_facing", "user")
    value = _CAMERA_COMPONENT(facing=st.session_state[facing_key], key=key, default=None)
    if not isinstance(value, dict):
        return None
    if value.get("facing") in ("user", "environment"):
        st.session_state["camera_facing"] = value["facing"]    # the next photo opens the same camera
    image = value.get("image")
    if not image or "," not in image:
        return None
    try:
        return CapturedPhoto(base64.b64decode(image.split(",", 1)[1]))
    except ValueError:
        return None


# ---------- Session handling ----------
def fresh_state() -> dict:
    return {"stage": "welcome", "name": "", "consent_version": None, "consent_at": None,
            "embeddings": [], "template": None, "shot": 0, "attempt": 0, "outcome": None,
            "last_analysis": None}


def init_state() -> None:
    for key, value in fresh_state().items():
        st.session_state.setdefault(key, value)


def wipe_session() -> None:
    """Zero every face number and forget everything about this visit."""
    for vector in st.session_state.get("embeddings") or []:
        vector[:] = 0
    template = st.session_state.get("template")
    if template is not None:
        template[:] = 0
    st.session_state.clear()


def try_again() -> None:
    st.session_state["attempt"] += 1                           # a new camera widget, so the old photo is dropped
    st.session_state["outcome"] = None
    st.session_state["last_analysis"] = None
    st.session_state["stage"] = "verify"


# ---------- Screens ----------
def screen_welcome() -> None:
    st.title("Face Verification Application")
    st.write(f"A consent-based demo: enrol with {ENROLL_PHOTOS} photos, then prove it is you with one more.")
    st.info("**Session-only demo.** Nothing is saved. When you close the page, the app forgets you completely.")

    typed = st.text_input("Your name", value=st.session_state["name"], max_chars=NAME_MAX_LENGTH,
                          placeholder="Type your name here")

    with st.container(border=True):
        st.subheader("Consent")
        for point in CONSENT_POINTS:
            st.markdown(f"- {point}")
        st.markdown("### I AGREE")
        st.write("Tapping the button means I agree to the points above.")
        agreed = st.button("Tap to agree", type="primary")

    if agreed:
        try:
            name = clean_name(typed)
        except ValueError as problem:
            st.error(str(problem))
            return
        st.session_state["name"] = name
        st.session_state["consent_version"] = CONSENT_VERSION
        st.session_state["consent_at"] = datetime.now(timezone.utc).isoformat()
        st.session_state["stage"] = "enroll"
        st.rerun()


def screen_enroll() -> None:
    name = st.session_state["name"]
    done = len(st.session_state["embeddings"])
    st.subheader(f"Hi {name}")
    st.progress(done / ENROLL_PHOTOS, text=f"Enrolment: photo {done + 1} of {ENROLL_PHOTOS}")
    st.write("Look straight at the camera in good light, then tap Take photo. The selfie camera opens first; tap Switch to back camera if you prefer.")

    photo = camera_photo(f"enroll_cam_{st.session_state['shot']}")
    if photo is not None:
        embedding, problem = analyse_once(photo)
        if problem:
            st.warning(f"{problem} Please retake the photo.")
        else:
            st.success("Good photo.")
            if st.button("Use this photo", type="primary"):
                st.session_state["embeddings"].append(embedding)
                st.session_state["shot"] += 1                  # a new camera widget for the next photo
                st.session_state["last_analysis"] = None
                if len(st.session_state["embeddings"]) >= ENROLL_PHOTOS:
                    st.session_state["template"] = average_template(st.session_state["embeddings"])
                    for vector in st.session_state["embeddings"]:
                        vector[:] = 0                          # the single photos' numbers are no longer needed
                    st.session_state["embeddings"] = []
                    st.session_state["stage"] = "verify"
                st.rerun()


def screen_verify() -> None:
    name = st.session_state["name"]
    st.subheader(f"You are enrolled, {name}")
    st.write("Now take one more photo, to check that the app recognises you.")

    photo = camera_photo(f"verify_cam_{st.session_state['attempt']}")
    if photo is not None:
        embedding, problem = analyse_once(photo)
        if problem:
            st.warning(f"{problem} Please retake the photo.")
        elif st.button("Verify me", type="primary"):
            decision = decide(st.session_state["template"], [embedding], LIVE_THRESHOLD, min_frames=1)
            st.session_state["outcome"] = describe_outcome(name, decision, LIVE_THRESHOLD)
            st.session_state["last_analysis"] = None
            st.session_state["stage"] = "result"
            st.rerun()
    st.button("Delete my data and start over", on_click=wipe_session)


def screen_result() -> None:
    name = st.session_state["name"]
    outcome = st.session_state["outcome"]
    if outcome.verified:
        st.success(outcome.headline)
        st.caption(outcome.detail)
        with st.container(border=True):
            st.subheader("Thank you")
            st.write(f"Thank you for taking part, {name}. Nothing from this visit was saved. "
                     "If you would like to see my other projects, feel free to visit my website.")
            st.link_button("Visit website", WEBSITE_URL, type="primary")
    else:
        st.error(outcome.headline)
        st.write(outcome.detail)
        st.button("Try again", type="primary", on_click=try_again)
    st.button("Delete my data and start over", on_click=wipe_session)


def footer() -> None:
    st.divider()
    st.caption(f"Created by Anade Acharo. [Portfolio]({WEBSITE_URL}). [Source code]({SOURCE_URL}) · MIT licence · "
               "A demonstration only, not a security system.")


SCREENS = {"welcome": screen_welcome, "enroll": screen_enroll, "verify": screen_verify, "result": screen_result}


def main() -> None:
    st.set_page_config(page_title="Face verification", page_icon="🔐")
    init_state()
    SCREENS[st.session_state["stage"]]()
    footer()


if __name__ == "__main__":
    main()

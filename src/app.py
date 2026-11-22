"""Consent-based face verification: enroll, verify, delete, list, and a session-only demo.

Run from the project root with the virtual environment active:

    python src/app.py demo                 # session-only: nothing is saved anywhere
    python src/app.py enroll anade         # saves an encrypted template on this computer
    python src/app.py verify anade         # compares the camera with that template
    python src/app.py list                 # who is enrolled, and when it expires
    python src/app.py delete anade         # removes a template
    python src/app.py purge                # removes every expired template
    python src/app.py init-key             # makes the encryption key (also done automatically)

Privacy rules built into the code: consent is asked before the camera opens, no photo or video is
ever saved, templates are encrypted and expire, and the camera preview stays on this screen.
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from camera import Camera, CameraError, Preview, UserQuit
from face_engine import FaceEngine, quality_problem
from matching import Decision, average_template, decide
from template_store import (DiskStore, MemoryStore, TemplateError, TemplateRecord,
                            create_key, load_key, validate_subject)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TEMPLATES_DIR = PROJECT_ROOT / "data" / "templates"
ENV_FILE = PROJECT_ROOT / ".env"
THRESHOLD_FILE = PROJECT_ROOT / "reports" / "threshold.json"

CONSENT_VERSION = "2026-11-v1"          # change this whenever the consent wording changes
CONSENT_PHRASE = "I AGREE"
DEFAULT_THRESHOLD = 0.363               # OpenCV's published starting value, used until we calibrate on LFW
DEFAULT_RETENTION_DAYS = 90
ENROLL_FRAMES = 5                       # clear frames collected when enrolling
VERIFY_FRAMES = 7                       # clear frames collected when verifying
MIN_VERIFY_FRAMES = 3                   # fewest clear frames needed for a decision

SESSION_BANNER = """
==================================================================
 SESSION-ONLY MODE
 This camera session is for testing only. Nothing is saved: no
 photos, no video and no face data. When you close this program it
 forgets you completely, so you will need to start again next time.
 This protects your privacy and security.
==================================================================
"""
SESSION_REMINDER = "Session-only: nothing is saved."
END_MESSAGE = "Session ended. Your face data has been erased from memory."

CONSENT_TEXT_DISK = """
CONSENT TO FACE ENROLLMENT
  - Your camera will be used to turn your face into a template: a list of 128 numbers.
  - No photo or video is saved.
  - The template is encrypted and stored only on this computer. It is never uploaded or shared.
  - It expires on {expiry}. You can delete it at any time with:  python src/app.py delete {name}
  - Taking part is voluntary.
Type  I AGREE  to continue, or just press Enter to cancel."""

CONSENT_TEXT_SESSION = """
CONSENT FOR THIS TEST SESSION
  - Your camera will be used for a short test of face verification.
  - Nothing is saved: no photo, no video and no face data. Everything is erased when you close the program.
  - Taking part is voluntary.
Type  I AGREE  to continue, or just press Enter to cancel."""


# ---------- Small helpers ----------
def ask_consent(text: str, input_fn=input, print_fn=print) -> bool:
    """Show the consent text. Only the exact phrase counts as agreement."""
    print_fn(text)
    return input_fn("> ").strip().upper() == CONSENT_PHRASE


def ask_name(input_fn=input, print_fn=print) -> str:
    """Ask until a valid name is given (names become file names, so they are restricted)."""
    while True:
        name = input_fn("What is your name or code name? (letters, digits, - and _ only): ").strip()
        try:
            return validate_subject(name)
        except ValueError as error:
            print_fn(f"  {error}")


def load_threshold(path: Path = THRESHOLD_FILE) -> tuple[float, str]:
    """The decision threshold from the LFW evaluation, or OpenCV's default if none exists yet."""
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data["threshold"]), f"calibrated on LFW at a false accept rate of {data['target_far']:.1%}"
    return DEFAULT_THRESHOLD, "OpenCV's default (run src/02_evaluate_lfw.py to calibrate)"


def format_decision(name: str, decision: Decision, show_score: bool, threshold: float) -> str:
    if decision.reason == "not_enough_frames":
        return "Could not see your face clearly enough. Face the camera in good light and try again."
    text = f"Verified: {name}" if decision.accepted else "Not verified"
    if show_score:
        text += f"   (similarity {decision.score:.2f}, needed {threshold:.2f})"
    return text


# ---------- Core flows (they take the engine and camera as arguments, so tests can use fakes) ----------
def collect_embeddings(engine, camera, needed, *, spacing=0.4, timeout=25.0, preview=None,
                       clock=time.monotonic, label="Hold still"):
    """Collect `needed` clear frames, at least `spacing` seconds apart. Returns (embeddings, last_problem)."""
    embeddings, problem = [], None
    start = clock()
    last_taken = start - spacing                     # the first clear frame can be taken straight away
    while len(embeddings) < needed and clock() - start < timeout:
        frame = camera.read()
        if frame is None:
            problem = "The camera returned no image."
            continue
        observation = engine.observe(frame)
        problem = quality_problem(observation)
        if preview is not None:
            text = f"{label} {len(embeddings)}/{needed}" if problem is None else problem
            preview.show(frame, text=text, ok=(problem is None),
                         box=observation.box if observation else None)
            if preview.quit_requested:
                raise UserQuit()
        if problem is None and clock() - last_taken >= spacing:
            embeddings.append(observation.embedding)
            last_taken = clock()
    return embeddings, problem


@dataclass
class EnrollResult:
    ok: bool
    message: str
    record: TemplateRecord | None = None


def enroll(subject, store, engine, camera, *, retention_days, now, preview=None, clock=time.monotonic):
    """Capture clear frames, average them into one template, and save it. No image is ever written."""
    embeddings, problem = collect_embeddings(engine, camera, ENROLL_FRAMES, preview=preview,
                                             clock=clock, label="Enrolling: hold still")
    if len(embeddings) < ENROLL_FRAMES:
        return EnrollResult(False, problem or "Not enough clear frames. Please try again.")
    record = TemplateRecord(
        subject=subject,
        embedding=average_template(embeddings),
        consent_version=CONSENT_VERSION,
        created_at=now,
        expires_at=now + timedelta(days=retention_days),
    )
    store.save(record)
    return EnrollResult(True, "ok", record)


def verify(subject, store, engine, camera, threshold, *, now, preview=None, clock=time.monotonic):
    """Compare the camera with the stored template. Returns a Decision."""
    record = store.load(subject, now)                # raises if missing, altered or expired
    probes, _ = collect_embeddings(engine, camera, VERIFY_FRAMES, spacing=0.2, timeout=10.0,
                                   preview=preview, clock=clock, label="Verifying")
    return decide(record.embedding, probes, threshold, min_frames=MIN_VERIFY_FRAMES)


def hold_result(preview, camera, text, ok, seconds=1.5):
    """Keep the result on the preview window briefly."""
    if preview is None or not preview.enabled:
        return
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        preview.show(camera.read(), text=text, ok=ok)


def run_demo(engine, camera, preview, *, input_fn=input, print_fn=print, clock=time.monotonic,
             now_fn=lambda: datetime.now(timezone.utc), threshold=DEFAULT_THRESHOLD, show_score=True) -> int:
    """The session-only demo: enroll in memory, verify as often as wanted, then erase everything."""
    store = MemoryStore()
    try:
        print_fn(SESSION_BANNER)
        input_fn("Press Enter to continue... ")
        name = ask_name(input_fn, print_fn)
        if not ask_consent(CONSENT_TEXT_SESSION, input_fn, print_fn):
            print_fn("No consent given. The camera was not used and nothing was captured.")
            return 1
        print_fn(f"\n{SESSION_REMINDER}  Look at the camera and hold still...")
        result = enroll(name, store, engine, camera, retention_days=1, now=now_fn(), preview=preview, clock=clock)
        if not result.ok:
            print_fn(f"Enrollment failed: {result.message}")
            return 1
        print_fn(f"Enrolled '{name}' for this session only.")
        while True:
            answer = input_fn("\nPress Enter to verify, or type q then Enter to quit: ").strip().lower()
            if answer == "q":
                break
            print_fn(SESSION_REMINDER)
            decision = verify(name, store, engine, camera, threshold, now=now_fn(), preview=preview, clock=clock)
            message = format_decision(name, decision, show_score, threshold)
            print_fn(message)
            hold_result(preview, camera, message, decision.accepted)
        return 0
    finally:
        store.wipe()                                 # runs even if the program is interrupted
        print_fn(END_MESSAGE)


# ---------- Command handlers ----------
def _store_for_use(create_if_missing: bool) -> DiskStore:
    key = load_key(ENV_FILE)
    if key is None:
        if not create_if_missing:
            raise TemplateError("No encryption key found, so nothing has been enrolled yet. Run: python src/app.py enroll NAME")
        key = create_key(ENV_FILE)
        print("Created an encryption key in .env (kept out of Git). Back it up: without it, templates cannot be read.")
    return DiskStore(TEMPLATES_DIR, key)


def cmd_demo(args) -> int:
    threshold, source = load_threshold()
    print(f"Decision threshold {threshold:.3f}: {source}")
    engine = FaceEngine(MODELS_DIR)
    preview = Preview(enabled=not args.no_preview)
    with Camera() as camera:
        return run_demo(engine, camera, preview, threshold=threshold, show_score=not args.hide_score)


def cmd_enroll(args) -> int:
    subject = validate_subject(args.name)
    expiry = datetime.now(timezone.utc) + timedelta(days=args.days)
    if not ask_consent(CONSENT_TEXT_DISK.format(expiry=expiry.date(), name=subject)):
        print("No consent given. The camera was not used and nothing was saved.")
        return 1
    store = _store_for_use(create_if_missing=True)
    engine = FaceEngine(MODELS_DIR)
    preview = Preview(enabled=not args.no_preview)
    with Camera() as camera:
        result = enroll(subject, store, engine, camera, retention_days=args.days,
                        now=datetime.now(timezone.utc), preview=preview)
    if not result.ok:
        print(f"Enrollment failed: {result.message}")
        return 1
    print(f"Enrolled '{subject}'. Only an encrypted template was saved (no photo). It expires on {result.record.expires_at.date()}.")
    return 0


def cmd_verify(args) -> int:
    subject = validate_subject(args.name)
    store = _store_for_use(create_if_missing=False)
    threshold, source = load_threshold()
    print(f"Decision threshold {threshold:.3f}: {source}")
    engine = FaceEngine(MODELS_DIR)
    preview = Preview(enabled=not args.no_preview)
    with Camera() as camera:
        decision = verify(subject, store, engine, camera, threshold, now=datetime.now(timezone.utc), preview=preview)
        message = format_decision(subject, decision, args.show_score, threshold)
        print(message)
        hold_result(preview, camera, message, decision.accepted)
    return 0 if decision.accepted else 1


def cmd_delete(args) -> int:
    store = _store_for_use(create_if_missing=False)
    deleted = store.delete(validate_subject(args.name))
    print(f"Deleted the template for '{args.name}'." if deleted else f"There is no template for '{args.name}'.")
    return 0


def cmd_list(args) -> int:
    store = _store_for_use(create_if_missing=False)
    subjects = store.list_subjects()
    if not subjects:
        print("No templates are stored.")
    for subject in subjects:
        try:
            record = store.load(subject)
            print(f"{subject:<20} enrolled {record.created_at.date()}  expires {record.expires_at.date()}  consent {record.consent_version}")
        except TemplateError as error:
            print(f"{subject:<20} {error}")
    return 0


def cmd_purge(args) -> int:
    store = _store_for_use(create_if_missing=False)
    print(f"Removed {store.purge_expired()} expired template(s).")
    return 0


def cmd_init_key(args) -> int:
    if load_key(ENV_FILE):
        print("An encryption key already exists. Nothing changed.")
        return 0
    create_key(ENV_FILE)
    print("Created an encryption key in .env (kept out of Git). Back it up: without it, templates cannot be read.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consent-based face verification.")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_camera_options(sub):
        sub.add_argument("--no-preview", action="store_true", help="do not open the live preview window")

    demo = commands.add_parser("demo", help="session-only demo: nothing is saved")
    add_camera_options(demo)
    demo.add_argument("--hide-score", action="store_true", help="show only Verified or Not verified")
    demo.set_defaults(handler=cmd_demo)

    enroll_parser = commands.add_parser("enroll", help="enroll a person (saves an encrypted template)")
    enroll_parser.add_argument("name", help="a name or code name, for example anade or friend_1")
    enroll_parser.add_argument("--days", type=int, default=DEFAULT_RETENTION_DAYS, help="days until the template expires")
    add_camera_options(enroll_parser)
    enroll_parser.set_defaults(handler=cmd_enroll)

    verify_parser = commands.add_parser("verify", help="check the camera against an enrolled person")
    verify_parser.add_argument("name")
    verify_parser.add_argument("--show-score", action="store_true", help="also print the similarity score")
    add_camera_options(verify_parser)
    verify_parser.set_defaults(handler=cmd_verify)

    delete_parser = commands.add_parser("delete", help="delete a person's template")
    delete_parser.add_argument("name")
    delete_parser.set_defaults(handler=cmd_delete)

    commands.add_parser("list", help="list enrolled people").set_defaults(handler=cmd_list)
    commands.add_parser("purge", help="delete all expired templates").set_defaults(handler=cmd_purge)
    commands.add_parser("init-key", help="create the encryption key").set_defaults(handler=cmd_init_key)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except UserQuit:
        print("Cancelled.")
        return 1
    except (TemplateError, CameraError, FileNotFoundError, ValueError) as error:
        print(error)
        return 2


if __name__ == "__main__":
    sys.exit(main())

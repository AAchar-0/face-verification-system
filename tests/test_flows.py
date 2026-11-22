"""End-to-end tests of enrolling, verifying and the session-only demo, using fake camera and engine
(no real face, no OpenCV model), so they run anywhere."""
import itertools
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from cryptography.fernet import Fernet

import app
from face_engine import Observation
from template_store import DiskStore, MemoryStore, TemplateExpired

NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
rng = np.random.default_rng(42)
PERSON_A = rng.normal(size=128)
PERSON_B = rng.normal(size=128)


class Frame:
    def __init__(self, vector, faces=1, size=200, score=0.99):
        self.vector, self.faces, self.size, self.score = vector, faces, size, score


class FakeCamera:
    def __init__(self, frames):
        self._frames = itertools.cycle(frames)

    def read(self):
        return next(self._frames)


class FakeEngine:
    def observe(self, frame, prefer_centre=False):
        if frame.vector is None:
            return None
        return Observation(np.asarray(frame.vector, dtype=np.float32), (0.0, 0.0, frame.size, frame.size),
                           frame.score, frame.faces)


class Clock:
    """Moves forward 0.5 s every time it is read, so frame spacing and timeouts behave."""
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.5
        return self.t


def person_frames(vector, n=12, noise=0.05):
    noise_rng = np.random.default_rng(7)
    return [Frame(vector + noise_rng.normal(scale=noise, size=128)) for _ in range(n)]


@pytest.fixture
def store(tmp_path):
    return DiskStore(tmp_path / "templates", Fernet.generate_key())


def enroll_a(store, subject="anade", **kwargs):
    return app.enroll(subject, store, FakeEngine(), FakeCamera(person_frames(PERSON_A)),
                      retention_days=30, now=NOW, clock=Clock(), **kwargs)


def test_enrollment_saves_one_encrypted_file_and_no_images(store, tmp_path):
    result = enroll_a(store)
    assert result.ok
    files = [p.name for p in tmp_path.rglob("*") if p.is_file()]
    assert files == ["anade.enc"]                         # no photos, no video, nothing else


def test_enrollment_fails_and_saves_nothing_if_the_face_is_too_small(store, tmp_path):
    tiny = [Frame(PERSON_A, size=30) for _ in range(10)]
    result = app.enroll("anade", store, FakeEngine(), FakeCamera(tiny), retention_days=30, now=NOW, clock=Clock())
    assert not result.ok and "closer" in result.message.lower()
    assert not list(tmp_path.rglob("*.enc"))


def test_enrollment_refuses_two_faces(store):
    crowd = [Frame(PERSON_A, faces=2) for _ in range(10)]
    result = app.enroll("anade", store, FakeEngine(), FakeCamera(crowd), retention_days=30, now=NOW, clock=Clock())
    assert not result.ok and "one person" in result.message.lower()


def test_enrollment_fails_when_no_face_is_seen(store):
    result = app.enroll("anade", store, FakeEngine(), FakeCamera([Frame(None)]), retention_days=30, now=NOW, clock=Clock())
    assert not result.ok and "no face" in result.message.lower()


def test_genuine_person_is_verified(store):
    enroll_a(store)
    decision = app.verify("anade", store, FakeEngine(), FakeCamera(person_frames(PERSON_A)), 0.5, now=NOW, clock=Clock())
    assert decision.accepted
    assert app.format_decision("anade", decision, False, 0.5) == "Verified: anade"


def test_impostor_is_not_verified(store):
    enroll_a(store)
    decision = app.verify("anade", store, FakeEngine(), FakeCamera(person_frames(PERSON_B)), 0.5, now=NOW, clock=Clock())
    assert not decision.accepted
    assert app.format_decision("anade", decision, False, 0.5) == "Not verified"


def test_verification_with_no_face_asks_to_try_again(store):
    enroll_a(store)
    decision = app.verify("anade", store, FakeEngine(), FakeCamera([Frame(None)]), 0.5, now=NOW, clock=Clock())
    assert not decision.accepted and decision.reason == "not_enough_frames"
    assert "try again" in app.format_decision("anade", decision, False, 0.5).lower()


def test_expired_template_cannot_be_used(store):
    enroll_a(store)
    with pytest.raises(TemplateExpired):
        app.verify("anade", store, FakeEngine(), FakeCamera(person_frames(PERSON_A)), 0.5,
                   now=NOW + timedelta(days=31), clock=Clock())


def test_consent_requires_the_exact_phrase():
    assert app.ask_consent("text", input_fn=lambda _: "I agree", print_fn=lambda *_: None)
    assert app.ask_consent("text", input_fn=lambda _: "  I AGREE ", print_fn=lambda *_: None)
    assert not app.ask_consent("text", input_fn=lambda _: "yes", print_fn=lambda *_: None)
    assert not app.ask_consent("text", input_fn=lambda _: "", print_fn=lambda *_: None)


def test_name_prompt_repeats_until_valid():
    answers = iter(["../evil", "a b", "anade"])
    assert app.ask_name(input_fn=lambda _: next(answers), print_fn=lambda *_: None) == "anade"


def run_demo_with(answers, frames, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    printed = []
    supplied = iter(answers)
    code = app.run_demo(FakeEngine(), FakeCamera(frames), None, input_fn=lambda _: next(supplied),
                        print_fn=printed.append, clock=Clock(), now_fn=lambda: NOW, threshold=0.5)
    return code, "\n".join(str(line) for line in printed)


def test_demo_shows_the_session_messages_verifies_and_writes_nothing(tmp_path, monkeypatch):
    code, output = run_demo_with(["", "anade", "I agree", "", "q"], person_frames(PERSON_A), tmp_path, monkeypatch)
    assert code == 0
    assert "SESSION-ONLY MODE" in output and "forgets you completely" in output
    assert "Verified: anade" in output
    assert app.END_MESSAGE in output
    assert list(tmp_path.rglob("*")) == []                # nothing at all was written to disk


def test_demo_without_consent_never_uses_the_camera(tmp_path, monkeypatch):
    code, output = run_demo_with(["", "anade", "no"], person_frames(PERSON_A), tmp_path, monkeypatch)
    assert code == 1
    assert "No consent given" in output and app.END_MESSAGE in output
    assert "Enrolled" not in output


def test_demo_erases_memory_even_when_interrupted(tmp_path, monkeypatch):
    def interrupted(_prompt, _answers=iter(["", "anade", "I AGREE"])):
        try:
            return next(_answers)
        except StopIteration:
            raise KeyboardInterrupt

    monkeypatch.chdir(tmp_path)
    printed = []
    with pytest.raises(KeyboardInterrupt):
        app.run_demo(FakeEngine(), FakeCamera(person_frames(PERSON_A)), None, input_fn=interrupted,
                     print_fn=printed.append, clock=Clock(), now_fn=lambda: NOW)
    assert app.END_MESSAGE in printed


def test_threshold_falls_back_to_the_default(tmp_path):
    threshold, source = app.load_threshold(tmp_path / "missing.json")
    assert threshold == app.DEFAULT_THRESHOLD and "default" in source


def test_threshold_is_read_from_the_report(tmp_path):
    path = tmp_path / "threshold.json"
    path.write_text('{"threshold": 0.4123, "target_far": 0.001}')
    threshold, source = app.load_threshold(path)
    assert threshold == pytest.approx(0.4123) and "0.1%" in source

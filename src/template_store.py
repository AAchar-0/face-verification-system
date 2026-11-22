"""Encrypted storage for face templates, on disk or in memory only.

Only the 128-number template is kept. No photo is ever written. Each template is encrypted with
Fernet from the `cryptography` library: AES (Advanced Encryption Standard) encryption plus an HMAC
(Hash-based Message Authentication Code), so a changed file is detected and refused.

The key lives in .env (ignored by Git), never in the code. Keeping the key next to the data is
fine for a demonstration. A real deployment should keep it in an operating-system or cloud key vault.
"""
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet, InvalidToken

KEY_NAME = "FACE_TEMPLATE_KEY"
SUBJECT_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class TemplateError(Exception):
    """Something is wrong with a stored template."""


class TemplateNotFound(TemplateError):
    pass


class TemplateExpired(TemplateError):
    pass


def validate_subject(subject: str) -> str:
    """Names become file names, so allow only letters, digits, - and _ (blocks paths like ../x)."""
    if not SUBJECT_PATTERN.fullmatch(subject or ""):
        raise ValueError("Use 1 to 40 letters, digits, hyphens or underscores only (for example: anade or friend_1)")
    return subject


# ---------- Encryption key ----------
def load_key(env_path: Path) -> bytes | None:
    """Find the key in the environment or in the .env file. Returns None if there is none."""
    value = os.environ.get(KEY_NAME)
    if value:
        return value.encode()
    if Path(env_path).exists():
        for line in Path(env_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{KEY_NAME}="):
                value = line.split("=", 1)[1].strip().strip('"')
                if value:
                    return value.encode()
    return None


def create_key(env_path: Path) -> bytes:
    """Make a new random key and append it to .env."""
    if load_key(env_path):
        raise TemplateError("An encryption key already exists. Creating a new one would lock you out of existing templates.")
    key = Fernet.generate_key()
    env_path = Path(env_path)
    needs_newline = env_path.exists() and env_path.stat().st_size > 0 and not env_path.read_bytes().endswith(b"\n")
    with open(env_path, "a", encoding="utf-8") as handle:
        handle.write(("\n" if needs_newline else "") + f"{KEY_NAME}={key.decode()}\n")
    return key


# ---------- A stored template ----------
@dataclass
class TemplateRecord:
    subject: str
    embedding: np.ndarray
    consent_version: str
    created_at: datetime
    expires_at: datetime

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at

    def to_bytes(self) -> bytes:
        return json.dumps({
            "subject": self.subject,
            "embedding": [float(x) for x in self.embedding],
            "consent_version": self.consent_version,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }).encode("utf-8")

    @staticmethod
    def from_bytes(raw: bytes) -> "TemplateRecord":
        data = json.loads(raw.decode("utf-8"))
        return TemplateRecord(
            subject=data["subject"],
            embedding=np.array(data["embedding"], dtype=np.float64),
            consent_version=data["consent_version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------- Disk store ----------
class DiskStore:
    """One encrypted file per person: <folder>/<name>.enc"""

    def __init__(self, directory: Path, key: bytes):
        self.directory = Path(directory)
        self._fernet = Fernet(key)

    def _path(self, subject: str) -> Path:
        return self.directory / f"{validate_subject(subject)}.enc"

    def save(self, record: TemplateRecord) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self._path(record.subject)
        temporary = target.with_suffix(".tmp")                  # write to a temporary name, then rename:
        temporary.write_bytes(self._fernet.encrypt(record.to_bytes()))   # a crash never leaves half a file
        temporary.replace(target)

    def load(self, subject: str, now: datetime | None = None) -> TemplateRecord:
        path = self._path(subject)
        if not path.exists():
            raise TemplateNotFound(f"No template for '{subject}'. Enroll first.")
        try:
            record = TemplateRecord.from_bytes(self._fernet.decrypt(path.read_bytes()))
        except InvalidToken:
            raise TemplateError("The template cannot be decrypted: wrong key, or the file was altered.")
        if record.is_expired(now or _utc_now()):
            self.delete(subject)                                # expired data is removed, not kept
            raise TemplateExpired(f"The template for '{subject}' has expired and was deleted. Enroll again.")
        return record

    def delete(self, subject: str) -> bool:
        """Best-effort overwrite, then remove. (On SSDs the old bits may linger: deleting the key is the stronger erase.)"""
        path = self._path(subject)
        if not path.exists():
            return False
        size = path.stat().st_size
        with open(path, "r+b") as handle:
            handle.write(b"\0" * size)
            handle.flush()
            os.fsync(handle.fileno())
        path.unlink()
        return True

    def list_subjects(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.enc")) if self.directory.exists() else []

    def purge_expired(self, now: datetime | None = None) -> int:
        """Delete every expired template. Returns how many were removed."""
        removed = 0
        for subject in self.list_subjects():
            try:
                self.load(subject, now)
            except TemplateExpired:
                removed += 1
            except TemplateError:
                pass
        return removed


# ---------- Memory-only store (session-only mode) ----------
class MemoryStore:
    """Keeps templates in RAM (Random Access Memory) only. Nothing touches the disk."""

    def __init__(self):
        self._records: dict[str, TemplateRecord] = {}

    def save(self, record: TemplateRecord) -> None:
        self._records[validate_subject(record.subject)] = record

    def load(self, subject: str, now: datetime | None = None) -> TemplateRecord:
        if subject not in self._records:
            raise TemplateNotFound(f"No template for '{subject}' in this session.")
        return self._records[subject]

    def delete(self, subject: str) -> bool:
        record = self._records.pop(subject, None)
        if record is not None:
            record.embedding[:] = 0
        return record is not None

    def list_subjects(self) -> list[str]:
        return sorted(self._records)

    def wipe(self) -> None:
        """Zero every template and forget it. Called when the session ends."""
        for record in self._records.values():
            record.embedding[:] = 0
        self._records.clear()

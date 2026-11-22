from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from cryptography.fernet import Fernet

from template_store import (DiskStore, MemoryStore, TemplateError, TemplateExpired, TemplateNotFound,
                            TemplateRecord, create_key, load_key, validate_subject)

NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)


def make_record(subject="anade", days=30, seed=0):
    vector = np.random.default_rng(seed).normal(size=128)
    return TemplateRecord(subject, vector, "2026-11-v1", NOW, NOW + timedelta(days=days))


def test_key_is_created_once_and_read_back(tmp_path, monkeypatch):
    monkeypatch.delenv("FACE_TEMPLATE_KEY", raising=False)
    env = tmp_path / ".env"
    assert load_key(env) is None
    key = create_key(env)
    assert load_key(env) == key
    with pytest.raises(TemplateError):                 # a second key would lock out existing templates
        create_key(env)


def test_environment_variable_overrides_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("FACE_TEMPLATE_KEY", "from-environment")
    assert load_key(tmp_path / ".env") == b"from-environment"


def test_save_and_load_round_trip(tmp_path):
    store = DiskStore(tmp_path, Fernet.generate_key())
    record = make_record()
    store.save(record)
    loaded = store.load("anade", NOW)
    assert np.allclose(loaded.embedding, record.embedding)
    assert loaded.consent_version == "2026-11-v1"


def test_stored_file_is_encrypted(tmp_path):
    store = DiskStore(tmp_path, Fernet.generate_key())
    store.save(make_record("anade"))
    raw = (tmp_path / "anade.enc").read_bytes()
    assert b"embedding" not in raw and b"anade" not in raw and b"consent" not in raw


def test_wrong_key_is_refused(tmp_path):
    DiskStore(tmp_path, Fernet.generate_key()).save(make_record())
    with pytest.raises(TemplateError):
        DiskStore(tmp_path, Fernet.generate_key()).load("anade", NOW)


def test_tampered_file_is_refused(tmp_path):
    store = DiskStore(tmp_path, Fernet.generate_key())
    store.save(make_record())
    path = tmp_path / "anade.enc"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01                       # flip one bit
    path.write_bytes(bytes(data))
    with pytest.raises(TemplateError):
        store.load("anade", NOW)


def test_expired_template_is_refused_and_deleted(tmp_path):
    store = DiskStore(tmp_path, Fernet.generate_key())
    store.save(make_record(days=30))
    with pytest.raises(TemplateExpired):
        store.load("anade", NOW + timedelta(days=31))
    assert not (tmp_path / "anade.enc").exists()


def test_missing_template(tmp_path):
    with pytest.raises(TemplateNotFound):
        DiskStore(tmp_path, Fernet.generate_key()).load("nobody", NOW)


def test_delete_removes_the_file(tmp_path):
    store = DiskStore(tmp_path, Fernet.generate_key())
    store.save(make_record())
    assert store.delete("anade") is True
    assert not (tmp_path / "anade.enc").exists()
    assert store.delete("anade") is False


def test_purge_removes_only_expired(tmp_path):
    store = DiskStore(tmp_path, Fernet.generate_key())
    store.save(make_record("old", days=1))
    store.save(make_record("fresh", days=60))
    assert store.purge_expired(NOW + timedelta(days=10)) == 1
    assert store.list_subjects() == ["fresh"]


@pytest.mark.parametrize("bad", ["../evil", "a b", "", "x" * 41, "name.enc", "a/b", "a\\b"])
def test_unsafe_names_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_subject(bad)


def test_safe_names_are_accepted():
    assert validate_subject("friend_1") == "friend_1"
    assert validate_subject("Anade-A") == "Anade-A"


def test_memory_store_wipe_zeroes_and_forgets():
    store = MemoryStore()
    record = make_record()
    store.save(record)
    assert store.list_subjects() == ["anade"]
    store.wipe()
    assert store.list_subjects() == []
    assert not record.embedding.any()                  # the numbers themselves were overwritten with zeros

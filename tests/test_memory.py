"""Tests for persistent memory."""

from __future__ import annotations

from atlas.memory.store import MemoryStore


def test_learn_and_resolve() -> None:
    with MemoryStore(":memory:") as store:
        assert store.resolve_alias("dob") is None
        store.learn_alias("dob", "date of birth")
        assert store.resolve_alias("dob") == "date of birth"
        assert store.all_aliases()["dob"] == "date of birth"


def test_learning_overwrites() -> None:
    with MemoryStore(":memory:") as store:
        store.learn_alias("phone", "phone number")
        store.learn_alias("phone", "mobile number")
        assert store.resolve_alias("phone") == "mobile number"


def test_learn_ignores_blank() -> None:
    with MemoryStore(":memory:") as store:
        store.learn_alias("", "x")
        store.learn_alias("y", "")
        assert store.all_aliases() == {}


def test_disabled_learning() -> None:
    with MemoryStore(":memory:", alias_learning=False) as store:
        store.learn_alias("dob", "date of birth")
        assert store.all_aliases() == {}


def test_forget() -> None:
    with MemoryStore(":memory:") as store:
        store.learn_alias("dob", "date of birth")
        store.forget_alias("dob")
        assert store.resolve_alias("dob") is None


def test_meta() -> None:
    with MemoryStore(":memory:") as store:
        assert store.get_meta("schema_version") is None
        store.set_meta("schema_version", "1")
        assert store.get_meta("schema_version") == "1"


def test_persists_to_disk(tmp_path) -> None:
    db = tmp_path / "memory.db"
    with MemoryStore(db) as store:
        store.learn_alias("mob", "mobile number")
    with MemoryStore(db) as store:
        assert store.resolve_alias("mob") == "mobile number"

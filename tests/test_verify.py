"""Tests for the verification strategies, especially file-upload read-back."""

from __future__ import annotations

from atlas.act.verify import TargetFieldVerifier, _file_match


def test_file_match_basename() -> None:
    assert _file_match(r"C:\fakepath\x.pdf", "C:/docs/x.pdf")
    assert _file_match(r"C:\fakepath\x.pdf", r"C:\docs\x.pdf")
    assert _file_match("x.pdf", "C:/docs/x.pdf")


def test_file_match_path_containment() -> None:
    assert _file_match("C:/docs/x.pdf", "docs/x.pdf")


def test_file_match_different_file() -> None:
    assert not _file_match(r"C:\fakepath\y.pdf", "C:/docs/x.pdf")
    assert not _file_match("", "C:/docs/x.pdf")
    assert not _file_match("x.pdf", "")


def test_file_match_case_insensitive() -> None:
    assert _file_match(r"C:\FakePath\X.PDF", "c:/docs/x.pdf")


def test_target_verifier_accepts_fakepath() -> None:
    verifier = TargetFieldVerifier(lambda _: r"C:\fakepath\resume.pdf")
    ok, _ = verifier.verify(None, "C:/docs/resume.pdf", "f0")
    assert ok is True


def test_target_verifier_rejects_wrong_file() -> None:
    verifier = TargetFieldVerifier(lambda _: r"C:\fakepath\other.pdf")
    ok, evidence = verifier.verify(None, "C:/docs/resume.pdf", "f0")
    assert ok is False
    assert "mismatch" in evidence

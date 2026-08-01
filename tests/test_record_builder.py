"""Tests for the Record Extraction stage (RecordBuilder)."""

from __future__ import annotations

from pathlib import Path

from atlas.core.record_builder import RecordBuilder, RecordBuildResult

DECLARED = {
    "Full Name": {"type": "textbox", "required": True},
    "Gender": {"type": "combobox", "options": ["Male", "Female", "Other"]},
    "Date Of Birth": {"type": "date_picker", "required": True},
    "Mobile Number": {"type": "textbox"},
}

ALIASES = {
    "name": "Full Name",
    "applicant name": "Full Name",
    "dob": "Date Of Birth",
    "date of birth": "Date Of Birth",
    "mobile": "Mobile Number",
}


def _builder() -> RecordBuilder:
    return RecordBuilder(declared_fields=DECLARED, aliases=ALIASES)


def test_build_happy_path() -> None:
    result = _builder().build([
        ("Application Number", "MPF-100"),
        ("Full Name", "KRISHNA"),
        ("Gender", "Male"),
        ("Date Of Birth", "21 March 1996"),
    ])
    assert result.ok
    assert result.record is not None
    assert result.record.pairs == {
        "Application Number": "MPF-100",
        "Full Name": "KRISHNA",
        "Gender": "Male",
        "Date Of Birth": "21 March 1996",
    }
    assert result.record.ordered_labels[0] == "Application Number"
    assert result.record.record_key == "MPF-100"
    assert result.missing_required == []
    assert result.reason == "ok"


def test_build_cleans_labels_and_values() -> None:
    result = _builder().build([
        ("Full Name :", "  KRISHNA  "),
        ("Gender:", " Male "),
    ])
    assert result.ok
    assert result.record is not None
    assert result.record.pairs["Full Name"] == "KRISHNA"
    assert result.record.pairs["Gender"] == "Male"


def test_build_skips_empty_labels() -> None:
    result = _builder().build([
        ("  ", "value"),
        ("", "other"),
        ("Full Name", "KRISHNA"),
    ])
    assert result.ok
    assert result.record is not None
    assert "Full Name" in result.record.pairs


def test_build_no_pairs_fails() -> None:
    result = _builder().build([])
    assert not result.ok
    assert result.record is None
    assert "no label/value pairs" in result.reason


def test_build_all_values_empty_fails() -> None:
    result = _builder().build([("Full Name", "  "), ("Gender", "")])
    assert not result.ok
    assert result.record is None
    assert "no values" in result.reason


def test_build_missing_required_reports_missing() -> None:
    result = _builder().build([
        ("Full Name", "KRISHNA"),
    ])
    # Record is still produced; the planner surfaces the incomplete fields.
    assert result.ok
    assert result.record is not None
    assert result.record.pairs == {"Full Name": "KRISHNA"}
    assert "date of birth" in result.missing_required
    assert "required fields missing" in result.reason


def test_build_required_satisfied_through_alias() -> None:
    result = _builder().build([
        ("Name", "KRISHNA"),
        ("DOB", "21 March 1996"),
    ])
    assert result.ok
    assert result.record is not None
    assert result.record.pairs["Name"] == "KRISHNA"
    assert result.missing_required == []


def test_write_no_record(tmp_path: Path) -> None:
    result = RecordBuildResult(
        None, reason="source panel has no values (all pairs empty)",
        labels=["Full Name"], values=[""],
    )
    out = _builder().write_no_record(tmp_path / "no_record.json", result, screen="current")
    assert out.exists()
    import json

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["no_record"] is True
    assert payload["labels"] == ["Full Name"]
    assert payload["screen"] == "current"


def test_write_no_record_without_result(tmp_path: Path) -> None:
    out = _builder().write_no_record(tmp_path / "no_record_default.json")
    assert out.exists()
    import json

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["reason"] == "no valid record detected"


def test_as_dict_exposes_declared_and_required() -> None:
    data = _builder().as_dict()
    assert data["required"] == ["full name", "date of birth"]
    assert "gender" in data["declared_fields"]

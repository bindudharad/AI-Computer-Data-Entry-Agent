"""End-to-end web integration tests using Playwright + a mock VLM.

These tests exercise the real chain against a real browser: DOM index,
field -> selector mapping, DOM control engine, verification read-back, and the
full workflow loop with a synthetic vision provider standing in for the VLM.
"""

from __future__ import annotations

import pytest

from atlas.config import load_config
from atlas.core.events import get_event_bus
from atlas.mapping.mapper import SemanticMapper
from atlas.target.web import WebTarget
from atlas.vision.models import ElementType, SceneDescription, ScreenElement
from atlas.vision.providers import MockVisionProvider
from atlas.vision.scene import SceneAnalyzer

try:
    import playwright  # noqa: F401

    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False


PAGE_HTML = """<!doctype html>
<html>
<head><title>Atlas Web Test</title></head>
<body>
  <h1>Record Panel</h1>
  <div id="source">
    <div><span class="lbl">Application No</span><span id="rec-no">1001</span></div>
    <div><span class="lbl">Applicant Name</span><span id="src-name">Ravi Kumar</span></div>
    <div><span class="lbl">Date of Birth</span><span id="src-dob">1990-05-15</span></div>
    <div><span class="lbl">Gender</span><span id="src-gender">Male</span></div>
    <div><span class="lbl">Agree</span><span id="src-agree">Yes</span></div>
  </div>
  <form id="form">
    <div><label for="f-name">Applicant Name</label>
      <input id="f-name" name="applicant_name" type="text"></div>
    <div><label for="f-dob">Date of Birth</label>
      <input id="f-dob" name="dob" type="date"></div>
    <div><label for="f-gender">Gender</label>
      <select id="f-gender" name="gender">
        <option value="">--</option>
        <option>Male</option>
        <option>Female</option>
      </select></div>
    <div><label for="f-agree">Agree</label>
      <input id="f-agree" name="agree" type="checkbox"></div>
    <div><label for="f-file">Attachment</label>
      <input id="f-file" name="attachment" type="file"></div>
    <button id="f-save" type="button">Save</button>
  </form>
</body>
</html>
"""


def _mock_scene() -> SceneDescription:
    """The VLM's view of the test page: source panel + form fields."""
    elements = [
        ScreenElement(element_id="s0", type=ElementType.LABEL, label="Application No", value="1001"),
        ScreenElement(element_id="s1", type=ElementType.LABEL, label="Applicant Name", value="Ravi Kumar"),
        ScreenElement(element_id="s2", type=ElementType.LABEL, label="Date of Birth", value="1990-05-15"),
        ScreenElement(element_id="s3", type=ElementType.LABEL, label="Gender", value="Male"),
        ScreenElement(element_id="s4", type=ElementType.LABEL, label="Agree", value="Yes"),
        ScreenElement(element_id="f0", type=ElementType.TEXTBOX, label="Applicant Name"),
        ScreenElement(element_id="f1", type=ElementType.DATE_PICKER, label="Date of Birth"),
        ScreenElement(element_id="f2", type=ElementType.COMBOBOX, label="Gender", options=["Male", "Female"]),
        ScreenElement(element_id="f3", type=ElementType.CHECKBOX, label="Agree"),
        ScreenElement(element_id="f4", type=ElementType.FILE_UPLOAD, label="Attachment"),
        ScreenElement(element_id="b0", type=ElementType.BUTTON, label="Save"),
    ]
    return SceneDescription(window_title="Atlas Web Test", elements=elements, confidence=1.0, provider="mock")


@pytest.mark.skipif(not _HAS_PLAYWRIGHT, reason="playwright not installed")
def test_web_observe_and_dom_mapping(tmp_path) -> None:
    page_file = tmp_path / "page.html"
    page_file.write_text(PAGE_HTML, encoding="utf-8")

    provider = MockVisionProvider()
    provider.register("atlas web test", _mock_scene())
    analyzer = SceneAnalyzer(provider)
    target = WebTarget(analyzer=analyzer, browser_type="chromium", headless=True)
    try:
        target.attach(page_file.as_uri())
        analysis = target.observe()
        assert analysis is not None
        scene = analysis.scene
        assert any(e.label == "Applicant Name" for e in scene.elements)
        assert target.locator_for("f0") == "#f-name"
        assert target.locator_for("b0") == "#f-save"
    finally:
        target.close()


@pytest.mark.skipif(not _HAS_PLAYWRIGHT, reason="playwright not installed")
def test_dom_control_engine_fills_and_reads(tmp_path) -> None:
    page_file = tmp_path / "page.html"
    page_file.write_text(PAGE_HTML, encoding="utf-8")

    provider = MockVisionProvider()
    provider.register("atlas web test", _mock_scene())
    target = WebTarget(analyzer=SceneAnalyzer(provider), browser_type="chromium", headless=True)
    try:
        target.attach(page_file.as_uri())
        target.observe()
        controls = target.controls

        assert controls.type_value(None, "Ravi Kumar", "f0").ok
        assert target.read_field_value("f0") == "Ravi Kumar"

        assert controls.choose_date(None, "1990-05-15", None, "f1").ok
        assert target.read_field_value("f1") == "1990-05-15"

        assert controls.select_option(None, "Male", ["Male", "Female"], "f2").ok
        assert target.read_field_value("f2") == "Male"

        assert controls.toggle(None, "Yes", "f3").ok
        assert target.read_field_value("f3") == "checked"

        # File upload: create a temp file and attach it to the file input.
        upload = tmp_path / "doc.txt"
        upload.write_text("hello", encoding="utf-8")
        assert controls.upload_file(None, str(upload), "f4").ok
    finally:
        target.close()


@pytest.mark.skipif(not _HAS_PLAYWRIGHT, reason="playwright not installed")
def test_full_loop_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORKFLOW_NEXT_RECORD_TIMEOUT", "0.8")
    monkeypatch.setenv("WORKFLOW_NEXT_RECORD_POLL", "0.1")
    monkeypatch.setenv("VISION_PROVIDER", "mock")
    get_event_bus().clear()

    from atlas.assistant import Assistant

    page_file = tmp_path / "page.html"
    page_file.write_text(PAGE_HTML, encoding="utf-8")

    with Assistant(load_config()) as assistant:
        assert isinstance(assistant.mapper, SemanticMapper)
        vision = assistant._analyzer.provider  # type: ignore[attr-defined]
        vision.register("atlas web test", _mock_scene())  # type: ignore[attr-defined]

        target = assistant.attach_web(url=page_file.as_uri(), browser="chromium", headless=True)
        assert isinstance(target, WebTarget)

        summary = assistant.run(max_records=1)
        assert len(summary.records) == 1
        record = summary.records[0]
        assert record.success is True, record.message
        assert record.record.record_key == "1001"
        assert target.read_field_value("f0") == "Ravi Kumar"
        assert target.read_field_value("f1") == "1990-05-15"
        assert target.read_field_value("f2") == "Male"
        assert target.read_field_value("f3") == "checked"

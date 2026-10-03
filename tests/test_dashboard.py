from copy import deepcopy
import json
from pathlib import Path
import threading
import urllib.error
import urllib.parse
import urllib.request

import pytest

from research_lab.dashboard_data import DashboardStore
from research_lab.framework import build_artifact_manifest
from research_lab.storage import write_json, write_text
from research_lab.usage_dashboard import make_server


class EmptyLedger:
    def snapshot(self):
        return {"models": {}, "recent_requests": []}


@pytest.fixture
def workspace(tmp_path):
    grid = tmp_path / "outputs/demo"
    write_json(grid / "state.json", {"schema_version": 1, "status": "running", "jobs": {
        "a__1": {"status": "completed", "attempts": ["one"]},
        "b__1": {"status": "running", "attempts": ["one", "two"]}}})
    write_json(grid / "inputs/plan.json", {"title": "Demo", "methods": ["a", "b"], "seeds": [1, 2], "primary_metric": "accuracy"})
    write_text(grid / "paper_draft.md", "# Demo\nA harmless draft.")
    write_text(grid / "events.jsonl", '{"time":"2026-09-29T01:00:00Z","event":"job_started","job":"b__1"}\n{"partial":')
    return tmp_path


def test_live_projection_tracks_changes_without_double_counting(workspace):
    store = DashboardStore(workspace, EmptyLedger())
    first = store.snapshot(force=True)
    assert first["totals"]["completed_jobs"] == 1
    assert first["runs"][0]["total_jobs"] == 4
    assert first["runs"][0]["attempts"] == 3
    assert len(first["events"]) == 1
    path = workspace / "outputs/demo/state.json"
    state = json.loads(path.read_text())
    state["jobs"]["b__1"]["status"] = "completed"
    write_json(path, state)
    assert store.snapshot(force=True)["totals"]["completed_jobs"] == 2


def test_artifacts_are_allowlisted_and_redacted(workspace):
    write_text(workspace / ".env.local", "SECRET=never-show-this")
    write_text(workspace / "outputs/demo/paper_draft.md", "credential sk-abcdefghijklmnopqrstuv")
    store = DashboardStore(workspace, EmptyLedger())
    assert "[REDACTED]" in store.artifact("outputs/demo/paper_draft.md")["text"]
    for path in (".env.local", "../.env.local", "outputs/demo/../../.env.local", "outputs/demo/state.json"):
        with pytest.raises(ValueError, match="allowlist"):
            store.artifact(path)


def test_partial_json_does_not_break_other_records(workspace):
    write_text(workspace / "outputs/broken/state.json", '{"status":')
    result = DashboardStore(workspace, EmptyLedger()).snapshot(force=True)
    assert len(result["runs"]) == 1
    assert any(x["source"] == "broken" for x in result["alerts"])


def test_reader_followup_projection_preserves_budget_and_hides_private_route(workspace):
    path = workspace / "projects/followup/reader-study-state.json"
    state = {"id": "followup", "question": "A fixed protocol", "status": "needs_attention", "phase": "readers",
        "reader_calls_scheduled": 1152, "completed_tasks": 14, "evaluation_exposed": True,
        "operations": {"batch": {"tasks": 12}}, "charged_or_reserved": {"deepseek": 500, "kimi": 800},
        "budget": {"deepseek": 1000, "kimi": 2000}, "last_error": "Receipt requires inspection",
        "worker_identity": {"base_url": "private-route", "ledger_directory": "private-ledger"}}
    write_json(path, state)
    view = DashboardStore(workspace, EmptyLedger()).snapshot(force=True)
    project = next(p for p in view["projects"] if p["id"] == "followup")
    assert project["reader_followup"] and project["progress"] == {"split": "readers", "completed": 14, "total": 1394}
    assert project["model_usage"] == state["charged_or_reserved"] and project["model_token_limit"] == 3000
    assert "private-route" not in json.dumps(view) and "private-ledger" not in json.dumps(view)
    assert any(a["source"] == "followup" for a in view["alerts"])


def test_live_integrity_check_detects_modified_evidence(workspace):
    root = workspace / "outputs/demo"
    write_json(root / "state.json", {"schema_version": 1, "status": "completed", "jobs": {}})
    write_json(root / "artifact_manifest.json", build_artifact_manifest(root, [root / "paper_draft.md"]))
    store = DashboardStore(workspace, EmptyLedger())
    assert store.snapshot(force=True)["runs"][0]["integrity"]["status"] == "valid"
    write_text(root / "paper_draft.md", "modified")
    store.integrity_cache.clear()
    assert store.snapshot(force=True)["runs"][0]["integrity"]["status"] == "invalid"


def test_planned_and_supervised_tasks_are_distinguished(workspace):
    task = {"task_id": "task1", "model_alias": "kimi", "objective": "bounded work"}
    write_json(workspace / "config/tasks/task.json", task)
    store = DashboardStore(workspace, EmptyLedger())
    assert store.snapshot(force=True)["tasks"][0]["status"] == "planned"
    write_json(workspace / "outputs/task-result/state.json", {**task, "status": "supervisor_reviewed",
        "supervisor_decision": "partially_accepted_as_design_proposals", "applied_to_research": False})
    result = store.snapshot(force=True)
    assert len(result["tasks"]) == 1 and result["totals"]["reviewed_tasks"] == 1
    assert not result["tasks"][0]["applied"]
    assert any(event["event"] == "supervisor_reviewed" for event in result["events"])


def test_nested_campaign_grids_count_once(workspace):
    campaign = workspace / "outputs/campaign"
    write_json(campaign / "state.json", {"schema_version": 2, "status": "searching", "history": []})
    write_json(campaign / "runs/trial_001/state.json", {"jobs": {"a__1": {"status": "completed", "attempts": ["one"]}}})
    result = DashboardStore(workspace, EmptyLedger()).snapshot(force=True)
    assert len(result["runs"]) == 2 and result["totals"]["completed_jobs"] == 2


def test_http_api_static_assets_and_private_paths(workspace):
    config = {"ledger_directory": str(workspace / ".local/usage"), "models": {}}
    server = make_server(config, 0, workspace)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for path in ("/", "/assets/dashboard.js", "/assets/dashboard.css", "/api/research", "/api/usage"):
            with opener.open(base + path) as response:
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
        with opener.open(base + "/api/research") as response:
            assert json.load(response)["totals"]["completed_jobs"] == 1
        with pytest.raises(urllib.error.HTTPError) as exc:
            opener.open(base + "/api/artifact?path=" + urllib.parse.quote(".env.local"))
        assert exc.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as exc:
            opener.open(urllib.request.Request(base + "/api/research", headers={"Host": "unexpected.example"}))
        assert exc.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        worker.join()

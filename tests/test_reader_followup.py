from copy import deepcopy
import json

import pytest

from research_lab.reader_followup import reader_tasks, judge_tasks, analyze, Followup, review_record


def test_failed_protocol_output_is_a_durable_nonapproval():
    record = review_record("kimi", None)
    assert record["alias"] == "kimi" and record["verdict"] == "unavailable"
    assert record["issues"] and "binding" not in record


def test_initialization_freezes_real_runtime_and_rejects_route_drift(tmp_path, monkeypatch):
    import hashlib
    import research_lab.reader_followup as module
    from research_lab.storage import write_json, read_json, digest
    records = [{"uid": f"{kind}-{i}", "type": kind, "M_new": "updated", "M_old": "old", "explanation": "logic",
                "probing_queries": {f"dim{d}_query": f"question {d}" for d in (1, 2, 3)}} for kind in ("T1", "T2") for i in range(2)]
    dataset = tmp_path / "dataset.json"
    write_json(dataset, records)
    v6 = tmp_path / "v6"
    artifacts = []
    for row in records:
        for dim, question in row["probing_queries"].items():
            path = v6 / "contexts" / (hashlib.sha256((row["uid"] + ":" + dim).encode()).hexdigest() + ".json")
            write_json(path, {"uid": row["uid"], "query": question, "contexts": [
                {"method": arm, "budget": 1024, "units": 10, "context": "Natural updated text"} for arm in ("bm25_32", "ce_32")]})
            artifacts.append({"path": path.relative_to(v6).as_posix(), "sha256": digest(path)})
    result = v6 / "artifacts/retrieval/1/result.json"
    write_json(result, {"artifacts": artifacts})
    models = {"ledger_directory": str(tmp_path / "ledger"), "models": {a: {
        "model": a + "-fixture", "input_token_reservation": 100, "max_output_tokens": 50,
        "timeout_seconds": 500, "max_input_bytes": 98304, "api_key": "NOT-TO-BE-FROZEN"} for a in ("deepseek", "kimi")}}
    monkeypatch.setattr(module, "load_models", lambda: deepcopy(models))
    config = tmp_path / "config.json"
    write_json(config, {"question": "fixture", "dataset_path": str(dataset), "dataset_sha256": digest(dataset),
        "v6_project": str(v6), "v6_result_sha256": digest(result), "readers": ["deepseek", "kimi"],
        "per_type": 2, "sample_seed": "fixture", "max_wall_seconds": 86400,
        "model_budget": {"deepseek": 10000, "kimi": 12000}, "prior_charged_or_reserved": {"deepseek": 20, "kimi": 30}})
    project = tmp_path / "followup"
    state = module.initialize(config, project)
    protocol = read_json(project / "inputs/protocol.json")
    assert state["reader_calls_scheduled"] == 48 and state["worker_identity"] is not None
    assert protocol["call_accounting"]["total_max"] == 66
    assert protocol["call_accounting"]["wall_seconds"] == 86400
    assert protocol["call_accounting"]["per_call_reservations"] == {"deepseek": 150, "kimi": 150}
    assert protocol["payload_preflight"]["max_request_bytes"]["judges"] > 24576
    assert "NOT-TO-BE-FROZEN" not in json.dumps([state, protocol])
    Followup(project).verify()
    models["models"]["deepseek"]["timeout_seconds"] = 600
    with pytest.raises(ValueError, match="profile/ledger changed"):
        Followup(project).verify()
    models["models"]["deepseek"]["max_input_bytes"] = 24576
    with pytest.raises(ValueError, match="Worst-case"):
        module.initialize(config, tmp_path / "too-small")
    assert not (tmp_path / "too-small").exists()


def cases():
    return [{"uid": uid, "type": "T1", "dimension": dim, "arm": arm, "context": "Only natural retrieved text.",
             "query": "A fixed probing question?", "new_present": int(arm == "ce_32"), "context_sha256": "fixture-hash",
             "M_new": "SECRET GOLD ANNOTATION", "explanation": "SECRET HIDDEN LOGIC"}
            for uid in ("first", "second") for dim in ("dim1_query", "dim2_query", "dim3_query") for arm in ("bm25_32", "ce_32")]


def test_reader_payload_is_label_blind_and_matched():
    source = cases()
    tasks, manifest = reader_tasks(source, ["deepseek", "kimi"])
    assert len(tasks) == len(manifest) == 24
    text = json.dumps(tasks)
    assert "SECRET GOLD" not in text and "SECRET HIDDEN" not in text
    assert "new_present" not in text and "bm25_32" not in text and "ce_32" not in text
    other = deepcopy(source)
    for r in other:
        r.update(M_new="changed gold", new_present=1-r["new_present"])
    assert tasks == reader_tasks(other, ["deepseek", "kimi"])[0]


def test_judges_blind_to_method_reader_and_preserve_all_assignments():
    tasks, manifest = reader_tasks(cases(), ["deepseek", "kimi"])
    answers = {t["id"]: {"answer": "A bounded answer."} for t in tasks}
    gold = [{"uid": uid, "M_old": "old", "M_new": "new", "explanation": "updated state logic",
             "probing_queries": {d: "question" for d in ("dim1_query", "dim2_query", "dim3_query")}} for uid in ("first", "second")]
    tasks, assignments = judge_tasks(gold, manifest, answers, ["deepseek", "kimi"], ["first"])
    assert len(tasks) == 6 and sum(a["kind"] == "primary" for a in assignments) == 4
    for t in tasks:
        assert all(s not in t["prompt"] for s in ("bm25_32", "ce_32", "deepseek", "kimi"))
        assert len(t["schema"]["required"]) == 6
    assert all(a["judge"] != a["reader"] for a in assignments if a["kind"] == "primary")


def test_paired_analysis_clusters_dimensions_and_readers_with_missing_bounds():
    tasks, manifest = reader_tasks(cases(), ["deepseek", "kimi"])
    responses = {t["id"]: {"answer": "Answer"} for t in tasks}
    assignments, grades = [], {}
    for i, row in enumerate(manifest):
        tid = str(i)
        assignments.append({"id": tid, "kind": "primary", "labels": {"a": row["id"]}})
        grades[tid] = {"a": {"pass": row["arm"] == "ce_32", "reason": "fixture"}}
    report = analyze(manifest, responses, assignments, grades)
    assert report["primary"]["scheduled_scenarios"] == 2
    assert report["primary"]["complete_scenarios"] == 2
    assert report["primary"]["missing_as_failure_gain"] == 1
    assert report["primary"]["paired_scenario_ci95"] == [1., 1.]
    sensitivity = analyze(manifest, responses, assignments, grades, {"first"})
    assert sensitivity["primary"]["scheduled_scenarios"] == 2
    assert sensitivity["prior_reader_overlap_sensitivity"]["contrast"]["scheduled_scenarios"] == 1
    assert analyze(manifest, responses, assignments, grades, {"first", "second"})["prior_reader_overlap_sensitivity"]["contrast"] is None
    missing = analyze(manifest, {}, [], {})
    assert missing["primary"]["complete_case_gain"] is None
    assert missing["primary"]["missingness_identification_bounds"] == [-1., 1.]
    assert missing["judge_calibration"]["agreement"] is None


def test_batch_budget_admission_precedes_native_work(tmp_path):
    runner = Followup.__new__(Followup)
    runner.root = tmp_path
    runner.state = {"operations": {}, "charged_or_reserved": {"deepseek": 90}, "budget": {"deepseek": 100}}
    runner.models = {"models": {"deepseek": {"input_token_reservation": 10, "max_output_tokens": 10}}}
    with pytest.raises(RuntimeError, match="allowance"):
        runner.batch("test", [{"model": "deepseek"}])
    assert runner.state["operations"] == {}


def test_uncertain_batch_cannot_be_repeated(tmp_path):
    from research_lab.project import identity
    runner = Followup.__new__(Followup)
    runner.root = tmp_path
    tasks = [{"model": "deepseek"}]
    runner.state = {"operations": {"attempt": {"signature": identity(tasks), "status": "running"}}}
    with pytest.raises(RuntimeError, match="uncertain"):
        runner.batch("attempt", tasks)

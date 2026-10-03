from copy import deepcopy
import json
from pathlib import Path

import pytest

from research_lab.open_research import OpenResearch, OpenFixtureWorker, initialize
from research_lab.discovery_literature import FixtureSearch, CrossrefSearch
from research_lab.discovery_domains import evaluate, validate_splits, validate_method
from research_lab.storage import read_json, write_json, digest

ROOT = Path(__file__).resolve().parents[1]


def setup(tmp_path, **changes):
    cfg = read_json(ROOT / "config/projects/open-topic-fixture.json")
    for a in cfg["assets"]:
        for split in ("training", "development", "evaluation"):
            a[split] = str(ROOT / "config/projects" / a[split])
    cfg.update(changes)
    path = tmp_path / "config.json"
    write_json(path, cfg)
    root = tmp_path / "projects/open-topic"
    initialize(path, root)
    return root


class Spy(OpenFixtureWorker):
    def __init__(self):
        self.calls = []

    def generate(self, alias, stage, payload, schema, limit):
        self.calls.append(deepcopy(payload))
        return super().generate(alias, stage, payload, schema, limit)


def test_broad_brief_to_manuscript_resume_without_duplicate_calls(tmp_path):
    root, worker = setup(tmp_path), Spy()
    state = OpenResearch(root, worker).run()
    assert state["status"] == "completed"
    assert state["question"] != read_json(root / "inputs/config.json")["brief"]
    assert len(state["topic_pool"]) == 2
    assert state["worker_calls"] == 17 and state["model_calls"] == 0
    assert state["analysis"]["metric"] == "classification_accuracy"
    assert state["evaluation_exposed"] and state["deliverables"]["fixture_only"]
    assert "eval_a" not in json.dumps(worker.calls) and "train_a" not in json.dumps(worker.calls)
    assert all(r["actor_type"] == "model_review_panel" for r in state["reviews"])
    before = digest(root / "autonomy-state.json")
    OpenResearch(root, worker).run()
    assert len(worker.calls) == 17 and before == digest(root / "autonomy-state.json")
    draft = (root / "evidence/deliverables/paper_draft.md").read_text("utf-8")
    assert "not scientific evidence" in draft and "training split only" in draft
    assert "STALE" not in draft
    assert (root / "evidence/deliverables/topic_audit.json").is_file()


def test_partial_run_resumes_same_topic_pool(tmp_path):
    root, worker = setup(tmp_path), Spy()
    first = OpenResearch(root, worker).run(max_steps=7)
    assert first["status"] == "ready" and not first["evaluation_exposed"]
    end = OpenResearch(root, worker).run()
    assert end["status"] == "completed" and len(worker.calls) == 17


def test_unsupported_topic_is_rejected_before_any_experiment(tmp_path):
    class Missing(Spy):
        def generate(self, alias, stage, payload, schema, limit):
            result, cost = super().generate(alias, stage, payload, schema, limit)
            if payload["role"] == "topics":
                for t in result["topics"]:
                    t.update(domain="robot_training", required_resources=["gpu", "robot"], asset_id="missing")
            return result, cost
    state = OpenResearch(setup(tmp_path), Missing()).run()
    assert state["status"] == "stopped" and state["topic_round"] == 2
    assert not state["evaluation_exposed"] and "baseline-development" not in state["operations"]
    assert len(state["topic_pool"]) == 4
    assert all(t["status"] == "infeasible" for t in state["topic_pool"])


def test_critical_review_replans_with_actual_feedback(tmp_path):
    class Critic(Spy):
        def generate(self, alias, stage, payload, schema, limit):
            result, cost = super().generate(alias, stage, payload, schema, limit)
            if "topic-round-0" in stage:
                result.update(verdict="revise", reason="The proposed gap is already answered by prior work.", issues=["Choose an unanswered falsifiable question."])
            if payload["role"] == "topics" and payload["history"]:
                for t in result["topics"]:
                    t["question"] += " Under an alternative independent mechanism?"
            return result, cost
    worker = Critic()
    state = OpenResearch(setup(tmp_path), worker).run()
    assert state["status"] == "completed" and state["topic_round"] == 1
    assert all(t["status"] == "rejected" for t in state["topic_pool"][:2])
    history = next(p["history"] for p in worker.calls if p["role"] == "topics" and p["history"])
    assert "already answered" in history[0]["review_feedback"][0]["reason"]


def test_highest_priority_infeasible_topic_does_not_win(tmp_path):
    class Mixed(Spy):
        def generate(self, alias, stage, payload, schema, limit):
            result, cost = super().generate(alias, stage, payload, schema, limit)
            if payload["role"] == "topics":
                result["topics"][0]["metric"] = "reader_answer_accuracy"
            return result, cost
    state = OpenResearch(setup(tmp_path), Mixed()).run()
    assert state["status"] == "completed" and state["selected_topic"]["id"] == "fixture_1"


def test_search_errors_retained_and_bounded(tmp_path):
    class Offline(FixtureSearch):
        def search(self, query, limit, folder):
            raise OSError("Offline")
    state = OpenResearch(setup(tmp_path), Spy(), Offline()).run()
    assert state["status"] == "stopped" and state["worker_calls"] == 2
    assert len([op for op in state["operations"] if op.startswith("search-result")]) == 4
    assert "topics-0" not in state["operations"]


def test_metadata_only_cannot_support_a_topic(tmp_path):
    class Titles(FixtureSearch):
        def search(self, query, limit, folder):
            result = super().search(query, limit, folder)
            result["sources"][0]["evidence_level"] = "metadata_only"
            return result
    state = OpenResearch(setup(tmp_path), Spy(), Titles()).run()
    assert state["status"] == "stopped" and not state["reviews"]
    assert any("Title-only" in issue for t in state["topic_pool"] for issue in t["issues"])


def test_campaign_allowance_not_reset_when_topic_selected(tmp_path):
    class PaidFixture(Spy):
        def reservation(self, alias):
            return 10
        def generate(self, *args):
            result, _ = super().generate(*args)
            return result, 10
    worker = PaidFixture()
    state = OpenResearch(setup(tmp_path, model_budget={"deepseek": 50, "kimi": 50}), worker).run()
    assert state["status"] == "budget_exhausted"
    assert state["selected_topic"] and not state["evaluation_exposed"]
    assert state["charged_or_reserved"]["kimi"] == 50
    assert len(worker.calls) == 8  # Topic discovery/reviews consumed the same allowance.


def test_unknown_paid_intent_is_not_automatically_repeated(tmp_path):
    class Crash(Spy):
        def reservation(self, alias):
            return 10
        def generate(self, *args):
            raise RuntimeError("Simulated outcome unknown")
    root = setup(tmp_path, model_budget={"deepseek": 100, "kimi": 100})
    state = OpenResearch(root, Crash()).run()
    assert state["status"] == "needs_attention" and state["charged_or_reserved"]["kimi"] == 10
    worker = Spy()
    assert OpenResearch(root, worker).run()["status"] == "needs_attention"
    assert not worker.calls


def test_gateway_failure_has_no_paid_intent_and_can_resume(tmp_path):
    class Offline(Spy):
        def ready(self, alias):
            raise OSError("No gateway")
    root = setup(tmp_path)
    state = OpenResearch(root, Offline()).run()
    assert state["status"] == "waiting_for_gateway" and not state["operations"]
    assert OpenResearch(root, Spy()).run()["status"] == "completed"


def test_formal_evaluation_forbids_topic_search(tmp_path):
    root = setup(tmp_path)
    state = read_json(root / "autonomy-state.json")
    state["evaluation_exposed"] = True
    write_json(root / "autonomy-state.json", state)
    worker = Spy()
    state = OpenResearch(root, worker).run()
    assert state["status"] == "needs_attention" and not worker.calls
    assert "one-way" in state["last_error"]


def test_asset_cannot_override_budget_or_roles(tmp_path):
    root = setup(tmp_path)
    config = read_json(tmp_path / "config.json")
    config["assets"][0]["model_budget"] = {"kimi": 1000000000}
    write_json(tmp_path / "poisoned.json", config)
    with pytest.raises(ValueError, match="override"):
        initialize(tmp_path / "poisoned.json", tmp_path / "poisoned")


def test_input_and_discovery_artifact_tampering_rejected(tmp_path):
    root = setup(tmp_path)
    OpenResearch(root, Spy()).run(max_steps=2)
    path = root / "evidence/search-result-0-0/result.json"
    data = read_json(path)
    data["sources"][0]["text"] = "changed"
    write_json(path, data)
    with pytest.raises(ValueError, match="Evidence changed"):
        OpenResearch(root, Spy()).run()


def test_fabricated_quote_prevents_topic_execution(tmp_path):
    class Fabricator(Spy):
        def generate(self, alias, stage, payload, schema, limit):
            result, cost = super().generate(alias, stage, payload, schema, limit)
            if payload["role"] == "topics":
                for topic in result["topics"]:
                    topic["sources"][0]["excerpt"] = "This fabricated quote was never retrieved."
            return result, cost
    state = OpenResearch(setup(tmp_path), Fabricator()).run()
    assert state["status"] == "stopped" and not state["evaluation_exposed"]


@pytest.mark.parametrize("algorithm", ["centroid", "knn", "softmax"])
def test_classification_predictions_independent_of_test_labels(tmp_path, algorithm):
    root = setup(tmp_path)
    asset = read_json(root / "inputs/config.json")["assets"][0]
    config = {**asset, "max_local_seconds": 60}
    method = {"algorithm": algorithm, "standardize": True, "k": 1, "l2": 0, "epochs": 10}
    first = tmp_path / "first"
    result = evaluate(root, config, method, "evaluation", first, float("inf"))
    assert result["score"] == 1.0
    evaluation = root / asset["asset_directory"] / "inputs/evaluation.json"
    rows = read_json(evaluation)
    for r in rows:
        r["y"] = 1 - r["y"]
    write_json(evaluation, rows)
    second = tmp_path / "second"
    assert evaluate(root, config, method, "evaluation", second, float("inf"))["score"] == 0
    assert read_json(first / "predictions.json") == read_json(second / "predictions.json")


def test_relabelled_duplicate_feature_split_is_rejected():
    train = [{"id": "a", "x": [0.], "y": 0}, {"id": "b", "x": [1.], "y": 1}]
    dev = [{"id": "c", "x": [0.], "y": 1}, {"id": "d", "x": [.5], "y": 1}]
    with pytest.raises(ValueError, match="overlap"):
        validate_splits("tabular_classification", {"training": train, "development": dev})


def test_inactive_method_parameters_cannot_create_fake_candidates():
    with pytest.raises(ValueError, match="Inactive"):
        validate_method("tabular_classification", {"algorithm": "centroid", "standardize": True, "k": 3, "l2": 0, "epochs": 10})


def test_crossref_preserves_abstract_level_and_snapshot(tmp_path, monkeypatch):
    import research_lab.discovery_literature as literature
    data = {"message": {"items": [{"DOI": "10.1/test", "title": ["A classifier paper"], "abstract": "<jats:p>Actual publisher abstract.</jats:p>"},
                                  {"DOI": "10.1/metadata", "title": ["Metadata only paper"]}]}}
    class Response:
        remaining = json.dumps(data).encode()
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit):
            value, self.remaining = self.remaining, b""
            return value
    class Opener:
        def open(self, req, timeout):
            assert req.full_url.startswith("https://api.crossref.org/works?")
            return Response()
    monkeypatch.setattr(literature, "build_opener", lambda *args: Opener())
    result = CrossrefSearch().search("classification", 2, tmp_path)
    assert [s["evidence_level"] for s in result["sources"]] == ["abstract", "metadata_only"]
    assert "<jats" not in result["sources"][0]["text"]
    assert read_json(tmp_path / "provider-response.json") == data


def test_open_topic_dashboard_hides_raw_worker_identity(tmp_path):
    from research_lab.dashboard_data import DashboardStore
    class Ledger:
        def snapshot(self): return {"models": {}, "recent_requests": []}
    root = setup(tmp_path)
    OpenResearch(root, Spy()).run()
    snapshot = DashboardStore(tmp_path, Ledger()).snapshot(force=True)
    project = next(p for p in snapshot["projects"] if p["id"] == root.name)
    assert project["open_topic"] and project["topic_counts"]["approved"] == 2
    assert "worker_identity" not in json.dumps(project)
    assert project["selected_topic"]["id"] == "fixture_0"


def test_arxiv_abstract_is_attributed_and_xml_saved(tmp_path, monkeypatch):
    import research_lab.discovery_literature as literature
    xml = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/1234.56789v1</id>
      <title>Classification paper</title><summary>A genuine abstract text in the test provider response.</summary></entry></feed>'''
    def fetch(url, accept):
        assert url.startswith("https://export.arxiv.org/api/query?") and "max_results=2" in url
        return xml
    monkeypatch.setattr(literature, "fetch", fetch)
    monkeypatch.setattr(literature.time, "sleep", lambda _: None)
    result = literature.ArxivSearch().search("classification scaling", 2, tmp_path)
    assert result["sources"][0]["evidence_level"] == "abstract"
    assert result["sources"][0]["url"].startswith("https://arxiv.org/abs/")
    assert (tmp_path / "provider-response.xml").read_bytes() == xml


def test_deadline_stops_before_discovery_and_preserves_zero_usage(tmp_path, monkeypatch):
    import research_lab.open_research as research
    root, worker = setup(tmp_path), Spy()
    monkeypatch.setattr(research.time, "time", lambda: float("inf"))
    state = OpenResearch(root, worker).run()
    assert state["status"] == "deadline_exceeded" and not worker.calls


def test_missing_compute_can_fall_back_to_a_supported_topic(tmp_path):
    class Worker(Spy):
        def generate(self, alias, stage, payload, schema, limit):
            result, cost = super().generate(alias, stage, payload, schema, limit)
            if payload["role"] == "topics":
                result["topics"][0]["required_resources"] = ["gpu"]
            return result, cost
    state = OpenResearch(setup(tmp_path), Worker()).run()
    assert state["status"] == "completed"
    assert state["topic_pool"][0]["status"] == "infeasible"
    assert state["selected_topic"]["id"] == "fixture_1"


def test_fixture_and_real_campaigns_cannot_be_mixed(tmp_path):
    root = setup(tmp_path)
    class Real(Spy):
        fixture = False
    with pytest.raises(ValueError, match="Synthetic worker"):
        OpenResearch(root, Real()).run()


def test_open_selection_routes_to_stale_executor_without_label_prompts(tmp_path):
    config = read_json(ROOT / "config/projects/open-topic-fixture.json")
    asset = config["assets"][0]
    asset.update(domain="stale_retrieval", baseline={"relevance": 1, "recency": 0, "change": 0})
    del asset["training"]
    for split, city in (("development", "Austin"), ("evaluation", "Paris")):
        path = tmp_path / (split + ".json")
        write_json(path, [{"uid": split, "type": "T1", "M_old": "I live in Seattle.", "M_new": "I moved to " + city + ".",
                          "timestamps": [], "probing_queries": {"dim1": "Where should I live?"},
                          "haystack_session": [[{"role": "user", "content": "I live in Seattle."}],
                                               [{"role": "user", "content": "I moved to " + city + "."}]]}])
        asset[split] = str(path)
    write_json(tmp_path / "stale-config.json", config)
    root = tmp_path / "stale-project"
    initialize(tmp_path / "stale-config.json", root)
    class Worker(Spy):
        def generate(self, alias, stage, payload, schema, limit):
            result, cost = super().generate(alias, stage, payload, schema, limit)
            if payload["role"] == "propose":
                result["method"] = {"relevance": .7, "recency": .2, "change": .1}
            return result, cost
    worker = Worker()
    state = OpenResearch(root, worker).run()
    assert state["status"] == "completed" and state["analysis"]["metric"] == "literal_update_inclusion"
    assert '"M_new"' not in json.dumps(worker.calls)
    assert (root / "evidence/selected-evaluation/contexts.json").is_file()


def test_planning_history_is_bounded_without_changing_full_audit(tmp_path):
    root = setup(tmp_path)
    controller = OpenResearch(root, Spy())
    controller.campaign = read_json(root / "inputs/config.json")
    controller.state = read_json(root / "autonomy-state.json")
    controller.state["topic_pool"] = [{"key": str(i), "topic": {"question": "q" * 2000}, "status": "rejected",
        "issues": [], "review_feedback": [{"alias": "kimi", "verdict": "revise", "reason": "r" * 2000,
                                           "issues": ["x" * 2000] * 12}]} for i in range(25)]
    before = deepcopy(controller.state)
    history = controller.planning_history()
    assert len(history) == 10 and len(json.dumps(history).encode()) < 20000
    assert controller.state == before
    assert controller.execution_contract()["minimum_gain"] == .01

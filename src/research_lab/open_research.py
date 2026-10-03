"""Open topic discovery followed by one evidence-bound autonomous research study."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import time

from .autonomous import Research, TEXT, object_schema, implementation as core_implementation, TERMINAL
from .autonomous import BudgetLimit, GatewayUnavailable, DRAFT
from .autonomy_worker import FixtureWorker, NativeWorker
from .discovery_domains import CAPABILITIES, method_schema, validate_method, validate_splits, evaluate
from .discovery_literature import CrossrefSearch, ArxivSearch, FixtureSearch
from .project import contained, identity
from .storage import digest, read_json, write_json, run_lock, utc_now, event

SAFE_ID = r"^[a-z][a-z0-9_]{0,39}$"
SEARCH = object_schema({"queries": {"type": "array", "minItems": 1, "maxItems": 3,
                                   "uniqueItems": True, "items": {"type": "string", "minLength": 5, "maxLength": 200}}, "reason": TEXT})
SOURCE_REF = object_schema({"id": {"type": "string"}, "excerpt": {"type": "string", "minLength": 20, "maxLength": 600}, "relevance": TEXT})
TOPIC = object_schema({"id": {"type": "string", "pattern": SAFE_ID}, "title": TEXT, "question": TEXT,
    "hypothesis": TEXT, "falsifier": TEXT, "gap": TEXT, "closest_work_difference": TEXT,
    "asset_id": {"type": "string"}, "domain": {"type": "string"}, "metric": {"type": "string"},
    "required_resources": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
    "priority": {"type": "integer", "minimum": 0, "maximum": 100}, "limitations": TEXT,
    "sources": {"type": "array", "minItems": 1, "maxItems": 5, "items": SOURCE_REF}})
TOPICS = object_schema({"topics": {"type": "array", "minItems": 2, "maxItems": 5, "items": TOPIC}})
EARLY = {"search_plan", "search", "topics", "topic_review", "select"}


def implementation():
    pins = core_implementation()
    for name in ("open_research.py", "discovery_domains.py", "discovery_literature.py", "discovery_export.py"):
        pins[name] = digest(Path(__file__).with_name(name))
    return pins


def initialize(config_path, project):
    path, root = Path(config_path).resolve(), Path(project).resolve()
    c = read_json(path)
    if root.exists() or not re.fullmatch(r"[a-zA-Z0-9_-]+", root.name):
        raise ValueError("Use a new safe project ID")
    if c.get("version") != 1 or not isinstance(c.get("brief"), str) or not 10 <= len(c["brief"]) <= 4000:
        raise ValueError("A broad research brief is required")
    if type(c.get("fixture")) is not bool or c.get("literature_provider") not in {"crossref", "arxiv", "local", "fixture"}:
        raise ValueError("Declare fixture mode and literature provider")
    if (c["literature_provider"] == "fixture") != c["fixture"]:
        raise ValueError("Synthetic search is exclusive to fixture campaigns")
    for key, upper in {"max_topic_rounds": 5, "max_candidates": 20, "patience": 20, "max_review_revisions": 5,
                       "max_model_calls": 200, "max_wall_seconds": 604800, "max_local_seconds": 86400,
                       "search_results": 8}.items():
        if type(c.get(key)) is not int or not 1 <= c[key] <= upper:
            raise ValueError("Invalid campaign bound: " + key)
    if type(c.get("min_gain")) not in (int, float) or not 0 <= c["min_gain"] <= 1:
        raise ValueError("Invalid minimum gain")
    if type(c.get("compile_pdf", False)) is not bool:
        raise ValueError("compile_pdf must be boolean")
    roles = c.get("roles", {})
    if set(roles) != {"director", "writer", "reviewers"} or not isinstance(roles["reviewers"], list) or len(roles["reviewers"]) != 2 or len(set(roles["reviewers"])) != 2:
        raise ValueError("Distinct reviewer aliases required")
    aliases = {roles["director"], roles["writer"], *roles["reviewers"]}
    if set(c.get("model_budget", {})) != aliases or any(type(v) is not int or v < (0 if c["fixture"] else 1) for v in c["model_budget"].values()):
        raise ValueError("Explicit per-model total campaign budgets required")
    if c.get("resources") != ["cpu"]:
        raise ValueError("Current installed executors declare CPU resources only")
    if not isinstance(c.get("assets"), list) or not 1 <= len(c["assets"]) <= 6:
        raise ValueError("Register one to six local data assets")
    snapshots, assets, ids = {}, [], set()
    for spec in c["assets"]:
        allowed = {"id", "domain", "description", "provenance", "license", "unit_of_analysis", "data_exposure",
                   "training", "development", "evaluation", "baseline", "context_bytes"}
        if set(spec) - allowed:
            raise ValueError("Asset cannot override campaign policy or budgets")
        domain, aid = spec["domain"], spec["id"]
        if domain not in CAPABILITIES or not re.fullmatch(SAFE_ID, aid) or aid in ids:
            raise ValueError("Unknown capability or duplicate/unsafe asset ID")
        ids.add(aid)
        if (spec["data_exposure"] == "synthetic_fixture") != c["fixture"] or spec["data_exposure"] not in {"synthetic_fixture", "reused", "unseen_to_this_workflow"}:
            raise ValueError("Asset provenance does not match campaign mode")
        if domain == "fixture_threshold" and not c["fixture"]:
            raise ValueError("Fixture executor cannot run real research")
        for key in ("description", "provenance", "license", "unit_of_analysis"):
            if not isinstance(spec.get(key), str) or len(spec[key].strip()) < 5:
                raise ValueError("Asset needs declared description, provenance, license and independent evaluation unit")
        validate_method(domain, spec["baseline"])
        if type(spec.get("context_bytes", 4096)) is not int or not 1 <= spec.get("context_bytes", 4096) <= 65536:
            raise ValueError("Invalid context budget")
        names = ["development", "evaluation"] + (["training"] if CAPABILITIES[domain]["needs_training"] else [])
        splits = {name: read_json((path.parent / spec[name]).resolve()) for name in names}
        validate_splits(domain, splits)
        directory = "inputs/assets/" + aid
        for name, rows in splits.items():
            snapshots[directory + "/inputs/" + name + ".json"] = rows
        assets.append({**{k: v for k, v in spec.items() if k not in names}, "asset_directory": directory,
                       "split_sizes": {k: len(v) for k, v in splits.items()}})
    sources, source_ids = [], set()
    for source in c.get("sources", []):
        sid = source["id"]
        if not re.fullmatch(SAFE_ID, sid) or sid in source_ids:
            raise ValueError("Unique safe source IDs required")
        source_ids.add(sid)
        text = (path.parent / source["path"]).resolve().read_text("utf-8-sig")
        if not 20 <= len(text.encode()) <= 20000 or source.get("evidence_level") not in {"full_text_extract", "abstract", "synthetic_fixture"}:
            raise ValueError("Bound source text and declare its evidence level")
        if (source["evidence_level"] == "synthetic_fixture") != c["fixture"] or not source["url"].startswith("https://"):
            raise ValueError("Source provenance/attribution is invalid")
        sources.append({**{k: v for k, v in source.items() if k != "path"}, "text": text,
                        "verification": "User supplied snapshot; provenance declared, semantic support not certified"})
    if sum(len(s["text"].encode()) for s in sources) > 40000:
        raise ValueError("Supplied source bundle exceeds 40KB")
    if c["literature_provider"] == "local" and not sources:
        raise ValueError("Local literature mode needs supplied sources")
    c["assets"] = assets
    c["sources"] = [{k: v for k, v in s.items() if k != "text"} for s in sources]
    root.mkdir(parents=True)
    for name, data in {**snapshots, "inputs/config.json": c, "inputs/sources.json": sources}.items():
        write_json(root / name, data)
    pins = {p.relative_to(root).as_posix(): digest(p) for p in (root / "inputs").rglob("*.json")}
    state = {"schema_version": 8, "id": root.name, "open_topic": True, "question": c["brief"],
        "status": "ready", "phase": "search_plan", "created_at": utc_now(), "deadline": time.time() + c["max_wall_seconds"],
        "input_hashes": pins, "implementation": implementation(), "worker_identity": None, "search_identity": None,
        "operations": {}, "reviews": [], "trials": [], "attempt": 0, "stagnant": 0, "draft_revision": 0,
        "best_method": None, "best_score": None, "evaluation_exposed": False, "model_calls": 0, "worker_calls": 0,
        "fixture_only": c["fixture"], "charged_or_reserved": {a: 0 for a in aliases}, "decisions": [],
        "topic_round": 0, "topic_pool": [], "search_index": 0,
        "scientific_success_not_implied": True, "external_submission": False}
    write_json(root / "autonomy-state.json", state)
    return state


class OpenResearch(Research):
    def __init__(self, root, worker, searcher=None):
        super().__init__(root, worker)
        provider = read_json(self.root / "inputs/config.json")["literature_provider"]
        self.searcher = searcher or {"fixture": FixtureSearch, "crossref": CrossrefSearch, "arxiv": ArxivSearch, "local": CrossrefSearch}[provider]()

    def verify(self):
        if self.state.get("schema_version") != 8 or self.state["implementation"] != implementation():
            raise ValueError("Use the campaign's recorded implementation checkout")
        for name, sha in self.state["input_hashes"].items():
            if digest(contained(self.root, name)) != sha:
                raise ValueError("Frozen input changed: " + name)
        for op in self.state["operations"].values():
            if op["status"] == "completed":
                for name, sha in op["files"].items():
                    if digest(contained(self.root, name)) != sha:
                        raise ValueError("Evidence changed: " + name)

    def restore_context(self):
        self.config = deepcopy(self.campaign)
        sources = {s["id"]: s for s in read_json(self.root / "inputs/sources.json")}
        for name, op in self.state["operations"].items():
            if name.startswith("search-result-") and op["status"] == "completed":
                for source in read_json(self.root / "evidence" / name / "result.json")["sources"]:
                    sources.setdefault(source["id"], source)
        # Deterministic bounded selection; full raw provider snapshots remain in evidence.
        self.sources, size = [], 0
        for source in sources.values():
            length = len(json.dumps(source, ensure_ascii=False).encode())
            if size + length <= 55000:
                self.sources.append(source)
                size += length
        if self.state.get("selected_topic"):
            topic = self.state["selected_topic"]
            asset = next(a for a in self.campaign["assets"] if a["id"] == topic["asset_id"])
            self.config.update(asset, question=topic["question"], context_bytes=asset.get("context_bytes", 4096))

    def feasibility(self, topic):
        issues = []
        asset = next((a for a in self.campaign["assets"] if a["id"] == topic["asset_id"]), None)
        if topic["domain"] not in CAPABILITIES:
            issues.append("No trusted executor for the proposed domain")
        if asset is None or asset["domain"] != topic["domain"]:
            issues.append("Missing or incompatible registered data asset")
        if not set(topic["required_resources"]) <= set(self.campaign["resources"]):
            issues.append("Required compute/tool resources are unavailable")
        if topic["domain"] in CAPABILITIES and topic["metric"] != CAPABILITIES[topic["domain"]]["metric"]:
            issues.append("The evaluator cannot measure the proposed primary claim")
        registry = {s["id"]: s for s in self.sources}
        for ref in topic["sources"]:
            if ref["id"] not in registry or ref["excerpt"] not in registry[ref["id"]]["text"]:
                issues.append("Source excerpt is not present in the retrieved evidence")
        if not any(registry.get(r["id"], {}).get("evidence_level") != "metadata_only" and r["id"] in registry for r in topic["sources"]):
            issues.append("Title-only metadata cannot support a research gap")
        fingerprint = identity(topic["question"].strip().casefold())
        if any(identity(t["topic"]["question"].strip().casefold()) == fingerprint for t in self.state["topic_pool"]):
            issues.append("Duplicate research question")
        return issues

    def planning_history(self):
        """Bound planning context without discarding the immutable full review evidence."""
        topics = self.state["topic_pool"]
        return [{"key": t["key"], "question": t["topic"]["question"][:400], "status": t["status"],
                 "issues": t.get("issues", [])[:4],
                 "review_feedback": [{"alias": r["alias"], "verdict": r["verdict"], "reason": r["reason"][:180],
                                      "issues": [issue[:220] for issue in r["issues"][:4]]}
                                     for r in t.get("review_feedback", [])]}
                for t in topics[-10:]]

    def execution_contract(self):
        return {"method_schemas": {a["id"]: method_schema(a["domain"]) for a in self.campaign["assets"]},
                "max_development_candidates": self.campaign["max_candidates"],
                "minimum_gain": self.campaign["min_gain"],
                "selection": "Adapt methods on development only; strict gain > minimum_gain replaces incumbent; ties retain incumbent. Baseline can remain selected.",
                "formal_evaluation": "One frozen selected-versus-baseline contrast, 10000 paired scenario bootstrap replicates, seed 20261003; scenario means average queries. Gain criterion: lower 95% CI > minimum_gain. No formal grid-wide testing.",
                "scope": "No subgroup regression, causal identification, per-type hypothesis tests, additional metrics or reader responses in this executor. Do not promise them in the question/falsifier.",
                "stale_packing": "User turns only. BM25 score divided by query maximum; recency is normalized turn index; change is a binary cue. Descending combined score, then later-index tie break. Greedily skip turns that exceed remaining UTF-8 bytes; emit selected turns chronologically. Literal matching uses lowercased collapsed whitespace.",
                "review_stage": "This is topic selection. Exact new method parameters are specified and separately reviewed at design stage before candidate execution; the primary endpoint, selection and inference rules above are already fixed."}

    def proposal(self):
        schema = object_schema({"title": TEXT, "hypothesis": TEXT, "falsifier": TEXT,
            "method": method_schema(self.config["domain"]), "limitations": TEXT,
            "sources": {"type": "array", "minItems": 1, "maxItems": 5, "items": SOURCE_REF}})
        return self.ask("proposal-" + str(self.state["attempt"]), self.config["roles"]["director"],
            {"role": "propose", "question": self.config["question"], "domain": self.config["domain"],
             "topic": self.state["selected_topic"], "sources": self.sources, "attempt": self.state["attempt"],
             "history": self.state["trials"], "feedback": self.state.get("feedback", []), "baseline": self.config["baseline"],
             "capability": CAPABILITIES[self.config["domain"]], "task": "Propose a falsifiable bounded method with exact source excerpts. "
             "Inactive parameters must be canonical: k=1 except knn; l2=0 and epochs=10 except softmax. "
             "No code execution, new metric, new split, or claims of established novelty."}, schema)

    def experiment(self, name, method, split):
        directory = self.config["asset_directory"] + "/inputs/"
        pins = {k: v for k, v in self.state["input_hashes"].items() if k.startswith(directory)}
        return self.memo(name, {"method": method, "split": split, "data_hashes": pins,
                               "topic_binding": identity(self.state["selected_topic"])},
            lambda folder, _: (evaluate(self.root, self.config, method, split, folder, self.state["deadline"]), 0))

    def transition(self):
        s, c = self.state, self.config
        phase, rnd = s["phase"], s["topic_round"]
        if s["evaluation_exposed"] and phase in EARLY | {"propose", "development", "next", "freeze", "design_review"}:
            raise ValueError("Formal evaluation is one-way; no return to topic/method search")
        if phase == "search_plan":
            plan = self.ask(f"search-plan-{rnd}", c["roles"]["director"], {"role": "search_plan", "brief": c["brief"],
                "assets": c["assets"], "capabilities": CAPABILITIES, "previous_topics": self.planning_history(),
                "execution_contract": self.execution_contract(),
                "task": "Generate diverse focused literature queries and one query looking for already solved/contradictory work. "
                "Use concise English scientific keyword queries (two to five terms), not sentences or URLs. "
                "Do not reveal secrets or dataset rows in public queries. Select questions independently of formal evaluation results."}, SEARCH)
            s.update(search_plan=plan, search_index=0, phase="search")
        elif phase == "search":
            if c["literature_provider"] == "local" or s["search_index"] >= len(s["search_plan"]["queries"]):
                s["phase"] = "topics"
                return
            index = s["search_index"]
            query = s["search_plan"]["queries"][index]
            if not self.worker.fixture and s["deadline"] - time.time() < 50:
                raise TimeoutError("Insufficient remaining time for a bounded literature request")
            def search(folder, _):
                try:
                    return self.searcher.search(query, c["search_results"], folder), 0
                except (OSError, ValueError) as exc:
                    # Read-only discovery failures are definitive; retain them and try the next bounded query.
                    return {"query": query, "sources": [], "error": type(exc).__name__,
                            "coverage": "Search failed; no evidence inferred"}, 0
            self.memo(f"search-result-{rnd}-{index}", {"query": query, "provider": self.searcher.identity, "limit": c["search_results"]},
                search)
            s["search_index"] += 1
            self.restore_context()
        elif phase == "topics":
            if not self.sources:
                s["topic_round"] += 1
                if s["topic_round"] >= c["max_topic_rounds"]:
                    s.update(status="stopped", stop_reason="No usable literature evidence; no invented references")
                else:
                    s["phase"] = "search_plan"
                return
            result = self.ask(f"topics-{rnd}", c["roles"]["director"], {"role": "topics", "brief": c["brief"],
                "assets": c["assets"], "capabilities": CAPABILITIES, "resources": c["resources"], "sources": self.sources,
                "history": self.planning_history(), "execution_contract": self.execution_contract(),
                "task": "Propose two to five substantively different falsifiable research questions. "
                "Every question and falsifier must match the fixed executable analysis contract; do not invent a different testing procedure. "
                "Give closest-work differences and exact source excerpts; an abstract is not full-text verification. "
                "Priority measures relevance and information value, never expected positive results. Unsupported topics must retain honest requirements. "
                "Do not assume novelty from missing search results. Independent reviewers must assess whether available source depth suffices."}, TOPICS)
            for topic in result["topics"]:
                issues = self.feasibility(topic)
                s["topic_pool"].append({"key": f"round-{rnd}-topic-{len(s['topic_pool'])}", "round": rnd,
                    "topic": topic, "status": "infeasible" if issues else "pending_review", "issues": issues})
            s["phase"] = "topic_review"
        elif phase == "topic_review":
            item = next((t for t in s["topic_pool"] if t["status"] == "pending_review"), None)
            if item is None:
                s["phase"] = "select"
                return
            verdict = self.review("topic-" + item["key"], {"topic": item["topic"], "brief": c["brief"],
                "capabilities": CAPABILITIES, "sources": self.sources, "assets": c["assets"],
                "execution_contract": self.execution_contract(),
                "task": "Judge significance, falsifiability, closest prior work, source depth and whether the registered metric actually tests the hypothesis. "
                "Reject trivial parameter sweeps presented as novel work, already answered questions, unjustified independence, unsupported gaps and resources. "
                "An abstract may justify an exploratory audit, but cannot establish a full-text novelty claim."})
            item["status"] = "approved" if verdict == "approve" else "rejected"
            item["review_id"] = "topic-" + item["key"]
            item["review_feedback"] = [{"alias": r["alias"], "verdict": r["verdict"], "reason": r["reason"][:600],
                                        "issues": [i[:300] for i in r["issues"][:4]]} for r in s["reviews"][-1]["reviews"]]
        elif phase == "select":
            eligible = [t for t in s["topic_pool"] if t["status"] == "approved"]
            if not eligible:
                s["topic_round"] += 1
                if s["topic_round"] >= c["max_topic_rounds"]:
                    s.update(status="stopped", stop_reason="No feasible independently reviewed topic within discovery allowance")
                else:
                    s["phase"] = "search_plan"
                return
            chosen = sorted(eligible, key=lambda t: (-t["topic"]["priority"], t["key"]))[0]
            selection = {"topic": chosen["topic"], "topic_key": chosen["key"], "pool": s["topic_pool"],
                         "rule": "Highest director priority among feasible dual-reviewed topics; stable key tie break; no evaluation feedback"}
            frozen = self.memo("selected-topic", selection, lambda folder, _: (selection, 0))
            s.update(selected_topic=frozen["topic"], question=frozen["topic"]["question"], phase="baseline")
            self.restore_context()
            s["best_method"] = self.config["baseline"]
        elif phase == "propose":
            if s["attempt"] >= c["max_candidates"] or s["stagnant"] >= c["patience"]:
                s["phase"] = "freeze"
                return
            proposal = self.proposal()
            try:
                validate_method(c["domain"], proposal["method"])
                fingerprint = identity(proposal["method"])
                if fingerprint == identity(c["baseline"]) or any(t.get("fingerprint") == fingerprint for t in s["trials"]):
                    raise ValueError("Duplicate method")
                registry = {x["id"]: x for x in self.sources}
                for ref in proposal["sources"]:
                    if ref["id"] not in registry or ref["excerpt"] not in registry[ref["id"]]["text"]:
                        raise ValueError("Unverified source excerpt")
            except ValueError as exc:
                self.reject(str(exc))
                return
            s.update(proposal=proposal, phase="design_review")
        elif phase == "freeze":
            if not any(t["status"] == "evaluated" for t in s["trials"]):
                s.update(status="stopped", stop_reason="No executable candidate passed design review")
                return
            protocol = {"question": c["question"], "selected": s["best_method"], "baseline": c["baseline"],
                "development_history": s["trials"], "data_exposure": c["data_exposure"], "asset_id": c["id"],
                "data_hashes": {k: v for k, v in s["input_hashes"].items() if k.startswith(c["asset_directory"])},
                "capability": CAPABILITIES[c["domain"]], "min_gain": c["min_gain"], "context_bytes": c["context_bytes"],
                "topic_binding": identity(s["selected_topic"]), "implementation": s["implementation"],
                "primary": "Selected minus baseline mean per independent unit; paired bootstrap 10000 samples seed20261003",
                "selection_rule": "Development only; strict gain > min_gain; no topic reselection after any formal evaluation",
                "unit_of_analysis": c["unit_of_analysis"]}
            if self.review("freeze-" + str(s["attempt"]), protocol) == "approve":
                s["frozen"] = self.memo("frozen-protocol", protocol, lambda folder, _: (protocol, 0))
                s["phase"] = "evaluation"
            else:
                s.update(status="stopped", stop_reason="Frozen protocol did not pass review")
        elif phase == "interpretation":
            verdict = self.review("interpretation", {"protocol": s["frozen"], "analysis": s["analysis"],
                "task": "Preserve negative and inconclusive results; restrict claims to the frozen metric, source depth and data provenance. "
                "No established novelty, causality, generalization or venue score."})
            s["interpretation_verdict"] = verdict
            if verdict == "approve":
                s["phase"] = "write"
            else:
                s.update(status="stopped", stop_reason="Interpretation not supported; all evidence retained")
        elif phase == "write":
            s["draft"] = self.ask("draft-" + str(s["draft_revision"]), c["roles"]["writer"],
                {"role": "write", "topic": s["selected_topic"], "protocol": s["frozen"], "analysis": s["analysis"],
                 "sources": self.sources, "feedback": s.get("writing_feedback", []),
                 "task": "Write concise English research prose grounded in the supplied evidence, explicitly distinguish abstract/metadata/full-text extracts. "
                 "Cite only supplied IDs. No claims of established novelty or publication quality. Preserve negative results. "
                 "Do not write digits or numerical claims in prose; use {{primary_result}} for deterministic quantitative results."}, DRAFT)
            prose = " ".join(s["draft"][k] for k in ("title", "abstract", "introduction", "discussion", "limitations"))
            if not set(s["draft"]["citations"]) <= {x["id"] for x in self.sources} or re.search(r"\d", prose) or "{{" in prose.replace("{{primary_result}}", ""):
                s["writing_feedback"] = ["Use only supplied citations and {{primary_result}}; no digits/quantities in prose."]
                self.rewrite_or_stop()
            else:
                s["phase"] = "paper_review"
        elif phase == "export":
            from .discovery_export import export
            s["deliverables"] = self.memo("deliverables", {"draft": s["draft"], "analysis": s["analysis"], "protocol": s["frozen"]},
                lambda folder, _: (export(folder, c, s, self.sources), 0))
            s.update(status="completed", phase="done")
        else:
            super().transition()

    def run(self, *, max_steps=None):
        with run_lock(self.root):
            self.state = read_json(self.path)
            self.verify()
            self.campaign = read_json(self.root / "inputs/config.json")
            self.restore_context()
            if self.worker.fixture != self.campaign["fixture"]:
                raise ValueError("Synthetic worker cannot run a real campaign")
            for key, current in (("worker_identity", self.worker.identity), ("search_identity", self.searcher.identity)):
                if self.state[key] is not None and self.state[key] != current:
                    raise ValueError("Campaign provider/ledger changed")
            if not self.worker.fixture:
                a, b = self.campaign["roles"]["reviewers"]
                routes = self.worker.identity["routes"]
                if (routes[a]["base_url"], routes[a]["model"]) == (routes[b]["base_url"], routes[b]["model"]):
                    raise ValueError("Independent review requires distinct models")
            if self.state["status"] in TERMINAL:
                return self.state
            self.state.update(status="running", last_error=None, worker_identity=self.worker.identity, search_identity=self.searcher.identity)
            self.save()
            steps = 0
            try:
                while self.state["status"] == "running" and (max_steps is None or steps < max_steps):
                    if time.time() >= self.state["deadline"]:
                        raise TimeoutError("Total discovery and research deadline exceeded")
                    self.transition()
                    self.save()
                    event(self.root / "events.jsonl", "open_research_transition", status=self.state["status"], decision=self.state["phase"])
                    steps += 1
                if self.state["status"] == "running":
                    self.state["status"] = "ready"
            except BudgetLimit as exc:
                self.state.update(status="budget_exhausted", last_error=str(exc))
            except GatewayUnavailable as exc:
                self.state.update(status="waiting_for_gateway", last_error=str(exc))
            except TimeoutError as exc:
                self.state.update(status="deadline_exceeded", last_error=str(exc))
            except Exception as exc:
                self.state.update(status="needs_attention", last_error=type(exc).__name__ + ": " + str(exc))
            self.save()
            return self.state


class OpenFixtureWorker(FixtureWorker):
    identity = {"backend": "synthetic-open-topic-fixture-v1"}

    def generate(self, alias, stage, payload, schema, limit):
        role = payload["role"]
        if role == "search_plan":
            return {"queries": ["synthetic classifier robustness", "synthetic classifier prior limitations"], "reason": "Scripted software acceptance queries only."}, 0
        if role == "topics":
            asset, source = payload["assets"][0], payload["sources"][0]
            topics = []
            for i in range(2):
                topics.append({"id": "fixture_" + str(i), "title": "Synthetic open-topic workflow test",
                    "question": "Does standardized classification improve synthetic accuracy?" if i == 0 else "Does local voting improve synthetic classification accuracy?",
                    "hypothesis": "The bounded candidate improves the synthetic development metric.",
                    "falsifier": "No improvement on the frozen evaluation metric.", "gap": "A scripted fixture gap, not a scientific novelty claim.",
                    "closest_work_difference": "Workflow validation only; no research contribution is established.",
                    "domain": asset["domain"], "asset_id": asset["id"], "metric": CAPABILITIES[asset["domain"]]["metric"],
                    "required_resources": ["cpu"], "priority": 80 - i,
                    "limitations": "Synthetic source and data; cannot validate autonomous scientific reasoning.",
                    "sources": [{"id": source["id"], "excerpt": source["text"][:100], "relevance": "Synthetic orchestration evidence only."}]})
            return {"topics": topics}, 0
        if role == "propose" and payload["domain"] == "tabular_classification":
            result, cost = super().generate(alias, stage, payload, schema, limit)
            result["method"] = {"algorithm": "knn", "standardize": True, "k": 1 if payload["attempt"] == 0 else 3, "l2": 0, "epochs": 10}
            return result, cost
        return super().generate(alias, stage, payload, schema, limit)


def status(project):
    from .autonomous import status as bounded_status
    state = read_json(Path(project) / "autonomy-state.json")
    return {**bounded_status(project), "open_topic": True, "topic_round": state["topic_round"],
            "topic_pool": state["topic_pool"], "selected_topic": state.get("selected_topic")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("project")
    init.add_argument("--config", required=True)
    run = commands.add_parser("run")
    run.add_argument("project")
    mode = run.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture", action="store_true")
    mode.add_argument("--allow-models", action="store_true")
    run.add_argument("--max-steps", type=int)
    commands.add_parser("status").add_argument("project")
    args = parser.parse_args(argv)
    if args.command == "init":
        initialize(args.config, args.project)
    elif args.command == "run":
        if args.max_steps is not None and args.max_steps < 1:
            parser.error("max-steps must be positive")
        OpenResearch(args.project, OpenFixtureWorker() if args.fixture else NativeWorker()).run(max_steps=args.max_steps)
    print(json.dumps(status(args.project), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

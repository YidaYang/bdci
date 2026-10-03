"""Read-only, bounded projection of research artifacts for the local dashboard."""

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

from .framework import verify_artifact_manifest
from .model_client import redact
from .storage import utc_now

ARTIFACT_NAMES = {"paper_draft.md", "summary.json", "claims.json", "audit.json", "selection.json",
    "search_history.json", "resource_report.json", "resource_report.md", "results.csv", "review.json",
    "review.md", "proposal.json", "proposal.md", "supervisor_review.md", "revision_plan.md", "assignment.json", "results.json"}
ACTIVE = {"running", "searching", "evaluating_holdout", "writing", "requesting"}


class DashboardStore:
    def __init__(self, root, ledger, *, background_integrity=False):
        self.root = Path(root).resolve()
        self.ledger = ledger
        self.lock = threading.RLock()
        self.cache = None
        self.cache_at = 0
        self.integrity_cache = {}
        self.artifacts = {}
        self.background_integrity = background_integrity
        self.integrity_pending = set()
        self.integrity_worker = threading.Semaphore(1)

    def relative(self, path):
        path = Path(path)
        resolved = path.resolve()
        resolved.relative_to(self.root)
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != self.root.parent):
            raise ValueError("Symlink artifacts are not exposed")
        return resolved.relative_to(self.root).as_posix()

    def read(self, path, default=None):
        if not path.exists():
            return default
        self.relative(path)
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Metadata exceeds dashboard size limit")
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def register(self, path):
        if path.is_file():
            name = self.relative(path)
            self.artifacts[name] = path
            return {"path": name, "name": path.name, "bytes": path.stat().st_size}

    def artifact(self, name):
        with self.lock:
            self.snapshot(force=True)
            path = self.artifacts.get(name)
            if path is None or self.relative(path) != name:
                raise ValueError("Artifact is not in the explicit dashboard allowlist")
            with path.open("r", encoding="utf-8-sig") as stream:
                text = stream.read(250001)
            return {"path": name, "text": redact(text[:250000]), "truncated": len(text) > 250000}

    def integrity(self, root, state):
        if self.background_integrity and state.get("status") == "completed":
            key = self.relative(root)
            cached = self.integrity_cache.get(key)
            if (not cached or time.monotonic() - cached[0] >= 15) and key not in self.integrity_pending:
                self.integrity_pending.add(key)
                def check():
                    try:
                        with self.integrity_worker:
                            self.integrity_sync(root, state)
                    finally:
                        self.integrity_pending.discard(key)
                threading.Thread(target=check, daemon=True).start()
            return cached[1] if cached else {"status":"pending", "note":"后台验证产物哈希；不阻塞实时任务进度"}
        return self.integrity_sync(root, state)

    def integrity_sync(self, root, state):
        if state.get("status") != "completed":
            return {"status": "pending", "note": "运行尚未完成"}
        name = "campaign_manifest.json" if state.get("schema_version") == 2 else "artifact_manifest.json"
        manifest = root / name
        if not manifest.is_file():
            return {"status": "unavailable", "note": "没有完整性清单"}
        key = self.relative(root)
        cached = self.integrity_cache.get(key)
        if cached and time.monotonic() - cached[0] < 15:
            return cached[1]
        try:
            value = self.read(manifest)
            # Validate containment before handing entries to the upstream verifier.
            for entry in value.get("artifacts", []):
                path = root / entry["path"]
                path.resolve().relative_to(root.resolve())
                self.relative(path)
            errors = verify_artifact_manifest(root, value)
            result = {"status": "invalid" if errors else "valid", "checked_at": utc_now(),
                      "note": "产物哈希检查；不等于科学结论认证", "errors": errors[:5]}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result = {"status": "invalid", "checked_at": utc_now(), "errors": [str(exc)]}
        self.integrity_cache[key] = (time.monotonic(), result)
        return result

    def grid(self, root):
        state = self.read(root / "state.json", {})
        plan = self.read(root / "inputs/plan.json", {})
        jobs = state.get("jobs", {})
        total = len(plan.get("methods", [])) * len(plan.get("seeds", []))
        rows = [{"id": key, "status": value.get("status"), "attempts": len(value.get("attempts", []))}
                for key, value in jobs.items()]
        return {"name": root.name, "status": state.get("status", "pending"), "total": total,
                "completed": sum(row["status"] == "completed" for row in rows),
                "attempts": sum(row["attempts"] for row in rows), "jobs": rows}

    def events(self, path, owner):
        if not path.is_file():
            return []
        self.relative(path)
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
        rows = []
        for line in lines[-100:]:
            try:
                record = json.loads(line)
                rows.append({"owner": owner, **{key: record[key] for key in
                    ("time", "event", "job", "candidate", "decision", "status", "error", "method") if key in record}})
            except (ValueError, TypeError):
                continue  # A writer may still be appending the last line.
        return rows

    def snapshot(self, force=False):
        with self.lock:
            if not force and self.cache and time.monotonic() - self.cache_at < 1:
                return self.cache
            self.artifacts = {}
            runs, tasks, events, warnings = [], [], [], []
            outputs = self.root / "outputs"
            for folder in sorted(outputs.iterdir()) if outputs.exists() else []:
                if not folder.is_dir() or folder.is_symlink() or not (folder / "state.json").is_file():
                    continue
                try:
                    self.relative(folder)
                    state = self.read(folder / "state.json")
                    artifacts = [value for name in sorted(ARTIFACT_NAMES)
                                 if (value := self.register(folder / name))]
                    modified = datetime.fromtimestamp((folder / "state.json").stat().st_mtime, timezone.utc).isoformat()
                    common = {"id": folder.name, "status": state.get("status", "unknown"), "modified_at": modified,
                              "artifacts": artifacts, "error": state.get("last_error", state.get("error"))}
                    if "jobs" in state or "history" in state:
                        campaign = "history" in state
                        protocol = self.read(folder / ("inputs/protocol.json" if campaign else "inputs/plan.json"), {})
                        grids = [self.grid(path) for path in sorted((folder / "runs").iterdir(), key=lambda p: (p.name == "holdout", p.name)) if path.is_dir() and not path.is_symlink()] if campaign and (folder / "runs").is_dir() else [self.grid(folder)]
                        run = {**common, "kind": "campaign" if campaign else "grid", "title": protocol.get("title", folder.name),
                               "question": protocol.get("question", ""), "metric": protocol.get("primary_metric"),
                               "direction": protocol.get("direction"), "grids": grids,
                               "completed_jobs": sum(x["completed"] for x in grids), "total_jobs": sum(x["total"] for x in grids),
                               "attempts": sum(x["attempts"] for x in grids),
                               "summary": self.read(folder / "summary.json", {}),
                               "resources": self.read(folder / "resource_report.json", {}),
                               "selection": self.read(folder / "selection.json", {}),
                               "history": [{k: v for k, v in item.items() if k != "manifest"} for item in state.get("history", [])],
                               "planned_trials": protocol.get("search", {}).get("max_trials"),
                               "budget": {k: v for k, v in state.get("budget", {}).items() if k != "reservations"},
                               "integrity": self.integrity(folder, state),
                               "audit": self.read(folder / "audit.json", None)}
                        runs.append(run)
                        events.extend(self.events(folder / "events.jsonl", folder.name))
                    elif state.get("schema_version") == 4:
                        tasks.append({**common,"task_id":folder.name,"model":"SwarmFlow / DeepSeek + Kimi",
                            "objective":f"原生工作流：{state.get('phase') or folder.name}；已完成 {state.get('completed_tasks',0)}/{state.get('total_tasks','?')} 个有限任务",
                            "supervisor_decision":"see_v4_evidence","applied":False,
                            "usage":{"total_tokens":state.get("reported_tokens",0)}})
                        events.extend(self.events(folder / "events.jsonl", folder.name))
                    else:
                        assignment = self.read(folder / "assignment.json", {})
                        tasks.append({**common, "task_id": state.get("task_id", folder.name),
                            "model": state.get("model_alias", state.get("alias", "unknown")),
                            "objective": assignment.get("objective", "现有实验的模型证据审查"),
                            "supervisor_decision": state.get("supervisor_decision", "pending"),
                            "applied": state.get("applied_to_research", False), "usage": state.get("usage"),
                            "request_id": state.get("request_id")})
                        events.append({"time": modified, "event": "supervisor_reviewed" if state.get("status") == "supervisor_reviewed" else "model_task_updated",
                                       "owner": folder.name, "status": state.get("status"), "error": state.get("error")})
                except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
                    warnings.append({"source": folder.name, "message": f"记录暂不可读: {type(exc).__name__}"})
            for path in sorted((self.root / "config/tasks").glob("*.json")):
                try:
                    task = self.read(path)
                    if not any(x["task_id"] == task["task_id"] for x in tasks):
                        tasks.append({"id": path.stem, "task_id": task["task_id"], "status": "planned",
                            "model": task["model_alias"], "objective": task["objective"], "supervisor_decision": "not_started",
                            "applied": False, "usage": None, "artifacts": [self.register(path)]})
                except (OSError, ValueError, TypeError, KeyError):
                    warnings.append({"source": path.name, "message": "任务定义暂不可读"})
            try:
                roadmap = self.read(self.root / "research/progress.json", {})
            except (OSError, ValueError):
                roadmap = {}
                warnings.append({"source": "research/progress.json", "message": "研究进度记录暂不可读"})
            documents = []
            for pattern in ("research/**/*.md", "docs/*看板*.md"):
                for path in sorted(self.root.glob(pattern)):
                    try:
                        item = self.register(path)
                        if item:
                            documents.append(item)
                    except (OSError, ValueError):
                        pass
            usage = self.ledger.snapshot()
            for request in usage["recent_requests"]:
                events.append({"time": request.get("finished_at") or request["started_at"],
                               "event": {"settled": "model_completed", "unknown": "model_usage_unknown"}.get(request["status"], "model_started"),
                               "owner": request["alias"], "job": request["purpose"], "error": request.get("error")})
            # Only public accounting fields; no gateway URL, credentials, prompts or raw responses.
            for info in usage["models"].values():
                info.pop("base_url", None)
            alerts = list(warnings)
            for alias, info in usage["models"].items():
                if info["unknown_requests"]:
                    alerts.append({"source": alias, "message": f"{info['unknown_requests']} 次请求用量未结算，保留预留额度"})
                recent = [r for r in usage["recent_requests"] if r["alias"] == alias]
                if recent and recent[-1].get("error"):
                    alerts.append({"source": alias, "message": recent[-1]["error"]})
            for task in tasks:
                if task["status"] in ("failed", "awaiting_supervisor"):
                    alerts.append({"source": task["id"], "message": task.get("error") or "模型建议等待监督者审核"})
            for run in runs:
                if run["integrity"]["status"] == "invalid":
                    alerts.append({"source": run["id"], "message": "产物完整性检查失败，请核对证据"})
            totals = {"runs": len(runs), "completed_runs": sum(r["status"] == "completed" for r in runs),
                      "completed_jobs": sum(r["completed_jobs"] for r in runs), "attempts": sum(r["attempts"] for r in runs),
                      "active": sum(x["status"] in ACTIVE for x in runs + tasks),
                      "pending_reviews": sum(x["status"] == "awaiting_supervisor" for x in tasks),
                      "reviewed_tasks": sum(x["status"] == "supervisor_reviewed" for x in tasks)}
            value = {"updated_at": utc_now(), "roadmap": roadmap, "runs": sorted(runs, key=lambda r: r["modified_at"], reverse=True),
                     "tasks": tasks, "events": sorted(events, key=lambda e: e.get("time", ""), reverse=True)[:80],
                     "usage": usage, "alerts": alerts, "documents": documents, "totals": totals,
                     "integrity_refresh_seconds": 15}
            value["research_v4"] = self.read(self.root / "research/v4/monitor.json", {})
            value["projects"] = []
            from .project import Project
            from .stale_adapters_v5 import registry as research_registry
            for project_path in sorted((self.root / "projects").glob("*/project-state.json")):
                try:
                    if project_path.is_symlink() or project_path.parent.is_symlink():
                        continue
                    project = Project(project_path.parent, research_registry()).status()
                    value["projects"].append(project)
                    value["totals"]["pending_reviews"] += sum(s["status"] == "awaiting_review" for s in project["stages"])
                    value["totals"]["reviewed_tasks"] += sum(d["action"] == "accept" for d in project["decisions"])
                    value["totals"]["active"] += sum(s["status"] == "running" for s in project["stages"])
                    value["events"].extend({"time": e["time"], "event": e["kind"], "owner": project["id"],
                        "job": e.get("stage"), "status": e.get("status")} for e in project["events"])
                    for stage in project["stages"]:
                        if stage["status"] in ("awaiting_review", "failed", "stale", "changes_requested", "rejected", "running"):
                            value["alerts"].append({"source": project["id"] + "/" + stage["id"],
                                "message": "项目阶段：" + stage["status"]})
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    value["alerts"].append({"source": project_path.parent.name,
                        "message": "项目记录暂不可读: " + type(exc).__name__})
            for path in sorted((self.root / "projects").glob("*/autonomy-state.json")):
                try:
                    state = self.read(path)
                    config = self.read(path.parent / "inputs/config.json")
                    project = {"id": state["id"], "autonomous": True, "revision": 1,
                        "question": state["question"], "status": state["status"], "phase": state["phase"],
                        "fixture_only": state.get("fixture_only", config.get("domain") == "fixture_threshold"),
                        "holdout_exposed": state["evaluation_exposed"], "attempt": state["attempt"],
                        "max_candidates": config["max_candidates"], "trials": state["trials"],
                        "best_score": state["best_score"], "model_calls": state["model_calls"],
                        "model_tokens_charged_or_reserved": sum(state["charged_or_reserved"].values()),
                        "model_token_limit": sum(config["model_budget"].values()),
                        "reviews": state["reviews"], "last_error": state.get("last_error"),
                        "stop_reason": state.get("stop_reason"),
                        "progress": self.read(path.parent / "progress.json", {}), "artifacts": []}
                    if state.get("open_topic"):
                        topics = state.get("topic_pool", [])
                        project.update(open_topic=True, topic_round=state["topic_round"],
                            max_topic_rounds=config["max_topic_rounds"], selected_topic=state.get("selected_topic"),
                            topics=[{"key": t["key"], "question": t["topic"]["question"], "domain": t["topic"]["domain"],
                                     "status": t["status"], "issues": t["issues"], "priority": t["topic"]["priority"]} for t in topics],
                            topic_counts={status: sum(t["status"] == status for t in topics)
                                          for status in ("infeasible", "pending_review", "approved", "rejected")},
                            model_budgets=config["model_budget"], model_usage=state["charged_or_reserved"])
                    for op in state["operations"].values():
                        if op["status"] != "completed":
                            continue
                        for name in op["files"]:
                            item = path.parent / name
                            if item.name in {"paper_draft.md", "paper.tex", "claims.json", "resource_report.json", "review_audit.json", "REPRODUCE.md", "topic_audit.json"}:
                                entry = self.register(item)
                                if entry:
                                    project["artifacts"].append(entry)
                    value["projects"].append(project)
                    value["totals"]["active"] += int(state["status"] == "running")
                    value["events"].extend(self.events(path.parent / "events.jsonl", state["id"]))
                    if state["status"] in ("waiting_for_gateway", "needs_attention", "budget_exhausted", "deadline_exceeded", "stopped"):
                        value["alerts"].append({"source": state["id"], "message": state.get("last_error") or state.get("stop_reason") or state["status"]})
                except (OSError, ValueError, KeyError, TypeError):
                    value["alerts"].append({"source": path.parent.name, "message": "自主研究记录暂不可读"})
            for path in sorted((self.root / "projects").glob("*/reader-study-state.json")):
                try:
                    state = self.read(path)
                    outputs = []
                    for name in ("analysis.json", "paper_draft.md", "resource_report.json"):
                        item = path.parent / "evidence" / name
                        if item.is_file() and (entry := self.register(item)):
                            outputs.append(entry)
                    scheduled = state["reader_calls_scheduled"]
                    total = scheduled + scheduled // 6 + min(24, scheduled // 12) * 2 + 2
                    project = {"id": state["id"], "autonomous": True, "revision": 1, "question": state["question"],
                        "reader_followup": True, "model_budgets": state["budget"], "model_usage": state["charged_or_reserved"],
                        "status": state["status"], "phase": state["phase"], "fixture_only": False,
                        "holdout_exposed": state["evaluation_exposed"], "attempt": state["completed_tasks"], "max_candidates": total,
                        "trials": [], "best_score": None, "model_calls": sum(o["tasks"] for o in state["operations"].values()),
                        "model_tokens_charged_or_reserved": sum(state["charged_or_reserved"].values()),
                        "model_token_limit": sum(state["budget"].values()), "reviews": [], "last_error": state.get("last_error"),
                        "stop_reason": state.get("reason"), "progress": {"split": state["phase"], "completed": state["completed_tasks"], "total": total},
                        "artifacts": outputs}
                    value["projects"].append(project)
                    value["totals"]["active"] += int(state["status"] == "running")
                    if state["status"] in {"needs_attention", "stopped"}:
                        value["alerts"].append({"source": state["id"], "message": state.get("last_error") or state.get("reason") or state["status"]})
                except (OSError, ValueError, KeyError, TypeError):
                    value["alerts"].append({"source": path.parent.name, "message": "读者实验记录暂不可读"})
            value["events"] = sorted(value["events"], key=lambda e: e.get("time", ""), reverse=True)[:80]
            self.cache = json.loads(redact(json.dumps(value, ensure_ascii=False)))
            self.cache_at = time.monotonic()
            return self.cache

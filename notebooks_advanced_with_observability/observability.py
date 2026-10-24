"""Observability switch for the notebooks: Arize Phoenix, DeepEval, both, or none.

Choose with ONE line in `.env`:

    OBSERVABILITY_BACKEND=phoenix     # traces every LLM call / judge / agent step in the Phoenix UI (localhost:6006)
    OBSERVABILITY_BACKEND=deepeval    # DeepEval metrics run in a local Docker container, dashboard on localhost:8765
    OBSERVABILITY_BACKEND=both
    OBSERVABILITY_BACKEND=none        # default if unset

Design rules: observability must NEVER break a notebook. If a backend's container is not running we print one warning
and carry on without it. `real_helpers.py` calls the hooks here, so the notebooks themselves only call
`h.obs_start("<name>")` once and `h.obs_finish()` at the end.
"""
from __future__ import annotations

import atexit
import contextlib
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

PHOENIX_URL = "http://localhost:6006"
DEEPEVAL_URL = "http://localhost:8765"

_state = {"backend": "none", "phoenix": False, "deepeval": False, "project": "agentic-graphrag", "tracer": None,
          "provider": None, "judgements": [], "started": False}


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for p in (HERE / ".env", HERE.parent / ".env"):
        if p.exists():
            load_dotenv(p, override=False)


def _http(method: str, url: str, payload: dict | None = None, timeout: float = 3.0):
    import urllib.request
    data = json.dumps(payload, default=str).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
        return json.loads(body) if body[:1] in "{[" else body


def enabled(name: str) -> bool:
    return bool(_state.get(name))


# ---------------------------------------------------------------- start
def start(project: str | None = None) -> dict:
    """Read OBSERVABILITY_BACKEND, connect to the chosen backend(s) and print what is active. Safe to call twice."""
    if _state["started"]:
        return status()
    _load_env()
    _state["started"] = True
    _state["project"] = project or _state["project"]
    backend = os.environ.get("OBSERVABILITY_BACKEND", "none").strip().lower()
    if backend not in ("phoenix", "deepeval", "both", "none"):
        print(f"[obs] unknown OBSERVABILITY_BACKEND={backend!r}; use phoenix | deepeval | both | none. Observability OFF.")
        backend = "none"
    _state["backend"] = backend
    if backend in ("phoenix", "both"):
        _start_phoenix()
    if backend in ("deepeval", "both"):
        _start_deepeval()
    print_status()
    return status()


def _start_phoenix() -> None:
    try:
        _http("GET", PHOENIX_URL + "/healthz")
    except Exception:
        print("[obs] Phoenix is not reachable on localhost:6006. From the repo root start it (with its own Neo4j) via:\n"
              "      docker compose -f docker-compose/docker-compose.phoenix.yml up -d")
        return
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        # Built by hand (instead of phoenix.otel.register) so a version quirk in that helper can't break the notebook.
        provider = TracerProvider(resource=Resource.create({"openinference.project.name": _state["project"]}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=PHOENIX_URL + "/v1/traces")))
        trace.set_tracer_provider(provider)
        OpenAIInstrumentor().instrument(tracer_provider=provider)
        _state["provider"] = provider
        _state["tracer"] = provider.get_tracer("agentic-graphrag-notebooks")
        _state["phoenix"] = True
        atexit.register(flush)
    except Exception as e:
        print(f"[obs] Phoenix tracing could not start ({type(e).__name__}: {e}). "
              "Run: pip install arize-phoenix-otel openinference-instrumentation-openai opentelemetry-exporter-otlp-proto-http")


def _start_deepeval() -> None:
    try:
        _http("GET", DEEPEVAL_URL + "/health")
        _state["deepeval"] = True
        _http("POST", DEEPEVAL_URL + "/session", {"project": _state["project"]})
    except Exception:
        print("[obs] DeepEval container is not reachable on localhost:8765. From the repo root start it (with its own Neo4j) via:\n"
              "      docker compose --env-file notebooks_advanced_with_observability/.env -f docker-compose/docker-compose.deepeval.yml up -d --build")


def status() -> dict:
    return {k: _state[k] for k in ("backend", "phoenix", "deepeval", "project")}


def print_status() -> None:
    s = _state
    print(f"[obs] backend={s['backend']}  project={s['project']}")
    if s["phoenix"]:
        print(f"[obs]   Phoenix  ON  -> {PHOENIX_URL}  (project '{s['project']}'): every LLM/embedding call, judge, eval and agent step is a trace")
    if s["deepeval"]:
        print(f"[obs]   DeepEval ON  -> {DEEPEVAL_URL}  : LLM calls + judge verdicts are logged, DeepEval metrics run at the end")
    if s["backend"] == "none":
        print("[obs]   (off) set OBSERVABILITY_BACKEND=phoenix|deepeval|both in .env to turn it on")


# ---------------------------------------------------------------- spans (Phoenix)
class _Span:
    def __init__(self, span=None):
        self._s = span

    def set(self, **attrs) -> None:
        if self._s is None:
            return
        for k, v in attrs.items():
            if v is None:
                continue
            if not isinstance(v, (str, int, float, bool)):
                v = json.dumps(v, default=str)
            self._s.set_attribute(k.replace("__", "."), v if not isinstance(v, str) else v[:4000])


@contextlib.contextmanager
def span(name: str, kind: str = "CHAIN", input=None):
    """Context manager: a named span (kind = CHAIN | AGENT | TOOL | EVALUATOR | RETRIEVER). No-op unless Phoenix is on."""
    if not _state["phoenix"] or _state["tracer"] is None:
        yield _Span()
        return
    with _state["tracer"].start_as_current_span(name) as s:
        s.set_attribute("openinference.span.kind", kind)
        if input is not None:
            s.set_attribute("input.value", input if isinstance(input, str) else json.dumps(input, default=str)[:4000])
        w = _Span(s)
        try:
            yield w
        except Exception as e:
            s.record_exception(e)
            raise


def flush() -> None:
    p = _state.get("provider")
    if p is not None:
        try:
            p.force_flush(timeout_millis=5000)
        except Exception:
            pass


# ---------------------------------------------------------------- hooks called from real_helpers
def on_llm_call(messages, message, usage, seconds: float, model: str) -> None:
    """Every chat completion. Phoenix already sees it through auto-instrumentation; DeepEval's container logs it."""
    if not _state["deepeval"]:
        return
    try:
        _http("POST", DEEPEVAL_URL + "/trace", {
            "project": _state["project"], "kind": "llm", "model": model, "seconds": round(seconds, 3),
            "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
            "input": messages[-1].get("content") if messages else "", "output": getattr(message, "content", "") or "",
            "tool_calls": [t.function.name for t in (getattr(message, "tool_calls", None) or [])]}, timeout=2)
    except Exception:
        pass


def on_judge(kind: str, question: str, answer: str, reference: str | None, result: dict) -> None:
    """A judge verdict (rubric / pairwise / repeat). Buffered so DeepEval can second-guess a sample at the end."""
    rec = {"project": _state["project"], "kind": kind, "question": question, "answer": answer,
           "reference": reference, "result": result}
    if kind == "rubric":
        _state["judgements"].append(rec)
    if _state["deepeval"]:
        try:
            _http("POST", DEEPEVAL_URL + "/judgement", rec, timeout=2)
        except Exception:
            pass


def on_eval_run(summary: dict) -> None:
    if _state["deepeval"]:
        try:
            s = {k: v for k, v in summary.items() if k != "results"}
            _http("POST", DEEPEVAL_URL + "/eval_run", {"project": _state["project"], **s,
                                                        "results": summary.get("results", [])[:50]}, timeout=3)
        except Exception:
            pass


# ---------------------------------------------------------------- finish
def finish(max_cases: int | None = None) -> dict:
    """End-of-notebook summary. Phoenix: flush traces + print the link. DeepEval: run DeepEval metrics (in the container)
    over a small sample of the judged answers and compare them with our own LLM-as-a-judge scores."""
    out: dict = {"backend": _state["backend"]}
    if _state["phoenix"]:
        flush()
        out["phoenix_url"] = f"{PHOENIX_URL}/projects"
        print(f"[obs] Phoenix: traces flushed. Open {PHOENIX_URL}/projects -> project '{_state['project']}'.")
    if _state["deepeval"]:
        n = int(max_cases or os.environ.get("DEEPEVAL_MAX_CASES", "6"))
        sample = _state["judgements"][:: max(1, len(_state["judgements"]) // n)][:n] if _state["judgements"] else []
        if sample:
            cases = [{"input": j["question"], "actual_output": j["answer"], "expected_output": j["reference"],
                      "own_judge_mean": j["result"].get("mean")} for j in sample]
            try:
                res = _http("POST", DEEPEVAL_URL + "/evaluate", {"project": _state["project"], "cases": cases}, timeout=600)
                out["deepeval"] = res
                print(f"[obs] DeepEval ({res.get('model')}) scored {len(res['results'])} judged answers "
                      "(own judge mean is 1-5; DeepEval scores are 0-1):")
                for r in res["results"]:
                    own = r.get("own_judge_mean")
                    print(f"      own={own if own is None else round(own, 2)}  "
                          f"correctness={r.get('correctness')}  relevancy={r.get('relevancy')}  | {r['input'][:60]!r}")
            except Exception as e:
                print(f"[obs] DeepEval evaluation failed: {type(e).__name__}: {e}")
        else:
            print("[obs] DeepEval: no judge_rubric verdicts were recorded in this notebook, so nothing to second-guess.")
        out["deepeval_url"] = DEEPEVAL_URL
        print(f"[obs] DeepEval dashboard: {DEEPEVAL_URL}")
    if _state["backend"] == "none":
        print("[obs] observability is off (OBSERVABILITY_BACKEND=none).")
    return out

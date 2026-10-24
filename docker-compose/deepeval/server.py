"""Tiny local DeepEval service + dashboard (stdlib HTTP server wrapping the open-source `deepeval` library).

POST /session     {project}                      start/mark a project
POST /trace       {project, kind, model, ...}    one LLM call
POST /judgement   {project, kind, question, ...} one LLM-as-a-Judge verdict from the notebook
POST /eval_run    {project, name, version, ...}  one stored eval run
POST /evaluate    {project, cases:[...]}         run DeepEval metrics (Correctness via GEval, AnswerRelevancy)
GET  /health, /api/state, and / (HTML dashboard, auto-refresh)
Everything is also appended to /data/*.jsonl so history survives restarts.
"""
import html
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DATA = Path(os.environ.get("DATA_DIR", "/data"))
DATA.mkdir(parents=True, exist_ok=True)
MODEL = os.environ.get("DEEPEVAL_MODEL", "gpt-4o-mini")
LOCK = threading.Lock()
STATE = {"sessions": [], "traces": [], "judgements": [], "eval_runs": [], "metrics": []}
FILES = {k: DATA / f"{k}.jsonl" for k in STATE}

for k, f in FILES.items():          # reload history
    if f.exists():
        STATE[k] = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()][-2000:]


def add(kind: str, rec: dict) -> None:
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
    with LOCK:
        STATE[kind].append(rec)
        STATE[kind][:] = STATE[kind][-2000:]
        with FILES[kind].open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")


def run_deepeval(project: str, cases: list) -> dict:
    from deepeval.metrics import AnswerRelevancyMetric, GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    out = []
    for c in cases:
        has_ref = bool(c.get("expected_output"))
        tc = LLMTestCase(input=c["input"], actual_output=c["actual_output"],
                         expected_output=c.get("expected_output") if has_ref else None)
        params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT] + ([LLMTestCaseParams.EXPECTED_OUTPUT] if has_ref else [])
        rec = {"project": project, "input": c["input"], "actual_output": c["actual_output"],
               "expected_output": c.get("expected_output"), "own_judge_mean": c.get("own_judge_mean")}
        try:
            g = GEval(name="Correctness", model=MODEL, async_mode=False, verbose_mode=False, evaluation_params=params,
                      criteria="Is the actual output factually correct and consistent with the expected output and the question? "
                               "Penalise invented facts." if has_ref else
                               "Is the actual output correct, specific and free of invented facts for the question?")
            g.measure(tc)
            rec["correctness"], rec["correctness_reason"] = round(g.score, 3), g.reason
        except Exception as e:
            rec["correctness"], rec["correctness_reason"] = None, f"error: {type(e).__name__}: {e}"[:300]
        try:
            r = AnswerRelevancyMetric(model=MODEL, async_mode=False, verbose_mode=False)
            r.measure(tc)
            rec["relevancy"], rec["relevancy_reason"] = round(r.score, 3), r.reason
        except Exception as e:
            rec["relevancy"], rec["relevancy_reason"] = None, f"error: {type(e).__name__}: {e}"[:300]
        add("metrics", rec)
        out.append(rec)
    return {"model": MODEL, "results": out}


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))[:220]


def table(rows, cols):
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{esc(r.get(c))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def page() -> str:
    with LOCK:
        s = {k: list(v) for k, v in STATE.items()}
    llm = s["traces"]
    tok_in, tok_out = sum(t.get("prompt_tokens", 0) for t in llm), sum(t.get("completion_tokens", 0) for t in llm)
    projects = sorted({r.get("project", "") for v in s.values() for r in v})
    return f"""<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=5><title>DeepEval (local)</title>
<style>body{{font:14px system-ui;margin:24px;color:#222}}table{{border-collapse:collapse;width:100%;margin:8px 0 24px}}
td,th{{border:1px solid #ddd;padding:4px 8px;text-align:left;vertical-align:top}}th{{background:#f3f3f3}}h2{{margin-top:28px}}
.k{{display:inline-block;background:#eef;padding:8px 14px;margin-right:10px;border-radius:6px}}</style>
<h1>DeepEval &mdash; local dashboard</h1>
<p><span class=k>projects: {esc(', '.join(projects))}</span><span class=k>LLM calls: {len(llm)}</span>
<span class=k>tokens in/out: {tok_in}/{tok_out}</span><span class=k>judgements: {len(s['judgements'])}</span>
<span class=k>eval runs: {len(s['eval_runs'])}</span><span class=k>DeepEval metric results: {len(s['metrics'])}</span>
<span class=k>metric model: {esc(MODEL)}</span></p>
<h2>DeepEval metrics vs the notebook's own LLM-as-a-Judge (own = 1-5 mean, DeepEval = 0-1)</h2>
{table(reversed(s['metrics'][-40:]), ['ts','project','own_judge_mean','correctness','relevancy','input','correctness_reason'])}
<h2>Eval runs</h2>{table(reversed(s['eval_runs'][-30:]), ['ts','project','name','version','n','pass_rate','mean_score','failures'])}
<h2>Judge verdicts (from the notebook)</h2>{table(reversed(s['judgements'][-40:]), ['ts','project','kind','question','result'])}
<h2>Recent LLM calls</h2>{table(reversed(llm[-40:]), ['ts','project','model','seconds','prompt_tokens','completion_tokens','tool_calls','input'])}
"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, json.dumps({"ok": True, "model": MODEL}))
        if self.path == "/api/state":
            with LOCK:
                return self._send(200, json.dumps({k: len(v) for k, v in STATE.items()}))
        return self._send(200, page(), "text/html")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n) or b"{}")
            route = self.path.strip("/")
            if route == "evaluate":
                return self._send(200, json.dumps(run_deepeval(p.get("project", ""), p.get("cases", [])), default=str))
            kind = {"session": "sessions", "trace": "traces", "judgement": "judgements", "eval_run": "eval_runs"}.get(route)
            if not kind:
                return self._send(404, '{"error":"unknown route"}')
            add(kind, p)
            return self._send(200, '{"ok":true}')
        except Exception as e:
            return self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))


if __name__ == "__main__":
    print(f"DeepEval service on :8765 (metric model {MODEL})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8765), H).serve_forever()

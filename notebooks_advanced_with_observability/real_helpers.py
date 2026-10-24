"""Helpers for the `notebooks_advanced_with_observability` tier (= notebooks_advanced + Phoenix / DeepEval observability): REAL LLM, REAL embeddings, REAL Neo4j RBAC.

Nothing here falls back silently. If OPENAI_API_KEY is missing, `get_openai()` raises with
instructions. Notebooks import it as:

    import sys; sys.path.insert(0, "."); import real_helpers as h
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _early_env() -> None:
    """Load .env BEFORE graph_helpers is imported (it reads NEO4J_URI at import time), and choose the Neo4j that
    belongs to the selected observability stack unless NEO4J_URI is set explicitly:
    phoenix/both -> 7688 (docker-compose.phoenix.yml), deepeval -> 7689 (docker-compose.deepeval.yml), none -> 7687."""
    try:
        from dotenv import load_dotenv
        for p in (HERE / ".env", HERE.parent / ".env"):
            if p.exists():
                load_dotenv(p, override=False)
    except ImportError:
        pass
    port = {"phoenix": 7688, "both": 7688, "deepeval": 7689}.get(os.environ.get("OBSERVABILITY_BACKEND", "none").strip().lower(), 7687)
    os.environ.setdefault("NEO4J_URI", f"bolt://localhost:{port}")


_early_env()
sys.path.insert(0, str(HERE.parent / "notebooks_advanced_simplified"))
sys.path.insert(0, str(HERE))

import observability as obs  # noqa: E402  (Phoenix / DeepEval switch, see .env OBSERVABILITY_BACKEND)

from adv_helpers import (  # noqa: E402,F401  (re-exported)
    AgentTrace, check, skill_catalog, run_benchmark,
    connect, run, show, clear_graph, load_devops_graph, load_skill, PROJECT_ROOT, SKILLS_DIR,
)

LLM_MODEL = os.environ.get("OPENAI_LLM_MODEL", "gpt-4o")
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")
EMBED_DIM = int(os.environ.get("OPENAI_EMBED_DIM", "3072"))  # text-embedding-3-large native size

_client = None
obs_start = obs.start      # notebooks call h.obs_start("<name>") once and h.obs_finish() at the end
obs_finish = obs.finish
usage = {"llm_calls": 0, "judge_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "embed_calls": 0, "embed_tokens": 0}


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for p in (HERE / ".env", HERE.parent / ".env"):
        if p.exists():
            load_dotenv(p, override=False)


def get_openai():
    """Return a cached OpenAI client. Raises (no fallback) when no key is configured."""
    global _client
    if _client is None:
        _load_env()
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy notebooks_advanced_with_observability/.env.example to .env, put your key "
                "in it, then restart the kernel. These notebooks intentionally have no offline fallback."
            )
        from openai import OpenAI
        _client = OpenAI()
    return _client


# ---------------------------------------------------------------- LLM
def chat(messages, *, model=None, temperature=0.0, tools=None, response_format=None, **kw):
    """Thin wrapper over chat.completions that tracks token usage. Returns the message object."""
    args = dict(model=model or LLM_MODEL, messages=messages, temperature=temperature, **kw)
    if tools:
        args["tools"] = tools
    if response_format:
        args["response_format"] = response_format
    t0 = time.time()
    for attempt in range(4):
        try:
            r = get_openai().chat.completions.create(**args)
            break
        except Exception as e:  # rate limit / transient
            if attempt == 3 or "rate" not in str(e).lower() and "timeout" not in str(e).lower():
                raise
            time.sleep(2 ** attempt)
    usage["llm_calls"] += 1
    usage["prompt_tokens"] += r.usage.prompt_tokens
    usage["completion_tokens"] += r.usage.completion_tokens
    obs.on_llm_call(messages, r.choices[0].message, r.usage, time.time() - t0, args["model"])
    return r.choices[0].message


def llm(prompt: str, system: str = "You are a precise assistant.", **kw) -> str:
    return chat([{"role": "system", "content": system}, {"role": "user", "content": prompt}], **kw).content


def llm_json(prompt: str, system: str = "You are a precise assistant. Reply with JSON only.", **kw) -> dict:
    m = chat([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
             response_format={"type": "json_object"}, **kw)
    return json.loads(m.content)


def judge(question: str, answer: str, rubric: str) -> dict:
    """LLM-as-judge: returns {"score": 1-5, "reason": str}. Use it to grade answers, not to trust them blindly."""
    usage["judge_calls"] += 1
    return llm_json(
        f"Question:\n{question}\n\nAnswer:\n{answer}\n\nRubric:\n{rubric}\n\n"
        'Return JSON {"score": <1-5 int>, "reason": "<one sentence>"}.')


# ---------------------------------------------------------------- evaluation toolkit (LLM-as-a-judge done carefully)
JUDGE_SYSTEM = ("You are a strict, impartial evaluator. Judge ONLY against the stated criteria and the reference "
                "(if any). Do not reward length, confidence or style. Give the reason BEFORE the score. "
                "Reply with JSON only.")


def judge_rubric(question: str, answer: str, criteria: dict, reference: str | None = None, model: str | None = None) -> dict:
    """Multi-criteria judge. criteria = {"name": "what a 5 means vs a 1"}.
    Returns {"scores": {name: 1-5}, "reasons": {name: str}, "mean": float}. Reasons are asked first (reason-then-score)."""
    usage["judge_calls"] += 1
    with obs.span("judge_rubric", "EVALUATOR", input={"question": question, "answer": answer, "reference": reference}) as sp:
        res = _judge_rubric(question, answer, criteria, reference, model)
        sp.set(eval__mean=res["mean"], eval__scores=res["scores"], output__value=json.dumps(res["reasons"], default=str))
    obs.on_judge("rubric", question, answer, reference, res)
    return res


def _judge_rubric(question, answer, criteria, reference, model):
    crit = "\n".join(f"- {k}: {v}" for k, v in criteria.items())
    ref = f"\nReference (ground truth):\n{reference}\n" if reference else ""
    out = llm_json(
        f"Question:\n{question}\n{ref}\nAnswer to evaluate:\n{answer}\n\nCriteria (score each 1-5):\n{crit}\n\n"
        'Return {"reasons": {<criterion>: "<one sentence>"}, "scores": {<criterion>: <int 1-5>}}.',
        system=JUDGE_SYSTEM, model=model)
    scores = {k: int(out["scores"][k]) for k in criteria}
    return {"scores": scores, "reasons": out.get("reasons", {}), "mean": sum(scores.values()) / len(scores)}


def judge_pairwise(question: str, a: str, b: str, criteria: str, reference: str | None = None) -> dict:
    """Pairwise judge with POSITION-SWAP: asks twice (A,B) and (B,A). If the verdict flips with order the
    result is 'inconsistent' (position bias) and counted as a tie. Returns {"winner": 'a'|'b'|'tie', "consistent": bool}."""
    def once(x, y):
        usage["judge_calls"] += 1
        ref = f"\nReference:\n{reference}\n" if reference else ""
        r = llm_json(f"Question:\n{question}\n{ref}\nCriteria: {criteria}\n\nResponse X:\n{x}\n\nResponse Y:\n{y}\n\n"
                     'Which is better? Return {"reason": "<one sentence>", "winner": "X" | "Y" | "tie"}.',
                     system=JUDGE_SYSTEM)
        return r["winner"]
    with obs.span("judge_pairwise", "EVALUATOR", input={"question": question, "a": a, "b": b}) as sp:
        w1, w2 = once(a, b), once(b, a)           # second call has the answers swapped
        m1 = {"X": "a", "Y": "b", "tie": "tie"}[w1]
        m2 = {"X": "b", "Y": "a", "tie": "tie"}[w2]
        res = ({"winner": m1, "consistent": True, "first": m1, "swapped": m2} if m1 == m2
               else {"winner": "tie", "consistent": False, "first": m1, "swapped": m2})
        sp.set(eval__winner=res["winner"], eval__position_consistent=res["consistent"])
    obs.on_judge("pairwise", question, f"A: {a}\n\nB: {b}", reference, res)
    return res


def judge_repeat(question: str, answer: str, rubric: str, n: int = 3, temperature: float = 0.7) -> dict:
    """Self-consistency check: judge the same answer n times at temperature>0. A high spread means the
    judge is unreliable on this item. Returns {"scores": [...], "mean": float, "spread": float}."""
    scores = []
    for _ in range(n):
        usage["judge_calls"] += 1
        r = llm_json(f"Question:\n{question}\n\nAnswer:\n{answer}\n\nRubric:\n{rubric}\n\n"
                     'Return {"reason": "<one sentence>", "score": <int 1-5>}.', system=JUDGE_SYSTEM, temperature=temperature)
        scores.append(int(r["score"]))
    return {"scores": scores, "mean": sum(scores) / n, "spread": max(scores) - min(scores)}


def agreement(human: list, judge_scores: list) -> dict:
    """How well does the judge match human/gold labels? Pure Python: exact agreement, within-1, Cohen's kappa
    (categorical) and Spearman rho. Calibrate a judge BEFORE trusting it."""
    n = len(human)
    assert n == len(judge_scores) and n > 1, "need two equal-length lists"
    exact = sum(a == b for a, b in zip(human, judge_scores)) / n
    within1 = sum(abs(a - b) <= 1 for a, b in zip(human, judge_scores)) / n
    cats = sorted(set(human) | set(judge_scores))
    pe = sum((human.count(c) / n) * (judge_scores.count(c) / n) for c in cats)
    kappa = (exact - pe) / (1 - pe) if pe < 1 else 1.0

    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rh, rj = ranks(list(human)), ranks(list(judge_scores))
    mh, mj = sum(rh) / n, sum(rj) / n
    cov = sum((a - mh) * (b - mj) for a, b in zip(rh, rj))
    den = (sum((a - mh) ** 2 for a in rh) * sum((b - mj) ** 2 for b in rj)) ** 0.5
    return {"n": n, "exact": round(exact, 3), "within1": round(within1, 3), "kappa": round(kappa, 3),
            "spearman": round(cov / den, 3) if den else 0.0}


def run_eval(name: str, cases: list[dict], system_fn, score_fn, version: str = "v1", store: bool = True) -> dict:
    """Run an evaluation. cases=[{"id","input",...gold fields}]; system_fn(case)->output;
    score_fn(case, output)->{"score": float 0-1 or 1-5, "passed": bool, "notes": str}.
    Stores (:EvalRun)-[:HAS_RESULT]->(:EvalResult) in Neo4j so runs can be compared with Cypher. Returns a summary dict."""
    results = []
    for c in cases:
        with obs.span(f"eval_case:{name}:{c['id']}", "CHAIN", input=c.get("input")) as sp:
            try:
                out = system_fn(c)
                s = score_fn(c, out)
            except Exception as e:  # a crash is a failed case, not a crashed eval
                out, s = f"ERROR {type(e).__name__}: {e}", {"score": 0.0, "passed": False, "notes": "exception"}
            sp.set(output__value=str(out)[:1000], eval__score=float(s["score"]), eval__passed=bool(s["passed"]),
                   eval__version=version)
        results.append({"case": c["id"], "score": float(s["score"]), "passed": bool(s["passed"]),
                        "notes": str(s.get("notes", ""))[:300], "output": str(out)[:500]})
    summary = {"name": name, "version": version, "n": len(results),
               "pass_rate": sum(r["passed"] for r in results) / max(1, len(results)),
               "mean_score": sum(r["score"] for r in results) / max(1, len(results)),
               "failures": [r["case"] for r in results if not r["passed"]]}
    if store:
        run("CREATE (r:EvalRun {name:$n, version:$v, ts:datetime(), pass_rate:$p, mean_score:$m, n:$k}) "
            "WITH r UNWIND $rows AS x CREATE (r)-[:HAS_RESULT]->(:EvalResult {case:x.case, score:x.score, "
            "passed:x.passed, notes:x.notes, output:x.output})",
            n=name, v=version, p=summary["pass_rate"], m=summary["mean_score"], k=summary["n"], rows=results)
    summary["results"] = results
    obs.on_eval_run(summary)
    return summary


def compare_runs(name: str) -> list[dict]:
    """Compare stored runs of the same eval across versions (prompt/model/retrieval changes) with one Cypher query."""
    return run("MATCH (r:EvalRun {name:$n}) RETURN r.version AS version, r.n AS n, "
               "round(r.pass_rate*100)/100.0 AS pass_rate, round(r.mean_score*100)/100.0 AS mean_score "
               "ORDER BY r.ts", n=name)


PRICES_PER_M = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60), "embed": 0.13}  # USD, checked 2026-10-24


def cost_report() -> str:
    """Token accounting plus a rough USD figure (gpt-4o rates; update PRICES_PER_M when prices change)."""
    u = usage
    pin, pout = PRICES_PER_M.get(LLM_MODEL, PRICES_PER_M["gpt-4o"])
    usd = u["prompt_tokens"] * pin / 1e6 + u["completion_tokens"] * pout / 1e6 + u["embed_tokens"] * PRICES_PER_M["embed"] / 1e6
    return (f"LLM calls={u['llm_calls']} (of which judge calls={u['judge_calls']}) prompt_tokens={u['prompt_tokens']} "
            f"completion_tokens={u['completion_tokens']} | embed calls={u['embed_calls']} tokens={u['embed_tokens']} "
            f"| approx USD=${usd:.3f}")


# ---------------------------------------------------------------- embeddings + vector index
def embed(texts: list[str]) -> list[list[float]]:
    r = get_openai().embeddings.create(model=EMBED_MODEL, input=texts, dimensions=EMBED_DIM)
    usage["embed_calls"] += 1
    usage["embed_tokens"] += r.usage.total_tokens
    return [d.embedding for d in r.data]


def ensure_vector_index(name: str, label: str, prop: str = "embedding") -> None:
    run(f"CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:{label}) ON (n.{prop}) "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {EMBED_DIM}, `vector.similarity_function`: 'cosine'}}}}")
    run("CALL db.awaitIndexes(60)")


def embed_nodes(label: str, text_prop: str = "text", prop: str = "embedding", batch: int = 64) -> int:
    """Embed every `label` node that lacks `prop`, writing the vector onto the node."""
    rows = run(f"MATCH (n:{label}) WHERE n.{prop} IS NULL AND n.{text_prop} IS NOT NULL "
               f"RETURN elementId(n) AS id, n.{text_prop} AS t")
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        vecs = embed([r["t"] for r in chunk])
        run("UNWIND $rows AS r MATCH (n) WHERE elementId(n)=r.id "
            f"CALL db.create.setNodeVectorProperty(n, '{prop}', r.v)",
            rows=[{"id": r["id"], "v": v} for r, v in zip(chunk, vecs)])
    return len(rows)


def vector_search(index: str, query: str, k: int = 5) -> list[dict]:
    qv = embed([query])[0]
    return run("CALL db.index.vector.queryNodes($i, $k, $v) YIELD node, score "
               "RETURN node, score ORDER BY score DESC", i=index, k=k, v=qv)


# ---------------------------------------------------------------- real tool-calling agent
class ToolAgent:
    """A real LLM agent using OpenAI function calling. Every step is recorded in an AgentTrace.

    tools: {name: (callable, json_schema_parameters, description)}
    """

    def __init__(self, tools: dict, system: str, max_steps: int = 8, trace: AgentTrace | None = None):
        self.tools, self.system, self.max_steps = tools, system, max_steps
        self.trace = trace
        self.messages: list[dict] = []

    def _schemas(self):
        def norm(p):  # OpenAI requires an object schema, even for tools without arguments
            p = dict(p or {})
            p.setdefault("type", "object")
            p.setdefault("properties", {})
            return p
        return [{"type": "function", "function": {"name": n, "description": d, "parameters": norm(p)}}
                for n, (_, p, d) in self.tools.items()]

    def run(self, task: str, verbose: bool = True) -> str:
        with obs.span("ToolAgent.run", "AGENT", input=task) as sp:
            final = self._run(task, verbose)
            sp.set(output__value=final)
        return final

    def _run(self, task: str, verbose: bool = True) -> str:
        self.messages = [{"role": "system", "content": self.system}, {"role": "user", "content": task}]
        for step in range(1, self.max_steps + 1):
            m = chat(self.messages, tools=self._schemas())
            self.messages.append(m.model_dump(exclude_none=True))
            if not m.tool_calls:
                final = m.content or ""
                if verbose:
                    print(f"[step {step}] FINAL: {final}")
                return final
            for tc in m.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                fn = self.tools.get(tc.function.name)
                with obs.span(f"tool:{tc.function.name}", "TOOL", input=args) as tsp:
                    try:
                        out = fn[0](**args) if fn else f"unknown tool {tc.function.name}"
                    except Exception as e:  # feed errors back so the model can recover
                        out = f"ERROR: {type(e).__name__}: {e}"
                    tsp.set(output__value=out if isinstance(out, str) else json.dumps(out, default=str)[:2000])
                text = out if isinstance(out, str) else json.dumps(out, default=str)
                if verbose:
                    print(f"[step {step}] {tc.function.name}({args}) -> {text[:200]}")
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": text[:6000]})
        # step budget exhausted: force a final answer from what was gathered instead of returning nothing
        self.messages.append({"role": "user", "content":
                              "Step budget exhausted. Using only what you gathered so far, give your best final answer now. "
                              "State what you verified and what remains uncertain."})
        final = chat(self.messages).content or ""
        if verbose:
            print(f"[step {self.max_steps + 1}] FINAL (forced after budget): {final}")
        return final


# ---------------------------------------------------------------- real Neo4j RBAC (Enterprise)
def sysrun(cypher: str, **params):
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                               auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "password")))
    try:
        with drv.session(database="system") as s:
            return [dict(r) for r in s.run(cypher, **params)]
    finally:
        drv.close()


def run_as(user: str, password: str, cypher: str, **params):
    """Run Cypher as a different Neo4j user, so role GRANT/DENY rules are really enforced."""
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(os.environ.get("NEO4J_URI", "bolt://localhost:7687"), auth=(user, password))
    try:
        with drv.session(database="neo4j") as s:
            return [dict(r) for r in s.run(cypher, **params)]
    finally:
        drv.close()

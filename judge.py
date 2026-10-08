"""One Decisions API call per LegalBench row. Writes results/decisions.jsonl.

Usage:
  python judge.py --limit-per-task 10     # STOP 1 pilot
  python judge.py                         # everything not yet done (resumes)
  python judge.py --probe                 # one call, print raw response, exit
"""
import argparse, json, os, sys, time, threading, random
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, urllib.error

API_URL = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"
PRICE_PER_M_INPUT = 0.10  # USD per 1M input tokens
SAMPLE = "data/sample.jsonl"
OUT = "results/decisions.jsonl"
KEY = os.environ.get("OPENAI_API_KEY")
if not KEY:
    sys.exit("Set OPENAI_API_KEY in the environment first (never in a file in this repo).")
lock = threading.Lock()


def call(text, question, retries=6):
    body = json.dumps({
        "model": MODEL,
        "input": text,
        "questions": [{"type": "predicate", "name": "answer", "instructions": question}],
    }).encode()
    delay = 2.0
    for attempt in range(retries):
        req = urllib.request.Request(API_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read()), None
        except urllib.error.HTTPError as e:
            payload = e.read().decode(errors="replace")
            if e.code == 429 or e.code >= 500:
                time.sleep(delay + random.random()); delay = min(delay * 2, 60); continue
            # keep only the error message; never the raw body (a 401 body can echo the key)
            try: msg = json.loads(payload).get("error", {}).get("message", "")[:300]
            except Exception: msg = payload[:120]
            return None, {"status": e.code, "message": msg}
        except Exception as e:  # network blip
            time.sleep(delay); delay = min(delay * 2, 60); last = str(e)
    return None, {"status": "exhausted", "body": locals().get("last", "")}


def find_prob(obj):
    """answers[0].probability per the docs; fall back to a recursive search."""
    for a in (obj.get("answers") or []) if isinstance(obj, dict) else []:
        if isinstance(a, dict) and a.get("type") == "predicate" and isinstance(a.get("probability"), (int, float)):
            return float(a["probability"])
    if isinstance(obj, dict):
        for k in ("probability", "p_true", "prob", "value"):
            if k in obj and isinstance(obj[k], (int, float)) and 0 <= obj[k] <= 1:
                return float(obj[k])
        for v in obj.values():
            r = find_prob(v)
            if r is not None: return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_prob(v)
            if r is not None: return r
    return None


def find_usage(obj):
    u = obj.get("usage") if isinstance(obj, dict) else None
    if not u: return 0
    return u.get("input_tokens") or u.get("prompt_tokens") or 0


def is_refusal(obj):
    # Docs: a refused question comes back as an answer with type "refusal".
    return any(a.get("type") == "refusal" for a in (obj.get("answers") or []) if isinstance(a, dict))


def run_row(row):
    resp, err = call(row["text"], row["question"])
    rec = {"id": row["id"], "task": row["task"], "human_label": row["human_label"]}
    if err:
        rec.update(probability=None, status="error", error=err, input_tokens=0)
        return rec
    p = find_prob(resp)
    toks = find_usage(resp)
    rec.update(input_tokens=toks, usd=toks / 1e6 * PRICE_PER_M_INPUT)
    if p is None:
        rec.update(probability=None, status="refusal" if is_refusal(resp) else "no_prob", raw=resp)
    else:
        rec.update(probability=p, status="ok")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-task", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(SAMPLE)]
    random.Random(0).shuffle(rows)  # mix labels so a per-task pilot is not one-sided
    if a.probe:
        resp, err = call(rows[0]["text"], rows[0]["question"])
        print(json.dumps(resp or err, indent=2)); return

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l)
            if r.get("status") == "ok" or r.get("status") == "refusal": done.add(r["id"])
    todo = []
    per_task = {}
    for r in rows:
        if r["id"] in done:
            per_task[r["task"]] = per_task.get(r["task"], 0) + 1; continue
        if a.limit_per_task and per_task.get(r["task"], 0) >= a.limit_per_task: continue
        per_task[r["task"]] = per_task.get(r["task"], 0) + 1
        todo.append(r)
    print(f"{len(done)} already done, {len(todo)} to run", file=sys.stderr)

    os.makedirs("results", exist_ok=True)
    n = ok = ref = errs = 0; toks = 0
    with open(OUT, "a") as f, ThreadPoolExecutor(a.concurrency) as ex:
        for fut in as_completed([ex.submit(run_row, r) for r in todo]):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec) + "\n"); f.flush()
            n += 1; toks += rec.get("input_tokens", 0)
            ok += rec["status"] == "ok"; ref += rec["status"] == "refusal"; errs += rec["status"] in ("error", "no_prob")
            if n % 20 == 0:
                print(f"{n}/{len(todo)} ok={ok} refusals={ref} errors={errs} tokens={toks} ${toks/1e6*PRICE_PER_M_INPUT:.4f}", file=sys.stderr)
    print(f"DONE {n} ok={ok} refusals={ref} errors={errs} input_tokens={toks} usd={toks/1e6*PRICE_PER_M_INPUT:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Capture per-option logits from a /v1/systemone tier for T calibration.

Mirrors the jevbench typesafe adapter's request mapping (same build_question),
posting the question block to /v1/systemone (whose answers carry
extra.backend.option_logits since 0.2.0). Emits two files:
  <output>       prediction rows: {id, option_ids, option_logits}
  <output>.gold  gold rows:       {id, options:[{id}], label (int index)}
Both in the shape semif/benchmarks/calibrate.py consumes.
"""
import argparse, json, sys, urllib.request

def load_tasks(path):
    """Minimal canonical-record loader: one JSON object per line."""
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def build_question(task):
    """The wire's single typed question (canonical record -> request)."""
    q = {"type": task["question"]["type"],
         "instructions": task["question"]["instructions"]}
    if task["question"].get("criteria") is not None:
        q["criteria"] = task["question"]["criteria"]
    return q


def post(url, body, timeout=120.0):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def extract(t, ans, backend):
    """-> (option_ids, logits list). Raw per-option scores live under
    extra.backend.option_logits (older servers: extra.<backend name>)."""
    extra = ans.get("extra") or {}
    block = extra.get("backend") or extra.get(backend) or {}
    logits = block.get("option_logits") or {}
    probs = ans.get("probabilities") or {}
    ids = list(logits.keys()) or list(probs.keys())
    return ids, [logits[i] for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--backend", default=None,
                    help="backend override sent per request (and used to "
                         "read older servers' extra.<backend>.option_logits)")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tasks = load_tasks(args.tasks)
    n_ok = 0
    no_gold = []
    with open(args.output, "w", encoding="utf-8") as out, \
         open(args.output + ".gold", "w", encoding="utf-8") as gf:
        for i, t in enumerate(tasks, 1):
            body = {"state": t["state"], "model": args.model,
                    "questions": {"decision": build_question(t)}}
            if args.backend:
                body["backend"] = args.backend
            try:
                resp = post(args.endpoint + "/v1/systemone", body)
                ans = resp["answers"]["decision"]
                ids, logits = extract(t, ans, args.backend)
                out.write(json.dumps({"id": t["id"], "option_ids": ids,
                                      "option_logits": logits}) + "\n")
                # score tasks store the expected level as an int; the
                # server's option ids are strings
                exp = str(t.get("expected"))
                if t["question"]["type"] == "noul":
                    exp = {"yes": "true", "no": "false"}.get(exp, exp)
                if exp not in ids:
                    no_gold.append(t["id"])
                else:
                    gf.write(json.dumps({"id": t["id"],
                                         "options": [{"id": x} for x in ids],
                                         "label": ids.index(exp),
                                         "family": t.get("family"),
                                         "group_id": t.get("group") or t["id"]}) + "\n")
                n_ok += 1
            except Exception as e:
                out.write(json.dumps({"id": t.get("id"), "error": str(e)[:200]}) + "\n")
            if i % 25 == 0:
                print(f"{i}/{len(tasks)}", flush=True)
    print(f"done: {n_ok}/{len(tasks)} -> {args.output} (+ .gold)")
    if no_gold:
        print(f"WARNING: {len(no_gold)} tasks have no gold row (expected answer "
              f"not among the returned options): {', '.join(no_gold[:10])}")


if __name__ == "__main__":
    main()

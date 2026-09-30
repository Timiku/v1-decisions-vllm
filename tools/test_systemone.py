#!/usr/bin/env python3
"""SystemOne wire validation: docs fixtures, shape compatibility, confidence bands.

Runs against a live /v1/systemone endpoint serving this overlay:

    python tools/test_systemone.py --endpoint http://localhost:8000 [--model jev-latest]
"""
import json
import math
import sys
import urllib.request
import urllib.error
from urllib.parse import urlparse

def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

ENDPOINT = _arg("--endpoint", "http://localhost:8000").rstrip("/")
BASE = ENDPOINT
MODEL_OVERRIDE = _arg("--model", None)

# Resolve the served model id unless overridden (the endpoint reports it).
if MODEL_OVERRIDE is None:
    try:
        with urllib.request.urlopen(ENDPOINT + "/v1/models", timeout=10) as _r:
            MODEL_OVERRIDE = json.loads(_r.read())["data"][0]["id"]
    except Exception:
        MODEL_OVERRIDE = "jev-latest"


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f" - {detail}" if detail else ""))
    return ok

results = []

# --- fixture 1: docs noul example ---
st, d = post("/v1/systemone", {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                                "criteria": {"true": "Explicitly time-sensitive",
                                             "false": "No urgency expressed"}}}})
a = d.get("answers", {}).get("is_urgent", {})
results.append(check("noul type field", a.get("type") == "noul"))
results.append(check("noul has no confidence", "confidence" not in a))
results.append(check("noul 0..1", isinstance(a.get("noul"), (int, float)) and 0 <= a["noul"] <= 1))
results.append(check("resolved model id", d.get("model") == "jev-1.13.0"))
results.append(check("usage shape", set(d.get("usage", {})) == {"input_tokens", "output_tokens"}
                     and d["usage"]["output_tokens"] == 0))

# --- fixture 2: docs choice example ---
st, d = post("/v1/systemone", {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?",
                                 "criteria": {"billing": "Payments, invoicing, refunds",
                                              "technical": "Bugs, outages, integrations",
                                              "sales": "Pricing, upgrades, new accounts"}}}})
a = d.get("answers", {}).get("department", {})
probs = a.get("probabilities", {})
results.append(check("choice fields", set(a) >= {"type", "choice", "probabilities", "confidence"}))
results.append(check("probs sum to 1", abs(sum(probs.values()) - 1.0) < 1e-6))
results.append(check("choice is argmax", a.get("choice") == max(probs, key=probs.get) if probs else False))
k = len(probs)
expected_conf = max(0.0, min(1.0, (k * max(probs.values()) - 1) / (k - 1))) if probs else None
results.append(check("confidence = normalized-peak", abs(a.get("confidence", -1) - expected_conf) < 1e-6))

# --- fixture 3: docs score example ---
st, d = post("/v1/systemone", {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {"frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                                  "criteria": ["Calm", "Frustrated", "Very angry"]}}})
a = d.get("answers", {}).get("frustration", {})
probs = a.get("probabilities", {})
results.append(check("score string keys", all(isinstance(k, str) for k in probs)))
results.append(check("legend maps levels", a.get("legend") == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}))
expected_score = sum(int(k) * v for k, v in probs.items())
results.append(check("score = sum(i*p)", abs(a.get("score", -1) - expected_score) < 1e-6))
results.append(check("score between levels", 0 <= a.get("score", -1) <= len(probs) - 1))

# --- mixed multi-question request (shared state) ---
st, d = post("/v1/systemone", {
    "state": "I have asked three times now. Can I please just talk to a real person?",
    "model": "jev-latest",
    "questions": {
        "is_human_escalation": {"type": "noul", "instructions": "Is the customer asking for a human agent?"},
        "is_repeat_contact": {"type": "noul", "instructions": "Has the customer contacted support about this before?",
                              "criteria": {"true": "Mentions a prior attempt, ticket, or that they have asked before",
                                           "false": "No sign of any previous contact"}},
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Payments, invoicing, refunds",
                                    "technical": "Bugs, outages, integrations",
                                    "sales": "Pricing, upgrades, new accounts"}}}})
results.append(check("multi-question keys", set(d.get("answers", {})) ==
                     {"is_human_escalation", "is_repeat_contact", "department"}))
results.append(check("all three types answered",
                     {a["type"] for a in d.get("answers", {}).values()} == {"noul", "choice"}))

# --- validation refusals ---
st, d = post("/v1/systemone", {"state": "x", "model": "jev-latest",
    "questions": {"big": {"type": "choice", "instructions": "?",
                          "criteria": {f"o{i}": "d" for i in range(256)}}}})
results.append(check("256 options refused", st == 422, f"got {st}"))
st, d = post("/v1/systemone", {"state": "x", "model": "jev-latest",
    "questions": {"s": {"type": "score", "instructions": "?", "criteria": ["a"]}}})
results.append(check("1-level score refused", st == 422, f"got {st}"))
st, d = post("/v1/systemone", {"state": "x", "model": "gpt-4",
    "questions": {"n": {"type": "noul", "instructions": "?"}}})
results.append(check("unknown model echoes through (validation relaxed)",
                     st == 200 and d["model"] == "gpt-4", f"got {st}"))

# --- confidence display-rounding bands (upstream PR27 correction) ---
# peak 0.8755..0.8849 displays as 0.88 -> confidence 0.81 at k=3 within rounding
for peak, band in ((0.8755, 0.81), (0.9455, 0.92)):
    c = (3 * peak - 1) / 2
    results.append(check(f"band peak {peak} -> {band}+-rounding",
                         abs(round(c, 2) - band) <= 0.01))

print(f"\n{sum(results)}/{len(results)} passed")
exit(0 if all(results) else 1)

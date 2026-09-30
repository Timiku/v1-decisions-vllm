#!/usr/bin/env python3
"""Live checks for /v1/decisions, mirroring test_systemone.py's style.

    python tools/test_decisions.py --endpoint http://localhost:8000 [--model X]

Covers: single form (answer key "decision"), multi form, audit
fields, calibration override (probabilities move, argmax doesn't),
malformed options (422), noul/score id rules (422), unknown backend
(400), partial failures, the temperature source in the audit block.
"""
import json
import sys
import urllib.error
import urllib.request

def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

ENDPOINT = _arg("--endpoint", "http://localhost:8000").rstrip("/")
MODEL = _arg("--model", None)

if MODEL is None:
    try:
        with urllib.request.urlopen(ENDPOINT + "/v1/models", timeout=10) as r:
            MODEL = json.loads(r.read())["data"][0]["id"]
    except Exception:
        MODEL = "decision-test"

PASS = 0
FAIL = 0

def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"PASS {name}")
    else:
        FAIL += 1
        print(f"FAIL {name} {detail}")

def post(body):
    req = urllib.request.Request(
        ENDPOINT + "/v1/decisions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


SINGLE = {
    "model": MODEL,
    "state": "Ticket: The payment gateway returned a 502 for every "
             "transaction in the last 10 minutes and checkout is fully "
             "down for customers.",
    "question": "Is this urgent?",
    "options": [
        {"id": "yes", "description": "Act now: on-call, immediate mitigation"},
        {"id": "no", "description": "Routine ticket for business hours"},
    ],
}

MULTI = {
    "model": MODEL,
    "state": "Ticket: My payouts have been failing for 3 days and two "
             "customers have escalated.",
    "questions": {
        "urgency": {"type": "noul", "instructions": "Does this convey urgency?",
                    "criteria": {"true": "Time-sensitive", "false": "Routine"}},
        "dept": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": "Payments, refunds",
                              "technical": "Bugs, outages"}},
        "frustration": {"type": "score", "instructions": "How frustrated?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]},
    },
}


def main():
    # 1. single form: answer under "decision"
    status, body = post(dict(SINGLE))
    ans = body.get("answers", {}).get("decision", {})
    check("single 200", status == 200, str(body)[:200])
    check("single type choice", ans.get("type") == "choice")
    probs = ans.get("probabilities", {})
    check("single probs sum 1",
          abs(sum(probs.values()) - 1.0) < 1e-6, str(probs))
    check("single argmax in answers",
          ans.get("choice") in probs, str(ans)[:200])

    # 2. audit block
    prov = ans.get("extra", {}).get("audit", {})
    check("audit state_sha256",
          isinstance(prov.get("state_sha256"), str)
          and len(prov["state_sha256"]) == 64)
    check("audit formula", prov.get("confidence_formula")
          == "normalized-peak-v1")
    check("audit forward_passes",
          isinstance(prov.get("forward_passes"), int))
    block = ans.get("extra", {}).get("backend", {})
    check("backend block named", isinstance(block.get("name"), str),
          str(block)[:150])
    check("backend option_logits", set(block.get("option_logits", {}))
          == set(probs), str(block)[:150])

    # 2b. extra levels: basic drops per-option lists, none drops extra
    status, b_basic = post({**SINGLE, "extra": "basic"})
    x = b_basic.get("answers", {}).get("decision", {}).get("extra", {})
    check("extra basic keeps audit, drops option lists",
          "audit" in x and all("option_logits" not in v
                               for v in x.values() if isinstance(v, dict)),
          str(x)[:150])
    status, b_none = post({**SINGLE, "extra": "none"})
    check("extra none has no extra block",
          "extra" not in b_none.get("answers", {}).get("decision", {}),
          str(b_none)[:150])

    # 3. calibration override: probabilities move, argmax holds
    status, body2 = post({**SINGLE, "calibration_temperature": 10.0})
    ans2 = body2.get("answers", {}).get("decision", {})
    p1 = [probs[k] for k in sorted(probs)]
    p2 = [ans2.get("probabilities", {}).get(k) for k in sorted(probs)]
    if all(isinstance(x, (int, float)) for x in p2):
        moved = any(abs(a - b) > 1e-6 for a, b in zip(p1, p2))
        same_argmax = ans.get("choice") == ans2.get("choice")
        check("calibration moves probs", moved, f"{p1} vs {p2}")
        check("calibration argmax invariant", same_argmax)
    else:
        check("calibration moves probs", False, str(body2)[:200])

    # 4. multi form: all three types
    status, body = post(dict(MULTI))
    answers = body.get("answers", {})
    check("multi 200", status == 200, str(body)[:200])
    check("multi noul present", "urgency" in answers
          and answers["urgency"].get("type") == "noul")
    check("multi choice present", "dept" in answers
          and answers["dept"].get("type") == "choice")
    check("multi score present", "frustration" in answers
          and answers["frustration"].get("type") == "score")
    usage = body.get("usage", {})
    check("multi usage output_tokens 0",
          usage.get("output_tokens") == 0, str(usage))

    # 6. malformed options -> 422
    status, _ = post({**SINGLE, "options": ["a", "b"]})
    check("string options 422", status == 422, f"got {status}")
    status, _ = post({**SINGLE, "options": [{"id": "a"}, {"id": "b"}]})
    check("id-only options 422", status == 422, f"got {status}")

    # 7. noul/score id rules -> 422
    bad_noul = {"model": MODEL, "state": "s", "questions": {
        "q": {"type": "noul", "instructions": "i",
              "criteria": {"yes": "y", "no": "n"}}}}
    status, _ = post(bad_noul)
    check("bad noul ids 422", status == 422, f"got {status}")

    # 8. unknown backend -> 400 (item 11)
    status, body = post({**SINGLE, "backend": "no-such-backend"})
    check("unknown backend 400", status == 400
          and "unknown decision backend" in body.get("error", {})
          .get("message", ""), f"got {status} {str(body)[:150]}")

    # 9. empty questions -> 422
    status, _ = post({"model": MODEL, "state": "s", "questions": {}})
    check("empty questions 422", status == 422, f"got {status}")

    # 10. partial failure: a direct read cannot hold 30 options, the other
    # question still answers
    body = dict(MULTI)
    body["backend_options"] = {"readout": "direct"}
    body["questions"] = dict(MULTI["questions"])
    body["questions"]["too_many"] = {
        "type": "choice", "instructions": "pick one",
        "criteria": {f"o{i}": f"option {i}" for i in range(30)}}
    status, body = post(body)
    check("partial failure listed", status == 200
          and "too_many" in (body.get("partial_failures") or {})
          and "dept" in body.get("answers", {}), str(body)[:200])

    # a bad readout must refuse with 422 (not a crash)
    status, body = post({**SINGLE, "backend_options": {"readout": "bogus"}})
    check("bad readout 422", status == 422
          and "readout" in body.get("error", {}).get("message", ""),
          f"got {status} {str(body)[:150]}")

    # every question failing returns the failure's own
    # code (400), not a 500 from a broken error path
    body = dict(MULTI)
    body["backend_options"] = {"readout": "direct"}
    body["questions"] = {"too_many": {
        "type": "choice", "instructions": "pick one",
        "criteria": {f"o{i}": f"option {i}" for i in range(30)}}}
    status, body = post(body)
    check("all questions fail 400", status == 400
          and "failed:" in body.get("error", {}).get("message", ""),
          f"got {status} {str(body)[:150]}")

    # 11. temperature source is reported
    status, body = post(dict(SINGLE))
    prov = (body.get("answers", {}).get("decision", {}).get("extra") or {}
            ).get("audit", {})
    # with the startup calibration on and no operator temperature, the
    # source is "calibrated"; "server" means the operator
    # hand-set T
    check("temperature source calibrated", prov.get("temperature_source")
          == "calibrated", str(prov)[:150])
    status, body = post({**SINGLE, "calibration_temperature": 2.0})
    prov = (body.get("answers", {}).get("decision", {}).get("extra") or {}
            ).get("audit", {})
    check("temperature source request", prov.get("temperature_source")
          == "request" and prov.get("calibration_temperature") == 2.0,
          str(prov)[:150])

    print(f"\n{PASS}/{PASS + FAIL} passed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

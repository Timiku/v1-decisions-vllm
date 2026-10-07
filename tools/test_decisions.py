#!/usr/bin/env python3
"""Live checks for /v1/decisions (OpenAI's Decisions format, plus
`extra`), mirroring test_systemone.py's style.

    python tools/test_decisions.py --endpoint http://localhost:8000 [--model X]

Covers: the three answer types in question order, choice values typed
(string vs boolean), audit and backend blocks, detail levels, the
calibration override (probabilities move, argmax doesn't), invalid
bodies (400), unknown backend (400), a failed question as a refusal,
every question failing (400), OpenAI's usage shape, the temperature
source in the audit block. With the `openai` package installed, the
OpenAI SDK must parse the response too.
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
    "input": "Ticket: The payment gateway returned a 502 for every "
             "transaction in the last 10 minutes and checkout is fully "
             "down for customers.",
    "questions": [{
        "type": "choice", "name": "urgent", "instructions": "Is this urgent?",
        "choices": [
            {"value": "yes",
             "description": "Act now: on-call, immediate mitigation"},
            {"value": "no", "description": "Routine ticket for business hours"},
        ]}],
}

MULTI = {
    "model": MODEL,
    "input": [{"role": "user", "content": [
        {"type": "input_text", "text": "Ticket: My payouts have been failing "
         "for 3 days and two customers have escalated."}]}],
    "questions": [
        {"type": "predicate", "name": "urgency",
         "instructions": "Does this convey urgency?"},
        {"type": "choice", "name": "dept", "instructions": "Which team?",
         "choices": [{"value": "billing", "description": "Payments, refunds"},
                     {"value": "technical", "description": "Bugs, outages"},
                     {"value": False, "description": "No team needed"}]},
        {"type": "score", "name": "frustration",
         "instructions": "How frustrated?",
         "levels": [{"label": "Calm"}, {"label": "Frustrated"},
                    {"label": "Very angry"}]},
    ],
}

TOO_MANY = {"type": "choice", "name": "too_many", "instructions": "pick one",
            "choices": [{"value": f"o{i}", "description": f"option {i}"}
                        for i in range(30)]}


def first(body):
    return (body.get("answers") or [{}])[0]


def probs_of(ans):
    return [p.get("probability") for p in ans.get("probabilities", [])]


def main():
    # 1. one choice question
    status, body = post(SINGLE)
    ans = first(body)
    check("single 200", status == 200, str(body)[:200])
    check("single type choice", ans.get("type") == "choice"
          and ans.get("name") == "urgent")
    probs = probs_of(ans)
    check("single probs sum 1",
          probs and abs(sum(probs) - 1.0) < 1e-6, str(probs))
    check("single choice is a value", ans.get("choice") in ("yes", "no"),
          str(ans)[:200])
    check("response envelope", set(body) == {"model", "answers", "usage",
                                             "extra"}, str(set(body)))

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
    check("backend option_logits", len(block.get("option_logits", {}))
          == 2, str(block)[:150])

    # 2b. detail levels: basic drops per-option lists, none drops extra
    status, b_basic = post({**SINGLE, "extra": {"detail": "basic"}})
    x = first(b_basic).get("extra", {})
    check("detail basic keeps audit, drops option lists",
          "audit" in x and all("option_logits" not in v
                               for v in x.values() if isinstance(v, dict)),
          str(x)[:150])
    status, b_none = post({**SINGLE, "extra": {"detail": "none"}})
    check("detail none has no extra block",
          "extra" not in first(b_none), str(b_none)[:150])

    # 3. calibration override: probabilities move, argmax holds
    status, body2 = post({**SINGLE,
                          "extra": {"calibration_temperature": 10.0}})
    ans2 = first(body2)
    p2 = probs_of(ans2)
    if p2 and all(isinstance(x, (int, float)) for x in p2):
        check("calibration moves probs",
              any(abs(a - b) > 1e-6 for a, b in zip(probs, p2)),
              f"{probs} vs {p2}")
        check("calibration argmax invariant",
              ans.get("choice") == ans2.get("choice"))
    else:
        check("calibration moves probs", False, str(body2)[:200])

    # 4. all three types, in question order, typed choice values
    status, body = post(MULTI)
    answers = body.get("answers", [])
    check("multi 200", status == 200, str(body)[:200])
    check("multi order and types",
          [(a.get("type"), a.get("name")) for a in answers]
          == [("predicate", "urgency"), ("choice", "dept"),
              ("score", "frustration")], str(answers)[:200])
    if len(answers) == 3:
        check("predicate probability",
              0.0 <= answers[0].get("probability", -1) <= 1.0)
        values = [p.get("value") for p in answers[1].get("probabilities", [])]
        check("choice values keep their types",
              values == ["billing", "technical", False], str(values))
        lv = answers[2].get("probabilities", [])
        check("score levels", [(p.get("value"), p.get("label")) for p in lv]
              == [(0, "Calm"), (1, "Frustrated"), (2, "Very angry")])
        check("score in range", 0.0 <= answers[2].get("score", -1) <= 2.0)
    usage = body.get("usage", {})
    check("usage shape",
          set(usage) == {"input_tokens", "input_tokens_details",
                         "output_tokens", "output_tokens_details",
                         "total_tokens"}
          and usage.get("output_tokens") == 0
          and usage.get("total_tokens") == usage.get("input_tokens"),
          str(usage))

    # 5. invalid bodies -> 400
    for name, bad in (
            ("unknown field", {**SINGLE, "state": "s"}),
            ("top-level setting", {**SINGLE, "calibration_temperature": 2}),
            ("duplicate choices", {**SINGLE, "questions": [{
                "type": "choice", "instructions": "i",
                "choices": [{"value": "a"}, {"value": "a"}]}]}),
            ("null name", {**SINGLE, "questions": [{
                "type": "predicate", "instructions": "i", "name": None}]}),
            ("image input", {**SINGLE, "input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": "data:,"}]}]}),
            ("no questions", {**SINGLE, "questions": []})):
        status, _ = post(bad)
        check(f"{name} 400", status == 400, f"got {status}")

    # 6. unknown backend -> 400
    status, body = post({**SINGLE, "extra": {"backend": "no-such-backend"}})
    check("unknown backend 400", status == 400
          and "unknown decision backend" in body.get("error", {})
          .get("message", ""), f"got {status} {str(body)[:150]}")

    # 7. a failed question is a refusal: a direct read cannot hold 30
    # options, the other questions still answer
    body = {**MULTI, "questions": MULTI["questions"] + [TOO_MANY],
            "extra": {"backend_options": {"readout": "direct"}}}
    status, body = post(body)
    answers = body.get("answers", [])
    check("refusal in place", status == 200 and len(answers) == 4
          and answers[3].get("type") == "refusal"
          and answers[3].get("name") == "too_many"
          and answers[3].get("extra", {}).get("error")
          and answers[1].get("type") == "choice", str(body)[:200])

    # a bad readout must refuse with 422 (not a crash)
    status, body = post({**SINGLE,
                         "extra": {"backend_options": {"readout": "bogus"}}})
    check("bad readout 422", status == 422
          and "readout" in body.get("error", {}).get("message", ""),
          f"got {status} {str(body)[:150]}")

    # every question failing is an error (the failure's own code, 400)
    status, body = post({**SINGLE, "questions": [TOO_MANY],
                         "extra": {"backend_options": {"readout": "direct"}}})
    check("all questions fail 400", status == 400
          and "'too_many' failed:" in body.get("error", {})
          .get("message", ""), f"got {status} {str(body)[:150]}")

    # 8. temperature source is reported
    status, body = post(SINGLE)
    prov = (first(body).get("extra") or {}).get("audit", {})
    # with the startup calibration on and no operator temperature, the
    # source is "calibrated"; "server" means the operator hand-set T
    check("temperature source calibrated", prov.get("temperature_source")
          == "calibrated", str(prov)[:150])
    status, body = post({**SINGLE, "extra": {"calibration_temperature": 2.0}})
    prov = (first(body).get("extra") or {}).get("audit", {})
    check("temperature source request", prov.get("temperature_source")
          == "request" and prov.get("calibration_temperature") == 2.0,
          str(prov)[:150])

    # 9. the OpenAI SDK parses the response (when installed)
    try:
        from openai import OpenAI
    except ImportError:
        print("SKIP openai SDK not installed")
    else:
        client = OpenAI(base_url=ENDPOINT + "/v1", api_key="unused")
        try:
            d = client.decisions.create(model=MODEL, input=MULTI["input"],
                                        questions=MULTI["questions"])
            check("sdk parses", [a.type for a in d.answers]
                  == ["predicate", "choice", "score"], str(d)[:200])
            d = client.decisions.create(
                model=MODEL, input=SINGLE["input"],
                questions=MULTI["questions"] + [TOO_MANY],
                extra_body={"extra": {"backend_options":
                                      {"readout": "direct"}}})
            check("sdk parses refusal", d.answers[3].type == "refusal")
        except Exception as e:  # noqa: BLE001 - report, don't crash
            check("sdk parses", False, repr(e)[:200])

    print(f"\n{PASS}/{PASS + FAIL} passed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

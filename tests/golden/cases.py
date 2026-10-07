"""Wire bodies whose prompt renders are pinned by tests/golden/.

renders-8bd7268.json was captured from the pre-unification code
(commit 8bd7268) with the same bodies; the unified code must render every
case byte-identically. Do not edit a case without re-deriving its golden
from 8bd7268.
"""

CASES = {
    "typed_noul_criteria": {
        "state": "Ticket: payouts failing for 3 days.",
        "questions": {"urgent": {
            "type": "noul", "instructions": "Does this convey urgency?",
            "criteria": {"true": "Explicitly time-sensitive",
                         "false": "No urgency expressed"}}}},
    "typed_noul_bare": {
        "state": "Ticket: payouts failing for 3 days.",
        "questions": {"urgent": {
            "type": "noul", "instructions": "Is it urgent?"}}},
    "typed_choice_mixed": {
        "state": "Ticket: payouts failing for 3 days.",
        "questions": {"dept": {
            "type": "choice", "instructions": "Which team?",
            "criteria": {"billing": "Payments, refunds",
                         "technical": None,
                         "legal": {"covers": ["disputes", "fraud"]}}}}},
    "typed_score": {
        "state": "Ticket: payouts failing for 3 days.",
        "questions": {"frustration": {
            "type": "score", "instructions": "How frustrated?",
            "criteria": ["Calm", "Frustrated", "Very angry"]}}},
    "typed_structured": {
        "state": {"ticket": "payouts failing", "days": 3,
                  "tags": ["billing", "ünïcode"]},
        "questions": {"dept": {
            "type": "choice",
            "instructions": {"task": "route", "field": "`ticket`"},
            "criteria": {"billing": "Payments", "technical": "Bugs"}}}},
    "typed_multi": {
        "state": "Ticket: payouts failing for 3 days.",
        "questions": {
            "urgent": {"type": "noul", "instructions": "Urgent?"},
            "dept": {"type": "choice", "instructions": "Which team?",
                     "criteria": {"billing": "Payments",
                                  "technical": "Bugs"}}}},
    # The three shorthand_* cases were one-question shorthand bodies (the
    # shorthand was removed in 0.2.0). Each is now the typed body the
    # shorthand expanded to, which renders byte-identically, so the keys
    # (and the golden file) stay.
    "shorthand_choice": {
        "state": "The patient reports chest pain radiating to the left arm.",
        "questions": {"decision": {
            "type": "choice", "instructions": "Which diagnosis fits best?",
            "criteria": {"cardiac": "Acute cardiac event",
                         "musculoskeletal": "Musculoskeletal",
                         "reflux": "GERD"}}}},
    "shorthand_noul": {
        "state": "s",
        "questions": {"decision": {
            "type": "noul", "instructions": "Is it urgent?",
            "criteria": {"true": "Yes, urgent", "false": "Not urgent"}}}},
    "shorthand_score": {
        "state": "s",
        "questions": {"decision": {
            "type": "score", "instructions": "How frustrated?",
            "criteria": ["Calm", "Angry"]}}},
}

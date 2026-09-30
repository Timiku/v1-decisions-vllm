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
    "shorthand_choice": {
        "state": "The patient reports chest pain radiating to the left arm.",
        "question": "Which diagnosis fits best?",
        "options": [
            {"id": "cardiac", "description": "Acute cardiac event"},
            {"id": "musculoskeletal", "description": "Musculoskeletal"},
            {"id": "reflux", "description": "GERD"}]},
    "shorthand_noul": {
        "state": "s", "question": "Is it urgent?",
        "options": [{"id": "true", "description": "Yes, urgent"},
                    {"id": "false", "description": "Not urgent"}]},
    "shorthand_score": {
        "state": "s", "question": "How frustrated?", "qtype": "score",
        "options": [{"id": "0", "description": "Calm"},
                    {"id": "1", "description": "Angry"}]},
}

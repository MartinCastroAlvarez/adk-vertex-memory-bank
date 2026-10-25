"""``OfflineConcierge``: a deterministic stand-in for the LLM, for the demo, CLI and CI.

It is not a language model. It reads what a real model would read - the user's
message and the memories ``preload_memory`` put in the request - and answers from
templates, so you can watch memory change the answers without running ollama:

* "forget everything about me" -> calls ``forget_me``, then confirms;
* anything else -> acknowledges preferences it just heard, and plans with the ones
  it recalls ("an aisle seat", "vegetarian restaurants", "departing from SFO").
"""

from __future__ import annotations

import re

from google.adk.models.llm_request import LlmRequest

from travel_concierge.callbacks import CLOSE_TAG
from travel_concierge.callbacks import OPEN_TAG
from travel_concierge.memory.extract import rule_candidates
from travel_concierge.memory.sanitize import WITHHELD
from travel_concierge.stub import Reply
from travel_concierge.stub import StubLlm
from travel_concierge.stub import ToolCall
from travel_concierge.stub import last_function_responses

FORGET_ME = re.compile(
    r"\b(forget (everything|all|me)\b|delete (my data|everything|all my)|erase (me|my data))", re.I
)
_PERSON = [
    (r"^The user's ", "your "),
    (r"^The user prefers ", "you prefer "),
    (r"^The user is ", "you are "),
    (r"^The user wants ", "you want "),
    (r"^The user usually travels with their ", "you usually travel with your "),
    (r"^The user asked to remember: ", "you asked me to remember: "),
    (r"^The user ", "you "),
]
_PLAN = [
    (re.compile(r"prefers (aisle|window) seats", re.I), "{0} seat on every flight"),
    (
        re.compile(r"is (vegetarian|vegan|pescatarian|gluten[- ]free|halal|kosher)", re.I),
        "{0} options at every meal",
    ),
    (re.compile(r"home airport is ([A-Z]{3})"), "departures from {0}"),
    (re.compile(r"flying with ([A-Z][\w ]+?)\.?$"), "{0} flights first"),
    (re.compile(r"prefers ([\w-]+) hotels", re.I), "{0} hotels"),
    (re.compile(r"budget is (.+?)\.?$"), "a budget of {0}"),
    (re.compile(r"avoid (.+?)\.?$"), "no {0}"),
]


def second_person(fact: str) -> str:
    for pattern, repl in _PERSON:
        if re.match(pattern, fact):
            return re.sub(pattern, repl, fact).rstrip(".")
    return fact.rstrip(".")


def recalled_facts(request: LlmRequest) -> list[str]:
    """Memory lines from the ``<PAST_CONVERSATIONS>`` block, minus withheld ones."""
    facts: list[str] = []
    for content in request.contents:
        for part in content.parts or []:
            text = part.text or ""
            if OPEN_TAG not in text:
                continue
            body = text[text.find(OPEN_TAG) + len(OPEN_TAG) : text.rfind(CLOSE_TAG)]
            for line in body.splitlines():
                prefix, _, rest = line.partition(": ")
                if prefix == "user" and rest and rest != WITHHELD and rest not in facts:
                    facts.append(rest)
    return facts


def user_message(request: LlmRequest) -> str:
    """The newest user text that is not a preloaded memory block."""
    for content in reversed(request.contents):
        if content.role != "user":
            continue
        text = " ".join(p.text for p in content.parts or [] if p.text)
        if text and OPEN_TAG not in text:
            return text.strip()
    return ""


def plan_key(fact: str) -> int | None:
    """Which planning slot a fact fills (seat, diet, ...), if any."""
    return next((i for i, (pattern, _) in enumerate(_PLAN) if pattern.search(fact)), None)


def plan_lines(facts: list[str]) -> list[str]:
    lines = []
    for fact in facts:
        key = plan_key(fact)
        if key is None:
            continue
        pattern, template = _PLAN[key]
        m = pattern.search(fact)
        if m:
            line = template.format(*m.groups())
            if line.startswith(("aisle", "window")):
                line = ("an " if line.startswith("aisle") else "a ") + line
            lines.append(line)
    return lines


class OfflineConcierge(StubLlm):
    model: str = "offline/concierge"

    def reply(self, request: LlmRequest) -> Reply:
        responses = last_function_responses(request)
        if responses and responses[0][0] == "forget_me":
            result = responses[0][1]
            if result.get("status") != "forgotten":
                return f"I couldn't delete your memories: {result.get('error', 'unknown error')}."
            return "Done. I deleted everything I remembered about you; next time we start fresh."
        message = user_message(request)
        if FORGET_ME.search(message):
            return ToolCall("forget_me")
        heard = [c.fact for c in rule_candidates(message) if not c.forget]
        # What the user just said wins over what was recalled for the same slot.
        fresh = {plan_key(f) for f in heard} - {None}
        remembered = [
            f for f in recalled_facts(request) if f not in heard and plan_key(f) not in fresh
        ]
        out: list[str] = []
        if heard:
            out.append("Noted for next time: " + "; ".join(second_person(f) for f in heard) + ".")
        if remembered:
            out.append(
                "From what I remember: " + "; ".join(second_person(f) for f in remembered) + "."
            )
        plan = plan_lines(heard + remembered)
        if plan:
            out.append("I'll plan with " + ", ".join(plan) + ".")
        if not out:
            out.append(
                "Happy to help. Tell me how you like to travel (seat, diet, airline, home "
                "airport, hotels) and I'll remember it for next time."
            )
        return " ".join(out)

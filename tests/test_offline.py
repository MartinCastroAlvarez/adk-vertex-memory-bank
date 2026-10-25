"""The offline concierge's templates (it is a stub, not a model)."""

from __future__ import annotations

from travel_concierge.offline import plan_lines
from travel_concierge.offline import second_person


def test_second_person() -> None:
    assert second_person("The user's name is Ana.") == "your name is Ana"
    assert second_person("The user prefers aisle seats.") == "you prefer aisle seats"
    assert second_person("The user asked to remember: x.") == "you asked me to remember: x"
    assert second_person("Likes trains.") == "Likes trains"


def test_plan_lines_cover_each_slot() -> None:
    facts = [
        "The user prefers window seats.",
        "The user is vegan.",
        "The user's home airport is LIS.",
        "The user prefers flying with TAP Portugal.",
        "The user prefers boutique hotels.",
        "The user's usual budget is $150 per night.",
        "The user wants to avoid layovers.",
        "The user's name is Ana.",
    ]
    assert plan_lines(facts) == [
        "a window seat on every flight",
        "vegan options at every meal",
        "departures from LIS",
        "TAP Portugal flights first",
        "boutique hotels",
        "a budget of $150 per night",
        "no layovers",
    ]

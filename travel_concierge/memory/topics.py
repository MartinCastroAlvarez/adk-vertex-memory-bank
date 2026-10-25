"""Memory topics: what is worth remembering, Memory Bank style.

Memory Bank extracts facts under *topics*: four managed ones
(``USER_PERSONAL_INFO``, ``USER_PREFERENCES``, ``KEY_CONVERSATION_DETAILS``,
``EXPLICIT_INSTRUCTIONS``) plus custom ones you describe in the bank's
``customization_configs``. This module defines the same managed topics and one
custom topic for the concierge, and the *slots* the local backend consolidates on:
a user has at most one live fact per slot, so a new seat preference replaces the old
one instead of piling up next to it.
"""

from __future__ import annotations

from dataclasses import dataclass

USER_PERSONAL_INFO = "USER_PERSONAL_INFO"
USER_PREFERENCES = "USER_PREFERENCES"
KEY_CONVERSATION_DETAILS = "KEY_CONVERSATION_DETAILS"
EXPLICIT_INSTRUCTIONS = "EXPLICIT_INSTRUCTIONS"
MANAGED_TOPICS = (
    USER_PERSONAL_INFO,
    USER_PREFERENCES,
    KEY_CONVERSATION_DETAILS,
    EXPLICIT_INSTRUCTIONS,
)

#: The custom topic, as ``custom_memory_topic.label`` in docs/vertex-setup.md.
TRAVEL_PREFERENCES = "travel_preferences"
TRAVEL_PREFERENCES_DESCRIPTION = (
    "Durable travel preferences: seat, diet, airline, home airport, hotel style, "
    "budget, pace, accessibility needs and things the user wants to avoid."
)
CUSTOM_TOPICS = {TRAVEL_PREFERENCES: TRAVEL_PREFERENCES_DESCRIPTION}
TOPICS = (*MANAGED_TOPICS, TRAVEL_PREFERENCES)


@dataclass(frozen=True)
class Slot:
    """One thing a user has a single current value for."""

    name: str
    topic: str
    description: str


SLOTS: dict[str, Slot] = {
    slot.name: slot
    for slot in (
        Slot("name", USER_PERSONAL_INFO, "what the user is called"),
        Slot("home_airport", TRAVEL_PREFERENCES, "the airport they usually fly from"),
        Slot("seat", TRAVEL_PREFERENCES, "aisle or window"),
        Slot("diet", TRAVEL_PREFERENCES, "dietary needs or restrictions"),
        Slot("airline", TRAVEL_PREFERENCES, "preferred airline"),
        Slot("hotel", TRAVEL_PREFERENCES, "preferred kind of hotel"),
        Slot("budget", TRAVEL_PREFERENCES, "usual budget per trip or per night"),
        Slot("avoid", TRAVEL_PREFERENCES, "things they want to avoid, e.g. red-eye flights"),
        Slot("companions", USER_PERSONAL_INFO, "who they usually travel with"),
    )
}


def topic_for(slot: str | None, fallback: str = USER_PREFERENCES) -> str:
    """The topic a slot belongs to (``fallback`` for a slotless fact)."""
    return SLOTS[slot].topic if slot in SLOTS else fallback

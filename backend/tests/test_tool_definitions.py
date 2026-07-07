"""Tool-schema invariants (spec §8, CLAUDE.md rule 2) — pure, no DB/API.

Two things are locked down here:

1. ``search_knowledge_base`` (§8.7) is registered and carries exactly one input:
   ``query``.
2. No tool schema anywhere exposes a user-scoping field. The verified user id is
   injected server-side by the orchestrator; its *structural absence* from every
   schema is the project's prompt-injection defense, so this test fails the build
   if anyone adds one "to be validated later".
"""

from __future__ import annotations

from app.agent.tools.definitions import ALL_TOOL_DEFINITIONS, SEARCH_KNOWLEDGE_BASE_TOOL


def test_search_knowledge_base_is_registered() -> None:
    assert SEARCH_KNOWLEDGE_BASE_TOOL in ALL_TOOL_DEFINITIONS
    names = [tool["name"] for tool in ALL_TOOL_DEFINITIONS]
    assert "search_knowledge_base" in names
    assert len(names) == len(set(names))  # no duplicate registrations


def test_search_knowledge_base_schema_has_only_query() -> None:
    schema = SEARCH_KNOWLEDGE_BASE_TOOL["input_schema"]
    assert set(schema["properties"].keys()) == {"query"}
    assert schema["required"] == ["query"]
    assert schema["properties"]["query"]["type"] == "string"


def _property_names(schema: dict) -> list[str]:
    """Recursively collect every property name in a JSON schema (incl. array items)."""
    names: list[str] = []
    for key, sub in schema.get("properties", {}).items():
        names.append(key)
        if isinstance(sub, dict):
            names.extend(_property_names(sub))
            if isinstance(sub.get("items"), dict):
                names.extend(_property_names(sub["items"]))
    return names


def test_no_user_scoping_field_in_any_tool_schema() -> None:
    for tool in ALL_TOOL_DEFINITIONS:
        for name in _property_names(tool["input_schema"]):
            assert "user" not in name.lower(), (
                f"tool {tool['name']!r} exposes user-scoping field {name!r} — "
                "the verified user id is injected server-side (CLAUDE.md rule 2)"
            )

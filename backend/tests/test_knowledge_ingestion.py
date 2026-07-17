"""Ingestion chunking/footer-parsing tests — pure functions only.

No network, no DB, and no real corpus: the ``knowledge_base/`` content is authored
separately, so these tests run entirely on fixture markdown strings (CLAUDE.md
rule 10 — CI must not hit live services, and must not depend on corpus files
existing). Covers the footer convention, section chunking, the leading
preamble rule, the prepended title/heading embed format, and index ordering.
"""

from __future__ import annotations

from seed.ingest_knowledge_base import chunk_document, parse_document

TITLE = "Common Shoulder Injuries in Lifters"

DOC_WITH_FOOTER = f"""# {TITLE}

Shoulder issues are common in pressing-heavy programs.

## Rotator Cuff Strain

Gradual-onset pain with overhead work.

## Impingement

Pinching sensation at the top of the press.

---
Source: Shoulder pain overview, Mayo Clinic — https://example.org/shoulder
Source: Rotator cuff injury guidance, NHS — https://example.org/cuff
"""

DOC_NO_FOOTER = """# Progressive Overload Principles

## Why Load Must Increase

Adaptation requires increasing demand.

## How Fast to Progress

Small jumps, sustained over months.
"""

DOC_SINGLE_SOURCE = """# Tendinopathy Basics

## What It Is

A load-tolerance problem in the tendon.

---
Source: Tendinopathy overview, APTA — https://example.org/tendon
"""

DOC_MID_RULE = """# Deload Protocols

## When

Every 4-8 weeks.

---

## How

Halve the volume.
"""


# --------------------------------------------------------------------------- #
# Footer parsing
# --------------------------------------------------------------------------- #
def test_footer_stripped_and_multiple_sources_joined() -> None:
    chunks = chunk_document(DOC_WITH_FOOTER, "injury_prevention")
    expected = (
        "Shoulder pain overview, Mayo Clinic — https://example.org/shoulder; "
        "Rotator cuff injury guidance, NHS — https://example.org/cuff"
    )
    assert all(c.source_citation == expected for c in chunks)
    # The footer is metadata, never embedded semantic content.
    assert all("Source:" not in c.text for c in chunks)
    assert all("---" not in c.text for c in chunks)


def test_single_source_footer_parsed_without_join() -> None:
    parsed = parse_document(DOC_SINGLE_SOURCE)
    assert parsed.source_citation == "Tendinopathy overview, APTA — https://example.org/tendon"


def test_doc_without_footer_has_none_citation() -> None:
    chunks = chunk_document(DOC_NO_FOOTER, "training")
    assert chunks
    assert all(c.source_citation is None for c in chunks)


def test_mid_document_rule_is_not_a_footer() -> None:
    # A "---" not followed by Source lines is document content, not a citation
    # footer — nothing is stripped and no citation is invented.
    chunks = chunk_document(DOC_MID_RULE, "training")
    assert all(c.source_citation is None for c in chunks)
    assert [c.section_heading for c in chunks] == ["When", "How"]
    assert "Halve the volume." in chunks[1].body


# --------------------------------------------------------------------------- #
# Section chunking
# --------------------------------------------------------------------------- #
def test_chunks_split_on_h2_headings() -> None:
    chunks = chunk_document(DOC_NO_FOOTER, "training")
    assert [c.section_heading for c in chunks] == [
        "Why Load Must Increase",
        "How Fast to Progress",
    ]
    assert "Adaptation requires increasing demand." in chunks[0].body
    assert "Small jumps, sustained over months." in chunks[1].body


def test_preamble_between_title_and_first_h2_is_kept_as_leading_chunk() -> None:
    chunks = chunk_document(DOC_WITH_FOOTER, "injury_prevention")
    assert len(chunks) == 3
    preamble = chunks[0]
    # The leading chunk uses the document title as its heading.
    assert preamble.section_heading == TITLE
    assert preamble.body == "Shoulder issues are common in pressing-heavy programs."


def test_doc_without_preamble_has_no_empty_leading_chunk() -> None:
    chunks = chunk_document(DOC_NO_FOOTER, "training")
    assert len(chunks) == 2
    assert chunks[0].section_heading == "Why Load Must Increase"


def test_chunk_text_prepends_title_and_heading() -> None:
    chunks = chunk_document(DOC_NO_FOOTER, "training")
    assert chunks[0].text == (
        "# Progressive Overload Principles\n"
        "## Why Load Must Increase\n"
        "Adaptation requires increasing demand."
    )


def test_chunk_index_is_zero_based_document_order() -> None:
    chunks = chunk_document(DOC_WITH_FOOTER, "injury_prevention")
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_category_and_title_propagate_to_every_chunk() -> None:
    chunks = chunk_document(DOC_WITH_FOOTER, "injury_prevention")
    assert all(c.category == "injury_prevention" for c in chunks)
    assert all(c.document_title == TITLE for c in chunks)


def test_parse_document_extracts_title_and_body() -> None:
    parsed = parse_document(DOC_NO_FOOTER)
    assert parsed.title == "Progressive Overload Principles"
    assert parsed.body.startswith("## Why Load Must Increase")

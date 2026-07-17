"""Offline knowledge-base ingestion.

Run from ``backend/`` with::

    python -m seed.ingest_knowledge_base

Walks ``knowledge_base/{training,nutrition,injury_prevention}/*.md``, parses the
optional citation footer, section-chunks each document, embeds every chunk via
Voyage, and wipe-and-reloads the ``knowledge_chunks`` table. Re-run whenever the
corpus changes; the corpus is tiny so a full wipe-and-reload is simpler and safer
than incremental diffing (one transaction, converges every time).

Design notes:
- The ``knowledge_base/`` path is anchored to the repo layout via ``Path(__file__)``,
  never CWD-relative — a CWD-relative path silently reads nothing when this runs
  from ``backend/`` (LESSONS.md env_file entry, same class of bug).
- Footer parsing and section chunking are pure functions (:func:`parse_document`,
  :func:`chunk_document`) with no network/DB dependency, so they're unit-testable
  on fixture strings (CLAUDE.md rule 10 — no live calls in CI).
- Embedding uses the SYNC ``voyageai.Client`` deliberately: this is a one-off
  offline script, not the request path, so blocking is fine. The async
  client is reserved for retrieval (:mod:`app.llm.voyage`).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import voyageai
from sqlalchemy import delete

from app.core.config import settings
from app.db.session import async_session_maker
from app.models.knowledge_chunk import KnowledgeChunk

# knowledge_base/ sits at the repo root, next to backend/. Anchor to this file's
# location, not the CWD (LESSONS.md): backend/seed/ -> parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_DIR = _REPO_ROOT / "knowledge_base"

CATEGORIES = ("training", "nutrition", "injury_prevention")

# Voyage accepts up to 128 inputs per embed call. The corpus fits in one or
# two batches today, but batch anyway so growth past 128 chunks needs no code change.
EMBED_BATCH_SIZE = 128


@dataclass(frozen=True)
class Chunk:
    """One section-chunk ready for embedding + insertion.

    ``text`` is the full ``"# title\n## heading\n{body}"`` string that is BOTH
    stored in ``chunk_text`` and embedded — heading provenance travels
    with the chunk so the embedding has context and Haiku can cite it.
    """

    document_title: str
    category: str
    section_heading: str
    body: str
    chunk_index: int
    source_citation: str | None

    @property
    def text(self) -> str:
        return f"# {self.document_title}\n## {self.section_heading}\n{self.body}"


@dataclass(frozen=True)
class ParsedDocument:
    """A markdown doc after footer stripping, before chunking."""

    title: str
    body: str  # everything after the "# title" line, footer removed
    source_citation: str | None


def parse_document(raw: str) -> ParsedDocument:
    """Strip the citation footer and split off the ``# title`` from a raw markdown doc.

    Footer convention: a final ``---`` line followed by one or more lines
    starting ``Source: ``. The footer is stripped from the text and the Source lines
    are joined with ``"; "`` into ``source_citation``. Parsed uniformly wherever a
    footer appears (only expected in injury_prevention, but not category-gated); a
    doc without a footer gets ``source_citation=None``.

    The ``# `` title line is located as the first line starting with ``"# "``; the
    returned ``body`` is everything after it (footer already removed).
    """
    body_without_footer, citation = _split_footer(raw)

    lines = body_without_footer.splitlines()
    title = ""
    title_line_index = None
    for i, line in enumerate(lines):
        if line.startswith("# "):
            title = line[2:].strip()
            title_line_index = i
            break

    if title_line_index is None:
        # No H1 — treat the whole thing as untitled body. Callers should author a
        # title, but don't drop content silently.
        remainder = body_without_footer
    else:
        remainder = "\n".join(lines[title_line_index + 1 :])

    return ParsedDocument(title=title, body=remainder.strip("\n"), source_citation=citation)


def _split_footer(raw: str) -> tuple[str, str | None]:
    """Return ``(text_without_footer, joined_citation_or_None)``.

    Recognizes a footer only at the very end of the document: a ``---`` line whose
    following non-empty lines all start with ``"Source: "``. Anything else (a ``---``
    used as a mid-document rule, or trailing prose) is left untouched.
    """
    lines = raw.rstrip("\n").splitlines()

    # Find the last standalone "---" line.
    sep_index = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "---":
            sep_index = i
            break
    if sep_index is None:
        return raw, None

    footer_lines = [ln for ln in lines[sep_index + 1 :] if ln.strip()]
    if not footer_lines or not all(ln.strip().startswith("Source: ") for ln in footer_lines):
        # Not a citation footer (mid-doc rule, or trailing non-Source text).
        return raw, None

    citations = [ln.strip()[len("Source: ") :].strip() for ln in footer_lines]
    text_without_footer = "\n".join(lines[:sep_index]).rstrip("\n")
    return text_without_footer, "; ".join(citations)


def chunk_document(raw: str, category: str) -> list[Chunk]:
    """Parse + section-chunk a raw markdown doc into ordered :class:`Chunk` objects.

    The doc body is split on ``## `` headings; each section
    becomes one chunk with ``chunk_index`` = its 0-based position. Non-empty prose
    between the ``# `` title and the first ``##`` is kept as its own leading
    (preamble) chunk — never silently dropped — using the document title as its
    heading (leading-chunk rule). The footer citation (if any) is
    attached to every chunk of the document.
    """
    doc = parse_document(raw)
    sections = _split_sections(doc.body)

    chunks: list[Chunk] = []
    for index, (heading, body) in enumerate(sections):
        chunks.append(
            Chunk(
                document_title=doc.title,
                category=category,
                # A leading preamble section has no ## heading; use the doc title so
                # the "# title\n## heading" format still holds and stays meaningful.
                section_heading=heading if heading is not None else doc.title,
                body=body,
                chunk_index=index,
                source_citation=doc.source_citation,
            )
        )
    return chunks


def _split_sections(body: str) -> list[tuple[str | None, str]]:
    """Split a doc body into ``(heading_or_None, section_body)`` in document order.

    A leading section (prose before the first ``## ``) is emitted with heading
    ``None`` only if it contains non-empty text. Each ``## `` heading starts a new
    section; empty sections (heading with no body) are still emitted so an authored
    heading isn't lost.
    """
    lines = body.splitlines()
    sections: list[tuple[str | None, list[str]]] = []
    current_heading: str | None = None
    current_body: list[str] = []
    started = False  # whether we've opened at least one section

    def flush() -> None:
        nonlocal current_body
        text = "\n".join(current_body).strip("\n")
        # Drop an empty leading preamble (no heading, no text); keep everything else.
        if current_heading is None and not text.strip():
            current_body = []
            return
        sections.append((current_heading, current_body))
        current_body = []

    for line in lines:
        if line.startswith("## "):
            if started:
                flush()
            current_heading = line[3:].strip()
            current_body = []
            started = True
        else:
            current_body.append(line)
            started = True
    if started:
        flush()

    return [(heading, "\n".join(b).strip("\n")) for (heading, b) in sections]


def _read_corpus(base_dir: Path) -> list[Chunk]:
    """Walk the category folders and chunk every ``*.md`` file (pure, no network)."""
    all_chunks: list[Chunk] = []
    for category in CATEGORIES:
        category_dir = base_dir / category
        if not category_dir.is_dir():
            continue
        for md_path in sorted(category_dir.glob("*.md")):
            raw = md_path.read_text(encoding="utf-8")
            all_chunks.extend(chunk_document(raw, category))
    return all_chunks


def _embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """Embed chunk texts via the SYNC Voyage client, in ≤128-input batches."""
    vo = voyageai.Client(api_key=settings.VOYAGE_API_KEY or None)
    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        result = vo.embed(
            [c.text for c in batch],
            model=settings.EMBED_MODEL_ID,
            input_type="document",
        )
        embeddings.extend(result.embeddings)
    return embeddings


async def _wipe_and_reload(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Delete all existing rows and insert the fresh set in one transaction."""
    async with async_session_maker() as session:
        await session.execute(delete(KnowledgeChunk))
        session.add_all(
            [
                KnowledgeChunk(
                    document_title=chunk.document_title,
                    category=chunk.category,
                    chunk_text=chunk.text,
                    chunk_index=chunk.chunk_index,
                    source_citation=chunk.source_citation,
                    embedding=embedding,
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ]
        )
        await session.commit()


def _summarize(chunks: list[Chunk]) -> str:
    """Human-readable ingestion summary: docs, chunks, per-category counts."""
    docs = {(c.category, c.document_title) for c in chunks}
    per_category: dict[str, int] = {}
    for c in chunks:
        per_category[c.category] = per_category.get(c.category, 0) + 1
    lines = [
        f"Ingested {len(docs)} document(s), {len(chunks)} chunk(s) total.",
        "Per category:",
    ]
    for category in CATEGORIES:
        lines.append(f"  {category}: {per_category.get(category, 0)} chunk(s)")
    return "\n".join(lines)


async def ingest() -> int:
    """Read + embed + wipe-and-reload the corpus; return the chunk count."""
    chunks = _read_corpus(KNOWLEDGE_BASE_DIR)
    if not chunks:
        print(f"No markdown found under {KNOWLEDGE_BASE_DIR}; nothing to ingest.")
        return 0

    embeddings = _embed_chunks(chunks)
    await _wipe_and_reload(chunks, embeddings)
    print(_summarize(chunks))
    return len(chunks)


if __name__ == "__main__":
    asyncio.run(ingest())

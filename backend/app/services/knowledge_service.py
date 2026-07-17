"""RAG retrieval + synthesis — the internals of ``search_knowledge_base``.

This is the implementation behind the tool contract. One call:

1. embeds the query via the async Voyage client (``input_type="query"``),
2. pulls the top-``top_k`` chunks by cosine distance from ``knowledge_chunks``,
3. has Haiku synthesize a 2-4 sentence answer grounded strictly in those chunks,
4. runs a second cheap Haiku entailment check, and
5. returns ``{"answer", "sources", "groundedness_passed}``.

Design decisions worth flagging:
- The knowledge base is global/unscoped — it holds no user data, so there is no
  ``user_id`` filter here (application-level access control applies to user-owned
  tables, not this one; CLAUDE.md rule 9). This is not a gap.
- Groundedness failure does NOT discard retrieval: we log a
  structured ``groundedness_failed`` warning and return a conservative answer WITH
  the sources so the user can read the material directly rather than being left
  with nothing.
- Deterministic bits (dedup by title, top-k ordering) live in code, not in the
  model (CLAUDE.md rule 3). Haiku only does the two language tasks (synthesis,
  entailment).

The client factories (:func:`get_voyage_client`, :func:`get_anthropic_client`) are
imported and called here so tests patch them at THIS module (their point of use)
and no live API is hit in CI (CLAUDE.md rule 10).
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.client import get_anthropic_client
from app.llm.voyage import get_voyage_client
from app.models.knowledge_chunk import KnowledgeChunk

log = get_logger(__name__)

DEFAULT_TOP_K = 5

_NO_RESULTS_ANSWER = "No relevant information found in the knowledge base."

_UNGROUNDED_ANSWER = (
    "I found related material but couldn't fully confirm a grounded answer — here "
    "are the sources I pulled, which you may want to read directly."
)

_SYNTHESIS_PROMPT = (
    "Answer the question using ONLY the reference material below. If the material "
    "doesn't contain enough information to answer, say so directly rather than "
    "using outside knowledge.\n\n"
    "Question: {query}\n\n"
    "Reference material:\n{reference}\n\n"
    "Give a 2-4 sentence answer in plain prose (no markdown headings or lists), "
    "grounded strictly in the material above."
)

_GROUNDEDNESS_PROMPT = (
    "Does the ANSWER rely only on information present in the REFERENCE material, or "
    "does it introduce claims not supported by it?\n\n"
    "Reference:\n{reference}\n\nAnswer:\n{answer}\n\n"
    "Respond with exactly one word: GROUNDED or UNGROUNDED."
)


def _dedup_sources(chunks: Sequence[KnowledgeChunk]) -> list[dict]:
    """Collapse chunks by ``document_title`` so one source isn't listed k times.

    Preserves first-seen (best-ranked) order and carries ``category`` +
    ``source_citation`` — injury answers must visibly show their citation.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for chunk in chunks:
        if chunk.document_title not in seen:
            seen.add(chunk.document_title)
            out.append(
                {
                    "document_title": chunk.document_title,
                    "category": chunk.category,
                    "source_citation": chunk.source_citation,
                }
            )
    return out


def _numbered_reference(chunks: Sequence[KnowledgeChunk]) -> str:
    """Build the ``[1] ... [2] ...`` reference block shared by both Haiku prompts."""
    return "\n".join(f"[{i + 1}] {chunk.chunk_text}" for i, chunk in enumerate(chunks))


def build_synthesis_prompt(query: str, chunks: Sequence[KnowledgeChunk]) -> str:
    """Render the synthesis prompt with numbered [1]..[k] reference chunks."""
    return _SYNTHESIS_PROMPT.format(query=query, reference=_numbered_reference(chunks))


async def _check_groundedness(answer: str, chunks: Sequence[KnowledgeChunk]) -> bool:
    """Second Haiku call: strict ``GROUNDED`` entailment check.

    Returns True only on an exact ``GROUNDED`` (after strip); any other word is
    treated as not-grounded, matching the classifier's strict one-word convention.
    """
    client = get_anthropic_client()
    response = await client.messages.create(
        model=settings.HAIKU_MODEL_ID,
        max_tokens=10,
        messages=[
            {
                "role": "user",
                "content": _GROUNDEDNESS_PROMPT.format(
                    reference=_numbered_reference(chunks), answer=answer
                ),
            }
        ],
    )
    return response.content[0].text.strip() == "GROUNDED"


async def search_knowledge_base(db: AsyncSession, query: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """Retrieve + synthesize a grounded answer for ``query``.

    Returns ``{"answer", "sources", "groundedness_passed"}``. Empty retrieval yields
    an explicit no-results answer; a failed groundedness check yields the
    conservative fallback answer *with* the sources and ``groundedness_passed=False``
    (never discards retrieval). Raises on provider/DB failure — the tool handler
    wraps that into ``{"error": ...}`` per CLAUDE.md rule 4.
    """
    voyage = get_voyage_client()
    embed_result = await voyage.embed([query], model=settings.EMBED_MODEL_ID, input_type="query")
    query_embedding = embed_result.embeddings[0]

    chunks = (
        (
            await db.execute(
                select(KnowledgeChunk)
                .order_by(KnowledgeChunk.embedding.cosine_distance(query_embedding))
                .limit(top_k)
            )
        )
        .scalars()
        .all()
    )
    if not chunks:
        return {"answer": _NO_RESULTS_ANSWER, "sources": [], "groundedness_passed": True}

    sources = _dedup_sources(chunks)

    client = get_anthropic_client()
    synthesis = await client.messages.create(
        model=settings.HAIKU_MODEL_ID,
        max_tokens=400,
        messages=[{"role": "user", "content": build_synthesis_prompt(query, chunks)}],
    )
    answer = synthesis.content[0].text

    if not await _check_groundedness(answer, chunks):
        # Groundedness failed — log for tuning, but keep the sources so the user can
        # read the underlying material rather than being left with nothing.
        log.warning(
            "groundedness_failed",
            extra={"query": query, "sources": [s["document_title"] for s in sources]},
        )
        return {
            "answer": _UNGROUNDED_ANSWER,
            "sources": sources,
            "groundedness_passed": False,
        }

    return {"answer": answer, "sources": sources, "groundedness_passed": True}

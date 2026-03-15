"""Shared memory service pipeline.

Used by both the MCP tool layer (tools/memory.py) and the gateway REST API.
Handles: content validation, date extraction, entity extraction,
importance scoring, embedding generation, surprise scoring,
storage, and auto-linking.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from ogham.database import auto_link_memory as db_auto_link
from ogham.database import get_profile_ttl as db_get_profile_ttl
from ogham.database import hybrid_search_memories, record_access
from ogham.database import store_memory as db_store
from ogham.embeddings import generate_embedding
from ogham.extraction import compute_importance, extract_dates, extract_entities

logger = logging.getLogger(__name__)


def store_memory_enriched(
    content: str,
    profile: str,
    source: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    auto_link: bool = True,
) -> dict[str, Any]:
    """Full store pipeline: validation, extraction, embedding, scoring, store, link.

    Returns the stored memory dict with id, created_at, links_created, etc.
    """
    # Lazy import to avoid circular dependency with tools/memory.py
    from ogham.tools.memory import _require_content

    _require_content(content)

    # Auto-extract dates into metadata
    dates = extract_dates(content)
    if dates:
        if metadata is None:
            metadata = {}
        metadata["dates"] = dates

    # Auto-extract entities as tags
    entity_tags = extract_entities(content)
    if entity_tags:
        if tags is None:
            tags = []
        else:
            tags = list(tags)
        tags.extend(entity_tags)

    # Compute importance score from content signals
    importance = compute_importance(content, tags)

    # Generate embedding
    embedding = generate_embedding(content)

    # Compute surprise score: how novel is this vs existing memories?
    surprise = 0.5
    try:
        existing = hybrid_search_memories(
            query_text=content[:200],
            query_embedding=embedding,
            profile=profile,
            limit=3,
        )
        if existing:
            max_sim = max(r.get("similarity", 0) for r in existing)
            surprise = round(1.0 - max_sim, 3)
    except Exception:
        logger.debug("Surprise scoring skipped: search failed, using default 0.5")

    # TTL
    ttl_days = db_get_profile_ttl(profile)
    expires_at = None
    if ttl_days is not None:
        expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()

    # Store
    result = db_store(
        content=content,
        embedding=embedding,
        profile=profile,
        metadata=metadata,
        source=source,
        tags=tags,
        expires_at=expires_at,
        importance=importance,
        surprise=surprise,
    )

    response: dict[str, Any] = {
        "status": "stored",
        "id": result["id"],
        "profile": profile,
        "created_at": result["created_at"],
        "expires_at": expires_at,
        "importance": importance,
        "surprise": surprise,
    }

    # Auto-link
    if auto_link:
        links_created = db_auto_link(
            memory_id=result["id"],
            embedding=embedding,
            profile=profile,
        )
        response["links_created"] = links_created

    return response


def search_memories_enriched(
    query: str,
    profile: str,
    limit: int = 10,
    tags: list[str] | None = None,
    source: str | None = None,
    graph_depth: int = 0,
) -> list[dict[str, Any]]:
    """Full search pipeline: embed query, search, optional graph traversal, record access."""
    embedding = generate_embedding(query)

    if graph_depth > 0:
        from ogham.database import graph_augmented_search

        results = graph_augmented_search(
            query_text=query,
            query_embedding=embedding,
            profile=profile,
            limit=limit,
            graph_depth=graph_depth,
            tags=tags,
            source=source,
        )
    else:
        results = hybrid_search_memories(
            query_text=query,
            query_embedding=embedding,
            profile=profile,
            limit=limit,
            tags=tags,
            source=source,
        )

    if results:
        record_access([r["id"] for r in results])

    return results

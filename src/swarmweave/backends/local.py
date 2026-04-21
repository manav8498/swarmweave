"""File-backed reference implementation of :class:`Backend`.

LocalBackend stores observations as JSONL under
``~/.swarmweave/<session_id>.jsonl``. Retrieval is embedding-based by default
(OpenAI ``text-embedding-3-small`` — small, cheap, fast) with a Jaccard +
recency fallback for offline use or when no API key is set.

Why ship embeddings as the default
----------------------------------

A reference implementation that uses keyword overlap stops being useful the
moment your shared log grows past a handful of observations. Real semantic
similarity is what makes the substrate work on real workloads. The embedding
model is small enough that the marginal cost is negligible (well under a
cent per swarm run on typical workloads) and the upgrade is invisible to
user code — same ``Backend`` interface, same ``ContextSlice`` shape.

Set ``mode="jaccard"`` to force the offline fallback (useful for tests,
examples, or environments without ``OPENAI_API_KEY``).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal

from swarmweave.backends.base import Backend
from swarmweave.types import ContextSlice, Observation, Outcome

logger = logging.getLogger("swarmweave")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_RECENCY_HALF_LIFE_S = 600.0  # 10 minutes — long enough for one swarm run.
_DEFAULT_EMBED_MODEL = "text-embedding-3-small"

LocalRetrievalMode = Literal["auto", "embedding", "jaccard"]


def _tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(text)}


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class LocalBackend(Backend):
    """Append-only JSONL log with embedding-based (or Jaccard) retrieval.

    Parameters
    ----------
    session_id:
        Logical name for this run. If omitted, a random hex id is used.
    root:
        Directory the log lives in. Defaults to ``~/.swarmweave``.
    persist:
        When ``False``, observations are kept in memory only and nothing is
        written to disk. Useful for tests and benchmarks that want full
        isolation between runs.
    mode:
        ``"auto"`` (default) — use embeddings when ``OPENAI_API_KEY`` is set,
        otherwise fall back to Jaccard. ``"embedding"`` forces embedding
        retrieval (raises if no key). ``"jaccard"`` forces the offline path.
    embedding_model:
        OpenAI embedding model. Defaults to ``text-embedding-3-small``.
    api_key:
        Optional override for the embedding-API key (otherwise uses
        ``OPENAI_API_KEY`` from the environment).
    """

    def __init__(
        self,
        session_id: str | None = None,
        root: Path | str | None = None,
        persist: bool = True,
        mode: LocalRetrievalMode = "auto",
        embedding_model: str = _DEFAULT_EMBED_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex
        self.persist = persist
        self.root = Path(root) if root is not None else Path.home() / ".swarmweave"
        self._log: list[Observation] = []
        self._outcome: Outcome | None = None
        self._lock = asyncio.Lock()
        self._embeddings: dict[str, list[float]] = {}
        self.embedding_model = embedding_model

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if mode == "embedding" and not resolved_key:
            raise RuntimeError(
                "LocalBackend(mode='embedding') requires OPENAI_API_KEY; set it "
                "or use mode='auto' / 'jaccard'."
            )
        if mode == "jaccard":
            self._mode: LocalRetrievalMode = "jaccard"
            self._embed_client: Any = None
        elif mode == "auto" and not resolved_key:
            self._mode = "jaccard"
            self._embed_client = None
        else:
            try:
                from openai import AsyncOpenAI

                self._embed_client = AsyncOpenAI(api_key=resolved_key)
                self._mode = "embedding"
            except ImportError:
                logger.warning("openai SDK not installed; LocalBackend falling back to Jaccard.")
                self._embed_client = None
                self._mode = "jaccard"

        if self.persist:
            self.root.mkdir(parents=True, exist_ok=True)
            self._log_path: Path | None = self.root / f"{self.session_id}.jsonl"
            if self._log_path.exists():
                self._load_existing()
        else:
            self._log_path = None

    @property
    def mode(self) -> LocalRetrievalMode:
        """Effective retrieval mode (``embedding`` or ``jaccard``) for this instance."""
        return self._mode

    def _load_existing(self) -> None:
        assert self._log_path is not None
        with self._log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._log.append(Observation.model_validate_json(line))
                except Exception as exc:
                    logger.warning("skipping malformed observation: %s", exc)

    async def _embed(self, text: str) -> list[float]:
        if self._embed_client is None:
            return []
        try:
            response = await self._embed_client.embeddings.create(
                model=self.embedding_model,
                input=text[:8000],  # token budget guard
            )
            vec: list[float] = list(response.data[0].embedding)
            return vec
        except Exception as exc:
            logger.warning(
                "embedding call failed (%s); falling back to Jaccard for this read.", exc
            )
            return []

    async def append(self, observation: Observation) -> None:
        async with self._lock:
            self._log.append(observation)
            if self.persist and self._log_path is not None:
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(observation.model_dump_json())
                    fh.write("\n")
        # Pre-compute observation embedding outside the lock so a slow API
        # call doesn't block other writers/readers.
        if self._mode == "embedding":
            vec = await self._embed(observation.content)
            if vec:
                self._embeddings[observation.id] = vec

    async def query(self, agent_id: str, task: str, limit: int = 8) -> ContextSlice:
        async with self._lock:
            snapshot = list(self._log)

        if not snapshot:
            return ContextSlice()

        candidates = [o for o in snapshot if o.agent_id != agent_id]
        if not candidates:
            return ContextSlice()

        if self._mode == "embedding":
            slice_ = await self._query_embedding(task, candidates, limit)
            if slice_ is not None:
                return slice_
            # If embedding path produced no usable scores, fall through to Jaccard.

        return self._query_jaccard(task, candidates, limit)

    async def _query_embedding(
        self, task: str, candidates: list[Observation], limit: int
    ) -> ContextSlice | None:
        task_vec = await self._embed(task)
        if not task_vec:
            return None
        scored: list[tuple[float, Observation]] = []
        now = max(o.timestamp for o in candidates)
        missing: list[Observation] = []
        for obs in candidates:
            obs_vec = self._embeddings.get(obs.id)
            if obs_vec is None:
                missing.append(obs)
                continue
            similarity = _cosine(task_vec, obs_vec)
            age = max(0.0, now - obs.timestamp)
            recency = math.exp(-age / _RECENCY_HALF_LIFE_S)
            scored.append((similarity + 0.05 * recency, obs))
        # Backfill any missing embeddings (e.g., loaded from disk on a fresh process).
        for obs in missing:
            obs_vec = await self._embed(obs.content)
            if obs_vec:
                self._embeddings[obs.id] = obs_vec
                similarity = _cosine(task_vec, obs_vec)
                age = max(0.0, now - obs.timestamp)
                recency = math.exp(-age / _RECENCY_HALF_LIFE_S)
                scored.append((similarity + 0.05 * recency, obs))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        top = [obs for _, obs in scored[:limit]]
        tokens_estimate = sum(len(_tokenize(o.content)) for o in top)
        return ContextSlice(observations=top, tokens_estimate=tokens_estimate)

    def _query_jaccard(self, task: str, candidates: list[Observation], limit: int) -> ContextSlice:
        task_tokens = _tokenize(task)
        now = max(o.timestamp for o in candidates)
        scored: list[tuple[float, Observation]] = []
        for obs in candidates:
            obs_tokens = _tokenize(obs.content) | _tokenize(obs.task)
            if not obs_tokens or not task_tokens:
                similarity = 0.0
            else:
                inter = len(task_tokens & obs_tokens)
                union = len(task_tokens | obs_tokens)
                similarity = inter / union if union else 0.0
            age = max(0.0, now - obs.timestamp)
            recency = math.exp(-age / _RECENCY_HALF_LIFE_S)
            scored.append((similarity + 0.15 * recency, obs))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = [obs for _, obs in scored[:limit]]
        tokens_estimate = sum(len(_tokenize(o.content)) for o in top)
        return ContextSlice(observations=top, tokens_estimate=tokens_estimate)

    async def trace(self) -> list[Observation]:
        async with self._lock:
            return list(self._log)

    async def finalize(self, outcome: Outcome) -> None:
        async with self._lock:
            self._outcome = outcome
            if self.persist and self._log_path is not None:
                outcome_path = self._log_path.with_suffix(".outcome.json")
                outcome_path.write_text(outcome.model_dump_json(indent=2))
        logger.info(
            "finalized session %s success=%s score=%s",
            self.session_id,
            outcome.success,
            outcome.score,
        )

    async def close(self) -> None:
        if self._embed_client is not None:
            import contextlib

            with contextlib.suppress(Exception):
                await self._embed_client.close()
            self._embed_client = None

    @property
    def last_outcome(self) -> Outcome | None:
        """Most recent outcome recorded via :meth:`finalize`."""
        return self._outcome

"""Self-improving swarm: lessons learned from past runs.

The idea
--------

Most multi-agent frameworks rerun a task as if they've never seen it before.
Lessons changes that. After every successful run, swarmweave asks the
synthesizer to extract a short list of *lessons* — concrete, reusable
patterns it would tell its future self to remember next time. These get
appended to a file under ``~/.swarmweave/lessons/<lesson_book>.jsonl``.

On the next run, before any worker fires, the supervisor:

1. retrieves the lessons most relevant to the current user task (semantic
   search if embeddings are available, lexical Jaccard otherwise),
2. injects them into the shared context as ``Observation(agent_id="lessons",
   kind="note")``,
3. so every worker reads from a context that already contains hard-won
   patterns from earlier swarms.

The result, demonstrable in a 60-second screen recording: **the same swarm,
on the same task, gets measurably better the second time it runs.**

Privacy / honesty
-----------------

Lessons are stored locally as plain JSONL — fully readable, fully editable,
fully deletable. No telemetry leaves your machine. You can edit lessons by
hand to teach the swarm domain knowledge directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from swarmweave._openai_client import OpenAIClient
from swarmweave.shared_context import SharedContext
from swarmweave.types import Observation

logger = logging.getLogger("swarmweave")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_DEFAULT_BOOK = "default"
_EXTRACT_SYSTEM = """You are a senior reviewer extracting transferable lessons
from a completed swarm run.

You will be given the user task, the final synthesized answer, and the
per-worker findings. Produce 1-5 short lessons that a future swarm should
read BEFORE attempting a similar task. Each lesson must be:

- one or two sentences,
- concrete and actionable (not platitudes),
- generalizable beyond this exact task (not a verbatim restatement),
- something a worker would benefit from knowing up front.

If the run had no useful lessons (trivial task, failed run, or nothing
generalizable), return an empty list. Do not fabricate.

Use the record_lessons tool. Do not respond in plain text."""

_RECORD_TOOL = {
    "type": "function",
    "function": {
        "name": "record_lessons",
        "description": "Record 0-5 short transferable lessons from this run.",
        "parameters": {
            "type": "object",
            "properties": {
                "lessons": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "lesson": {
                                "type": "string",
                                "description": "One or two sentences. Concrete and reusable.",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Short keywords for retrieval.",
                            },
                        },
                        "required": ["lesson", "tags"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["lessons"],
            "additionalProperties": False,
        },
    },
}


def _tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(text)}


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class LessonBook:
    """Persistent, append-only store of lessons learned from past runs.

    Parameters
    ----------
    name:
        Lesson-book identifier. Use one book per project / domain.
    root:
        Directory the book lives in. Defaults to ``~/.swarmweave/lessons``.
    use_embeddings:
        When True (default), retrieval is semantic via OpenAI embeddings;
        when False or no API key, falls back to Jaccard token overlap.
    embedding_model:
        OpenAI embedding model. Default ``text-embedding-3-small``.
    """

    def __init__(
        self,
        name: str = _DEFAULT_BOOK,
        root: Path | str | None = None,
        use_embeddings: bool = True,
        embedding_model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        self.name = name
        self.root = Path(root) if root is not None else Path.home() / ".swarmweave" / "lessons"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{name}.jsonl"
        self._lock = asyncio.Lock()
        self.embedding_model = embedding_model

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if use_embeddings and resolved_key:
            try:
                from openai import AsyncOpenAI

                self._embed_client: Any = AsyncOpenAI(api_key=resolved_key)
            except ImportError:
                self._embed_client = None
        else:
            self._embed_client = None

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("skipping malformed lesson: %s", exc)
        return rows

    def _append(self, lesson: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(lesson))
            fh.write("\n")

    def count(self) -> int:
        return len(self._read_all())

    def all(self) -> list[dict[str, Any]]:
        """Return every lesson in this book (most recent last)."""
        return self._read_all()

    async def _embed(self, text: str) -> list[float]:
        if self._embed_client is None:
            return []
        try:
            r = await self._embed_client.embeddings.create(
                model=self.embedding_model,
                input=text[:8000],
            )
            vec: list[float] = list(r.data[0].embedding)
            return vec
        except Exception as exc:
            logger.warning("lesson embedding failed (%s); using Jaccard.", exc)
            return []

    async def retrieve(self, task: str, limit: int = 5) -> list[str]:
        """Return the lessons most relevant to ``task`` as plain strings."""
        async with self._lock:
            rows = self._read_all()
        if not rows:
            return []

        if self._embed_client is not None:
            task_vec = await self._embed(task)
            if task_vec:
                scored: list[tuple[float, str]] = []
                for r in rows:
                    vec = r.get("embedding")
                    if not vec:
                        continue
                    score = _cosine(task_vec, vec)
                    scored.append((score, r["lesson"]))
                if scored:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    return [lesson for _, lesson in scored[:limit] if _ > 0.2]

        # Jaccard fallback (also runs if embeddings missing for legacy lessons).
        task_tokens = _tokenize(task)
        lex_scored: list[tuple[float, str]] = []
        for r in rows:
            lesson_tokens = _tokenize(r["lesson"]) | _tokenize(" ".join(r.get("tags", [])))
            if not task_tokens or not lesson_tokens:
                continue
            inter = len(task_tokens & lesson_tokens)
            union = len(task_tokens | lesson_tokens)
            if union:
                lex_scored.append((inter / union, r["lesson"]))
        lex_scored.sort(key=lambda x: x[0], reverse=True)
        return [lesson for score, lesson in lex_scored[:limit] if score > 0.05]

    async def record_lesson(self, lesson: str, tags: list[str] | None = None) -> None:
        """Append a single lesson (with optional tags). Embed if possible."""
        vec = await self._embed(lesson)
        row = {
            "id": uuid.uuid4().hex,
            "timestamp": time.time(),
            "lesson": lesson,
            "tags": tags or [],
        }
        if vec:
            row["embedding"] = vec
        async with self._lock:
            self._append(row)


async def preload_lessons(
    book: LessonBook, shared_context: SharedContext, task: str, limit: int = 5
) -> int:
    """Read relevant lessons from ``book`` and write them into shared context.

    Returns the number of lessons preloaded. Workers will then see them on
    their first ``SharedContext.aread`` call. Designed to be called by user
    code before ``Supervisor.arun(task)``.
    """
    lessons = await book.retrieve(task, limit=limit)
    for lesson in lessons:
        obs = Observation(
            agent_id="lessons",
            task=task,
            content=f"LESSON FROM PAST RUN: {lesson}",
            kind="note",
            metadata={"source": "lessons", "book": book.name},
        )
        await shared_context.awrite("lessons", obs)
    return len(lessons)


async def extract_and_record_lessons(
    book: LessonBook,
    user_task: str,
    final_output: str,
    per_worker_outputs: dict[str, str],
    client: OpenAIClient,
    max_lessons: int = 5,
) -> list[str]:
    """Ask the model to distill lessons from a completed run, then persist them.

    Returns the list of lesson strings recorded.
    """
    block = "\n\n".join(f"### {n}\n{o}" for n, o in per_worker_outputs.items())
    user_msg = (
        f"User task:\n{user_task}\n\n"
        f"Final synthesized answer:\n{final_output}\n\n"
        f"Per-worker findings:\n{block}\n\n"
        f"Extract up to {max_lessons} short transferable lessons via the "
        "record_lessons tool now."
    )
    result = await client.call(
        system=_EXTRACT_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
        tools=[_RECORD_TOOL],
        tool_choice={"type": "function", "function": {"name": "record_lessons"}},
    )
    recorded: list[str] = []
    for tc in result.tool_calls:
        if tc.name == "record_lessons":
            for entry in tc.arguments.get("lessons", []):
                lesson = entry.get("lesson", "").strip()
                if not lesson:
                    continue
                tags = list(entry.get("tags", []) or [])
                await book.record_lesson(lesson, tags=tags)
                recorded.append(lesson)
            break
    return recorded


class Mentor:
    """Convenience wrapper around the lessons workflow.

    Two-line API to make a swarm self-improving::

        mentor = Mentor(book="my_project")
        await mentor.before_run(supervisor, user_task)
        result = await supervisor.arun(user_task)
        await mentor.after_run(supervisor, user_task, result)
    """

    def __init__(
        self,
        book: str = _DEFAULT_BOOK,
        root: Path | str | None = None,
        use_embeddings: bool = True,
    ) -> None:
        self.book = LessonBook(name=book, root=root, use_embeddings=use_embeddings)
        self.preloaded: int = 0

    async def before_run(self, supervisor: Any, user_task: str, limit: int = 5) -> int:
        self.preloaded = await preload_lessons(
            self.book, supervisor.shared_context, user_task, limit=limit
        )
        if self.preloaded:
            logger.info("preloaded %d lessons into shared context", self.preloaded)
        return self.preloaded

    async def after_run(
        self,
        supervisor: Any,
        user_task: str,
        result: Any,
        max_lessons: int = 5,
    ) -> list[str]:
        lessons = await extract_and_record_lessons(
            self.book,
            user_task=user_task,
            final_output=result.final_output,
            per_worker_outputs=result.per_worker_outputs,
            client=supervisor.client,
            max_lessons=max_lessons,
        )
        if lessons:
            logger.info("recorded %d new lessons", len(lessons))
        return lessons

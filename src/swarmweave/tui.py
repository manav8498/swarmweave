"""Live terminal UI for watching a swarm think.

Renders a split-screen Rich dashboard:

    ┌──────── workers ────────┐  ┌────────── shared context ──────────┐
    │ scout       ⠋ thinking  │  │ 14:22:01 [scout] FINDING: ...      │
    │ investigator ⠋ tool:read │  │ 14:22:14 [investigator] reads ... │
    │ fixer        idle        │  │ 14:22:30 [fixer] FINDING: ...     │
    └──────────────────────────┘  └────────────────────────────────────┘
    ┌─ metrics ────────────────────────────────────────────────────────┐
    │ 02:14 elapsed │ 12 calls │ 8,420 tokens in / 23,991 out │ ~$0.04 │
    └──────────────────────────────────────────────────────────────────┘

Use it like::

    from swarmweave import EventBus, Supervisor
    from swarmweave.tui import LiveDashboard

    events = EventBus()
    sup = Supervisor(workers=..., events=events, mode="sequential")

    async with LiveDashboard(events) as dash:
        await sup.arun("...")

The dashboard streams everything the swarm publishes — tool calls,
shared-context reads/writes, per-worker timing, accumulating cost — and
renders it at 8 fps. When the run finishes the dashboard prints a clean
summary card and returns control.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from swarmweave.events import EventBus, SwarmEvent

# OpenAI gpt-4o-mini pricing (USD per 1M tokens) used for the live cost meter.
# These are best-effort and update over time; they are NOT a contract.
_PRICE_IN_PER_M = 0.15
_PRICE_OUT_PER_M = 0.60

_ACCENT = "#1f6feb"
_DIM = "grey50"
_OK = "green"
_WARN = "yellow"
_ERR = "red"


@dataclass
class WorkerState:
    name: str
    role: str = ""
    status: str = "idle"  # idle | thinking | tool | wrap-up | done
    current_tool: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    last_finding: str = ""
    tool_calls: int = 0
    context_reads: int = 0

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at


@dataclass
class RunState:
    user_task: str = ""
    mode: str = "parallel"
    model: str = ""
    started_at: float = field(default_factory=time.time)
    finished: bool = False
    workers: dict[str, WorkerState] = field(default_factory=dict)
    log: list[tuple[float, str, str]] = field(default_factory=list)  # (ts, actor, msg)
    input_tokens: int = 0
    output_tokens: int = 0
    worker_calls: int = 0

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def cost(self) -> float:
        return (
            self.input_tokens * _PRICE_IN_PER_M / 1_000_000
            + self.output_tokens * _PRICE_OUT_PER_M / 1_000_000
        )


class LiveDashboard:
    """Async context manager that subscribes to an EventBus and renders live."""

    def __init__(self, bus: EventBus, refresh_per_second: int = 8) -> None:
        self.bus = bus
        self.state = RunState()
        self._console = Console()
        self._live: Live | None = None
        self._consumer_task: asyncio.Task[None] | None = None
        self._spinner = Spinner("dots", style=_ACCENT)
        self._refresh = refresh_per_second

    async def __aenter__(self) -> LiveDashboard:
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=self._refresh,
            screen=False,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.__enter__()
        self._consumer_task = asyncio.create_task(self._consume())
        return self

    async def __aexit__(self, *exc: Any) -> None:
        # Wait briefly for the bus to close (the supervisor calls aclose on run_done).
        if self._consumer_task is not None:
            try:
                await asyncio.wait_for(self._consumer_task, timeout=2.0)
            except TimeoutError:
                self._consumer_task.cancel()
        if self._live is not None:
            self._live.update(self._render(), refresh=True)
            self._live.__exit__(None, None, None)
            self._live = None
        # Print a final summary card below the live region.
        self._console.print(self._summary_card())

    async def _consume(self) -> None:
        async for event in self.bus:
            self._apply(event)
            if self._live is not None:
                self._live.update(self._render())
            if event.kind == "run_done":
                self.state.finished = True
                break

    # ------------------------------------------------------------------ apply

    def _ensure_worker(self, name: str) -> WorkerState:
        if name not in self.state.workers:
            self.state.workers[name] = WorkerState(name=name)
        return self.state.workers[name]

    def _log(self, actor: str, msg: str) -> None:
        self.state.log.append((time.time(), actor, msg))
        if len(self.state.log) > 200:
            self.state.log = self.state.log[-200:]

    def _apply(self, e: SwarmEvent) -> None:
        p = e.payload
        if e.kind == "run_start":
            self.state.user_task = p.get("user_task", "")
            self.state.mode = p.get("mode", "parallel")
            self.state.model = p.get("model", "")
            for w in p.get("workers", []):
                ws = self._ensure_worker(w["name"])
                ws.role = w.get("role", "")
            self._log("supervisor", f"run_start mode={self.state.mode} model={self.state.model}")
        elif e.kind == "decompose_start":
            self._log("supervisor", "decomposing task...")
        elif e.kind == "decompose_done":
            for a in p.get("assignments", []):
                ws = self._ensure_worker(a["worker"])
                ws.role = ws.role or a["subtask"][:60]
            self._log("supervisor", f"decompose_done -> {len(p.get('assignments', []))} subtasks")
        elif e.kind == "worker_start":
            ws = self._ensure_worker(e.actor)
            ws.status = "thinking"
            ws.started_at = time.time()
            self._log(e.actor, "starting subtask")
        elif e.kind == "worker_tool_call":
            ws = self._ensure_worker(e.actor)
            ws.status = "tool"
            ws.current_tool = p.get("tool", "?")
            ws.tool_calls += 1
            self._log(e.actor, f"tool_call: {ws.current_tool}")
        elif e.kind == "worker_tool_result":
            ws = self._ensure_worker(e.actor)
            ws.status = "thinking"
            ws.current_tool = None
        elif e.kind == "worker_wrap_up":
            ws = self._ensure_worker(e.actor)
            ws.status = "wrap-up"
            self._log(e.actor, "tool budget exhausted; forcing wrap-up")
        elif e.kind == "worker_done":
            ws = self._ensure_worker(e.actor)
            ws.status = "done"
            ws.finished_at = time.time()
            ws.last_finding = p.get("content_preview", "")
            self._log(e.actor, f"done in {p.get('wall_clock', 0)}s")
        elif e.kind == "context_write":
            self._log(e.actor, f"WRITE → {p.get('content_preview', '')[:120]}")
        elif e.kind == "context_read":
            ws = self._ensure_worker(e.actor)
            ws.context_reads += 1
            self._log(e.actor, f"READ ← {p.get('hit_count', 0)} prior observations")
        elif e.kind == "synthesize_start":
            self._log("supervisor", "synthesizing...")
        elif e.kind == "synthesize_done":
            self._log("supervisor", "synthesis done")
        elif e.kind == "run_done":
            self.state.finished = True
            self.state.input_tokens = p.get("input_tokens", self.state.input_tokens)
            self.state.output_tokens = p.get("output_tokens", self.state.output_tokens)
            self.state.worker_calls = p.get("worker_calls", self.state.worker_calls)
            self._log(
                "supervisor",
                f"run_done in {p.get('wall_clock', 0)}s, {p.get('worker_calls', 0)} calls",
            )

    # ----------------------------------------------------------------- render

    def _worker_panel(self, ws: WorkerState) -> Panel:
        indicator: Any
        if ws.status == "done":
            indicator = Text("✓", style=_OK)
            title_style = _OK
        elif ws.status in ("thinking", "tool", "wrap-up"):
            indicator = self._spinner.render(time.time())
            title_style = _ACCENT
        else:
            indicator = Text("·", style=_DIM)
            title_style = _DIM

        body_lines: list[Text] = []
        role = ws.role[:80] + ("…" if len(ws.role) > 80 else "")
        body_lines.append(Text(f"role  {role}", style=_DIM))
        if ws.current_tool:
            body_lines.append(Text(f"tool  {ws.current_tool}", style=_WARN))
        body_lines.append(
            Text(
                f"calls {ws.tool_calls}   reads {ws.context_reads}   elapsed {ws.elapsed():4.1f}s",
                style=_DIM,
            )
        )
        if ws.last_finding:
            preview = ws.last_finding.replace("\n", " ")[:160]
            body_lines.append(Text(""))
            body_lines.append(Text(preview, style="white"))

        title = Text.assemble(indicator, " ", (ws.name, title_style), " · ", (ws.status, _DIM))
        return Panel(Group(*body_lines), title=title, border_style=title_style, padding=(0, 1))

    def _workers_panel(self) -> Panel:
        if not self.state.workers:
            return Panel(Text("waiting for workers...", style=_DIM), title="workers")
        panels: list[Panel] = [self._worker_panel(ws) for ws in self.state.workers.values()]
        return Panel(Group(*panels), title=Text("workers", style=_ACCENT), border_style=_ACCENT)

    def _context_panel(self) -> Panel:
        rows: list[Text] = []
        for ts, actor, msg in self.state.log[-40:]:
            t = time.strftime("%H:%M:%S", time.localtime(ts))
            style = (
                _OK
                if "WRITE" in msg
                else _ACCENT
                if "READ" in msg
                else _WARN
                if "wrap-up" in msg
                else _DIM
            )
            rows.append(
                Text.assemble(
                    (t, _DIM),
                    "  ",
                    (f"[{actor}] ", style),
                    msg,
                )
            )
        body = Group(*rows) if rows else Text("(no events yet)", style=_DIM)
        return Panel(body, title=Text("shared-context flow", style=_ACCENT), border_style=_ACCENT)

    def _metrics_panel(self) -> Panel:
        elapsed = self.state.elapsed()
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        cost = self.state.cost()
        table = Table.grid(expand=True, padding=(0, 2))
        table.add_column(justify="left")
        table.add_column(justify="left")
        table.add_column(justify="left")
        table.add_column(justify="left")
        table.add_column(justify="right")
        table.add_row(
            Text.assemble(("elapsed ", _DIM), (elapsed_str, "white")),
            Text.assemble(("calls ", _DIM), (str(self.state.worker_calls), "white")),
            Text.assemble(("in ", _DIM), (f"{self.state.input_tokens:,}", "white")),
            Text.assemble(("out ", _DIM), (f"{self.state.output_tokens:,}", "white")),
            Text.assemble(("~$", _DIM), (f"{cost:.4f}", _OK if cost < 1 else _WARN)),
        )
        return Panel(table, border_style=_DIM)

    def _header_panel(self) -> Panel:
        title = Text("swarmweave — live", style=f"bold {_ACCENT}")
        if self.state.user_task:
            title.append(f"   {self.state.user_task[:120]}", style=_DIM)
        sub = Text(
            f"mode={self.state.mode}  model={self.state.model}",
            style=_DIM,
        )
        return Panel(Group(title, sub), border_style=_ACCENT)

    def _render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._header_panel(), size=4, name="header"),
            Layout(name="body", ratio=1),
            Layout(self._metrics_panel(), size=3, name="metrics"),
        )
        layout["body"].split_row(
            Layout(self._workers_panel(), name="workers", ratio=1),
            Layout(self._context_panel(), name="context", ratio=1),
        )
        return layout

    def _summary_card(self) -> Panel:
        elapsed = self.state.elapsed()
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        cost = self.state.cost()
        rows = []
        for ws in self.state.workers.values():
            status_color = _OK if ws.status == "done" else _WARN
            rows.append(
                Text.assemble(
                    ("✓ " if ws.status == "done" else "· ", status_color),
                    (f"{ws.name:<14}", "white"),
                    (f"{ws.elapsed():5.1f}s   ", _DIM),
                    (f"{ws.tool_calls} tools   ", _DIM),
                    (f"{ws.context_reads} reads", _DIM),
                )
            )
        body = Group(
            Text(""),
            Text("Workers", style=f"bold {_ACCENT}"),
            *rows,
            Text(""),
            Text.assemble(
                ("wall-clock  ", _DIM),
                (f"{elapsed_str}", "white"),
                ("    tokens  ", _DIM),
                (f"{self.state.input_tokens:,} in / {self.state.output_tokens:,} out", "white"),
                ("    cost  ", _DIM),
                (f"~${cost:.4f}", _OK),
            ),
        )
        return Panel(body, title=Text("run summary", style=f"bold {_OK}"), border_style=_OK)

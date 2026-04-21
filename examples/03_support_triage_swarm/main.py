"""Support triage swarm: sequential classify -> resolve -> verify over shared context."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared import (
    enable_progress_logging,
    print_banner,
    print_section,
    render_trace,
    require_api_key,
)

from swarmweave import SharedContext, Supervisor, Worker
from swarmweave.backends import LocalBackend

HERE = Path(__file__).resolve().parent

TICKET = """\
Subject: refund please?

Hi — I bought your Pro plan back in February and I just realized I haven't
used it once. Can I get my money back? Order number is ACME-9821. My name
is Sam.

Thanks,
Sam
"""

USER_TASK = "Triage and respond to this ticket."


def build_swarm(events=None):  # type: ignore[no-untyped-def]
    """Used by `swarmweave watch`. Returns a fresh Supervisor."""
    policies = (HERE / "policies.txt").read_text(encoding="utf-8")

    def get_policies() -> str:
        """Return the full company support response policy."""
        return policies

    def get_ticket() -> str:
        """Return the original customer ticket text."""
        return TICKET

    workers = [
        Worker(
            name="classifier",
            role="Read the ticket and classify it as billing / technical / abuse / feature.",
            tools=[get_ticket],
        ),
        Worker(
            name="resolver",
            role=(
                "Read the ticket and the classifier's category from shared "
                "context, then draft a customer-facing response."
            ),
            tools=[get_ticket],
        ),
        Worker(
            name="verifier",
            role=(
                "Read the resolver's draft from shared context and verify it "
                "against the company policy. List any policy violations. "
                "If the draft looks fine, say so explicitly."
            ),
            tools=[get_policies],
        ),
    ]
    return Supervisor(
        workers=workers,
        shared_context=SharedContext(backend=LocalBackend(persist=False), events=events),
        model="gpt-4o-mini",
        mode="sequential",
        events=events,
    )


def main() -> None:
    require_api_key()
    enable_progress_logging()
    policies = (HERE / "policies.txt").read_text(encoding="utf-8")

    print_banner("support triage swarm")
    print("ticket:")
    print(TICKET)

    def get_policies() -> str:
        """Return the full company support response policy."""
        return policies

    def get_ticket() -> str:
        """Return the original customer ticket text."""
        return TICKET

    ctx = SharedContext(backend=LocalBackend(persist=False))

    workers = [
        Worker(
            name="classifier",
            role="Read the ticket and classify it as billing / technical / abuse / feature.",
            tools=[get_ticket],
        ),
        Worker(
            name="resolver",
            role=(
                "Read the ticket and the classifier's category from shared "
                "context, then draft a customer-facing response."
            ),
            tools=[get_ticket],
        ),
        Worker(
            name="verifier",
            role=(
                "Read the resolver's draft from shared context and verify it "
                "against the company policy. List any policy violations. "
                "If the draft looks fine, say so explicitly."
            ),
            tools=[get_policies],
        ),
    ]

    # Sequential mode: classifier runs first; resolver reads the
    # classifier's category from shared context and drafts the response;
    # verifier reads the draft from shared context and checks policy.
    # Each step builds on the previous step's write — this is what shared
    # context is *for*.
    supervisor = Supervisor(
        workers=workers,
        shared_context=ctx,
        model="gpt-4o-mini",
        mode="sequential",
    )

    print_section("running swarm")
    result = supervisor.run("Triage and respond to this ticket.")

    print_section("final output")
    print(result.final_output)

    render_trace(result.trace)

    print_section("metrics")
    m = result.metrics
    print(
        f"  workers: {len(result.per_worker_outputs)}  "
        f"calls: {m.worker_calls}  "
        f"tokens in/out: {m.input_tokens}/{m.output_tokens}  "
        f"wall: {m.wall_clock_seconds}s"
    )


if __name__ == "__main__":
    main()

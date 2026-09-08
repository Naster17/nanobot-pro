"""Run-loop routing tests for mid-turn follow-ups.

Regression coverage for the follow-up pile-up: messages that arrive while a
turn is active must be diverted into that turn's injection queue, and a burst
of re-published leftovers must chain into exactly one follow-up turn instead
of spawning a crowd of competing lock-waiting tasks.
"""

import asyncio
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.bus.events import InboundMessage
from nanobot.events import FollowUpEvent
from nanobot.providers.base import LLMResponse


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = SimpleNamespace(
        max_tokens=4096,
        temperature=0.1,
        reasoning_effort=None,
    )
    provider.estimate_prompt_tokens = MagicMock(return_value=(10_000, "test"))
    return provider


def _msg(text: str, message_id: str | None = None) -> InboundMessage:
    metadata = {"message_id": message_id} if message_id is not None else {}
    return InboundMessage(
        channel="telegram",
        sender_id="u1",
        chat_id="chat1",
        content=text,
        metadata=metadata,
    )


async def _wait_until(predicate, timeout: float = 15.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.02)


async def _stop_run(loop, run_task: asyncio.Task) -> None:
    loop.stop()
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_followups_divert_into_running_turn(loop_factory) -> None:
    """Messages sent mid-turn join the active turn instead of spawning tasks."""
    loop = loop_factory(provider=_provider())
    key = "telegram:chat1"
    first_seen = asyncio.Event()
    first_release = asyncio.Event()
    model_messages: list[list[dict]] = []

    async def chat(*args, **kwargs):
        model_messages.append(list(kwargs.get("messages") or []))
        if not first_release.is_set():
            first_seen.set()
            await asyncio.wait_for(first_release.wait(), timeout=10)
        return LLMResponse(content="done", tool_calls=[])

    loop.provider.chat_with_retry = AsyncMock(side_effect=chat)
    tracked: list[str] = []
    original_track = loop._track_active_task

    def track(task_key, task):
        tracked.append(task_key)
        original_track(task_key, task)

    loop._track_active_task = track

    run_task = asyncio.create_task(loop.run())
    try:
        await loop.bus.publish_inbound(_msg("start", message_id="100"))
        await asyncio.wait_for(first_seen.wait(), timeout=10)

        await loop.bus.publish_inbound(_msg("second", message_id="101"))
        await loop.bus.publish_inbound(_msg("third", message_id="102"))
        await _wait_until(lambda: loop._pending_queues[key].qsize() == 2)
        # Both follow-ups were diverted into the running turn: one task only.
        assert tracked == [key]

        first_release.set()
        await _wait_until(lambda: (
            not loop._active_tasks
            and not loop._pending_queues
            and loop.bus.inbound_size == 0
        ))
    finally:
        await _stop_run(loop, run_task)

    assert tracked == [key]
    user_texts = [
        str(message.get("content"))
        for call in model_messages
        for message in call
        if message.get("role") == "user"
    ]
    assert any("second" in text for text in user_texts)
    assert any("third" in text for text in user_texts)

    # Lifecycle events: both diverted messages ran queued -> processing -> done.
    phases: list[tuple[str, str | None]] = []
    while loop.bus.outbound_size > 0:
        outbound = await loop.bus.consume_outbound()
        if isinstance(outbound.event, FollowUpEvent):
            phases.append((outbound.event.phase, outbound.event.message_id))
    assert Counter(phase for phase, _ in phases) == Counter(
        {"queued": 2, "processing": 2, "done": 2}
    )
    assert {mid for phase, mid in phases if phase == "processing"} == {"101", "102"}


@pytest.mark.asyncio
async def test_leftover_burst_chains_into_one_followup_turn(loop_factory) -> None:
    """A burst that exhausts the injection cap chains into one follow-up turn.

    With a cap of 5 injection cycles x 3 messages, 17 follow-ups leave 2
    queued when the turn finalizes. The re-published leftovers must chain
    into exactly one follow-up turn (which absorbs the rest), not one
    competing task per message.
    """
    loop = loop_factory(provider=_provider())
    key = "telegram:chat1"
    first_seen = asyncio.Event()
    first_release = asyncio.Event()

    async def chat(*args, **kwargs):
        if not first_release.is_set():
            first_seen.set()
            await asyncio.wait_for(first_release.wait(), timeout=10)
        return LLMResponse(content="done", tool_calls=[])

    loop.provider.chat_with_retry = AsyncMock(side_effect=chat)
    tracked: list[str] = []
    original_track = loop._track_active_task

    def track(task_key, task):
        tracked.append(task_key)
        original_track(task_key, task)

    loop._track_active_task = track

    run_task = asyncio.create_task(loop.run())
    try:
        await loop.bus.publish_inbound(_msg("start"))
        await asyncio.wait_for(first_seen.wait(), timeout=10)

        for index in range(17):
            await loop.bus.publish_inbound(_msg(f"flood-{index}"))
        await _wait_until(lambda: loop._pending_queues[key].qsize() == 17)

        first_release.set()
        await _wait_until(lambda: (
            not loop._active_tasks
            and not loop._pending_queues
            and loop.bus.inbound_size == 0
        ))
    finally:
        await _stop_run(loop, run_task)

    # Initial turn + exactly one chained follow-up turn: no pile-up.
    assert tracked == [key, key]
    assert not loop._pending_queues

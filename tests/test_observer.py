# Copyright 2026 Tenacity Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the explicit attempt observer (``attempt_observer=``).

All tests use *virtual* sleeps (callables that record but never actually
wait), so the whole suite is deterministic and fast.
"""

import asyncio
import dataclasses
import threading
import typing as t

import pytest

from tenacity import (
    AsyncRetrying,
    AttemptEvent,
    AttemptEventType,
    RetryError,
    Retrying,
    retry,
    stop_after_attempt,
    wait_fixed,
)


def virtual_sleep(seconds: float) -> None:
    """A synchronous sleep over virtual time: never actually waits."""


async def virtual_async_sleep(seconds: float) -> None:
    """An asynchronous sleep over virtual time: never actually waits."""


def collect() -> tuple[list[AttemptEvent], t.Callable[[AttemptEvent], None]]:
    """Return an (events, observer) pair recording every observed moment."""
    events: list[AttemptEvent] = []
    return events, events.append


def types_of(events: list[AttemptEvent]) -> list[AttemptEventType]:
    return [event.type for event in events]


def test_observer_sees_only_type_and_moment() -> None:
    """The observer payload is exactly (call_id, type, timestamp)."""
    events, observer = collect()

    @retry(stop=stop_after_attempt(1), attempt_observer=observer)
    def succeed() -> str:
        return "ok"

    assert succeed() == "ok"
    assert events, "expected at least one event"
    for event in events:
        assert isinstance(event.call_id, int)
        assert isinstance(event.type, AttemptEventType)
        assert isinstance(event.timestamp, float)
        # No retry state, result, or exception ever leaks to the observer.
        assert {f.name for f in dataclasses.fields(event)} == {
            "call_id",
            "type",
            "timestamp",
        }
    # Timestamps are monotonic within a call.
    timestamps = [event.timestamp for event in events]
    assert timestamps == sorted(timestamps)


def test_fail_once_then_succeed_sync() -> None:
    """One failure then a success: two attempts, one wait, closed stream."""
    events, observer = collect()
    attempts = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        sleep=virtual_sleep,
        attempt_observer=observer,
    )
    def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("boom")
        return "ok"

    assert flaky() == "ok"
    assert attempts == 2
    assert types_of(events) == [
        AttemptEventType.ATTEMPT_START,
        AttemptEventType.ATTEMPT_RESULT,
        AttemptEventType.STOP_DECISION,
        AttemptEventType.WAIT_START,
        AttemptEventType.WAIT_END,
        AttemptEventType.ATTEMPT_START,
        AttemptEventType.ATTEMPT_RESULT,
        AttemptEventType.CALL_END,
    ]
    # All events belong to a single call.
    assert len({event.call_id for event in events}) == 1


def test_final_success_has_closing_event() -> None:
    """Even an immediate success must close its event stream."""
    events, observer = collect()

    @retry(stop=stop_after_attempt(3), attempt_observer=observer)
    def succeed() -> int:
        return 42

    assert succeed() == 42
    assert types_of(events) == [
        AttemptEventType.ATTEMPT_START,
        AttemptEventType.ATTEMPT_RESULT,
        AttemptEventType.CALL_END,
    ]


def test_fail_once_then_succeed_async() -> None:
    """The async path reports the same correlated sequence as the sync one."""
    events, observer = collect()
    attempts = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        sleep=virtual_async_sleep,
        attempt_observer=observer,
    )
    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("boom")
        return "ok"

    async def main() -> None:
        assert await flaky() == "ok"

    asyncio.run(main())
    assert attempts == 2
    assert types_of(events) == [
        AttemptEventType.ATTEMPT_START,
        AttemptEventType.ATTEMPT_RESULT,
        AttemptEventType.STOP_DECISION,
        AttemptEventType.WAIT_START,
        AttemptEventType.WAIT_END,
        AttemptEventType.ATTEMPT_START,
        AttemptEventType.ATTEMPT_RESULT,
        AttemptEventType.CALL_END,
    ]
    assert len({event.call_id for event in events}) == 1


def test_concurrent_calls_finishing_in_reverse_order_do_not_cross() -> None:
    """Two calls, the second finishing first, must not share events."""
    events: list[AttemptEvent] = []
    lock = threading.Lock()

    def observer(event: AttemptEvent) -> None:
        with lock:
            events.append(event)

    release_a = threading.Event()
    a_entered = threading.Event()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        sleep=virtual_sleep,
        attempt_observer=observer,
    )
    def call_a() -> str:
        if not a_entered.is_set():
            # First attempt of call A: park until call B has completed.
            a_entered.set()
            assert release_a.wait(timeout=10)
            raise ValueError("a fails once")
        return "A"

    @retry(stop=stop_after_attempt(3), attempt_observer=observer)
    def call_b() -> str:
        return "B"

    result_a: list[str] = []
    thread = threading.Thread(target=lambda: result_a.append(call_a()))
    thread.start()
    assert a_entered.wait(timeout=10)

    # Call B starts after A but completes before it.
    assert call_b() == "B"
    release_a.set()
    thread.join(timeout=10)
    assert result_a == ["A"]

    by_call: dict[int, list[AttemptEventType]] = {}
    for event in events:
        by_call.setdefault(event.call_id, []).append(event.type)

    # Two isolated calls, each with its own well-formed event stream.
    assert len(by_call) == 2
    sequences = sorted(by_call.values(), key=len)
    assert sequences[0] == [
        AttemptEventType.ATTEMPT_START,
        AttemptEventType.ATTEMPT_RESULT,
        AttemptEventType.CALL_END,
    ]
    assert sequences[1] == [
        AttemptEventType.ATTEMPT_START,
        AttemptEventType.ATTEMPT_RESULT,
        AttemptEventType.STOP_DECISION,
        AttemptEventType.WAIT_START,
        AttemptEventType.WAIT_END,
        AttemptEventType.ATTEMPT_START,
        AttemptEventType.ATTEMPT_RESULT,
        AttemptEventType.CALL_END,
    ]


def test_concurrent_async_calls_do_not_cross() -> None:
    """Async tasks completing in reverse order keep isolated event streams."""
    events, observer = collect()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        sleep=virtual_async_sleep,
        attempt_observer=observer,
    )
    async def call_a(release: asyncio.Event, entered: asyncio.Event) -> str:
        if not entered.is_set():
            entered.set()
            await release.wait()
            raise ValueError("a fails once")
        return "A"

    @retry(stop=stop_after_attempt(3), attempt_observer=observer)
    async def call_b() -> str:
        return "B"

    async def main() -> tuple[str, str]:
        release = asyncio.Event()
        entered = asyncio.Event()
        task_a = asyncio.create_task(call_a(release, entered))
        await entered.wait()
        result_b = await call_b()
        release.set()
        return await task_a, result_b

    result_a, result_b = asyncio.run(main())
    assert (result_a, result_b) == ("A", "B")

    by_call: dict[int, list[AttemptEventType]] = {}
    for event in events:
        by_call.setdefault(event.call_id, []).append(event.type)
    assert len(by_call) == 2
    for sequence in by_call.values():
        assert sequence[0] is AttemptEventType.ATTEMPT_START
        assert sequence[-1] is AttemptEventType.CALL_END
        # Waits are always opened before they are closed.
        assert sequence.count(AttemptEventType.WAIT_START) == sequence.count(
            AttemptEventType.WAIT_END
        )


def test_observer_failure_does_not_change_business_result() -> None:
    """A crashing observer must not alter the return value of a success."""

    def bad_observer(event: AttemptEvent) -> None:
        raise RuntimeError("observer is broken")

    attempts = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        sleep=virtual_sleep,
        attempt_observer=bad_observer,
    )
    def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("boom")
        return "ok"

    assert flaky() == "ok"
    assert attempts == 2


def test_observer_failure_does_not_change_retry_error() -> None:
    """A crashing observer must not alter the RetryError or its cause."""

    def bad_observer(event: AttemptEvent) -> None:
        raise RuntimeError("observer is broken")

    @retry(
        stop=stop_after_attempt(2),
        sleep=virtual_sleep,
        attempt_observer=bad_observer,
    )
    def always_fails() -> None:
        raise ValueError("boom")

    with pytest.raises(RetryError) as excinfo:
        always_fails()
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert isinstance(excinfo.value.last_attempt.exception(), ValueError)


def test_observer_failure_does_not_change_reraise() -> None:
    """A crashing observer must not alter reraise=True behaviour."""

    def bad_observer(event: AttemptEvent) -> None:
        raise RuntimeError("observer is broken")

    @retry(
        stop=stop_after_attempt(2),
        sleep=virtual_sleep,
        reraise=True,
        attempt_observer=bad_observer,
    )
    def always_fails() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        always_fails()


def test_final_exception_cause_is_preserved() -> None:
    """The RetryError keeps the last exception as its __cause__."""
    events, observer = collect()

    @retry(stop=stop_after_attempt(2), sleep=virtual_sleep, attempt_observer=observer)
    def always_fails() -> None:
        raise ValueError("boom")

    with pytest.raises(RetryError) as excinfo:
        always_fails()
    cause = excinfo.value.__cause__
    assert isinstance(cause, ValueError)
    assert str(cause) == "boom"
    # The failed call still closed its event stream.
    assert events[-1].type is AttemptEventType.CALL_END
    assert types_of(events) == [
        AttemptEventType.ATTEMPT_START,
        AttemptEventType.ATTEMPT_RESULT,
        AttemptEventType.STOP_DECISION,
        AttemptEventType.WAIT_START,
        AttemptEventType.WAIT_END,
        AttemptEventType.ATTEMPT_START,
        AttemptEventType.ATTEMPT_RESULT,
        AttemptEventType.STOP_DECISION,
        AttemptEventType.CALL_END,
    ]


def test_reraise_keeps_original_exception_and_closes_stream() -> None:
    events, observer = collect()

    @retry(
        stop=stop_after_attempt(2),
        sleep=virtual_sleep,
        reraise=True,
        attempt_observer=observer,
    )
    def always_fails() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        always_fails()
    assert events[-1].type is AttemptEventType.CALL_END


def test_cancelled_during_attempt_async() -> None:
    """Cancelling the task mid-attempt marks the attempt phase."""
    events, observer = collect()
    entered = asyncio.Event()

    @retry(stop=stop_after_attempt(3), attempt_observer=observer)
    async def hangs() -> None:
        entered.set()
        await asyncio.Event().wait()

    async def main() -> None:
        task = asyncio.create_task(hangs())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert types_of(events) == [
        AttemptEventType.ATTEMPT_START,
        AttemptEventType.CANCELLED_DURING_ATTEMPT,
        AttemptEventType.ATTEMPT_RESULT,
        AttemptEventType.CALL_END,
    ]


def test_cancelled_during_wait_async() -> None:
    """Cancelling the task mid-wait marks the wait phase."""
    events, observer = collect()
    attempted = asyncio.Event()

    async def blocking_sleep(seconds: float) -> None:
        await asyncio.Event().wait()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        sleep=blocking_sleep,
        attempt_observer=observer,
    )
    async def fails_once_then_hangs() -> None:
        attempted.set()
        raise ValueError("boom")

    async def main() -> None:
        task = asyncio.create_task(fails_once_then_hangs())
        await attempted.wait()
        # Let the retry loop reach the blocking sleep.
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert types_of(events) == [
        AttemptEventType.ATTEMPT_START,
        AttemptEventType.ATTEMPT_RESULT,
        AttemptEventType.STOP_DECISION,
        AttemptEventType.WAIT_START,
        AttemptEventType.CANCELLED_DURING_WAIT,
        AttemptEventType.CALL_END,
    ]


def test_keyboard_interrupt_marks_attempt_phase() -> None:
    """A KeyboardInterrupt out of the wrapped function is a cancellation."""
    events, observer = collect()

    @retry(stop=stop_after_attempt(3), attempt_observer=observer)
    def interrupted() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupted()
    assert types_of(events) == [
        AttemptEventType.ATTEMPT_START,
        AttemptEventType.CANCELLED_DURING_ATTEMPT,
        AttemptEventType.ATTEMPT_RESULT,
        AttemptEventType.CALL_END,
    ]


def test_sync_iterator_path_reports_events() -> None:
    """The context-manager (Retrying) path emits the same event kinds."""
    events, observer = collect()
    retrying = Retrying(
        stop=stop_after_attempt(2),
        sleep=virtual_sleep,
        attempt_observer=observer,
    )
    with pytest.raises(RetryError):
        for attempt in retrying:
            with attempt:
                raise ValueError("boom")
    assert types_of(events) == [
        AttemptEventType.ATTEMPT_START,
        AttemptEventType.ATTEMPT_RESULT,
        AttemptEventType.STOP_DECISION,
        AttemptEventType.WAIT_START,
        AttemptEventType.WAIT_END,
        AttemptEventType.ATTEMPT_START,
        AttemptEventType.ATTEMPT_RESULT,
        AttemptEventType.STOP_DECISION,
        AttemptEventType.CALL_END,
    ]
    assert len({event.call_id for event in events}) == 1


def test_async_iterator_path_reports_events() -> None:
    """The async context-manager (AsyncRetrying) path emits events too."""
    events, observer = collect()

    async def main() -> None:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(2),
            sleep=virtual_async_sleep,
            attempt_observer=observer,
        )
        with pytest.raises(RetryError):
            async for attempt in retrying:
                with attempt:
                    raise ValueError("boom")

    asyncio.run(main())
    assert types_of(events) == [
        AttemptEventType.ATTEMPT_START,
        AttemptEventType.ATTEMPT_RESULT,
        AttemptEventType.STOP_DECISION,
        AttemptEventType.WAIT_START,
        AttemptEventType.WAIT_END,
        AttemptEventType.ATTEMPT_START,
        AttemptEventType.ATTEMPT_RESULT,
        AttemptEventType.STOP_DECISION,
        AttemptEventType.CALL_END,
    ]
    assert len({event.call_id for event in events}) == 1


def test_observer_survives_copy_via_retry_with() -> None:
    """retry_with copies keep reporting to the configured observer."""
    events, observer = collect()

    @retry(stop=stop_after_attempt(1), attempt_observer=observer)
    def always_fails() -> None:
        raise ValueError("boom")

    with pytest.raises(RetryError):
        always_fails.retry_with(stop=stop_after_attempt(2))()
    assert events
    assert events[-1].type is AttemptEventType.CALL_END


def test_no_observer_means_no_events_and_classic_behaviour() -> None:
    """Without an observer the behaviour is exactly the classic one."""

    @retry(stop=stop_after_attempt(2), sleep=virtual_sleep)
    def always_fails() -> None:
        raise ValueError("boom")

    with pytest.raises(RetryError):
        always_fails()

# Copyright 2026 Julien Danjou
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
"""Tests for the explicit attempt observer.

These tests use a *virtual sleep* (``sleep=lambda seconds: None``) so that no
real time is spent waiting between attempts.
"""

from __future__ import annotations

import asyncio
import threading
import typing as t
from functools import wraps

import pytest

import tenacity
from tenacity import (
    AsyncRetrying,
    AttemptObserver,
    Event,
    EventType,
    RetryError,
    Retrying,
    retry,
    stop_after_attempt,
    wait_fixed,
)


def asynctest(callable_: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
    @wraps(callable_)
    def wrapper(*a: t.Any, **kw: t.Any) -> t.Any:
        return asyncio.run(callable_(*a, **kw))

    return wrapper


def virtual_sleep(seconds: float) -> None:
    """A sleep strategy that does not actually sleep (virtual sleep)."""
    return


def async_virtual_sleep(seconds: float) -> t.Awaitable[None]:
    async def _sleep() -> None:
        return

    return _sleep()


# ---------------------------------------------------------------------------
# Event specification
# ---------------------------------------------------------------------------


def test_event_only_carries_type_and_timestamp() -> None:
    obs = AttemptObserver()
    call_id = obs.begin_call()
    obs.event(EventType.ATTEMPT_START)
    obs.end_call(call_id)
    events = obs.events
    assert len(events) == 3
    for event in events:
        assert isinstance(event, Event)
        assert isinstance(event.type, EventType)
        assert isinstance(event.timestamp, float)
        assert isinstance(event.call_id, int)
    # The observer never receives the retry state: the dataclass only has
    # these three fields.
    assert set(events[0].__dataclass_fields__) == {
        "type",
        "timestamp",
        "call_id",
    }


def test_event_types_are_explicit() -> None:
    # The event spec: every phase has its own explicit type.
    assert EventType.CALL_START.value == "call_start"
    assert EventType.CALL_END.value == "call_end"
    assert EventType.ATTEMPT_START.value == "attempt_start"
    assert EventType.ATTEMPT_RESULT.value == "attempt_result"
    assert EventType.STOP_DECISION.value == "stop_decision"
    assert EventType.WAIT_START.value == "wait_start"
    assert EventType.WAIT_END.value == "wait_end"
    assert EventType.CANCELLED_DURING_ATTEMPT.value == "cancelled_during_attempt"
    assert EventType.CANCELLED_DURING_WAIT.value == "cancelled_during_wait"


# ---------------------------------------------------------------------------
# Fail once then succeed: two attempts and one wait
# ---------------------------------------------------------------------------


def test_fail_once_then_succeed_has_two_attempts_and_one_wait() -> None:
    obs = AttemptObserver()
    calls = [0]

    def fn() -> str:
        calls[0] += 1
        if calls[0] == 1:
            raise ValueError("boom")
        return "ok"

    r = Retrying(
        observer=obs,
        stop=stop_after_attempt(3),
        wait=wait_fixed(0),
        sleep=virtual_sleep,
    )
    assert r(fn) == "ok"

    types = [e.type for e in obs.events]
    assert types == [
        EventType.CALL_START,
        EventType.ATTEMPT_START,
        EventType.ATTEMPT_RESULT,
        EventType.STOP_DECISION,
        EventType.WAIT_START,
        EventType.WAIT_END,
        EventType.ATTEMPT_START,
        EventType.ATTEMPT_RESULT,
        EventType.CALL_END,
    ]
    # Two attempts ...
    assert types.count(EventType.ATTEMPT_START) == 2
    assert types.count(EventType.ATTEMPT_RESULT) == 2
    # ... and one wait.
    assert types.count(EventType.WAIT_START) == 1
    assert types.count(EventType.WAIT_END) == 1
    # The final success has a closing event.
    assert types[0] is EventType.CALL_START
    assert types[-1] is EventType.CALL_END


def test_final_success_has_closing_event() -> None:
    obs = AttemptObserver()
    r = Retrying(
        observer=obs,
        stop=stop_after_attempt(1),
        wait=wait_fixed(0),
        sleep=virtual_sleep,
    )
    r(lambda: "done")
    types = [e.type for e in obs.events]
    assert types[0] is EventType.CALL_START
    assert types[-1] is EventType.CALL_END


def test_final_failure_has_closing_event() -> None:
    obs = AttemptObserver()
    r = Retrying(
        observer=obs,
        stop=stop_after_attempt(1),
        wait=wait_fixed(0),
        sleep=virtual_sleep,
    )
    with pytest.raises(RetryError):
        r(lambda: (_ for _ in ()).throw(ValueError("nope")))
    types = [e.type for e in obs.events]
    assert types[0] is EventType.CALL_START
    assert types[-1] is EventType.CALL_END


# ---------------------------------------------------------------------------
# Stop decision is recorded
# ---------------------------------------------------------------------------


def test_stop_decision_recorded_after_every_failed_attempt() -> None:
    obs = AttemptObserver()
    calls = [0]

    def fn() -> str:
        calls[0] += 1
        if calls[0] <= 2:
            raise ValueError("boom")
        return "ok"

    r = Retrying(
        observer=obs,
        stop=stop_after_attempt(3),
        wait=wait_fixed(0),
        sleep=virtual_sleep,
    )
    r(fn)
    types = [e.type for e in obs.events]
    # The stop strategy runs after every failed attempt (a retry is being
    # considered); on the final successful attempt no stop decision is needed.
    assert types.count(EventType.STOP_DECISION) == 2


# ---------------------------------------------------------------------------
# Observer failures do not change the business result
# ---------------------------------------------------------------------------


class _RaisingObserver(AttemptObserver):
    def event(self, event_type: EventType, timestamp: float | None = None) -> None:
        raise RuntimeError("observer failure")


def test_observer_failure_does_not_change_return_value() -> None:
    obs = _RaisingObserver()
    r = Retrying(
        observer=obs,
        stop=stop_after_attempt(1),
        wait=wait_fixed(0),
        sleep=virtual_sleep,
    )
    assert r(lambda: 42) == 42


def test_observer_failure_does_not_change_retry_error() -> None:
    obs = _RaisingObserver()
    r = Retrying(
        observer=obs,
        stop=stop_after_attempt(1),
        wait=wait_fixed(0),
        sleep=virtual_sleep,
    )
    with pytest.raises(RetryError) as exc_info:
        r(lambda: (_ for _ in ()).throw(ValueError("business")))
    # The RetryError is unchanged and its cause is preserved.
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == "business"


def test_observer_failure_does_not_change_reraise() -> None:
    obs = _RaisingObserver()
    r = Retrying(
        observer=obs,
        stop=stop_after_attempt(1),
        wait=wait_fixed(0),
        sleep=virtual_sleep,
        reraise=True,
    )
    with pytest.raises(KeyError):
        r(lambda: (_ for _ in ()).throw(KeyError("business")))


def test_observer_failure_does_not_change_retry_error_callback() -> None:
    obs = _RaisingObserver()

    def callback(retry_state: tenacity.RetryCallState) -> str:
        return "callback-result"

    r = Retrying(
        observer=obs,
        stop=stop_after_attempt(1),
        wait=wait_fixed(0),
        sleep=virtual_sleep,
        retry_error_callback=callback,
    )
    assert r(lambda: (_ for _ in ()).throw(ValueError("x"))) == "callback-result"


# ---------------------------------------------------------------------------
# Cause preservation
# ---------------------------------------------------------------------------


def test_final_exception_cause_preserved() -> None:
    obs = AttemptObserver()
    r = Retrying(
        observer=obs,
        stop=stop_after_attempt(1),
        wait=wait_fixed(0),
        sleep=virtual_sleep,
    )
    with pytest.raises(RetryError) as exc_info:
        r(lambda: (_ for _ in ()).throw(ValueError("the cause")))
    cause = exc_info.value.__cause__
    assert isinstance(cause, ValueError)
    assert str(cause) == "the cause"


# ---------------------------------------------------------------------------
# Concurrent call state isolation
# ---------------------------------------------------------------------------


def test_concurrent_sync_calls_do_not_interleave_events() -> None:
    obs = AttemptObserver()
    results: dict[str, str] = {}
    barrier = threading.Barrier(2)

    def make_fn(name: str) -> t.Callable[[], str]:
        calls = [0]

        def fn() -> str:
            calls[0] += 1
            if calls[0] == 1:
                raise ValueError(name)
            barrier.wait()
            return name

        return fn

    def run(name: str) -> None:
        r = Retrying(
            observer=obs,
            stop=stop_after_attempt(3),
            wait=wait_fixed(0),
            sleep=virtual_sleep,
        )
        results[name] = r(make_fn(name))

    ta = threading.Thread(target=run, args=("A",))
    tb = threading.Thread(target=run, args=("B",))
    ta.start()
    tb.start()
    ta.join()
    tb.join()

    assert results == {"A": "A", "B": "B"}
    grouped = obs.calls()
    assert len(grouped) == 2
    for call_id, events in grouped.items():
        # Every event in a call must carry that call's id.
        assert {e.call_id for e in events} == {call_id}
        types = [e.type for e in events]
        assert types[0] is EventType.CALL_START
        assert types[-1] is EventType.CALL_END


@asynctest
async def test_concurrent_async_calls_do_not_interleave_events() -> None:
    obs = AttemptObserver()

    async def make_fn(
        name: str, barrier: asyncio.Event
    ) -> t.Callable[[], t.Awaitable[str]]:
        calls = [0]

        async def fn() -> str:
            calls[0] += 1
            if calls[0] == 1:
                raise ValueError(name)
            await barrier.wait()
            return name

        return fn

    async def run(name: str, barrier: asyncio.Event) -> str:
        r = AsyncRetrying(
            observer=obs,
            stop=stop_after_attempt(3),
            wait=wait_fixed(0),
            sleep=async_virtual_sleep,
        )
        fn = await make_fn(name, barrier)
        return await r(fn)

    barrier = asyncio.Event()
    ta = asyncio.create_task(run("A", barrier))
    tb = asyncio.create_task(run("B", barrier))
    await asyncio.sleep(0.01)
    barrier.set()
    assert await ta == "A"
    assert await tb == "B"

    grouped = obs.calls()
    assert len(grouped) == 2
    for call_id, events in grouped.items():
        assert {e.call_id for e in events} == {call_id}
        types = [e.type for e in events]
        assert types[0] is EventType.CALL_START
        assert types[-1] is EventType.CALL_END


# ---------------------------------------------------------------------------
# Cancellation marks the specific phase
# ---------------------------------------------------------------------------


@asynctest
async def test_cancellation_during_attempt_marks_phase() -> None:
    obs = AttemptObserver()

    async def fn() -> str:
        raise asyncio.CancelledError

    r = AsyncRetrying(
        observer=obs,
        stop=stop_after_attempt(3),
        wait=wait_fixed(0),
        sleep=async_virtual_sleep,
    )
    with pytest.raises(asyncio.CancelledError):
        await r(fn)

    types = [e.type for e in obs.events]
    assert EventType.CANCELLED_DURING_ATTEMPT in types
    assert EventType.CANCELLED_DURING_WAIT not in types
    assert types[-1] is EventType.CANCELLED_DURING_ATTEMPT


@asynctest
async def test_cancellation_during_wait_marks_phase() -> None:
    obs = AttemptObserver()
    calls = [0]

    async def fn() -> str:
        calls[0] += 1
        if calls[0] == 1:
            raise ValueError("boom")
        return "ok"

    async def cancelling_sleep(seconds: float) -> None:
        raise asyncio.CancelledError

    r = AsyncRetrying(
        observer=obs,
        stop=stop_after_attempt(3),
        wait=wait_fixed(100),
        sleep=cancelling_sleep,
    )
    with pytest.raises(asyncio.CancelledError):
        await r(fn)

    types = [e.type for e in obs.events]
    assert EventType.CANCELLED_DURING_WAIT in types
    assert EventType.CANCELLED_DURING_ATTEMPT not in types
    assert types[-1] is EventType.CANCELLED_DURING_WAIT


# ---------------------------------------------------------------------------
# Sync and async paths share call ids
# ---------------------------------------------------------------------------


@asynctest
async def test_sync_and_async_paths_share_call_ids() -> None:
    obs = AttemptObserver()

    sync_r = Retrying(
        observer=obs,
        stop=stop_after_attempt(1),
        wait=wait_fixed(0),
        sleep=virtual_sleep,
    )
    sync_r(lambda: "sync")

    async_r = AsyncRetrying(
        observer=obs,
        stop=stop_after_attempt(1),
        wait=wait_fixed(0),
        sleep=async_virtual_sleep,
    )

    async def async_fn() -> str:
        return "async"

    await async_r(async_fn)

    grouped = obs.calls()
    assert len(grouped) == 2
    call_ids = list(grouped.keys())
    assert call_ids[0] != call_ids[1]
    for events in grouped.values():
        types = [e.type for e in events]
        assert types[0] is EventType.CALL_START
        assert types[-1] is EventType.CALL_END


# ---------------------------------------------------------------------------
# Decorator integration
# ---------------------------------------------------------------------------


def test_retry_decorator_accepts_observer() -> None:
    obs = AttemptObserver()
    calls = [0]

    @retry(
        observer=obs,
        stop=stop_after_attempt(3),
        wait=wait_fixed(0),
        sleep=virtual_sleep,
    )
    def fn() -> str:
        calls[0] += 1
        if calls[0] == 1:
            raise ValueError("boom")
        return "ok"

    assert fn() == "ok"
    types = [e.type for e in obs.events]
    assert types.count(EventType.ATTEMPT_START) == 2
    assert types.count(EventType.WAIT_START) == 1


# ---------------------------------------------------------------------------
# Context manager integration
# ---------------------------------------------------------------------------


def test_context_manager_records_events() -> None:
    obs = AttemptObserver()
    r = Retrying(
        observer=obs,
        stop=stop_after_attempt(3),
        wait=wait_fixed(0),
        sleep=virtual_sleep,
    )
    calls = [0]
    for attempt in r:
        with attempt:
            calls[0] += 1
            if calls[0] == 1:
                raise ValueError("boom")
    assert calls[0] == 2
    types = [e.type for e in obs.events]
    assert types.count(EventType.ATTEMPT_START) == 2
    assert types.count(EventType.WAIT_START) == 1
    assert types[0] is EventType.CALL_START
    assert types[-1] is EventType.CALL_END


# ---------------------------------------------------------------------------
# Observer clock and reset
# ---------------------------------------------------------------------------


def test_observer_uses_provided_clock() -> None:
    times = iter([10.0, 20.0, 30.0])
    obs = AttemptObserver(clock=lambda: next(times))
    r = Retrying(
        observer=obs,
        stop=stop_after_attempt(1),
        wait=wait_fixed(0),
        sleep=virtual_sleep,
    )
    r(lambda: "ok")
    timestamps = [e.timestamp for e in obs.events]
    assert timestamps == [10.0, 20.0, 30.0]


def test_observer_reset() -> None:
    obs = AttemptObserver()
    r = Retrying(
        observer=obs,
        stop=stop_after_attempt(1),
        wait=wait_fixed(0),
        sleep=virtual_sleep,
    )
    r(lambda: "ok")
    assert len(obs.events) > 0
    obs.reset()
    assert obs.events == []
    assert obs.calls() == {}

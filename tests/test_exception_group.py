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

"""Tests for the explicit exception group retry predicate."""

import asyncio
import builtins
import typing
import unittest
from functools import wraps

import pytest

import tenacity
from tenacity import (
    AsyncRetrying,
    NotAnExceptionGroup,
    RetryCallState,
    Retrying,
    retry_if_exception,
    retry_if_exception_group,
    stop_after_attempt,
)

# Exception groups are a Python 3.11 feature; on older interpreters use the
# ``exceptiongroup`` backport when it is available, otherwise skip the module.
try:
    from exceptiongroup import BaseExceptionGroup, ExceptionGroup
except ImportError:  # pragma: no cover
    BaseExceptionGroup = getattr(builtins, "BaseExceptionGroup", None)
    ExceptionGroup = getattr(builtins, "ExceptionGroup", None)
    if ExceptionGroup is None:
        pytest.skip(
            "ExceptionGroup requires Python 3.11+ or the 'exceptiongroup' backport",
            allow_module_level=True,
        )


class Transient(Exception):
    """A retryable leaf."""


class Fatal(Exception):
    """A non-retryable leaf."""


def make_retry_state(exception: BaseException | None = None) -> RetryCallState:
    retry_state = RetryCallState(Retrying(), fn=None, args=(), kwargs={})
    if exception is not None:
        retry_state.outcome = tenacity.Future.construct(1, exception, True)
    return retry_state


def asynctest(
    callable_: typing.Callable[..., typing.Any],
) -> typing.Callable[..., typing.Any]:
    @wraps(callable_)
    def wrapper(*a: typing.Any, **kw: typing.Any) -> typing.Any:
        return asyncio.run(callable_(*a, **kw))

    return wrapper


def is_transient(exception: BaseException) -> bool:
    return isinstance(exception, Transient)


# ---------------------------------------------------------------------------
# any / all evaluation of real leaves
# ---------------------------------------------------------------------------


def test_any_mode_retries_when_one_leaf_matches() -> None:
    predicate = retry_if_exception_group(is_transient, mode="any")
    state = make_retry_state(ExceptionGroup("g", [Transient(), Fatal()]))
    assert predicate(state) is True


def test_all_mode_does_not_retry_when_one_leaf_rejects() -> None:
    predicate = retry_if_exception_group(is_transient, mode="all")
    state = make_retry_state(ExceptionGroup("g", [Transient(), Fatal()]))
    assert predicate(state) is False


def test_all_mode_retries_when_every_leaf_matches() -> None:
    predicate = retry_if_exception_group(is_transient, mode="all")
    state = make_retry_state(ExceptionGroup("g", [Transient(), Transient()]))
    assert predicate(state) is True


def test_any_mode_does_not_retry_when_no_leaf_matches() -> None:
    predicate = retry_if_exception_group(is_transient, mode="any")
    state = make_retry_state(ExceptionGroup("g", [Fatal(), Fatal()]))
    assert predicate(state) is False


def test_default_mode_is_any() -> None:
    predicate = retry_if_exception_group(is_transient)
    state = make_retry_state(ExceptionGroup("g", [Transient(), Fatal()]))
    assert predicate(state) is True
    assert predicate.mode == "any"


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        retry_if_exception_group(is_transient, mode="one")


# ---------------------------------------------------------------------------
# nested groups keep their paths
# ---------------------------------------------------------------------------


def test_nested_group_paths_are_preserved() -> None:
    predicate = retry_if_exception_group(is_transient, mode="all")
    group = ExceptionGroup(
        "outer",
        [
            ExceptionGroup("inner-a", [Transient(), Fatal()]),
            ExceptionGroup("inner-b", [Transient()]),
        ],
    )
    state = make_retry_state(group)
    assert predicate(state) is False
    decision = predicate.last_decision
    assert decision is not None
    assert [leaf.path for leaf in decision.leaves] == [(0, 0), (0, 1), (1, 0)]


def test_nested_all_mode_names_the_deciding_leaf() -> None:
    predicate = retry_if_exception_group(is_transient, mode="all")
    group = ExceptionGroup(
        "outer",
        [
            ExceptionGroup("inner", [Transient(), Fatal()]),
        ],
    )
    state = make_retry_state(group)
    assert predicate(state) is False
    decision = predicate.last_decision
    assert decision is not None
    assert decision.retry is False
    # The Fatal leaf at path (0, 1) is the one that decides against retrying.
    assert [leaf.path for leaf in decision.rejected] == [(0, 1)]
    assert [leaf.path for leaf in decision.matched] == [(0, 0)]


def test_nested_any_mode_names_the_deciding_leaf() -> None:
    predicate = retry_if_exception_group(is_transient, mode="any")
    group = ExceptionGroup(
        "outer",
        [
            ExceptionGroup("inner", [Fatal(), Transient()]),
        ],
    )
    state = make_retry_state(group)
    assert predicate(state) is True
    decision = predicate.last_decision
    assert decision is not None
    assert [leaf.path for leaf in decision.matched] == [(0, 1)]
    # The Fatal leaf at path (0, 0) rejected the predicate.
    assert [leaf.path for leaf in decision.rejected] == [(0, 0)]


# ---------------------------------------------------------------------------
# each leaf predicate runs exactly once
# ---------------------------------------------------------------------------


def test_each_leaf_predicate_runs_once() -> None:
    calls: list[BaseException] = []

    def counting_predicate(exception: BaseException) -> bool:
        calls.append(exception)
        return isinstance(exception, Transient)

    predicate = retry_if_exception_group(counting_predicate, mode="all")
    group = ExceptionGroup("g", [Transient(), Fatal(), Transient()])
    state = make_retry_state(group)
    assert predicate(state) is False
    assert len(calls) == 3


def test_duplicate_leaf_object_is_evaluated_once() -> None:
    calls: list[BaseException] = []

    def counting_predicate(exception: BaseException) -> bool:
        calls.append(exception)
        return isinstance(exception, Transient)

    predicate = retry_if_exception_group(counting_predicate, mode="all")
    transient = Transient()
    group = ExceptionGroup("g", [transient, Fatal(), transient])
    state = make_retry_state(group)
    assert predicate(state) is False
    assert len(calls) == 2
    assert calls == [transient, group.exceptions[1]]


# ---------------------------------------------------------------------------
# cancellation and system termination leaves are never retried
# ---------------------------------------------------------------------------


def test_cancellation_leaf_vetoes_retry() -> None:
    calls: list[BaseException] = []

    def counting_predicate(exception: BaseException) -> bool:
        calls.append(exception)
        return True

    predicate = retry_if_exception_group(counting_predicate, mode="any")
    group = BaseExceptionGroup("g", [asyncio.CancelledError()])
    state = make_retry_state(group)
    assert predicate(state) is False
    # The business predicate is never consulted for a cancellation leaf.
    assert calls == []
    decision = predicate.last_decision
    assert decision is not None
    assert decision.fatal is True
    assert decision.retry is False
    assert len(decision.fatal_leaves) == 1


def test_cancellation_among_leaves_vetoes_retry() -> None:
    predicate = retry_if_exception_group(lambda e: True, mode="any")
    group = BaseExceptionGroup("g", [Transient(), asyncio.CancelledError(), Fatal()])
    state = make_retry_state(group)
    assert predicate(state) is False
    decision = predicate.last_decision
    assert decision is not None
    assert decision.fatal is True
    assert [leaf.path for leaf in decision.fatal_leaves] == [(1,)]


def test_keyboard_interrupt_vetoes_retry() -> None:
    predicate = retry_if_exception_group(lambda e: True, mode="any")
    group = BaseExceptionGroup("g", [KeyboardInterrupt()])
    state = make_retry_state(group)
    assert predicate(state) is False
    assert predicate.last_decision is not None
    assert predicate.last_decision.fatal is True


def test_system_exit_vetoes_retry() -> None:
    predicate = retry_if_exception_group(lambda e: True, mode="any")
    group = BaseExceptionGroup("g", [SystemExit(1)])
    state = make_retry_state(group)
    assert predicate(state) is False
    assert predicate.last_decision is not None
    assert predicate.last_decision.fatal is True


def test_nested_fatal_leaf_vetoes_retry() -> None:
    predicate = retry_if_exception_group(lambda e: True, mode="any")
    group = BaseExceptionGroup(
        "g",
        [
            ExceptionGroup("inner", [Transient()]),
            BaseExceptionGroup("deep", [GeneratorExit()]),
        ],
    )
    state = make_retry_state(group)
    assert predicate(state) is False
    decision = predicate.last_decision
    assert decision is not None
    assert decision.fatal is True
    assert [leaf.path for leaf in decision.fatal_leaves] == [(1, 0)]


def test_fatal_exceptions_are_configurable() -> None:
    predicate = retry_if_exception_group(
        lambda e: True,
        mode="any",
        fatal_exceptions=(Fatal,),
    )
    state = make_retry_state(ExceptionGroup("g", [Fatal()]))
    assert predicate(state) is False
    assert predicate.last_decision is not None
    assert predicate.last_decision.fatal is True


# ---------------------------------------------------------------------------
# unknown chains refuse to judge; predicate failures are not faked as False
# ---------------------------------------------------------------------------


def test_non_group_exception_refuses_to_judge() -> None:
    predicate = retry_if_exception_group(is_transient, mode="any")
    state = make_retry_state(ValueError("not a group"))
    with pytest.raises(NotAnExceptionGroup):
        predicate(state)
    # No decision is recorded for a chain that cannot be judged.
    assert predicate.last_decision is None


def test_predicate_failure_propagates() -> None:
    def failing_predicate(exception: BaseException) -> bool:
        raise RuntimeError("predicate boom")

    predicate = retry_if_exception_group(failing_predicate, mode="any")
    state = make_retry_state(ExceptionGroup("g", [Transient()]))
    with pytest.raises(RuntimeError, match="predicate boom"):
        predicate(state)


def test_successful_outcome_is_not_retried() -> None:
    predicate = retry_if_exception_group(is_transient, mode="any")
    retry_state = RetryCallState(Retrying(), fn=None, args=(), kwargs={})
    retry_state.outcome = tenacity.Future.construct(1, "ok", False)
    assert predicate(retry_state) is False


def test_missing_outcome_raises() -> None:
    predicate = retry_if_exception_group(is_transient, mode="any")
    with pytest.raises(RuntimeError):
        predicate(RetryCallState(Retrying(), fn=None, args=(), kwargs={}))


# ---------------------------------------------------------------------------
# integration with the retrying loops
# ---------------------------------------------------------------------------


def test_retrying_retries_any_group() -> None:
    attempts: list[int] = []

    def flaky() -> None:
        attempts.append(1)
        raise ExceptionGroup("g", [Transient(), Fatal()])

    retrying = Retrying(
        retry=retry_if_exception_group(is_transient, mode="any"),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    with pytest.raises(ExceptionGroup):
        retrying(flaky)
    assert len(attempts) == 3


def test_retrying_all_group_stops_after_success() -> None:
    attempts: list[int] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 2:
            raise ExceptionGroup("g", [Transient(), Transient()])
        return "ok"

    retrying = Retrying(
        retry=retry_if_exception_group(is_transient, mode="all"),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    assert retrying(flaky) == "ok"
    assert len(attempts) == 2


def test_retrying_fatal_group_propagates_without_retry() -> None:
    attempts: list[int] = []

    def cancelled() -> None:
        attempts.append(1)
        raise BaseExceptionGroup("g", [asyncio.CancelledError()])

    retrying = Retrying(
        retry=retry_if_exception_group(lambda e: True, mode="any"),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    with pytest.raises(BaseExceptionGroup):
        retrying(cancelled)
    assert len(attempts) == 1


def test_retrying_predicate_failure_propagates() -> None:
    attempts: list[int] = []

    def flaky() -> None:
        attempts.append(1)
        raise ExceptionGroup("g", [Transient()])

    def failing_predicate(exception: BaseException) -> bool:
        raise RuntimeError("predicate boom")

    retrying = Retrying(
        retry=retry_if_exception_group(failing_predicate, mode="any"),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    with pytest.raises(RuntimeError, match="predicate boom"):
        retrying(flaky)
    assert len(attempts) == 1


def test_default_retry_if_exception_behavior_is_unchanged() -> None:
    # The default predicate still judges plain exceptions directly.
    predicate = retry_if_exception(lambda e: isinstance(e, Transient))
    state = make_retry_state(Transient())
    assert predicate(state) is True
    state = make_retry_state(Fatal())
    assert predicate(state) is False


# ---------------------------------------------------------------------------
# asyncio integration
# ---------------------------------------------------------------------------


class TestAsyncRetryIfExceptionGroup(unittest.TestCase):
    @asynctest
    async def test_async_retries_any_group(self) -> None:
        attempts: list[int] = []

        async def flaky() -> None:
            attempts.append(1)
            raise ExceptionGroup("g", [Transient(), Fatal()])

        retrying = AsyncRetrying(
            retry=retry_if_exception_group(is_transient, mode="any"),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        with pytest.raises(ExceptionGroup):
            await retrying(flaky)
        assert len(attempts) == 3

    @asynctest
    async def test_async_fatal_group_propagates_without_retry(self) -> None:
        attempts: list[int] = []

        async def cancelled() -> None:
            attempts.append(1)
            raise BaseExceptionGroup("g", [asyncio.CancelledError()])

        retrying = AsyncRetrying(
            retry=retry_if_exception_group(lambda e: True, mode="any"),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        with pytest.raises(BaseExceptionGroup):
            await retrying(cancelled)
        assert len(attempts) == 1

    @asynctest
    async def test_async_predicate_failure_propagates(self) -> None:
        attempts: list[int] = []

        async def flaky() -> None:
            attempts.append(1)
            raise ExceptionGroup("g", [Transient()])

        def failing_predicate(exception: BaseException) -> bool:
            raise RuntimeError("predicate boom")

        retrying = AsyncRetrying(
            retry=retry_if_exception_group(failing_predicate, mode="any"),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        with pytest.raises(RuntimeError, match="predicate boom"):
            await retrying(flaky)
        assert len(attempts) == 1

    @asynctest
    async def test_async_nested_group_paths(self) -> None:
        predicate = retry_if_exception_group(is_transient, mode="all")
        group = ExceptionGroup(
            "outer",
            [ExceptionGroup("inner", [Transient(), Fatal()])],
        )
        state = make_retry_state(group)
        assert predicate(state) is False
        decision = predicate.last_decision
        assert decision is not None
        assert [leaf.path for leaf in decision.leaves] == [(0, 0), (0, 1)]

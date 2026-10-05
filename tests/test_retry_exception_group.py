# Copyright 2026 Tenacity contributors
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

import asyncio
import builtins
import sys
import unittest
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, TypeVar

import pytest

import tenacity
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    UndecidableRetryError,
    retry_if_exception,
    retry_if_exception_group,
    retry_if_exception_type,
    stop_after_attempt,
)

if sys.version_info < (3, 11):
    pytest.skip("ExceptionGroup requires Python 3.11+", allow_module_level=True)

# Aliases for the 3.11+ builtins so ruff/mypy (target py310) accept them;
# the module-level skip above guarantees they exist at runtime.
ExceptionGroup = builtins.ExceptionGroup  # type: ignore[attr-defined]
BaseExceptionGroup = builtins.BaseExceptionGroup  # type: ignore[attr-defined]


class TransientError(Exception):
    """Business error considered worth retrying."""


class FatalError(Exception):
    """Business error considered not worth retrying."""


def is_transient(exc: BaseException) -> bool:
    return isinstance(exc, TransientError)


def make_failed_state(exc: BaseException) -> RetryCallState:
    retry_state = RetryCallState(None, None, (), {})  # type: ignore[arg-type]
    retry_state.set_exception((type(exc), exc, exc.__traceback__))
    return retry_state


def make_success_state() -> RetryCallState:
    retry_state = RetryCallState(None, None, (), {})  # type: ignore[arg-type]
    retry_state.set_result("ok")
    return retry_state


_F = TypeVar("_F", bound=Callable[..., Coroutine[Any, Any, Any]])


def asynctest(callable_: _F) -> Callable[..., Any]:
    @wraps(callable_)
    def wrapper(*a: Any, **kw: Any) -> Any:
        return asyncio.run(callable_(*a, **kw))

    return wrapper


class TestExceptionGroupPredicate:
    def test_any_mode_true_with_transient_and_fatal_leaves(self) -> None:
        strategy = retry_if_exception_group(is_transient, mode="any")
        group = ExceptionGroup("mixed", [TransientError("t"), FatalError("f")])
        assert strategy(make_failed_state(group)) is True
        report = strategy.last_report
        assert report is not None
        assert report.decision is True
        assert report.deciding_path == (0,)
        assert report.blocked_paths == ()
        assert "path (0,)" in report.explanation

    def test_all_mode_false_with_transient_and_fatal_leaves(self) -> None:
        strategy = retry_if_exception_group(is_transient, mode="all")
        group = ExceptionGroup("mixed", [TransientError("t"), FatalError("f")])
        assert strategy(make_failed_state(group)) is False
        report = strategy.last_report
        assert report is not None
        assert report.decision is False
        # The FatalError leaf at index 1 is what vetoed the retry.
        assert report.deciding_path == (1,)
        assert "FatalError" in report.explanation

    def test_all_mode_true_when_every_leaf_matches(self) -> None:
        strategy = retry_if_exception_group(is_transient, mode="all")
        group = ExceptionGroup(
            "all-transient", [TransientError("a"), TransientError("b")]
        )
        assert strategy(make_failed_state(group)) is True
        report = strategy.last_report
        assert report is not None
        assert report.deciding_path == (1,)

    def test_nested_group_paths_are_preserved(self) -> None:
        strategy = retry_if_exception_group(is_transient, mode="any")
        inner = ExceptionGroup("inner", [FatalError("f0"), TransientError("t1")])
        group = ExceptionGroup("outer", [FatalError("f1"), inner])
        assert strategy(make_failed_state(group)) is True
        report = strategy.last_report
        assert report is not None
        assert [v.path for v in report.verdicts] == [(0,), (1, 0), (1, 1)]
        assert report.deciding_path == (1, 1)
        assert isinstance(report.verdicts[2].exception, TransientError)

    def test_each_leaf_predicate_runs_exactly_once(self) -> None:
        calls: list[int] = []

        def counting_predicate(exc: BaseException) -> bool:
            calls.append(id(exc))
            return isinstance(exc, TransientError)

        strategy = retry_if_exception_group(counting_predicate, mode="any")
        leaves = [TransientError(), FatalError(), TransientError()]
        group = ExceptionGroup(
            "nested", [leaves[0], ExceptionGroup("inner", leaves[1:])]
        )
        strategy(make_failed_state(group))
        assert len(calls) == 3
        assert sorted(calls) == sorted(id(leaf) for leaf in leaves)

    @pytest.mark.parametrize(
        "terminal",
        [
            asyncio.CancelledError(),
            KeyboardInterrupt(),
            SystemExit(1),
            GeneratorExit(),
        ],
    )
    def test_terminal_leaf_is_never_retried(self, terminal: BaseException) -> None:
        calls: list[BaseException] = []

        def spy_predicate(exc: BaseException) -> bool:
            calls.append(exc)
            return True  # business rule would retry everything

        for mode in ("any", "all"):
            strategy = retry_if_exception_group(spy_predicate, mode=mode)
            group = BaseExceptionGroup("cancel", [TransientError("t"), terminal])
            assert strategy(make_failed_state(group)) is False
            report = strategy.last_report
            assert report is not None
            assert report.decision is False
            assert report.blocked_paths == ((1,),)
            assert report.deciding_path == (1,)
            assert "terminal leaf" in report.explanation
            # The terminal leaf was never shown to the business predicate.
            assert terminal not in calls
            terminal_verdict = report.verdicts[1]
            assert terminal_verdict.terminal is True
            assert terminal_verdict.matched is None

    def test_predicate_error_propagates_instead_of_false(self) -> None:
        class Boom(Exception):
            pass

        def broken_predicate(exc: BaseException) -> bool:
            raise Boom("predicate failed")

        strategy = retry_if_exception_group(broken_predicate, mode="any")
        group = ExceptionGroup("g", [TransientError("t")])
        with pytest.raises(Boom):
            strategy(make_failed_state(group))

    def test_non_group_exception_is_refused(self) -> None:
        strategy = retry_if_exception_group(is_transient)
        with pytest.raises(UndecidableRetryError):
            strategy(make_failed_state(TransientError("plain")))

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            retry_if_exception_group(is_transient, mode="maybe")

    def test_successful_outcome_is_not_retried(self) -> None:
        strategy = retry_if_exception_group(is_transient)
        assert strategy(make_success_state()) is False

    def test_missing_outcome_raises(self) -> None:
        strategy = retry_if_exception_group(is_transient)
        retry_state = RetryCallState(None, None, (), {})  # type: ignore[arg-type]
        with pytest.raises(RuntimeError):
            strategy(retry_state)

    def test_default_retry_if_exception_behaviour_unchanged(self) -> None:
        # Plain exceptions still go through the classic predicate path.
        classic = retry_if_exception_type(TransientError)
        assert classic(make_failed_state(TransientError("t"))) is True
        assert classic(make_failed_state(FatalError("f"))) is False
        # A group is not a TransientError instance: no implicit unwrapping.
        group = ExceptionGroup("g", [TransientError("t")])
        assert classic(make_failed_state(group)) is False
        # retry_if_exception keeps calling its predicate with the raw object.
        seen: list[BaseException] = []

        def recording_predicate(exc: BaseException) -> bool:
            seen.append(exc)
            return True

        strategy = retry_if_exception(recording_predicate)
        assert strategy(make_failed_state(group)) is True
        assert seen == [group]


class TestExceptionGroupRetryingIntegration:
    def test_sync_retrying_retries_on_any_match(self) -> None:
        attempts = 0

        @tenacity.retry(
            retry=retry_if_exception_group(is_transient, mode="any"),
            stop=stop_after_attempt(3),
        )
        def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ExceptionGroup("mixed", [TransientError("t"), FatalError("f")])
            return "done"

        assert flaky() == "done"
        assert attempts == 3

    def test_sync_retrying_does_not_retry_on_all_mismatch(self) -> None:
        attempts = 0

        @tenacity.retry(
            retry=retry_if_exception_group(is_transient, mode="all"),
            stop=stop_after_attempt(3),
        )
        def flaky() -> None:
            nonlocal attempts
            attempts += 1
            raise ExceptionGroup("mixed", [TransientError("t"), FatalError("f")])

        with pytest.raises(ExceptionGroup):
            flaky()
        assert attempts == 1

    def test_sync_retrying_does_not_retry_cancelled_leaf(self) -> None:
        attempts = 0

        @tenacity.retry(
            retry=retry_if_exception_group(lambda e: True, mode="any"),
            stop=stop_after_attempt(3),
        )
        def flaky() -> None:
            nonlocal attempts
            attempts += 1
            raise BaseExceptionGroup(
                "cancel", [TransientError("t"), asyncio.CancelledError()]
            )

        with pytest.raises(BaseExceptionGroup):
            flaky()
        assert attempts == 1


class TestExceptionGroupAsync:
    @asynctest
    async def test_async_retrying_retries_on_any_match(self) -> None:
        attempts = 0

        @tenacity.retry(
            retry=retry_if_exception_group(is_transient, mode="any"),
            stop=stop_after_attempt(3),
        )
        async def flaky() -> str:
            nonlocal attempts
            attempts += 1
            await asyncio.sleep(0)
            if attempts < 3:
                raise ExceptionGroup("mixed", [TransientError("t"), FatalError("f")])
            return "done"

        assert await flaky() == "done"
        assert attempts == 3

    @asynctest
    async def test_async_retrying_does_not_retry_on_all_mismatch(self) -> None:
        attempts = 0

        @tenacity.retry(
            retry=retry_if_exception_group(is_transient, mode="all"),
            stop=stop_after_attempt(3),
        )
        async def flaky() -> None:
            nonlocal attempts
            attempts += 1
            await asyncio.sleep(0)
            raise ExceptionGroup("mixed", [TransientError("t"), FatalError("f")])

        with pytest.raises(ExceptionGroup):
            await flaky()
        assert attempts == 1

    @asynctest
    async def test_async_retrying_does_not_retry_cancelled_leaf(self) -> None:
        attempts = 0

        async def flaky() -> None:
            nonlocal attempts
            attempts += 1
            await asyncio.sleep(0)
            raise BaseExceptionGroup(
                "cancel", [TransientError("t"), asyncio.CancelledError()]
            )

        retrying = AsyncRetrying(
            retry=retry_if_exception_group(lambda e: True, mode="any"),
            stop=stop_after_attempt(3),
        )
        with pytest.raises(BaseExceptionGroup):
            await retrying(flaky)
        assert attempts == 1

    @asynctest
    async def test_async_nested_group_report_path(self) -> None:
        strategy = retry_if_exception_group(is_transient, mode="any")

        async def flaky() -> None:
            await asyncio.sleep(0)
            raise ExceptionGroup(
                "outer",
                [
                    FatalError("f"),
                    ExceptionGroup("inner", [TransientError("t")]),
                ],
            )

        retrying = AsyncRetrying(retry=strategy, stop=stop_after_attempt(1))
        with pytest.raises(RetryError):
            await retrying(flaky)
        report = strategy.last_report
        assert report is not None
        assert report.decision is True
        assert report.deciding_path == (1, 0)


class TestExceptionGroupUnittestStyle(unittest.TestCase):
    def test_report_explanation_names_deciding_leaf(self) -> None:
        strategy = retry_if_exception_group(is_transient, mode="any")
        group = ExceptionGroup("g", [FatalError("f"), TransientError("t")])
        self.assertTrue(strategy(make_failed_state(group)))
        report = strategy.last_report
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report.deciding_path, (1,))
        self.assertIn("TransientError", report.explanation)
        self.assertIn("allowed", report.explanation)

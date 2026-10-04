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

import asyncio
import threading

import pytest

import tenacity
from tenacity import (
    AsyncRetrying,
    AttemptQuota,
    AttemptQuotaExhausted,
    RetryCallState,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    stop_never,
    wait_fixed,
)


class Counter:
    """Thread-safe counter."""

    def __init__(self) -> None:
        self.value = 0
        self._lock = threading.Lock()

    def incr(self) -> None:
        with self._lock:
            self.value += 1


def make_sync_retrying(quota, calls, before_calls):
    def before(retry_state: RetryCallState) -> None:
        before_calls.incr()

    retrying = Retrying(
        stop=stop_never,
        wait=wait_fixed(0.001),
        retry=retry_if_exception_type(ValueError),
        before=before,
        attempt_quota=quota,
    )

    def always_fails() -> None:
        calls.incr()
        raise ValueError("always fails")

    return retrying, always_fails


class TestAttemptQuotaUnit:
    def test_acquire_is_atomic_and_terminal(self):
        quota = AttemptQuota(2)
        assert quota.limit == 2
        assert quota.remaining == 2
        assert quota.acquire() is True
        assert quota.acquire() is True
        assert quota.acquire() is False
        assert quota.remaining == 0
        assert quota.acquired == 2
        # No refill, no automatic reset: still exhausted afterwards.
        assert quota.acquire() is False

    def test_negative_limit_rejected(self):
        with pytest.raises(ValueError):
            AttemptQuota(-1)

    def test_repr(self):
        assert repr(AttemptQuota(1)) == "AttemptQuota(limit=1, remaining=1)"


class TestSharedQuotaSync:
    def test_two_concurrent_calls_share_quota_of_three(self):
        quota = AttemptQuota(3)
        calls = Counter()
        before_calls = Counter()
        retrying, always_fails = make_sync_retrying(quota, calls, before_calls)

        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def worker() -> None:
            barrier.wait(timeout=10)
            try:
                retrying(always_fails)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
            assert not thread.is_alive()

        # The quota caps the number of *actual function executions* across
        # both concurrent retry loops, not the sum of their retry counts.
        assert calls.value == 3
        # `before` only ran for attempts that were granted a token.
        assert before_calls.value == 3
        assert quota.remaining == 0
        # Both workers exited with the dedicated terminal error instead of
        # looping forever in sleep/retry.
        assert len(errors) == 2
        assert all(isinstance(exc, AttemptQuotaExhausted) for exc in errors)

    def test_exhausted_quota_skips_before_and_function(self):
        quota = AttemptQuota(0)
        calls = Counter()
        before_calls = Counter()
        retrying, always_fails = make_sync_retrying(quota, calls, before_calls)

        with pytest.raises(AttemptQuotaExhausted):
            retrying(always_fails)

        assert calls.value == 0
        assert before_calls.value == 0

    def test_quota_error_is_not_wrapped_in_retry_error(self):
        quota = AttemptQuota(1)
        calls = Counter()
        before_calls = Counter()
        retrying, always_fails = make_sync_retrying(quota, calls, before_calls)

        # First (and only) token: the function runs once and fails; with
        # stop_after_attempt(1) the call ends with a regular RetryError.
        retrying_stop1 = retrying.copy(stop=stop_after_attempt(1))
        with pytest.raises(RetryError):
            retrying_stop1(always_fails)
        assert calls.value == 1

        # Quota is now exhausted and never resets: the dedicated terminal
        # error surfaces directly, even with reraise=True.
        retrying_reraise = retrying.copy(reraise=True)
        with pytest.raises(AttemptQuotaExhausted):
            retrying_reraise(always_fails)
        assert calls.value == 1
        assert before_calls.value == 1

    def test_quota_state_is_separate_from_retry_call_state(self):
        quota = AttemptQuota(1)
        calls = Counter()
        before_calls = Counter()
        retrying, always_fails = make_sync_retrying(quota, calls, before_calls)

        # The quota object is shared by reference across copies and is not
        # part of any RetryCallState.
        assert retrying.copy().attempt_quota is quota
        assert not hasattr(RetryCallState, "attempt_quota")

        seen_states = []

        def spy_before(retry_state: RetryCallState) -> None:
            seen_states.append(retry_state)

        retrying_spy = retrying.copy(before=spy_before)
        with pytest.raises(ValueError, match="always fails"):
            retrying_spy.copy(stop=stop_after_attempt(1), reraise=True)(always_fails)
        assert len(seen_states) == 1
        assert not hasattr(seen_states[0], "attempt_quota")
        assert quota.remaining == 0

    def test_no_quota_default_behaviour_unchanged(self):
        calls = Counter()

        @tenacity.retry(stop=stop_after_attempt(3), reraise=True)
        def always_fails() -> None:
            calls.incr()
            raise ValueError("always fails")

        with pytest.raises(ValueError):
            always_fails()
        assert calls.value == 3


class TestSharedQuotaAsync:
    def test_two_concurrent_coroutines_share_quota_of_three(self):
        async def main():
            quota = AttemptQuota(3)
            calls = Counter()
            before_calls = Counter()

            def before(retry_state: RetryCallState) -> None:
                before_calls.incr()

            async def always_fails() -> None:
                calls.incr()
                raise ValueError("always fails")

            retrying = AsyncRetrying(
                stop=stop_never,
                wait=wait_fixed(0.001),
                retry=retry_if_exception_type(ValueError),
                before=before,
                attempt_quota=quota,
            )

            results = await asyncio.gather(
                retrying(always_fails),
                retrying(always_fails),
                return_exceptions=True,
            )
            return quota, calls, before_calls, results

        quota, calls, before_calls, results = asyncio.run(main())

        assert calls.value == 3
        assert before_calls.value == 3
        assert quota.remaining == 0
        assert len(results) == 2
        assert all(isinstance(exc, AttemptQuotaExhausted) for exc in results)

    def test_sync_and_async_share_one_quota(self):
        quota = AttemptQuota(3)
        sync_calls = Counter()
        async_calls = Counter()

        retrying_sync, always_fails_sync = make_sync_retrying(
            quota, sync_calls, Counter()
        )

        async def always_fails_async() -> None:
            async_calls.incr()
            raise ValueError("always fails")

        retrying_async = AsyncRetrying(
            stop=stop_never,
            wait=wait_fixed(0.001),
            retry=retry_if_exception_type(ValueError),
            attempt_quota=quota,
        )

        sync_errors: list[BaseException] = []

        def sync_worker() -> None:
            try:
                retrying_sync(always_fails_sync)
            except BaseException as exc:
                sync_errors.append(exc)

        async def amain():
            return await asyncio.gather(
                retrying_async(always_fails_async),
                retrying_async(always_fails_async),
                return_exceptions=True,
            )

        thread = threading.Thread(target=sync_worker)
        thread.start()
        async_results = asyncio.run(amain())
        thread.join(timeout=30)
        assert not thread.is_alive()

        # One shared budget across sync and async retrying objects.
        assert sync_calls.value + async_calls.value == 3
        assert quota.remaining == 0
        assert len(sync_errors) == 1
        assert isinstance(sync_errors[0], AttemptQuotaExhausted)
        assert all(isinstance(exc, AttemptQuotaExhausted) for exc in async_results)

    def test_cancel_before_acquire_consumes_nothing(self):
        async def main():
            quota = AttemptQuota(5)
            calls = Counter()
            # Controllable barrier: the retry loop parks here inside its
            # sleep, i.e. after a failed attempt but before the next quota
            # acquisition.
            barrier = asyncio.Event()

            async def sleep_on_barrier(seconds: float) -> None:
                await barrier.wait()

            async def always_fails() -> None:
                calls.incr()
                raise ValueError("always fails")

            retrying = AsyncRetrying(
                sleep=sleep_on_barrier,
                stop=stop_never,
                retry=retry_if_exception_type(ValueError),
                attempt_quota=quota,
            )

            task = asyncio.create_task(retrying(always_fails))
            # Wait until the first attempt executed and the loop parked on
            # the barrier before its second acquisition.
            for _ in range(1000):
                await asyncio.sleep(0)
                if calls.value == 1:
                    break
            assert calls.value == 1
            assert quota.remaining == 4

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            # Cancelled while parked before acquiring: nothing more spent.
            assert calls.value == 1
            assert quota.remaining == 4

            # A call cancelled before it ever runs consumes nothing either.
            task2 = asyncio.create_task(retrying(always_fails))
            task2.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task2
            assert calls.value == 1
            assert quota.remaining == 4

        asyncio.run(main())

    def test_exhausted_quota_skips_before_and_function_async(self):
        async def main():
            quota = AttemptQuota(0)
            calls = Counter()
            before_calls = Counter()

            def before(retry_state: RetryCallState) -> None:
                before_calls.incr()

            async def always_fails() -> None:
                calls.incr()
                raise ValueError("always fails")

            retrying = AsyncRetrying(
                stop=stop_never,
                retry=retry_if_exception_type(ValueError),
                before=before,
                attempt_quota=quota,
            )
            with pytest.raises(AttemptQuotaExhausted):
                await retrying(always_fails)
            return calls, before_calls

        calls, before_calls = asyncio.run(main())
        assert calls.value == 0
        assert before_calls.value == 0

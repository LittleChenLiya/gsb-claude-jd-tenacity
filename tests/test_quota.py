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
"""Tests for the shared retry quota."""

import asyncio
import contextlib
import threading
import unittest
from collections.abc import Callable

import pytest

import tenacity
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    RetryQuota,
    RetryQuotaExceeded,
    retry,
    wait_none,
)


class TestRetryQuotaBasic(unittest.TestCase):
    def test_claim_and_remaining(self) -> None:
        quota = RetryQuota(max_attempts=3)
        assert quota.claimed == 0
        assert quota.remaining == 3
        assert quota.claim() is True
        assert quota.claimed == 1
        assert quota.remaining == 2
        assert quota.claim() is True
        assert quota.claim() is True
        assert quota.claimed == 3
        assert quota.remaining == 0
        assert quota.claim() is False
        assert quota.claimed == 3

    def test_invalid_max_attempts(self) -> None:
        with pytest.raises(ValueError):
            RetryQuota(max_attempts=0)
        with pytest.raises(ValueError):
            RetryQuota(max_attempts=-1)

    def test_claim_or_raise(self) -> None:
        quota = RetryQuota(max_attempts=1)
        quota.claim_or_raise(1)
        with pytest.raises(RetryQuotaExceeded) as exc_info:
            quota.claim_or_raise(2)
        err = exc_info.value
        assert err.quota is quota
        assert err.attempt_number == 2
        assert quota.claimed == 1

    def test_quota_does_not_replenish(self) -> None:
        quota = RetryQuota(max_attempts=2)
        quota.claim()
        quota.claim()
        assert quota.claim() is False
        assert quota.remaining == 0
        assert quota.claim() is False


class TestRetryQuotaSync(unittest.TestCase):
    def test_sync_retry_claims_until_exhausted(self) -> None:
        quota = RetryQuota(max_attempts=3)
        call_count = 0

        def fn() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        r = Retrying(
            retry=tenacity.retry_if_exception_type(),
            wait=wait_none(),
            retry_quota=quota,
        )
        with pytest.raises(RetryQuotaExceeded):
            r(fn)
        assert call_count == 3
        assert quota.claimed == 3

    def test_failures_consume_quota(self) -> None:
        quota = RetryQuota(max_attempts=2)
        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("fail")
            return "ok"

        r = Retrying(
            retry=tenacity.retry_if_exception_type(),
            wait=wait_none(),
            stop=tenacity.stop_after_attempt(5),
            retry_quota=quota,
        )
        result = r(fn)
        assert result == "ok"
        assert call_count == 2
        assert quota.claimed == 2
        assert quota.remaining == 0

    def test_no_quota_unchanged(self) -> None:
        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "ok"

        r = Retrying(
            retry=tenacity.retry_if_exception_type(),
            wait=wait_none(),
            stop=tenacity.stop_after_attempt(5),
        )
        result = r(fn)
        assert result == "ok"
        assert call_count == 3

    def test_decorator_with_quota(self) -> None:
        quota = RetryQuota(max_attempts=2)
        call_count = 0

        @retry(
            retry=tenacity.retry_if_exception_type(),
            wait=wait_none(),
            retry_quota=quota,
        )
        def fn() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        with pytest.raises(RetryQuotaExceeded):
            fn()
        assert call_count == 2
        assert quota.claimed == 2


class TestRetryQuotaConcurrent(unittest.TestCase):
    def test_shared_quota_concurrent_failing(self) -> None:
        quota = RetryQuota(max_attempts=3)
        call_count = 0
        lock = threading.Lock()
        errors = []

        def make_fn() -> Callable[[], None]:
            def fn() -> None:
                nonlocal call_count
                with lock:
                    call_count += 1
                raise ValueError("fail")

            return fn

        def run(fn: Callable[[], None]) -> None:
            r = Retrying(
                retry=tenacity.retry_if_exception_type(),
                wait=wait_none(),
                retry_quota=quota,
            )
            try:
                r(fn)
            except RetryQuotaExceeded as e:
                errors.append(e)

        t1 = threading.Thread(target=run, args=(make_fn(),))
        t2 = threading.Thread(target=run, args=(make_fn(),))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert call_count == 3, f"Expected exactly 3 calls, got {call_count}"
        assert len(errors) == 2
        assert all(isinstance(e, RetryQuotaExceeded) for e in errors)
        assert quota.claimed == 3
        assert quota.remaining == 0

    def test_quota_state_shared_across_calls(self) -> None:
        quota = RetryQuota(max_attempts=3)

        def make_fn() -> Callable[[], None]:
            def fn() -> None:
                raise ValueError("fail")

            return fn

        r1 = Retrying(
            retry=tenacity.retry_if_exception_type(),
            wait=wait_none(),
            retry_quota=quota,
        )
        r2 = Retrying(
            retry=tenacity.retry_if_exception_type(),
            wait=wait_none(),
            retry_quota=quota,
        )

        with pytest.raises(RetryQuotaExceeded):
            r1(make_fn())
        assert quota.claimed == 3

        with pytest.raises(RetryQuotaExceeded):
            r2(make_fn())
        assert quota.claimed == 3


class TestRetryQuotaExhaustionProtocol(unittest.TestCase):
    def test_before_not_called_when_exhausted(self) -> None:
        quota = RetryQuota(max_attempts=2)
        before_count = 0
        call_count = 0

        def before(rs: RetryCallState) -> None:
            nonlocal before_count
            before_count += 1

        def fn() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        r = Retrying(
            retry=tenacity.retry_if_exception_type(),
            wait=wait_none(),
            before=before,
            retry_quota=quota,
        )
        with pytest.raises(RetryQuotaExceeded):
            r(fn)

        assert call_count == 2
        assert before_count == 2

    def test_retry_error_callback_not_called_on_quota_exhaustion(self) -> None:
        quota = RetryQuota(max_attempts=1)
        callback_count = 0

        def callback(rs: RetryCallState) -> str:
            nonlocal callback_count
            callback_count += 1
            return "callback"

        def fn() -> None:
            raise ValueError("fail")

        r = Retrying(
            retry=tenacity.retry_if_exception_type(),
            wait=wait_none(),
            retry_error_callback=callback,
            retry_quota=quota,
        )
        with pytest.raises(RetryQuotaExceeded):
            r(fn)
        assert callback_count == 0

    def test_reraise_ignored_on_quota_exhaustion(self) -> None:
        quota = RetryQuota(max_attempts=1)

        def fn() -> None:
            raise ValueError("fail")

        r = Retrying(
            retry=tenacity.retry_if_exception_type(),
            wait=wait_none(),
            reraise=True,
            retry_quota=quota,
        )
        with pytest.raises(RetryQuotaExceeded):
            r(fn)

    def test_function_not_called_when_exhausted(self) -> None:
        quota = RetryQuota(max_attempts=1)
        call_count = 0

        def fn() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        r = Retrying(
            retry=tenacity.retry_if_exception_type(),
            wait=wait_none(),
            retry_quota=quota,
        )
        with pytest.raises(RetryQuotaExceeded):
            r(fn)
        assert call_count == 1


class TestRetryQuotaAsync(unittest.TestCase):
    def test_async_retry_claims_until_exhausted(self) -> None:
        quota = RetryQuota(max_attempts=3)
        call_count = 0

        async def fn() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        r = AsyncRetrying(
            retry=tenacity.retry_if_exception_type(),
            wait=wait_none(),
            retry_quota=quota,
        )

        async def main() -> None:
            with pytest.raises(RetryQuotaExceeded):
                await r(fn)

        asyncio.run(main())
        assert call_count == 3
        assert quota.claimed == 3

    def test_async_shared_quota_concurrent(self) -> None:
        quota = RetryQuota(max_attempts=3)
        call_count = 0
        errors = []

        async def make_fn() -> Callable[[], None]:
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        async def run() -> None:
            r = AsyncRetrying(
                retry=tenacity.retry_if_exception_type(),
                wait=wait_none(),
                retry_quota=quota,
            )
            try:
                await r(make_fn)
            except RetryQuotaExceeded as e:
                errors.append(e)

        async def main() -> None:
            await asyncio.gather(run(), run())

        asyncio.run(main())
        assert call_count == 3
        assert len(errors) == 2
        assert quota.claimed == 3

    def test_async_no_quota_unchanged(self) -> None:
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "ok"

        r = AsyncRetrying(
            retry=tenacity.retry_if_exception_type(),
            wait=wait_none(),
            stop=tenacity.stop_after_attempt(5),
        )

        async def main() -> None:
            result: str = await r(fn)
            assert result == "ok"

        asyncio.run(main())
        assert call_count == 3

    def test_cancel_before_claim_does_not_consume(self) -> None:
        reached = asyncio.Event()
        release = asyncio.Event()

        async def barrier() -> None:
            reached.set()
            await release.wait()

        quota = RetryQuota(max_attempts=3, barrier=barrier)

        async def fn() -> None:
            raise ValueError("fail")

        async def main() -> None:
            r = AsyncRetrying(
                retry=tenacity.retry_if_exception_type(),
                wait=wait_none(),
                retry_quota=quota,
            )
            task: asyncio.Task[None] = asyncio.create_task(r(fn))
            await reached.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()

        asyncio.run(main())
        assert quota.claimed == 0
        assert quota.remaining == 3

    def test_cancel_after_claim_consumes_quota(self) -> None:
        quota = RetryQuota(max_attempts=3)
        call_count = 0

        async def fn() -> None:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(10)

        async def main() -> None:
            r = AsyncRetrying(
                retry=tenacity.retry_if_exception_type(),
                wait=wait_none(),
                retry_quota=quota,
            )
            task: asyncio.Task[None] = asyncio.create_task(r(fn))
            for _ in range(100):
                if call_count >= 1:
                    break
                await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(main())
        assert quota.claimed == 1
        assert call_count == 1


class TestRetryQuotaMixed(unittest.TestCase):
    def test_sync_and_async_share_quota(self) -> None:
        quota = RetryQuota(max_attempts=3)
        call_count = 0

        def sync_fn() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        async def async_fn() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        sync_r = Retrying(
            retry=tenacity.retry_if_exception_type(),
            wait=wait_none(),
            retry_quota=quota,
        )
        async_r = AsyncRetrying(
            retry=tenacity.retry_if_exception_type(),
            wait=wait_none(),
            retry_quota=quota,
        )

        sync_error = []

        def run_sync() -> None:
            try:
                sync_r(sync_fn)
            except RetryQuotaExceeded as e:
                sync_error.append(e)

        t = threading.Thread(target=run_sync)
        t.start()

        async def run_async() -> None:
            with contextlib.suppress(RetryQuotaExceeded):
                await async_r(async_fn)

        async def main() -> None:
            await run_async()

        asyncio.run(main())
        t.join()

        assert call_count == 3
        assert quota.claimed == 3
        assert len(sync_error) == 1

    def test_claim_async_and_sync_share_counter(self) -> None:
        quota = RetryQuota(max_attempts=3)

        assert quota.claim() is True
        assert quota.claimed == 1

        async def async_claim() -> bool:
            return await quota.claim_async()

        assert asyncio.run(async_claim()) is True
        assert quota.claimed == 2

        assert quota.claim() is True
        assert quota.claimed == 3

        assert quota.claim() is False
        assert asyncio.run(async_claim()) is False
        assert quota.claimed == 3


class TestRetryQuotaStateSeparate(unittest.TestCase):
    def test_quota_state_not_stored_in_retry_call_state(self) -> None:
        quota = RetryQuota(max_attempts=3)

        def fn() -> None:
            raise ValueError("fail")

        r = Retrying(
            retry=tenacity.retry_if_exception_type(),
            wait=wait_none(),
            retry_quota=quota,
        )
        with pytest.raises(RetryQuotaExceeded):
            r(fn)

        assert not hasattr(RetryCallState, "quota")
        assert not hasattr(RetryCallState, "retry_quota")
        assert quota.claimed == 3
        assert quota.remaining == 0

    def test_quota_object_holds_state(self) -> None:
        quota = RetryQuota(max_attempts=5)
        assert quota.claimed == 0
        assert quota.remaining == 5
        quota.claim()
        quota.claim()
        assert quota.claimed == 2
        assert quota.remaining == 3
        assert isinstance(quota.claimed, int)
        assert isinstance(quota.remaining, int)


if __name__ == "__main__":
    unittest.main()

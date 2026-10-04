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
"""Shared attempt quota for concurrent retries.

A :class:`RetryQuota` is a fixed budget of *actual function executions*
that is shared across every retrying call that references it. Each
attempt atomically claims one quota before the target function runs;
failures consume quota too. When the quota is exhausted, the next claim
raises :class:`RetryQuotaExceeded` immediately, terminating the retry
loop without running the ``before`` callback or the target function and
without any further sleep.

The quota state lives in the :class:`RetryQuota` object itself, separate
from any :class:`~tenacity.RetryCallState`. It is not stored per-call, so
multiple concurrent retries (sync, async, or mixed) share the same
counter. The quota does not replenish or reset on its own.
"""

import threading
import typing as t


class RetryQuotaExceeded(Exception):
    """Raised when a shared :class:`RetryQuota` is exhausted.

    This is a dedicated termination error, not a retry error. When it is
    raised, the retry loop terminates immediately: the ``before``
    callback and the target function are not executed, no sleep occurs,
    and ``retry_error_callback`` is not called. The ``reraise`` flag is
    ignored because quota exhaustion is not a retryable condition.

    :param quota: The exhausted quota.
    :param attempt_number: The attempt number that would have been made
        had the claim succeeded.
    """

    def __init__(self, quota: "RetryQuota", attempt_number: int) -> None:
        self.quota = quota
        self.attempt_number = attempt_number
        super().__init__(
            f"Retry quota exhausted: attempt {attempt_number} would exceed "
            f"the shared quota of {quota.max_attempts} "
            f"({quota.claimed} already claimed)"
        )


class RetryQuota:
    """Shared attempt quota for concurrent retries.

    Limits the total number of actual function executions across all
    retrying calls that share this quota. Each attempt atomically claims
    one quota before executing. Failures consume quota too.

    The quota is a fixed budget: it does not replenish or reset.

    **Claim timing.** A claim is attempted at the start of each attempt
    iteration, *before* the ``before`` callback and *before* the target
    function. The claim is the commit point: once claimed, the quota is
    consumed regardless of the outcome (success, failure, or cancellation
    after the claim).

    **Exhaustion protocol.** When a claim fails (the quota is exhausted),
    :class:`RetryQuotaExceeded` is raised immediately. The ``before``
    callback and the target function are not executed, no sleep occurs,
    and the retry loop terminates.

    **Cancellation before claim.** In async mode, :meth:`claim_async`
    awaits the optional ``barrier`` before the atomic claim. Cancellation
    while awaiting the barrier (before the atomic claim) does not consume
    quota. Cancellation after the atomic claim does consume quota.

    :param max_attempts: The total number of attempts allowed across all
        retries that share this quota. Must be >= 1.
    :param barrier: An optional async callable that is awaited before the
        atomic claim in async mode. This creates an await point where
        cancellation can occur before the quota is consumed. It is only
        used by :meth:`claim_async`; :meth:`claim` (sync) does not await
        it. Intended for testing cancellation-before-claim.
    """

    def __init__(
        self,
        max_attempts: int,
        barrier: t.Callable[[], t.Awaitable[None]] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.max_attempts = max_attempts
        self._claimed = 0
        self._lock = threading.Lock()
        self._barrier = barrier

    def claim(self) -> bool:
        """Atomically claim one attempt.

        This is the sync claim path. It is atomic across threads.

        :return: ``True`` if the claim succeeded, ``False`` if the quota
            is exhausted.
        """
        with self._lock:
            if self._claimed >= self.max_attempts:
                return False
            self._claimed += 1
            return True

    async def claim_async(self) -> bool:
        """Async claim path.

        Awaits the ``barrier`` (if set) before the atomic claim. This
        creates an await point where cancellation can occur before the
        quota is consumed. Cancellation while awaiting the barrier does
        not consume quota.

        :return: ``True`` if the claim succeeded, ``False`` if the quota
            is exhausted.
        """
        if self._barrier is not None:
            await self._barrier()
        return self.claim()

    def claim_or_raise(self, attempt_number: int) -> None:
        """Claim one attempt, raising :class:`RetryQuotaExceeded` if exhausted.

        This is the sync entry point used by the retry loop.

        :param attempt_number: The attempt number that would be made.
        :raises RetryQuotaExceeded: If the quota is exhausted.
        """
        if not self.claim():
            raise RetryQuotaExceeded(self, attempt_number)

    async def claim_or_raise_async(self, attempt_number: int) -> None:
        """Async claim, raising :class:`RetryQuotaExceeded` if exhausted.

        This is the async entry point used by the retry loop. It awaits
        the ``barrier`` (if set) before the atomic claim, so cancellation
        before the claim does not consume quota.

        :param attempt_number: The attempt number that would be made.
        :raises RetryQuotaExceeded: If the quota is exhausted.
        """
        if not await self.claim_async():
            raise RetryQuotaExceeded(self, attempt_number)

    @property
    def remaining(self) -> int:
        """The number of attempts still available."""
        with self._lock:
            return self.max_attempts - self._claimed

    @property
    def claimed(self) -> int:
        """The number of attempts already claimed."""
        with self._lock:
            return self._claimed


__all__ = ["RetryQuota", "RetryQuotaExceeded"]

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
"""Shared attempt quota for retrying objects.

A quota caps the number of *actual function executions* performed across
every retrying object (and every concurrent call) that shares it.  It does
not limit the number of retry loops: each attempt -- successful or not --
consumes exactly one token, atomically, right before the ``before``
callbacks and the wrapped function would run.

The quota is never refilled on a timer and never resets itself; once it is
exhausted every further acquisition fails until the caller replaces it.
"""

import threading

from ._utils import override


class AttemptQuotaExhausted(Exception):
    """Terminal error raised when a shared attempt quota is exhausted.

    This error is raised directly out of the retrying call: it is never
    recorded as the attempt outcome, never triggers another retry, and is
    never wrapped in :class:`tenacity.RetryError`.  When it is raised the
    ``before`` callbacks and the wrapped function have *not* been executed
    for that attempt.
    """


class AttemptQuota:
    """A shared, thread-safe budget of function-execution attempts.

    The same instance can be handed to several retrying objects (sync and
    async alike); every attempt performed by any of them atomically
    consumes one token via :meth:`acquire`.  The quota state lives entirely
    in this object and is kept separate from each call's
    :class:`tenacity.RetryCallState`.

    The quota is not replenished over time and is never reset
    automatically; once :attr:`remaining` reaches zero every subsequent
    acquisition fails.
    """

    __slots__ = ("_limit", "_lock", "_remaining")

    def __init__(self, limit: int) -> None:
        if limit < 0:
            raise ValueError("quota limit must be >= 0")
        self._limit = int(limit)
        self._remaining = int(limit)
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        """The total number of attempts this quota was created with."""
        return self._limit

    @property
    def remaining(self) -> int:
        """The number of attempts still available."""
        with self._lock:
            return self._remaining

    @property
    def acquired(self) -> int:
        """The number of attempts consumed so far."""
        with self._lock:
            return self._limit - self._remaining

    def acquire(self) -> bool:
        """Atomically take one token, returning False when none are left.

        A token that has been handed out is gone for good: it is consumed
        whether the attempt it authorises succeeds, fails, or never gets to
        run because the caller is cancelled *after* this method returned.
        Calls cancelled *before* this method runs consume nothing.
        """
        with self._lock:
            if self._remaining <= 0:
                return False
            self._remaining -= 1
            return True

    @override
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(limit={self._limit}, "
            f"remaining={self.remaining})"
        )


__all__ = [
    "AttemptQuota",
    "AttemptQuotaExhausted",
]

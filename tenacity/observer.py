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

"""Explicit attempt observer events for retry calls.

An *attempt observer* is a plain callable receiving :class:`AttemptEvent`
objects. It is notified about the meaningful moments of a retry call --
attempt start, attempt result, stop decision, wait start and wait end --
plus a closing ``CALL_END`` event for *every* terminal outcome (including
final success) and a cancellation event naming the exact phase when the
call is cancelled.

Observers only ever learn the *type* of the moment and *when* it happened,
correlated by ``call_id`` across the synchronous and asynchronous paths.
They never receive the retry state, the wrapped function, its arguments,
its result, or its exception. Observer failures are swallowed: they can
never change the business return value, the raised ``RetryError``, or the
``reraise`` behaviour of the retried call.
"""

import dataclasses
import enum
import sys
import typing as t


class AttemptEventType(enum.Enum):
    """Kinds of moments reported to an attempt observer."""

    #: An attempt to call the wrapped function is about to start.
    ATTEMPT_START = "attempt_start"
    #: An attempt finished and its outcome (result or exception) was recorded.
    ATTEMPT_RESULT = "attempt_result"
    #: The stop strategy was evaluated and a stop/continue decision was made.
    STOP_DECISION = "stop_decision"
    #: The retry loop is about to sleep before the next attempt.
    WAIT_START = "wait_start"
    #: The retry loop finished sleeping and the next attempt may begin.
    WAIT_END = "wait_end"
    #: The call finished, either successfully or by giving up. Every call
    #: emits exactly one closing event, including final success.
    CALL_END = "call_end"
    #: The call was cancelled while the wrapped function was running.
    CANCELLED_DURING_ATTEMPT = "cancelled_during_attempt"
    #: The call was cancelled while waiting between attempts.
    CANCELLED_DURING_WAIT = "cancelled_during_wait"


@dataclasses.dataclass(frozen=True, slots=True)
class AttemptEvent:
    """A single observed moment of a retry call.

    The payload is deliberately minimal: the *type* of the moment and the
    *time* it happened, correlated by ``call_id``. No business data
    (arguments, results, exceptions) ever reaches the observer.
    """

    #: Identifier of the retry call this event belongs to. Unique per
    #: invocation of the wrapped callable and shared by the sync and async
    #: paths, so events of concurrent calls never get mixed up.
    call_id: int
    #: What happened.
    type: AttemptEventType
    #: When it happened, as a :func:`time.monotonic` timestamp.
    timestamp: float


#: An attempt observer is any callable accepting a single AttemptEvent.
AttemptObserver = t.Callable[[AttemptEvent], None]


def is_cancellation(exc: BaseException) -> bool:
    """Return True if *exc* looks like a cancellation of the running call."""
    if isinstance(exc, KeyboardInterrupt):
        return True
    # Only consult asyncio/trio when they are already in use: importing them
    # here would be expensive and pointless for purely synchronous callers.
    asyncio_mod = sys.modules.get("asyncio")
    if asyncio_mod is not None and isinstance(exc, asyncio_mod.CancelledError):
        return True
    trio_mod = sys.modules.get("trio")
    return trio_mod is not None and isinstance(exc, trio_mod.Cancelled)

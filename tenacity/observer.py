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
"""Explicit attempt observer.

The observer is a passive, structured event sink for retry calls.  Unlike the
``before``/``after``/``before_sleep`` callbacks -- which receive the whole
:class:`~tenacity.RetryCallState` and whose return values steer the retry
loop -- the observer only ever receives an :class:`EventType` and a timestamp.
Every event is tagged with a call id so that concurrent calls (threads or
asyncio tasks) can be told apart.

Observer failures are swallowed: a misbehaving observer must never change the
business return value, the raised :class:`~tenacity.RetryError`, or the
``reraise`` behaviour.

Event specification
-------------------

A retry call emits the following events, in order:

1. :data:`EventType.CALL_START` -- emitted once when the call begins.
2. For every attempt:

   a. :data:`EventType.ATTEMPT_START` -- right before the wrapped function is
      run.
   b. :data:`EventType.ATTEMPT_RESULT` -- right after the wrapped function
      returned or raised.  This is emitted for **every** attempt, including
      the final successful one; it is never piggybacked on ``before_sleep``.
   c. :data:`EventType.STOP_DECISION` -- after the stop strategy ran.
   d. :data:`EventType.WAIT_START` / :data:`EventType.WAIT_END` -- only when a
      wait actually happens between two attempts.

3. :data:`EventType.CALL_END` -- emitted once when the call finishes, on both
   the success and the non-cancellation error paths.
4. :data:`EventType.CANCELLED_DURING_ATTEMPT` /
   :data:`EventType.CANCELLED_DURING_WAIT` -- the closing event when the call
   is cancelled, marking the phase during which the cancellation occurred.

The observer only receives the event type and a timestamp; it never sees the
retry state.  The call id is taken from the current context
(:class:`contextvars.ContextVar`), which isolates threads and asyncio tasks.
"""

from __future__ import annotations

import contextvars
import enum
import itertools
import threading
import time
import typing as t
from dataclasses import dataclass


class EventType(enum.Enum):
    """The kind of event observed during a retry call."""

    #: A retry call has started.
    CALL_START = "call_start"
    #: A retry call has finished (successfully or with a non-cancellation error).
    CALL_END = "call_end"
    #: An attempt is about to be run.
    ATTEMPT_START = "attempt_start"
    #: An attempt has produced a result (return value or exception).
    #: Emitted for every attempt, including the final successful one.
    ATTEMPT_RESULT = "attempt_result"
    #: The stop strategy has made a decision.
    STOP_DECISION = "stop_decision"
    #: The wait between two attempts has started.
    WAIT_START = "wait_start"
    #: The wait between two attempts has ended.
    WAIT_END = "wait_end"
    #: The call was cancelled while running an attempt.
    CANCELLED_DURING_ATTEMPT = "cancelled_during_attempt"
    #: The call was cancelled while waiting between attempts.
    CANCELLED_DURING_WAIT = "cancelled_during_wait"


@dataclass(frozen=True, slots=True)
class Event:
    """A single observed event.

    The observer only records the :class:`EventType`, the timestamp (in the
    clock's units, default ``time.monotonic``) and the call id.
    """

    type: EventType
    timestamp: float
    call_id: int


class AttemptObserver:
    """A passive observer recording timestamped attempt events.

    The observer is safe to share between threads and asyncio tasks.  Each
    retry call is assigned a unique call id; events are tagged with it so that
    concurrent calls can be correlated.

    :param clock: zero-argument callable returning a monotonic timestamp.
    """

    def __init__(self, clock: t.Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._events: list[Event] = []
        self._lock = threading.Lock()
        self._call_ids = itertools.count(1)
        self._tokens: dict[int, contextvars.Token[int | None]] = {}
        self._current_call: contextvars.ContextVar[int | None] = contextvars.ContextVar(
            "tenacity_observer_current_call", default=None
        )

    def _timestamp(self) -> float:
        return self._clock()

    def begin_call(self) -> int:
        """Record the start of a retry call and return its call id."""
        call_id = next(self._call_ids)
        token = self._current_call.set(call_id)
        self._tokens[call_id] = token
        self._record(EventType.CALL_START, call_id)
        return call_id

    def end_call(
        self, call_id: int, event_type: EventType = EventType.CALL_END
    ) -> None:
        """Record the end of a retry call.

        ``event_type`` is the closing event: :data:`EventType.CALL_END` for a
        normal finish, or one of the ``CANCELLED_*`` types when the call was
        cancelled.
        """
        self._record(event_type, call_id)
        token = self._tokens.pop(call_id, None)
        if token is not None:
            self._current_call.reset(token)

    def event(self, event_type: EventType, timestamp: float | None = None) -> None:
        """Record an event for the current call.

        The observer only receives the event type and a timestamp; the call id
        is taken from the current context.  Calls made outside of a retry call
        are ignored.
        """
        call_id = self._current_call.get()
        if call_id is None:
            return
        self._record(event_type, call_id, timestamp)

    def _record(
        self,
        event_type: EventType,
        call_id: int,
        timestamp: float | None = None,
    ) -> None:
        if timestamp is None:
            timestamp = self._timestamp()
        with self._lock:
            self._events.append(Event(event_type, timestamp, call_id))

    @property
    def events(self) -> list[Event]:
        """All recorded events, in recording order."""
        with self._lock:
            return list(self._events)

    def calls(self) -> dict[int, list[Event]]:
        """Recorded events grouped by call id."""
        grouped: dict[int, list[Event]] = {}
        for event in self.events:
            grouped.setdefault(event.call_id, []).append(event)
        return grouped

    def reset(self) -> None:
        """Remove all recorded events."""
        with self._lock:
            self._events.clear()

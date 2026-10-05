Attempt Observer
================

Tenacity can report the lifecycle of every retry call to an explicit
*attempt observer*: a plain callable passed as ``attempt_observer`` to
``@retry``, ``Retrying``, or ``AsyncRetrying``. The observer is notified
about each meaningful moment of a call -- attempt start, attempt result,
stop decision, wait start and wait end -- plus a closing ``CALL_END`` event
for *every* terminal outcome (including final success) and a cancellation
event naming the exact phase when the call is cancelled.

This is deliberately **not** built on top of ``before_sleep``:
``before_sleep`` only fires before a sleep between retries, so it can never
report attempts that do not lead to a retry, the final outcome, or a
cancellation. The attempt observer is a real event stream instead.


Event specification
-------------------

The observer receives exactly one argument: an
:class:`tenacity.AttemptEvent`. It carries only the *type* of the moment
and *when* it happened -- never the retry state, the wrapped function, its
arguments, its result, or its exception.

.. code-block:: python

    @dataclasses.dataclass(frozen=True, slots=True)
    class AttemptEvent:
        call_id: int                 # correlates all events of one call
        type: AttemptEventType       # what happened
        timestamp: float             # time.monotonic() of the moment

``call_id`` is unique per invocation of the wrapped callable and is shared
by the synchronous and asynchronous paths, so events of concurrent calls
(threads or asyncio tasks) never get mixed up: group events by ``call_id``
to reconstruct each individual call.

The event types (:class:`tenacity.AttemptEventType`) are:

============================ ===============================================
Event                        Meaning
============================ ===============================================
``ATTEMPT_START``            An attempt to call the wrapped function starts.
``ATTEMPT_RESULT``           The attempt finished; its outcome was recorded.
``STOP_DECISION``            The stop strategy was evaluated.
``WAIT_START``               The loop is about to sleep before retrying.
``WAIT_END``                 The sleep finished; the next attempt may begin.
``CALL_END``                 The call finished (success *or* giving up).
                             Every call emits exactly one closing event.
``CANCELLED_DURING_ATTEMPT`` The call was cancelled inside the wrapped
                             function (e.g. ``asyncio.CancelledError``,
                             ``KeyboardInterrupt``).
``CANCELLED_DURING_WAIT``    The call was cancelled while sleeping between
                             attempts.
============================ ===============================================

A call that fails once and then succeeds produces exactly two
``ATTEMPT_START``/``ATTEMPT_RESULT`` pairs and one
``WAIT_START``/``WAIT_END`` pair:

.. code-block:: python

    ATTEMPT_START, ATTEMPT_RESULT,          # attempt 1 fails
    STOP_DECISION,                          # keep going
    WAIT_START, WAIT_END,                   # one wait
    ATTEMPT_START, ATTEMPT_RESULT,          # attempt 2 succeeds
    CALL_END,                               # stream closed


Usage
-----

.. code-block:: python

    from tenacity import retry, stop_after_attempt, wait_fixed

    events = []

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1),
           attempt_observer=events.append)
    def flaky():
        ...

The same keyword works on ``Retrying`` and ``AsyncRetrying`` (including
their context-manager/iterator forms) and is preserved by ``retry_with``.

Safety guarantees
-----------------

* The observer only ever receives the event type and the moment it
  happened (correlated by ``call_id``); no business data is exposed.
* Observer failures are swallowed: a crashing observer changes neither
  the business return value, nor the raised ``RetryError`` (whose
  ``__cause__`` still chains the last exception), nor ``reraise=True``
  behaviour.
* Concurrent calls are fully isolated: each invocation gets its own
  ``call_id`` and its own event sequence.

Testing with virtual sleeps
---------------------------

Observer tests do not need real waiting: pass a *virtual* ``sleep``
callable that records (or ignores) the requested delay instead of
sleeping, making the whole suite deterministic and fast:

.. code-block:: python

    def virtual_sleep(seconds: float) -> None:
        pass  # virtual time: never actually waits

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1),
           sleep=virtual_sleep, attempt_observer=events.append)
    def flaky():
        ...

See ``tests/test_observer.py`` for complete examples, including
cancellation of asyncio tasks and concurrent calls.

Commands
--------

Run the observer acceptance tests (virtual sleeps, no real waiting)::

    python -m pytest tests/test_observer.py -q

Run the whole test suite::

    python -m pytest tests/ -q

Lint and format checks::

    python -m ruff check .
    python -m ruff format --check .

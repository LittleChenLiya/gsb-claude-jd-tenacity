===============
 API Reference
===============

Retry Main API
--------------

.. autofunction:: tenacity.retry
   :noindex:

.. autoclass:: tenacity.Retrying
   :members:

.. autoclass:: tenacity.AsyncRetrying
   :members:

.. autoclass:: tenacity.tornadoweb.TornadoRetrying
   :members:

.. autoclass:: tenacity.RetryCallState
   :members:

Shared Retry Quota
------------------

A :class:`tenacity.RetryQuota` is a fixed budget of *actual function
executions* shared across every retrying call that references it. Each
attempt atomically claims one quota before the target function runs;
failures consume quota too. The quota state lives in the
:class:`tenacity.RetryQuota` object itself, separate from any
:class:`tenacity.RetryCallState`, so multiple concurrent retries (sync,
async, or mixed) share the same counter. The quota does not replenish or
reset on its own.

**Claim timing.** A claim is attempted at the start of each attempt
iteration, *before* the ``before`` callback and *before* the target
function. The claim is the commit point: once claimed, the quota is
consumed regardless of the outcome (success, failure, or cancellation
after the claim).

**Exhaustion protocol.** When a claim fails (the quota is exhausted),
:class:`tenacity.RetryQuotaExceeded` is raised immediately. The ``before``
callback and the target function are not executed, no sleep occurs, and
the retry loop terminates. ``retry_error_callback`` is not called, and
``reraise`` is ignored because quota exhaustion is not a retryable
condition.

**Cancellation before claim.** In async mode, the quota's ``barrier`` (if
set) is awaited before the atomic claim. Cancellation while awaiting the
barrier (before the atomic claim) does not consume quota. Cancellation
after the atomic claim does consume quota.

.. autoclass:: tenacity.RetryQuota
   :members:

.. autoexception:: tenacity.RetryQuotaExceeded

After Functions
---------------

Those functions can be used as the `after` keyword argument of
:py:func:`tenacity.retry`.

.. automodule:: tenacity.after
   :members:

Before Functions
----------------

Those functions can be used as the `before` keyword argument of
:py:func:`tenacity.retry`.

.. automodule:: tenacity.before
   :members:

Before Sleep Functions
----------------------

Those functions can be used as the `before_sleep` keyword argument of
:py:func:`tenacity.retry`.

.. automodule:: tenacity.before_sleep
   :members:

Nap Functions
-------------

Those functions can be used as the `sleep` keyword argument of
:py:func:`tenacity.retry`.

.. automodule:: tenacity.nap
   :members:

Retry Functions
---------------

Those functions can be used as the `retry` keyword argument of
:py:func:`tenacity.retry`.

.. automodule:: tenacity.retry
   :members:

Stop Functions
--------------

Those functions can be used as the `stop` keyword argument of
:py:func:`tenacity.retry`.

.. automodule:: tenacity.stop
   :members:

Wait Functions
--------------

Those functions can be used as the `wait` keyword argument of
:py:func:`tenacity.retry`.

.. automodule:: tenacity.wait
   :members:

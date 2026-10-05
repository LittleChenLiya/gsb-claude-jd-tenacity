# Copyright 2016–2021 Julien Danjou
# Copyright 2016 Joshua Harlow
# Copyright 2013-2014 Ray Holder
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

import abc
import asyncio
import builtins
import re
import typing

from tenacity._utils import override

if typing.TYPE_CHECKING:
    from tenacity import RetryCallState

# ``BaseExceptionGroup`` is a builtin only on Python 3.11+. On 3.10 groups
# cannot exist at runtime, so the group strategy simply refuses to judge.
_BaseExceptionGroup: "type[BaseException] | None" = getattr(
    builtins, "BaseExceptionGroup", None
)


class _ExceptionGroupLike(typing.Protocol):
    """Structural view of ``BaseExceptionGroup`` for leaf traversal."""

    @property
    def exceptions(self) -> tuple[BaseException, ...]: ...


class retry_base(abc.ABC):
    """Abstract base class for retry strategies."""

    @abc.abstractmethod
    def __call__(self, retry_state: "RetryCallState") -> bool:
        pass

    def __and__(self, other: "RetryBaseT") -> "retry_all":
        if isinstance(other, retry_base):
            return other.__rand__(self)
        # Plain callable: flatten if self is already a retry_all
        if isinstance(self, retry_all):
            return retry_all(*self.retries, other)
        return retry_all(self, other)

    def __rand__(self, other: "RetryBaseT") -> "retry_all":
        # Flatten if other is already a retry_all
        if isinstance(other, retry_all):
            return retry_all(*other.retries, self)
        return retry_all(other, self)

    def __or__(self, other: "RetryBaseT") -> "retry_any":
        if isinstance(other, retry_base):
            return other.__ror__(self)
        # Plain callable: flatten if self is already a retry_any
        if isinstance(self, retry_any):
            return retry_any(*self.retries, other)
        return retry_any(self, other)

    def __ror__(self, other: "RetryBaseT") -> "retry_any":
        # Flatten if other is already a retry_any
        if isinstance(other, retry_any):
            return retry_any(*other.retries, self)
        return retry_any(other, self)


RetryBaseT = retry_base | typing.Callable[["RetryCallState"], bool]


class _retry_never(retry_base):
    """Retry strategy that never rejects any result."""

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        return False


retry_never = _retry_never()


class _retry_always(retry_base):
    """Retry strategy that always rejects any result."""

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        return True


retry_always = _retry_always()


class retry_if_exception(retry_base):
    """Retry strategy that retries if an exception verifies a predicate."""

    def __init__(self, predicate: typing.Callable[[BaseException], bool]) -> None:
        self.predicate = predicate

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome is None:
            raise RuntimeError("__call__() called before outcome was set")

        if retry_state.outcome.failed:
            exception = retry_state.outcome.exception()
            if exception is None:
                raise RuntimeError("outcome failed but the exception is None")
            return self.predicate(exception)
        return False


class retry_if_exception_type(retry_if_exception):
    """Retries if an exception has been raised of one or more types."""

    def __init__(
        self,
        exception_types: type[BaseException]
        | tuple[type[BaseException], ...] = Exception,
    ) -> None:
        self.exception_types = exception_types
        super().__init__(self._check)

    def _check(self, e: BaseException) -> bool:
        return isinstance(e, self.exception_types)


class retry_if_not_exception_type(retry_if_exception):
    """Retries except an exception has been raised of one or more types."""

    def __init__(
        self,
        exception_types: type[BaseException]
        | tuple[type[BaseException], ...] = Exception,
    ) -> None:
        self.exception_types = exception_types
        super().__init__(self._check)

    def _check(self, e: BaseException) -> bool:
        return not isinstance(e, self.exception_types)


class retry_unless_exception_type(retry_if_exception):
    """Retries until an exception is raised of one or more types."""

    def __init__(
        self,
        exception_types: type[BaseException]
        | tuple[type[BaseException], ...] = Exception,
    ) -> None:
        self.exception_types = exception_types
        super().__init__(self._check)

    def _check(self, e: BaseException) -> bool:
        return not isinstance(e, self.exception_types)

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome is None:
            raise RuntimeError("__call__() called before outcome was set")

        # always retry if no exception was raised
        if not retry_state.outcome.failed:
            return True

        exception = retry_state.outcome.exception()
        if exception is None:
            raise RuntimeError("outcome failed but the exception is None")
        return self.predicate(exception)


class retry_if_exception_cause_type(retry_base):
    """Retries if any of the causes of the raised exception is of one or more types.

    The check on the type of the cause of the exception is done recursively (until finding
    an exception in the chain that has no ``__cause__``, or a cycle is detected).
    """

    def __init__(
        self,
        exception_types: type[BaseException]
        | tuple[type[BaseException], ...] = Exception,
    ) -> None:
        self.exception_cause_types = exception_types

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome is None:
            raise RuntimeError("__call__ called before outcome was set")

        if retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            # Guard against cyclic __cause__ chains (e.g. ``raise e from e``),
            # which would otherwise spin forever inside the predicate and
            # prevent stop conditions from ever running (see #658).
            seen: set[int] = set()
            while exc is not None and id(exc) not in seen:
                seen.add(id(exc))
                if isinstance(exc.__cause__, self.exception_cause_types):
                    return True
                exc = exc.__cause__

        return False


class UndecidableRetryError(RuntimeError):
    """Raised when a retry strategy explicitly refuses to judge an exception.

    This is never swallowed by tenacity: it propagates out of the retry
    machinery so that misconfiguration surfaces loudly instead of being
    silently treated as "do not retry".
    """


class LeafVerdict(typing.NamedTuple):
    """Outcome of evaluating a single leaf exception of an exception group.

    Attributes:
        path: Index path of the leaf inside the (possibly nested) group,
            e.g. ``(1, 0)`` is the first leaf of the second sub-group.
        exception: The leaf exception instance itself.
        matched: Result of the leaf predicate, or ``None`` when the
            predicate was deliberately not run (terminal leaf).
        terminal: True for cancellation / system-termination leaves
            (``KeyboardInterrupt``, ``SystemExit``, ``GeneratorExit``,
            ``asyncio.CancelledError``) which business rules must never
            swallow into a retry.
    """

    path: tuple[int, ...]
    exception: BaseException
    matched: bool | None
    terminal: bool


class ExceptionGroupRetryReport(typing.NamedTuple):
    """Explanation of a :class:`retry_if_exception_group` decision.

    Attributes:
        decision: The boolean returned to the retry machinery.
        mode: ``"any"`` or ``"all"``.
        verdicts: One :class:`LeafVerdict` per leaf, in traversal order.
        deciding_path: Path of the leaf that determined the decision
            (first match for ``any``, first non-match for ``all``, first
            terminal leaf when blocked, last leaf for a fully-matching
            ``all``); ``None`` when no single leaf decided (e.g. ``any``
            with no match at all).
        blocked_paths: Paths of terminal leaves that forced ``decision``
            to ``False`` regardless of the business predicate.
    """

    decision: bool
    mode: str
    verdicts: tuple[LeafVerdict, ...]
    deciding_path: tuple[int, ...] | None
    blocked_paths: tuple[tuple[int, ...], ...]

    @property
    def explanation(self) -> str:
        """Human-readable reason naming the leaf that decided."""
        if self.blocked_paths:
            return (
                f"retry denied: terminal leaf at path {self.blocked_paths[0]} "
                "(cancellation/system-exit exceptions are never retried)"
            )
        if self.deciding_path is None:
            return (
                f"retry {'allowed' if self.decision else 'denied'}: "
                f"mode={self.mode!r}, no single deciding leaf "
                f"({len(self.verdicts)} leaves evaluated)"
            )
        verdict = next(v for v in self.verdicts if v.path == self.deciding_path)
        return (
            f"retry {'allowed' if self.decision else 'denied'}: "
            f"mode={self.mode!r}, decided by leaf at path "
            f"{self.deciding_path} ({type(verdict.exception).__name__}, "
            f"matched={verdict.matched})"
        )


class retry_if_exception_group(retry_base):
    """Retries based on the real leaf exceptions of an ``ExceptionGroup``.

    The leaf ``predicate`` is applied to every non-terminal leaf exception
    of the group (nested groups are flattened, keeping index paths) and the
    per-leaf booleans are combined with ``mode``:

    - ``mode="any"``: retry if at least one leaf matches;
    - ``mode="all"``: retry only if every leaf matches.

    Rules (deliberately strict, unlike :class:`retry_if_exception`):

    - The raised exception must be a ``BaseExceptionGroup``; anything else
      (plain exceptions, unknown cause chains) raises
      :class:`UndecidableRetryError` instead of guessing.
    - Terminal leaves (``KeyboardInterrupt``, ``SystemExit``,
      ``GeneratorExit``, ``asyncio.CancelledError``) are never passed to
      the business predicate and force the whole decision to ``False``:
      cancellation and system termination must not be swallowed as retries.
    - Each leaf predicate runs exactly once per leaf per call.
    - Exceptions raised by the predicate itself propagate unchanged; they
      are never converted into a ``False`` ("do not retry") answer.
    - The decision is explained via :attr:`last_report`, including the
      index path of the leaf that decided.

    The default :class:`retry_if_exception` behaviour is untouched.
    """

    TERMINAL_EXCEPTION_TYPES: tuple[type[BaseException], ...] = (
        KeyboardInterrupt,
        SystemExit,
        GeneratorExit,
        asyncio.CancelledError,
    )

    def __init__(
        self,
        predicate: typing.Callable[[BaseException], bool],
        mode: str = "any",
    ) -> None:
        if mode not in ("any", "all"):
            raise ValueError(f"mode must be 'any' or 'all', got {mode!r}")
        self.predicate = predicate
        self.mode = mode
        self.last_report: ExceptionGroupRetryReport | None = None

    def _iter_leaves(
        self,
        exc: BaseException,
        path: tuple[int, ...],
    ) -> typing.Iterator[tuple[tuple[int, ...], BaseException]]:
        if _BaseExceptionGroup is not None and isinstance(exc, _BaseExceptionGroup):
            group = typing.cast("_ExceptionGroupLike", exc)
            for index, sub in enumerate(group.exceptions):
                yield from self._iter_leaves(sub, (*path, index))
        else:
            yield path, exc

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome is None:
            raise RuntimeError("__call__() called before outcome was set")

        if not retry_state.outcome.failed:
            return False

        exception = retry_state.outcome.exception()
        if exception is None:
            raise RuntimeError("outcome failed but the exception is None")

        if _BaseExceptionGroup is None or not isinstance(
            exception, _BaseExceptionGroup
        ):
            raise UndecidableRetryError(
                f"{self.__class__.__name__} only judges BaseExceptionGroup "
                f"instances; got {type(exception).__name__}. Use "
                "retry_if_exception for plain exceptions."
            )

        verdicts: list[LeafVerdict] = []
        blocked_paths: list[tuple[int, ...]] = []
        for path, leaf in self._iter_leaves(exception, ()):
            if isinstance(leaf, self.TERMINAL_EXCEPTION_TYPES):
                # Never let business rules see cancellation/system-exit
                # leaves: they cannot be "retried away".
                verdicts.append(LeafVerdict(path, leaf, None, terminal=True))
                blocked_paths.append(path)
                continue
            # Predicate errors propagate: a failing predicate must not be
            # silently read as "do not retry".
            matched = bool(self.predicate(leaf))
            verdicts.append(LeafVerdict(path, leaf, matched, terminal=False))

        deciding_path: tuple[int, ...] | None
        if blocked_paths:
            decision = False
            deciding_path = blocked_paths[0]
        elif self.mode == "any":
            decision = any(v.matched for v in verdicts)
            deciding_path = next((v.path for v in verdicts if v.matched), None)
        else:
            decision = all(v.matched for v in verdicts)
            if decision:
                deciding_path = verdicts[-1].path if verdicts else None
            else:
                deciding_path = next((v.path for v in verdicts if not v.matched), None)

        self.last_report = ExceptionGroupRetryReport(
            decision=decision,
            mode=self.mode,
            verdicts=tuple(verdicts),
            deciding_path=deciding_path,
            blocked_paths=tuple(blocked_paths),
        )
        return decision


class retry_if_result(retry_base):
    """Retries if the result verifies a predicate."""

    def __init__(self, predicate: typing.Callable[[typing.Any], bool]) -> None:
        self.predicate = predicate

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome is None:
            raise RuntimeError("__call__() called before outcome was set")

        if not retry_state.outcome.failed:
            return self.predicate(retry_state.outcome.result())
        return False


class retry_if_not_result(retry_base):
    """Retries if the result refutes a predicate."""

    def __init__(self, predicate: typing.Callable[[typing.Any], bool]) -> None:
        self.predicate = predicate

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome is None:
            raise RuntimeError("__call__() called before outcome was set")

        if not retry_state.outcome.failed:
            return not self.predicate(retry_state.outcome.result())
        return False


class retry_if_exception_message(retry_if_exception):
    """Retries if an exception message equals or matches."""

    def __init__(
        self,
        message: str | None = None,
        match: str | re.Pattern[str] | None = None,
    ) -> None:
        if message is not None and match is not None:
            raise TypeError(
                f"{self.__class__.__name__}() takes either 'message' or 'match', not both"
            )

        if message is None and match is None:
            raise TypeError(
                f"{self.__class__.__name__}() missing 1 required argument 'message' or 'match'"
            )

        self.message = message
        self.match: re.Pattern[str] | None = (
            re.compile(match) if match is not None else None
        )
        super().__init__(self._check)

    def _check(self, exception: BaseException) -> bool:
        if self.message is not None:
            return self.message == str(exception)
        assert self.match is not None
        return bool(self.match.match(str(exception)))


class retry_if_not_exception_message(retry_if_exception_message):
    """Retries until an exception message equals or matches."""

    @override
    def _check(self, exception: BaseException) -> bool:
        return not super()._check(exception)

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome is None:
            raise RuntimeError("__call__() called before outcome was set")

        if not retry_state.outcome.failed:
            return True

        exception = retry_state.outcome.exception()
        if exception is None:
            raise RuntimeError("outcome failed but the exception is None")
        return self.predicate(exception)


class retry_any(retry_base):
    """Retries if any of the retries condition is valid."""

    def __init__(self, *retries: "RetryBaseT") -> None:
        self.retries = retries

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        return any(r(retry_state) for r in self.retries)

    @override
    def __ror__(self, other: "RetryBaseT") -> "retry_any":
        if isinstance(other, retry_any):
            return retry_any(*other.retries, *self.retries)
        return retry_any(other, *self.retries)


class retry_all(retry_base):
    """Retries if all the retries condition are valid."""

    def __init__(self, *retries: "RetryBaseT") -> None:
        self.retries = retries

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        return all(r(retry_state) for r in self.retries)

    @override
    def __rand__(self, other: "RetryBaseT") -> "retry_all":
        if isinstance(other, retry_all):
            return retry_all(*other.retries, *self.retries)
        return retry_all(other, *self.retries)

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
import dataclasses
import re
import typing

from tenacity._utils import override

if typing.TYPE_CHECKING:
    from tenacity import RetryCallState

# Exception groups are a Python 3.11 feature; on older interpreters use the
# ``exceptiongroup`` backport when it is available. When neither is available
# the predicate refuses to judge rather than guessing.
try:
    from exceptiongroup import BaseExceptionGroup  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    BaseExceptionGroup = getattr(builtins, "BaseExceptionGroup", None)  # type: ignore[assignment]


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


#: Exceptions that represent cancellation or interpreter/system shutdown and
#: that business retry rules must never be able to turn into a retry.
DEFAULT_FATAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    asyncio.CancelledError,
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
)


class NotAnExceptionGroup(TypeError):
    """Raised when a retry predicate is asked to judge an exception that is
    not an :class:`ExceptionGroup` (or :class:`BaseExceptionGroup`).

    The group predicates are explicit about the chain they understand: rather
    than guessing on an unknown exception chain, they refuse to judge.
    """


@dataclasses.dataclass(frozen=True)
class ExceptionGroupLeaf:
    """A single leaf of an exception group as evaluated by
    :class:`retry_if_exception_group`.

    :param path: indexes of the nested groups that lead to this leaf
    :param exception: the real leaf exception
    :param result: the predicate result for this leaf, or ``None`` when the
        predicate was not consulted because the leaf is a fatal/system exception
    :param fatal: whether this leaf is a fatal/system exception that vetoed
        the retry without the predicate being consulted
    """

    path: tuple[int, ...]
    exception: BaseException
    result: bool | None
    fatal: bool = False


@dataclasses.dataclass(frozen=True)
class ExceptionGroupDecision:
    """Explanation of a :class:`retry_if_exception_group` decision.

    :param retry: whether the group should be retried
    :param mode: ``"any"`` or ``"all"``
    :param fatal: whether a fatal/system leaf vetoed the retry
    :param leaves: every leaf, with the result of the predicate for each
    """

    retry: bool
    mode: str
    fatal: bool
    leaves: tuple[ExceptionGroupLeaf, ...]

    @property
    def matched(self) -> tuple[ExceptionGroupLeaf, ...]:
        """Leaves for which the predicate returned ``True``."""
        return tuple(leaf for leaf in self.leaves if leaf.result is True)

    @property
    def rejected(self) -> tuple[ExceptionGroupLeaf, ...]:
        """Leaves for which the predicate returned ``False``."""
        return tuple(leaf for leaf in self.leaves if leaf.result is False)

    @property
    def fatal_leaves(self) -> tuple[ExceptionGroupLeaf, ...]:
        """Fatal/system leaves that were not submitted to the predicate."""
        return tuple(leaf for leaf in self.leaves if leaf.fatal)


def _iter_group_leaves(
    exception: BaseExceptionGroup,
    path: tuple[int, ...] = (),
) -> typing.Iterator[tuple[tuple[int, ...], BaseException]]:
    """Yield ``(path, leaf_exception)`` for every leaf of an exception group.

    Nested groups are flattened in order; the path records the index of each
    nested group on the way to the leaf, so the exact position of every leaf
    in the original group is preserved.
    """
    for index, sub in enumerate(exception.exceptions):
        sub_path = (*path, index)
        if isinstance(sub, BaseExceptionGroup):
            yield from _iter_group_leaves(sub, sub_path)
        else:
            yield sub_path, sub


class retry_if_exception_group(retry_base):
    """Retry strategy that evaluates the real leaves of an exception group.

    The predicate is called once per leaf exception (a leaf appearing more
    than once in the group is only evaluated a single time); nested groups are
    flattened and each leaf keeps the path of group indexes that led to it.

    :param predicate: callable invoked with each leaf exception; must return
        a truthy value to request a retry for that leaf
    :param mode: ``"any"`` to retry as soon as one leaf matches, ``"all"`` to
        only retry when every leaf matches
    :param fatal_exceptions: exception types that represent cancellation or
        system termination; when such a leaf is present the group is never
        retried and the predicate is not consulted for any leaf

    The predicate is an explicit group predicate: if the raised exception is
    not a :class:`BaseExceptionGroup`, :class:`NotAnExceptionGroup` is raised
    instead of guessing. If the predicate itself raises, the exception
    propagates -- it is never swallowed and turned into a ``False`` decision.

    The latest decision is explained on the :attr:`last_decision` attribute.
    """

    def __init__(
        self,
        predicate: typing.Callable[[BaseException], bool],
        mode: str = "any",
        *,
        fatal_exceptions: tuple[type[BaseException], ...] = DEFAULT_FATAL_EXCEPTIONS,
    ) -> None:
        if mode not in ("any", "all"):
            raise ValueError(
                f"{self.__class__.__name__}() mode must be 'any' or 'all', got {mode!r}"
            )
        self.predicate = predicate
        self.mode = mode
        self.fatal_exceptions = fatal_exceptions
        self.last_decision: ExceptionGroupDecision | None = None

    @override
    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome is None:
            raise RuntimeError("__call__() called before outcome was set")

        if not retry_state.outcome.failed:
            return False

        exception = retry_state.outcome.exception()
        if exception is None:
            raise RuntimeError("outcome failed but the exception is None")

        if BaseExceptionGroup is None or not isinstance(exception, BaseExceptionGroup):
            # This predicate is explicit about the chain it understands:
            # rather than guessing on an unknown exception chain, refuse to
            # judge.
            raise NotAnExceptionGroup(
                f"{self.__class__.__name__}() expects an ExceptionGroup, "
                f"got {type(exception).__name__}"
            )

        leaves = tuple(_iter_group_leaves(exception))

        # Cancellation and system-termination exceptions must never be
        # swallowed into a retry by a business rule: veto without consulting
        # the predicate for any leaf.
        if any(
            isinstance(leaf_exception, self.fatal_exceptions)
            for _, leaf_exception in leaves
        ):
            decision = ExceptionGroupDecision(
                retry=False,
                mode=self.mode,
                fatal=True,
                leaves=tuple(
                    ExceptionGroupLeaf(
                        path,
                        leaf_exception,
                        None,
                        isinstance(leaf_exception, self.fatal_exceptions),
                    )
                    for path, leaf_exception in leaves
                ),
            )
            self.last_decision = decision
            return False

        # Evaluate every leaf exactly once. A leaf object appearing more than
        # once in the group is only evaluated a single time. A predicate that
        # raises propagates: its failure must not be faked as a False result.
        results: dict[int, bool] = {}
        evaluated: list[ExceptionGroupLeaf] = []
        for path, leaf_exception in leaves:
            leaf_id = id(leaf_exception)
            if leaf_id not in results:
                results[leaf_id] = self.predicate(leaf_exception)
            evaluated.append(ExceptionGroupLeaf(path, leaf_exception, results[leaf_id]))

        if self.mode == "any":
            retry = any(leaf.result for leaf in evaluated)
        else:
            retry = all(leaf.result for leaf in evaluated)

        decision = ExceptionGroupDecision(
            retry=retry,
            mode=self.mode,
            fatal=False,
            leaves=tuple(evaluated),
        )
        self.last_decision = decision
        return retry

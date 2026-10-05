# Copyright 2017 Elisey Zanko
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

import sys
import typing

from tornado import gen

from tenacity import (
    AttemptEventType,
    BaseRetrying,
    DoAttempt,
    DoSleep,
    RetryCallState,
)
from tenacity._utils import override
from tenacity.observer import is_cancellation

if typing.TYPE_CHECKING:
    from tornado.concurrent import Future

_RetValT = typing.TypeVar("_RetValT")


class TornadoRetrying(BaseRetrying):
    sleep: typing.Callable[..., "Future[None]"]

    def __init__(
        self,
        sleep: "typing.Callable[[float], Future[None]]" = gen.sleep,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sleep = sleep

    @gen.coroutine
    @override
    def __call__(  # type: ignore[override]
        self,
        fn: "typing.Callable[..., typing.Generator[typing.Any, typing.Any, _RetValT] | Future[_RetValT]]",
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> "typing.Generator[typing.Any, typing.Any, _RetValT]":
        self.begin()

        retry_state = RetryCallState(retry_object=self, fn=fn, args=args, kwargs=kwargs)
        while True:
            try:
                do = self.iter(retry_state=retry_state)
            except BaseException:
                # The retry pipeline bailed out (RetryError, reraise, ...):
                # close the event stream before propagating.
                self._emit_attempt_event(retry_state.call_id, AttemptEventType.CALL_END)
                raise
            if isinstance(do, DoAttempt):
                self._emit_attempt_event(
                    retry_state.call_id, AttemptEventType.ATTEMPT_START
                )
                try:
                    result = yield fn(*args, **kwargs)
                except BaseException as exc:
                    if is_cancellation(exc):
                        self._emit_attempt_event(
                            retry_state.call_id,
                            AttemptEventType.CANCELLED_DURING_ATTEMPT,
                        )
                    retry_state.set_exception(sys.exc_info())  # type: ignore[arg-type]
                else:
                    retry_state.set_result(result)
            elif isinstance(do, DoSleep):
                retry_state.prepare_for_next_attempt()
                self._emit_attempt_event(
                    retry_state.call_id, AttemptEventType.WAIT_START
                )
                try:
                    yield self.sleep(do)
                except BaseException as exc:
                    if is_cancellation(exc):
                        self._emit_attempt_event(
                            retry_state.call_id,
                            AttemptEventType.CANCELLED_DURING_WAIT,
                        )
                    # The wait never completed, so the call ends here.
                    self._emit_attempt_event(
                        retry_state.call_id, AttemptEventType.CALL_END
                    )
                    raise
                self._emit_attempt_event(retry_state.call_id, AttemptEventType.WAIT_END)
            else:
                self._emit_attempt_event(retry_state.call_id, AttemptEventType.CALL_END)
                raise gen.Return(do)

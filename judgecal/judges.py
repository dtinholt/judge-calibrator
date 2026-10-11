"""Judges: anything that returns a verdict for an item under a rubric."""
from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

from .items import Item, _id, read_jsonl
from .rubric import Rubric


@dataclass(frozen=True)
class Verdict:
    item: str
    judge: str
    repeat: int
    cls: Optional[str]  # None when the reply could not be read as one of the rubric's classes
    confidence: Optional[str] = None
    raw: str = ""
    input_tokens: Optional[int] = None  # usage.prompt_tokens from the endpoint, when it sent one
    output_tokens: Optional[int] = None  # usage.completion_tokens
    failed: bool = False  # True when the call failed after its retries (raw holds the error)


@runtime_checkable
class Judge(Protocol):
    name: str

    def judge(self, item: Item, rubric: Rubric, repeat: int) -> Optional[Verdict]:
        """Return a verdict, or None when this judge has nothing for the item on this repeat."""
        ...


# A transport takes (url, headers, body) and returns (status, response body).
Transport = Callable[[str, dict, bytes], tuple[int, bytes]]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect. urllib's default handler would re-send the request, Bearer
    header included, to whatever origin the Location names, and turn a POST into a GET on
    301, 302 and 303. With this handler a 3xx reaches the caller as an HTTPError."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _open(req: urllib.request.Request, timeout: float):
    return _OPENER.open(req, timeout=timeout)


def urllib_transport(timeout: float = 120.0) -> Transport:
    """The default transport. It never follows redirects: a 3xx comes back as its status
    code, which ChatCompletionsJudge treats as an error. It accepts a `timeout` keyword per
    call (its `accepts_timeout` attribute says so), which ChatCompletionsJudge uses to keep
    each attempt inside the item's deadline."""
    def send(url: str, headers: dict, body: bytes, timeout: float = timeout) -> tuple[int, bytes]:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with _open(req, timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read()

    send.accepts_timeout = True  # type: ignore[attr-defined]
    return send


def _tokens(usage: object, key: str) -> Optional[int]:
    if not isinstance(usage, dict):
        return None
    v = usage.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v < 0 or v == float("inf"):
        return None
    return int(v)


class JudgeError(RuntimeError):
    pass


class ChatCompletionsJudge:
    """Posts to `{base_url}/chat/completions` on any endpoint that speaks the chat-completions protocol.

    The key is read from the environment variable named by `api_key_env` at call time, so it
    never sits in a config file. Pass `api_key_env=None` for a local server that takes no key.
    Each repeat is a fresh, independent request; nothing from one repeat reaches the next.

    `item_timeout` caps the wall time one item may take, retries and backoff included. When
    the next attempt or backoff would cross it the call fails as a timeout, and under
    Calibration's default the item is recorded as unreadable. With the default transport each
    attempt's socket timeout is also cut to the time left. `concurrent = True` tells
    Calibration it may call this judge from a thread pool; the object holds no state that
    changes between calls.
    """

    concurrent = True

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: Optional[str] = "JUDGE_API_KEY",
        name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 400,
        transport: Optional[Transport] = None,
        retries: int = 3,
        backoff: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        item_timeout: Optional[float] = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if temperature is not None and not (math.isfinite(temperature) and temperature >= 0):
            raise ValueError("temperature must be a finite number of 0 or more")
        if item_timeout is not None and not (math.isfinite(item_timeout) and item_timeout > 0):
            raise ValueError("item_timeout must be a positive number of seconds")
        if retries < 0:
            raise ValueError("retries must be 0 or more")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.name = name or model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.transport = transport or urllib_transport()
        self.retries = retries
        self.backoff = backoff
        self.sleep = sleep
        self.item_timeout = item_timeout
        self.clock = clock

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise JudgeError(f"environment variable {self.api_key_env} is not set")
            h["Authorization"] = f"Bearer {key}"
        return h

    def request_body(self, item: Item, rubric: Rubric) -> dict:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": rubric.system_prompt()},
                {"role": "user", "content": item.text},
            ],
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return body

    def judge(self, item: Item, rubric: Rubric, repeat: int) -> Optional[Verdict]:
        """One verdict. Raises JudgeError once every retry is spent.

        Transport failures (refused or reset connections, timeouts, DNS errors) and HTTP 408,
        429 and 5xx are retried `retries` times with exponential backoff. A 200 whose body is
        malformed gets one extra attempt of its own. Under Calibration's default
        on_error="record" the final JudgeError becomes an unreadable verdict.
        """
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(self.request_body(item, rubric), allow_nan=False).encode()
        headers = self._headers()
        deadline = None if self.item_timeout is None else self.clock() + self.item_timeout
        pass_timeout = getattr(self.transport, "accepts_timeout", False)
        attempt = 0  # retryable failures used
        malformed_retry_used = False

        def left() -> Optional[float]:
            return None if deadline is None else deadline - self.clock()

        def backoff_or_raise(why: str) -> None:
            nonlocal attempt
            wait = self.backoff * (2 ** attempt)
            rem = left()
            if rem is not None and wait >= rem:
                raise JudgeError(f"{self.name}: item timeout ({self.item_timeout:g} s) after {why}")
            self.sleep(wait)
            attempt += 1

        while True:
            rem = left()
            if rem is not None and rem <= 0:
                raise JudgeError(f"{self.name}: item timeout ({self.item_timeout:g} s)")
            try:
                if pass_timeout and rem is not None:
                    status, payload = self.transport(url, headers, data, timeout=rem)
                else:
                    status, payload = self.transport(url, headers, data)
            except (OSError, TimeoutError, ConnectionError) as e:  # URLError is an OSError
                if attempt < self.retries:
                    backoff_or_raise(f"transport failure: {e}")
                    continue
                raise JudgeError(f"{self.name}: transport failure after {attempt + 1} attempts: {e}") from e
            if status == 200:
                try:
                    body = json.loads(payload)
                    text = body["choices"][0]["message"]["content"] or ""
                    if not isinstance(text, str):
                        raise TypeError("content is not a string")
                except (ValueError, KeyError, IndexError, TypeError, RecursionError) as e:
                    if not malformed_retry_used:
                        malformed_retry_used = True
                        continue
                    raise JudgeError(f"{self.name}: unreadable response: {e}") from e
                cls_, conf = rubric.parse(text)
                usage = body.get("usage") if isinstance(body, dict) else None
                return Verdict(item.id, self.name, repeat, cls_, conf, text,
                               _tokens(usage, "prompt_tokens"), _tokens(usage, "completion_tokens"))
            if 300 <= status < 400:
                raise JudgeError(f"{self.name}: HTTP {status} redirect refused; point base_url at the "
                                 f"final address (redirects would re-send the key to another origin)")
            if (status in (408, 429) or status >= 500) and attempt < self.retries:
                backoff_or_raise(f"HTTP {status}")
                continue
            raise JudgeError(f"{self.name}: HTTP {status}: {payload[:200]!r}")



class ReplayJudge:
    """Reads recorded verdicts from JSONL instead of calling a model.

    Each row: `item` (or `packet`, or `id`; a trailing ".md" is dropped, as for every id in
    judgecal), `class`, optional `confidence`, optional
    `repeat` (default 0), optional `judge`. Several paths may be given; with
    `repeat_per_file=True` the n-th file supplies repeat n and any `repeat` field in it is
    ignored. With `judge_field` set, only rows whose `judge` equals it are read, so the file
    written by `judgecal report --verdicts-out` replays directly. Set `fixture=True` when the
    file is a recorded test fixture; reports built from it then carry the fixture label.
    Optional `input_tokens` and `output_tokens` (written by --verdicts-out) are read back so
    a replayed run can still show what its original calls cost.
    """

    replayed = True

    ID_KEYS = ("item", "packet", "id")

    def __init__(self, paths, name: str, repeat_per_file: bool = False, judge_field: Optional[str] = None,
                 fixture: bool = False):
        self.fixture = fixture
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.name = name
        self._rows: dict[tuple[str, int], dict] = {}
        for i, p in enumerate(paths):
            for n, d in read_jsonl(p):
                if judge_field is not None and d.get("judge") != judge_field:
                    continue
                key = next((d[k] for k in self.ID_KEYS if k in d), None)
                if key is None:
                    raise ValueError(f"{p}:{n}: row has no item id")
                key = _id(key, f"{p}:{n}")
                rep = i if repeat_per_file else _repeat(d.get("repeat", 0), f"{p}:{n}")
                if (key, rep) in self._rows:
                    raise ValueError(f"{p}:{n}: duplicate verdict for {key} repeat {rep}")
                self._rows[(key, rep)] = d

    def repeats_available(self) -> int:
        return 1 + max((r for _, r in self._rows), default=-1)

    def judge(self, item: Item, rubric: Rubric, repeat: int) -> Optional[Verdict]:
        d = self._rows.get((item.id, repeat))
        if d is None:
            return None
        conf = d.get("confidence")
        return Verdict(
            item.id,
            self.name,
            repeat,
            rubric.normalise(d.get("class")),
            None if conf is None else str(conf).strip().lower(),
            json.dumps(d),
            _tokens(d, "input_tokens"),
            _tokens(d, "output_tokens"),
        )


def _repeat(value: object, where: str) -> int:
    """A repeat index must be a whole number of 0 or more ("2" and 2.0 are accepted)."""
    try:
        if isinstance(value, bool):
            raise ValueError
        f = float(value)  # type: ignore[arg-type]
        if not f.is_integer() or f < 0:
            raise ValueError
        return int(f)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{where}: repeat must be a whole number of 0 or more, got {value!r}") from None

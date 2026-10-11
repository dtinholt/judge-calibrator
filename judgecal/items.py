"""Items: what the judges are asked to classify, one JSON object per line."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


def read_jsonl(path: str | Path) -> Iterator[tuple[int, dict]]:
    """Yield (line number, object) for every non-blank line of a JSONL file.

    Reads UTF-8 with or without a byte-order mark. Lines end at "\\n" only, with any "\\r"
    stripped, so CRLF, CR-LF mixes and raw U+2028 or U+0085 inside a JSON string all read
    cleanly (str.splitlines would cut a line at those). Every failure is a ValueError that
    names the file and line: malformed JSON, nesting too deep to decode, or a row that is not
    a JSON object.
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    for n, line in enumerate(text.split("\n"), 1):
        line = line.strip(" \t\r\f\v")
        if not line:
            continue
        try:
            d = json.loads(line)
        except RecursionError:
            raise ValueError(f"{path}:{n}: JSON nested too deeply to read") from None
        except ValueError as e:
            raise ValueError(f"{path}:{n}: not valid JSON ({e.msg} at column {e.colno})") from None
        if not isinstance(d, dict):
            raise ValueError(f"{path}:{n}: each line must be a JSON object, got {type(d).__name__}")
        yield n, d


def _id(value: object, where: str) -> str:
    """Every item id in judgecal goes through here: items, replayed verdicts, strata and
    saved verdict files. A trailing ".md" is dropped, so "a1.md" and "a1" name one item (case
    files are often keyed by file name in one place and by stem in another)."""
    if value is None or isinstance(value, (dict, list, bool)):
        raise ValueError(f"{where}: id must be a string or number, got {json.dumps(value)[:40]}")
    s = str(value)
    if s.endswith(".md"):
        s = s[:-3]
    if not s.strip():
        raise ValueError(f"{where}: id is empty")
    return s


@dataclass(frozen=True)
class Item:
    id: str
    text: str = ""
    meta: dict = field(default_factory=dict, compare=False, hash=False)


class Items:
    """An ordered set of items with unique ids.

    Each JSONL row needs an `id` (a trailing ".md" is dropped). The text the judge sees comes from `text`, or `input`, or
    `prompt`, in that order. Every other key is kept in `meta` and never sent to a judge.
    """

    TEXT_KEYS = ("text", "input", "prompt")

    def __init__(self, items: list[Item]):
        seen = set()
        for it in items:
            if it.id in seen:
                raise ValueError(f"duplicate item id {it.id!r}")
            seen.add(it.id)
        self._items = list(items)

    @classmethod
    def load(cls, path: str | Path) -> "Items":
        out = []
        seen: dict[str, int] = {}
        for n, d in read_jsonl(path):
            if "id" not in d:
                raise ValueError(f"{path}:{n}: row has no 'id'")
            iid = _id(d["id"], f"{path}:{n}")
            if iid in seen:
                raise ValueError(f"{path}:{n}: duplicate item id {iid!r} (first on line {seen[iid]})")
            seen[iid] = n
            text = next((d[k] for k in cls.TEXT_KEYS if k in d), "")
            meta = {k: v for k, v in d.items() if k != "id" and k not in cls.TEXT_KEYS}
            out.append(Item(iid, "" if text is None else str(text), meta))
        if not out:
            raise ValueError(f"{path}: no items (the file is empty or holds only blank lines)")
        return cls(out)

    def __iter__(self) -> Iterator[Item]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def ids(self) -> list[str]:
        return [i.id for i in self._items]


def load_strata(path: str | Path) -> dict[str, str]:
    """Read a user-supplied JSONL mapping item id to stratum label.

    Each row: `id` (or `item`) and `stratum`. The labels come from the user's own design
    (task family, source, difficulty set in advance); judgecal never derives them from any
    verdict or reference answer. Duplicate ids and empty labels are errors.
    """
    out: dict[str, str] = {}
    for n, d in read_jsonl(path):
        key = d.get("id", d.get("item"))
        iid = _id(key, f"{path}:{n}")
        if "stratum" not in d:
            raise ValueError(f"{path}:{n}: row has no 'stratum'")
        lab = d["stratum"]
        if lab is None or isinstance(lab, (dict, list)) or not str(lab).strip():
            raise ValueError(f"{path}:{n}: stratum must be a non-empty string")
        if iid in out:
            raise ValueError(f"{path}:{n}: duplicate stratum row for item {iid!r}")
        out[iid] = str(lab).strip()
    if not out:
        raise ValueError(f"{path}: no strata rows")
    return out

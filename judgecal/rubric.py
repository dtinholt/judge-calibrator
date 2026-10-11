"""Rubric: the classes a judge may choose and the instruction it is given."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_FORMAT = (
    "Answer with: CLASS: <{classes}>, then CONFIDENCE: <0-100>, then one sentence of reasoning."
)


@dataclass
class Rubric:
    """Classes in display order, an optional one-line meaning for each, and the instruction.

    `instruction` is the text the judge sees before the item. `answer_format` tells the judge
    how to reply; the default asks for CLASS and CONFIDENCE lines, which `parse` reads back.
    """

    classes: list[str]
    instruction: str = ""
    descriptions: dict[str, str] = field(default_factory=dict)
    name: str = "rubric"
    answer_format: str = DEFAULT_FORMAT

    def __post_init__(self) -> None:
        if isinstance(self.classes, (str, bytes)) or not hasattr(self.classes, "__iter__"):
            raise ValueError("rubric classes must be a list of names")
        self.classes = [str(c).strip().upper() for c in self.classes]
        if any(not c for c in self.classes):
            raise ValueError("rubric classes must be non-empty names")
        if len(self.classes) < 2:
            raise ValueError("a rubric needs at least two classes")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("rubric classes must be unique")
        self.descriptions = {str(k).strip().upper(): v for k, v in self.descriptions.items()}

    @classmethod
    def load(cls, path: str | Path) -> "Rubric":
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except RecursionError:
            raise ValueError(f"{path}: JSON nested too deeply to read") from None
        except ValueError as e:
            raise ValueError(f"{path}: not valid JSON ({e})") from None
        if not isinstance(d, dict) or "classes" not in d:
            raise ValueError(f"{path}: a rubric file is a JSON object with a 'classes' list")
        desc = d.get("descriptions", {})
        if not isinstance(desc, dict):
            raise ValueError(f"{path}: 'descriptions' must be an object of class: meaning")
        fmt = str(d.get("answer_format", DEFAULT_FORMAT))
        try:
            fmt.format(classes="")
        except (KeyError, IndexError, ValueError):
            raise ValueError(f"{path}: answer_format may hold only the {{classes}} placeholder; "
                             f"write literal braces as {{{{ and }}}}") from None
        return cls(
            classes=d["classes"],
            instruction=str(d.get("instruction", "")),
            descriptions={k: str(v) for k, v in desc.items()},
            name=str(d.get("name", Path(path).stem)),
            answer_format=fmt,
        )

    def system_prompt(self) -> str:
        lines = [self.instruction.strip()] if self.instruction.strip() else []
        lines.append("Classes:")
        for c in self.classes:
            desc = self.descriptions.get(c, "")
            lines.append(f"- {c}: {desc}" if desc else f"- {c}")
        lines.append(self.answer_format.format(classes="|".join(self.classes)))
        return "\n".join(lines)

    def normalise(self, label: object) -> Optional[str]:
        s = str(label or "").strip().upper()
        return s if s in self.classes else None

    def parse(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """Read (class, confidence) from a judge reply. Either may be None."""
        cls_ = conf = None
        t = text.strip()
        if t.startswith("{"):
            try:
                d = json.loads(t)
                cls_ = self.normalise(d.get("class"))
                c = d.get("confidence")
                conf = None if c is None else str(c).strip().lower()
                if cls_:
                    return cls_, conf
            except (ValueError, AttributeError):
                pass
        alts = "|".join(re.escape(c) for c in sorted(self.classes, key=len, reverse=True))
        m = re.search(rf"CLASS\s*[:=]\s*\**\s*({alts})\b", t, re.IGNORECASE)
        if m:
            cls_ = m.group(1).upper()
        m = re.search(r"CONFIDENCE\s*[:=]\s*\**\s*([A-Za-z-]+|\d+(?:\.\d+)?)", t, re.IGNORECASE)
        if m:
            conf = m.group(1).lower()
        return cls_, conf

import re


class ReplyPostProcessor:
    _line_patterns = [
        re.compile(r"^\s*THINK:.*$", re.IGNORECASE),
        re.compile(r"^\s*ACTION:.*$", re.IGNORECASE),
        re.compile(r"^\s*OBSERVE:.*$", re.IGNORECASE),
        re.compile(r"^\s*OBSERVATION:.*$", re.IGNORECASE),
    ]

    def clean(self, content: str | None) -> str:
        if not content:
            return ""

        lines = [
            line
            for line in content.splitlines()
            if not any(pattern.match(line.strip()) for pattern in self._line_patterns)
        ]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines).strip())

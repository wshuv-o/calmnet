r"""Make the mathtext literals in make_fig_arch.py raw strings.

Writing that file through a shell heredoc turned every "\\times" into "\times",
which Python then reads as a TAB, and "\rightarrow" into a carriage return. The
figure fails to render with an unhelpful mathtext ParseException.

This is the fifth time tonight that shell quoting has eaten a backslash. The fix
is mechanical: any double-quoted literal in the drawing code that contains a
backslash becomes a raw string, so the bytes reach matplotlib unchanged.
"""
from __future__ import annotations

import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "make_fig_arch.py")
BS = chr(92)


def main():
    s = io.open(SRC, encoding="utf-8").read()
    i = s.index("def main():")
    head, tail = s[:i], s[i:]

    def fix(m):
        body = m.group(1)
        return 'r"' + body + '"' if BS in body else m.group(0)

    # a literal already prefixed r or f is left alone
    tail2 = re.sub(r'(?<![rfRF])"([^"\n]*' + re.escape(BS) + r'[^"\n]*)"',
                   fix, tail)
    io.open(SRC, "w", encoding="utf-8", newline="\n").write(head + tail2)
    print("raw literals now: %d" % tail2.count('r"'))


if __name__ == "__main__":
    main()

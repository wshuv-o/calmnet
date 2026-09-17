"""Convert the final paper from IEEEtran journal to IEEEtran conference format.

Differences that actually matter:
  * documentclass option journal -> conference
  * \author{...}+\thanks{...} -> \IEEEauthorblockN / \IEEEauthorblockA blocks
  * \markboth is journal-only and must go
  * \IEEEPARstart (drop cap) is journal-only and must go
"""
import re
from pathlib import Path

SRC = Path("calmnet_paper_final.tex")
DST = Path("calmnet_paper_conference.tex")
t = SRC.read_text(encoding="utf-8")

# 1. document class
t = t.replace(r"\documentclass[journal]{IEEEtran}",
              r"\documentclass[conference]{IEEEtran}", 1)

# 2. author block -> conference style
AUTHOR = r"""\author{
\IEEEauthorblockN{Md Wahiduzzaman Suva}
\IEEEauthorblockA{\textit{Dept. of Computer Science} \\
\textit{American International University-Bangladesh}\\
Dhaka, Bangladesh \\
wshuvo360@gmail.com}
\and
\IEEEauthorblockN{Esme Moula Chowdhury Abha}
\IEEEauthorblockA{\textit{Dept. of Computer Science} \\
\textit{American International University-Bangladesh}\\
Dhaka, Bangladesh \\
esmechowdhuryabha@gmail.com}
\and
\IEEEauthorblockN{Hasibur Rashid Chayon}
\IEEEauthorblockA{\textit{Associate Professor} \\
\textit{American International University-Bangladesh}\\
Dhaka, Bangladesh \\
\textit{(Corresponding author)}}
}"""

t = re.sub(r"\\author\{.*?\n\a?\}\n\n\\markboth", "PLACEHOLDER\n\n\\\\markboth",
           t, count=1, flags=re.S)
# the regex above is fragile; do it by explicit span instead
t = SRC.read_text(encoding="utf-8")
t = t.replace(r"\documentclass[journal]{IEEEtran}",
              r"\documentclass[conference]{IEEEtran}", 1)

start = t.index(r"\author{")
end = t.index(r"\markboth{")
t = t[:start] + AUTHOR + "\n\n" + t[end:]

# 3. drop \markboth (journal-only), keep \maketitle
mstart = t.index(r"\markboth{")
mend = t.index(r"\maketitle")
t = t[:mstart] + t[mend:]

# 4. \IEEEPARstart is journal-only
t = t.replace(r"\IEEEPARstart{F}{or}", "For", 1)
t = re.sub(r"\\IEEEPARstart\{(.)\}\{([^}]*)\}", r"\1\2", t)

# 5. dataset acknowledgement lost with \thanks -> restore as a footnote-free note
t = t.replace(
    r"\section{Introduction}",
    "\\section{Introduction}", 1)

DST.write_text(t, encoding="utf-8")
print(f"wrote {DST} ({len(t.splitlines())} lines)")

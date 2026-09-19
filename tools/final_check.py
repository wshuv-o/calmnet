r"""Pre-submission check for the manuscript.

Each test here exists because the thing it tests broke at least once during
this project, silently, while the PDF still built:

  bibliography    a Conclusion rewrite once replaced everything to
                  \end{document}, taking the CRediT block, the declarations and
                  \bibliography with it. The build succeeded and produced a
                  23-page paper with no references.
  condition       a float-removal pass deleted the \begin{condition} block, so
                  ten cross-references resolved to ??.
  abstract        a bare % after a number commented out the rest of the line
                  and truncated the abstract. That happened three times.
  components      align, gate and the cross-epoch transformer are not in this
                  architecture and must not appear in the text.
  floats          every float needs a caption and a label, and every label
                  needs to resolve.

    python tools/final_check.py
"""
from __future__ import annotations

import io
import os
import re
import sys

import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
PDF = os.path.join(ROOT, "paper", "cas_calmnet.pdf")
BS = chr(92)

fails, warns = [], []


def check(ok, msg, hard=True):
    print("  %-4s %s" % ("ok" if ok else ("FAIL" if hard else "warn"), msg))
    if not ok:
        (fails if hard else warns).append(msg)


def main():
    tex = io.open(TEX, encoding="utf-8").read()
    doc = pymupdf.open(PDF)
    txt = "".join(doc[i].get_text() for i in range(doc.page_count))
    flat = re.sub(r"\s+", " ", txt)

    print("STRUCTURE")
    check(doc.page_count >= 15, "page count %d" % doc.page_count)
    n_ref = len(re.findall(r"\[\d+\]", txt))
    check(n_ref > 80, "%d bracketed citations rendered" % n_ref)
    check(txt.count("??") == 0, "%d unresolved cross-references"
          % txt.count("??"))
    for macro in ("bibliography", "end{document}"):
        check((BS + macro) in tex, "%s present in source" % macro)
    for name in ("CRediT", "Declaration of competing interest",
                 "Data availability"):
        check(name.lower() in flat.lower(), "%s section present" % name)
    check(BS + "begin{condition}" in tex, "Condition 1 block present")
    check("prop:band" in tex, "Condition 1 label present")

    print("\nFRONT MATTER")
    ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    check(bool(ab), "abstract environment found")
    if ab:
        n = len(ab.group(1).split())
        check(120 <= n <= 320, "abstract %d words" % n)
    check(BS + "begin{keywords}" in tex, "keywords present")
    check(BS + "begin{highlights}" in tex, "highlights present")

    print("\nCONTENT")
    for w in ("align + gate", "no-align", "selective head", "cross-epoch",
              "24,181"):
        check(w not in flat, "no %r in the rendered text" % w)
    check(flat.count("\u2014") == 0, "%d em-dashes rendered"
          % flat.count("\u2014"))
    check("290,943" in flat or "290{,}943" in tex, "parameter total stated")

    print("\nFLOATS")
    caps = len(re.findall(r"\\caption\{", tex)) + len(
        re.findall(r"\\caption\{", io.open(
            os.path.join(ROOT, "paper", "tables_auto.tex"),
            encoding="utf-8").read()))
    labs = set(re.findall(r"\\label\{((?:tab|fig):[^}]*)\}", tex))
    labs |= set(re.findall(r"\\label\{((?:tab|fig):[^}]*)\}", io.open(
        os.path.join(ROOT, "paper", "tables_auto.tex"),
        encoding="utf-8").read()))
    check(caps == len(labs), "%d captions, %d float labels" % (caps, len(labs)))
    refs = set(re.findall(r"\\ref\{((?:tab|fig):[^}]*)\}", tex))
    orphan = sorted(labs - refs)
    check(not orphan, "every float is referenced%s"
          % ("" if not orphan else "; unreferenced: %s" % orphan), hard=False)
    missing = sorted(refs - labs)
    check(not missing, "every float reference resolves%s"
          % ("" if not missing else "; missing: %s" % missing))

    print("\n%s" % ("FAILED: " + "; ".join(fails) if fails
                    else "all hard checks passed"))
    if warns:
        print("warnings: %s" % "; ".join(warns))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

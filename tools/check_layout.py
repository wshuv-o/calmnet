r"""Fail if anything in the compiled PDF overprints anything else.

Three tables in this paper have silently overflowed their column. Table 7 was
printed with its last column cut off, and Tables 12 and 13 printed on top of
each other, so the page read "Higher Cohortp". LaTeX reports an overfull hbox
as a warning among hundreds of others, and the PDF still builds, so the only
reliable check is to look at where the glyphs actually land.

Two tests, both on the rendered page:

  collisions  two words whose boxes overlap. Text set normally never overlaps,
              so any hit is one object printed over another.
  out of band the text block runs from about x=47 to x=548 on this template.
              A word outside that is in the margin or off the page.

    python tools/check_layout.py            # exits non-zero on any finding
"""
from __future__ import annotations

import os
import sys

import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "paper", "cas_calmnet.pdf")
LEFT, RIGHT = 40.0, 556.0          # text block, with a small tolerance
MIN_AREA = 2.0                     # ignore hairline touches from kerning


def is_math(t):
    """True for a token set in the maths font.

    The CAS class sets maths in Unicode math-italic and math-bold, so anything
    outside basic Latin is maths. Their reported boxes overlap neighbouring
    lines legitimately: a superscript, a summation limit or a fraction extends
    the box above the baseline while nothing is printed over anything. These
    are counted separately rather than reported as findings.
    """
    return any(ord(c) > 0x2000 or 0x1D400 <= ord(c) <= 0x1D7FF for c in t)


def boxes(page):
    out = []
    for w in page.get_text("words"):
        x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], w[4]
        if txt.strip():
            out.append((x0, y0, x1, y1, txt))
    return out


def main():
    doc = pymupdf.open(PDF)
    collisions, outside, mathy = [], [], []
    for pno in range(doc.page_count):
        page = doc[pno]
        ws = boxes(page)
        for x0, y0, x1, y1, t in ws:
            if x0 < LEFT or x1 > RIGHT:
                outside.append((pno + 1, t, round(x0), round(x1)))
        # sort by y so only nearby words are compared
        ws.sort(key=lambda w: (w[1], w[0]))
        for i, a in enumerate(ws):
            for b in ws[i + 1:]:
                if b[1] >= a[3]:
                    break                     # no vertical overlap beyond here
                ox = min(a[2], b[2]) - max(a[0], b[0])
                oy = min(a[3], b[3]) - max(a[1], b[1])
                if ox > 0 and oy > 0 and ox * oy > MIN_AREA:
                    row = (pno + 1, a[4], b[4], round(ox * oy, 1))
                    if is_math(a[4]) or is_math(b[4]):
                        mathy.append(row)
                    else:
                        collisions.append(row)

    print("pages %d" % doc.page_count)
    print("collisions: %d" % len(collisions))
    for c in collisions[:25]:
        print("   page %2d  %r over %r  (%.1f pt2)" % c)
    print("maths boxes ignored: %d "
          "(superscripts and summation limits overlap by design)" % len(mathy))
    print("outside the text block: %d" % len(outside))
    for o in outside[:25]:
        print("   page %2d  %r at x %d..%d" % o)

    if collisions or outside:
        sys.exit(1)
    print("\nno overprinting, nothing outside the text block")


if __name__ == "__main__":
    main()

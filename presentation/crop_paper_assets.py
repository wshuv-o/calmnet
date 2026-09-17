"""Crop tables and the architecture figure out of the compiled paper as images,
so the midterm deck shows exactly what the manuscript prints.

Each asset is found by a phrase from its caption; the crop runs from the caption
down to the last horizontal rule of the table (or to the figure's caption end),
within the caption's column.
"""
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "paper" / "cas_calmnet.pdf"
OUT = Path(__file__).resolve().parent / "paper_assets"
OUT.mkdir(exist_ok=True)

TABLES = {
    "tab_compare": "The proposed decoder and eight published decoders trained in one",
    "tab_ablation3": "Component ablation on cohort A over three",
    "tab_ratecurve": "Cohort B over three",
    "tab_prereg": "Predictions registered before any model",
    "tab_artefact": "Artefact control on cohort",
    "tab_drift": "Per-participant session drift",
    "tab_blocks": "block structure",
}


def find(doc, phrase):
    for page in doc:
        hits = page.search_for(phrase)
        if hits:
            return page, hits[0]
    raise SystemExit("not found: " + phrase)


def crop_table(doc, name, phrase, dpi=300):
    page, hit = find(doc, phrase)
    pw = page.rect.width
    # column: full width if the caption starts near the left margin and runs past
    # the middle, otherwise the half the caption sits in
    blocks = [b for b in page.get_text("blocks") if b[1] <= hit.y1 + 2 and b[3] >= hit.y0 - 2
              and b[0] <= hit.x0 + 2 and b[2] >= hit.x1 - 2]
    cap = fitz.Rect(blocks[0][:4]) if blocks else hit
    label_top = cap.y0 - 14                      # "Table N" sits just above
    wide = cap.x1 - cap.x0 > pw * 0.55
    x0, x1 = (cap.x0 - 4, cap.x1 + 4) if not wide else (40, pw - 40)
    rules = []
    for d in page.get_drawings():
        r = d["rect"]
        if r.height <= 2.5 and r.width > 60 and r.y0 > cap.y1 - 2 and r.x0 >= x0 - 30 and r.x1 <= x1 + 30:
            rules.append(r)
    rules.sort(key=lambda r: r.y0)
    # the table's own rules share the first rule's horizontal extent
    last = None
    for r in rules:
        if last is None:
            first, last = r, r
            continue
        if abs(r.x0 - first.x0) < 6 and abs(r.x1 - first.x1) < 6 and r.y0 - last.y0 < 200:
            last = r
        elif r.y0 - last.y0 >= 200:
            break
    bottom = (last.y1 + 4) if last else cap.y1 + 200
    clip = fitz.Rect(x0, label_top, x1, bottom)
    if last is not None:          # table body only: first rule to last rule
        clip = fitz.Rect(first.x0 - 4, first.y0 - 4, first.x1 + 4, last.y1 + 4)
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    path = OUT / (name + ".png")
    pix.save(str(path))
    return path, clip, page.number + 1


def crop_arch(doc, dpi=300):
    """Figure 1 is vector artwork: crop from below the running header down to
    its caption, over the drawings' horizontal extent."""
    page, hit = find(doc, "The decoder. The adaptive alignment layer")
    ds = [d["rect"] for d in page.get_drawings() if d["rect"].y1 < hit.y0 and d["rect"].y0 > 50]
    x0 = min(r.x0 for r in ds) - 6
    x1 = max(r.x1 for r in ds) + 6
    y0 = min(r.y0 for r in ds) - 6
    r = fitz.Rect(x0, y0, x1, hit.y0 - 2)
    pix = page.get_pixmap(dpi=dpi, clip=r)
    path = OUT / "fig_arch_paper.png"
    pix.save(str(path))
    return path, r, page.number + 1


if __name__ == "__main__":
    doc = fitz.open(str(PDF))
    for name, phrase in TABLES.items():
        path, clip, pg = crop_table(doc, name, phrase)
        print("%-14s page %2d  clip %s" % (name, pg, tuple(round(v) for v in clip)))
    path, clip, pg = crop_arch(doc)
    print("%-14s page %2d  clip %s" % ("fig_arch", pg, tuple(round(v) for v in clip)))

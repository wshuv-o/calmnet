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



def drop_rows(src_name, phrase, labels, out_name, dpi=300):
    """Cut table rows out of a cropped table image, at the midpoints of the gaps
    to the neighbouring lines, so the rest is the paper's own rendering."""
    from PIL import Image
    page, _ = find(doc_global, phrase)
    path, clip, _ = crop_table(doc_global, src_name, phrase, dpi=dpi)
    img = Image.open(path)
    scale = dpi / 72.0
    lines = sorted({round(w[1], 1): w for w in page.get_text("words", clip=clip)}.values(), key=lambda w: w[1])
    ys = sorted({(round(w[1], 1), round(w[3], 1)) for w in page.get_text("words", clip=clip)})
    cuts = []
    import re as _re
    rows = {}
    for w in page.get_text("words", clip=clip):
        rows.setdefault((w[5], w[6]), []).append(w)
    for lab in labels:
        lab = lab.strip()
        hit = None
        for ws in rows.values():
            text = " ".join(x[4] for x in sorted(ws, key=lambda x: x[0]))
            if _re.match(_re.escape(lab) + r"\s+[0-9]", text) or text == lab or _re.match(_re.escape(lab) + r"\s+\[", text):
                hit = fitz.Rect(min(x[0] for x in ws), min(x[1] for x in ws), max(x[2] for x in ws), max(x[3] for x in ws))
                break
        assert hit is not None, lab
        r = hit
        above = max(y1 for y0, y1 in ys if y1 <= r.y0 + 0.5)
        below = min((y0 for y0, y1 in ys if y0 >= r.y1 - 0.5), default=None)
        if below is None:          # last row: keep the bottom rule below it
            below = r.y1 + (r.y0 - above)
        cuts.append(((above + r.y0) / 2, (r.y1 + below) / 2))
    keep, prev = [], clip.y0
    for a, b in sorted(cuts):
        keep.append((prev, a)); prev = b
    keep.append((prev, clip.y1))
    parts = [img.crop((0, int(round((a - clip.y0) * scale)), img.width, int(round((b - clip.y0) * scale)))) for a, b in keep]
    out = Image.new("RGB", (img.width, sum(pp.height for pp in parts)), "white")
    y = 0
    for pp in parts:
        out.paste(pp, (0, y)); y += pp.height
    out.save(OUT / (out_name + ".png"))
    return cuts



def keep_columns_until(src_name, phrase, first_dropped_header, out_name, dpi=300):
    """Keep the table columns left of a header word; the rest is cut off."""
    from PIL import Image
    page, _ = find(doc_global, phrase)
    path, clip, _ = crop_table(doc_global, src_name, phrase, dpi=dpi)
    img = Image.open(path)
    r = page.search_for(first_dropped_header, clip=clip)[0]
    prev_right = max(w[2] for w in page.get_text("words", clip=clip) if w[2] < r.x0 - 1)
    cut = ((prev_right + r.x0) / 2 - clip.x0) * dpi / 72.0
    img.crop((0, 0, int(cut), img.height)).save(OUT / (out_name + ".png"))


if __name__ == "__main__":
    doc = fitz.open(str(PDF))
    doc_global = doc
    for name, phrase in TABLES.items():
        path, clip, pg = crop_table(doc, name, phrase)
        print("%-14s page %2d  clip %s" % (name, pg, tuple(round(v) for v in clip)))
    print("tab_compare_ppt  EEGTCNet row cut at", drop_rows("tab_compare", TABLES["tab_compare"], ["EEGTCNet"], "tab_compare_ppt"))
    drop_rows("tab_ablation3", TABLES["tab_ablation3"], ["align + ctx", "ctx + gate", "stem only"], "tab_ablation3_ppt")
    keep_columns_until("tab_ratecurve", TABLES["tab_ratecurve"], "Cost", "tab_ratecurve_ppt")
    path, clip, pg = crop_arch(doc)
    print("%-14s page %2d  clip %s" % ("fig_arch", pg, tuple(round(v) for v in clip)))

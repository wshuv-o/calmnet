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
    # Captions as the current manuscript prints them. The deck shows the
    # paper's own rendering of each table, so these must track the paper.
    "tab_blocks":    "Contiguous single-class block structure",
    "tab_compare":   "The reported model, the stem it is built on, and eight published",
    "tab_noise":     "Measurement noise, for the arms this paper reports",
    "tab_ablation":  "Ablation of the reported model on cohort",
    "tab_artefact":  "Artefact control on cohort",
    "tab_tangent":   "The tangent-space branch against the convolutional stem",
    "tab_tanref":    "Where the tangent reference comes from",
    "tab_allcoh":    "The tangent-space branch against the stem alone",
    "tab_rate":      "The estimator memory against the class-block length",
    "tab_deploy":    "What a wearer would experience",
    "tab_newbase":   "Published decoders on the two cohorts added last",
    "tab_cost":      "Inference cost at cohort",
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


def crop_col(doc, name, phrase, dpi=300, gap=150):
    """Crop a table by its own rules, grouped by vertical gap.

    crop_table walks rules by shared horizontal extent, which fails for tables
    that resizebox has widened past their column (Tables 7, 12 and 13 all do).
    Here the rules are those starting inside the caption's column and below it,
    cut at the first gap wider than `gap`, and the clip takes x from the caption
    block so a table* spanning both columns works too.
    """
    page, hit = find(doc, phrase)
    blocks = [b for b in page.get_text("blocks")
              if b[1] <= hit.y1 + 2 and b[3] >= hit.y0 - 2
              and b[0] <= hit.x0 + 2 and b[2] >= hit.x1 - 2]
    cap = fitz.Rect(blocks[0][:4]) if blocks else hit
    rules = sorted((d["rect"] for d in page.get_drawings()
                    if d["rect"].height <= 2.5 and d["rect"].width > 60
                    and d["rect"].y0 > cap.y1 - 2
                    and cap.x0 - 30 <= d["rect"].x0 <= cap.x1),
                   key=lambda r: r.y0)
    if not rules:
        raise SystemExit("no rules under the caption for " + name)
    grp = [rules[0]]
    for r in rules[1:]:
        if r.y0 - grp[-1].y0 > gap:
            break
        grp.append(r)
    # Some bottom rules are not reported by get_drawings (Table 14's is not),
    # which cuts the final rows. Extend past the last rule over text rows that
    # follow without a paragraph-sized gap.
    bottom = grp[-1].y1
    ys = sorted({round(w[1]) for w in page.get_text("words")
                 if w[1] > bottom - 12 and cap.x0 - 4 <= w[0] <= cap.x1 + 4})
    prev = bottom - 12
    for y in ys:
        if y - prev > 20:
            break
        prev = y
    bottom = max(bottom, prev + 15)
    clip = fitz.Rect(cap.x0 - 4, grp[0].y0 - 4, cap.x1 + 4, bottom) & page.rect
    page.get_pixmap(dpi=dpi, clip=clip).save(str(OUT / (name + ".png")))
    return OUT / (name + ".png"), clip, page.number + 1


def crop_between(doc, name, phrase, next_phrase, dpi=300):
    """Crop a table whose body is wider than its own column.

    crop_table filters candidate rules to the caption's horizontal extent, which
    drops tables that 
esizebox has widened past the column (Table 7 runs to
    x=662 in a column ending at 548). Here the rules are taken from the caption
    down to the next caption instead, with no width filter, and the clip is
    their union.
    """
    page, hit = find(doc, phrase)
    stop = page.search_for(next_phrase)
    y_stop = stop[0].y0 if stop else page.rect.height
    rules = [d["rect"] for d in page.get_drawings()
             if d["rect"].height <= 2.5 and d["rect"].width > 60
             and hit.y1 < d["rect"].y0 < y_stop
             and abs(d["rect"].x0 - hit.x0) < 25]
    if not rules:
        raise SystemExit("no rules between captions for " + name)
    # resizebox reports rule geometry before the transform, so the rule
    # extent can run past the page. Take y from the rules and x from the text
    # actually rendered in that band.
    y0 = min(r.y0 for r in rules) - 4
    y1 = max(r.y1 for r in rules) + 4
    # Both rule and word coordinates come back pre-transform, so neither gives
    # the rendered width. Use the caption's own column, which is what the
    # reader sees, and intersect with the page so nothing is clipped away.
    blocks = [b for b in page.get_text("blocks")
              if b[1] <= hit.y1 + 2 and b[3] >= hit.y0 - 2
              and b[0] <= hit.x0 + 2 and b[2] >= hit.x1 - 2]
    cap = fitz.Rect(blocks[0][:4]) if blocks else hit
    clip = fitz.Rect(cap.x0 - 4, y0, cap.x1 + 4, y1) & page.rect
    page.get_pixmap(dpi=dpi, clip=clip).save(str(OUT / (name + ".png")))
    return OUT / (name + ".png"), clip, page.number + 1


def crop_arch(doc=None, dpi=300):
    """The architecture figure is hand-maintained in drawio and exported to
    results/fig_arch.pdf. Render that file rather than cropping the page: it is
    the same artwork without the page's margins or caption."""
    src = ROOT / "results" / "fig_arch.pdf"
    fd = fitz.open(str(src))
    pix = fd[0].get_pixmap(dpi=dpi)
    path = OUT / "fig_arch_paper.png"
    pix.save(str(path))
    return path, fd[0].rect, 1


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
    path, clip, pg = crop_between(doc, "tab_ablation",
                                  "Ablation of the reported model on cohort",
                                  "Artefact control on cohort")
    print("%-14s page %2d  clip %s (re-cropped)" % ("tab_ablation", pg,
                                                    tuple(round(v) for v in clip)))
    # tab_rate now sits directly above tab_deploy in the same column, and
    # the gap between their rules is under the grouping threshold, so it
    # has to be bounded by the next caption instead.
    path, clip, pg = crop_between(doc, "tab_rate", TABLES["tab_rate"],
                                  TABLES["tab_deploy"])
    print("%-14s page %2d  clip %s (bounded)"
          % ("tab_rate", pg, tuple(round(v) for v in clip)))
    for nm, ph in (("tab_deploy", TABLES["tab_deploy"]),
                   ("tab_newbase", "mean and standard deviation over three seeds")):
        path, clip, pg = crop_col(doc, nm, ph)
        print("%-14s page %2d  clip %s (by column)"
              % (nm, pg, tuple(round(v) for v in clip)))
    path, clip, pg = crop_arch(doc)
    print("%-14s page %2d  clip %s" % ("fig_arch", pg, tuple(round(v) for v in clip)))

# Figure 1 (architecture) — reproduction

The architecture diagram is authored in draw.io and rendered to vector PDF by a
script in this repository. There is one source of truth: editing the `.drawio`
and re-running the renderer is the only supported way to change the figure.

---

## Files

| file | role |
|---|---|
| `paper/fig_arch.drawio` | **the source.** mxGraph XML, 820 × 560 px canvas, 37 vertices + 16 edges |
| `src/render_drawio.py` | renders the `.drawio` to PDF + PNG |
| `src/figstyle.py` | supplies fonts, colours and the target column widths |
| `results/fig_arch.pdf` | vector output the manuscript includes (~96 KB) |
| `results/fig_arch.png` | 400 dpi raster, for previews only (~320 KB) |
| `tools/figures.sh` | regenerates this figure and the other six |

`paper/cas_calmnet.tex` includes it as `../results/fig_arch.pdf`.

---

## Regenerate

```bash
cd src
python render_drawio.py ../paper/fig_arch.drawio ../results/fig_arch.pdf
```

Or regenerate every figure in the paper:

```bash
bash tools/figures.sh
```

Then rebuild the manuscript:

```bash
cd paper && ../tools/tex/tectonic.exe -X compile cas_calmnet.tex --outdir .
```

Requirements: Python 3 with `matplotlib` and `numpy`. Nothing else. No draw.io
installation, no Node, no network.

Expected output:

```
  canvas 820 x 560 px, natural 8.54 in, rendering at 6.84 in, font scale 0.601
rendered fig_arch.drawio -> fig_arch.pdf (+ .png)
```

---

## Edit

Open `paper/fig_arch.drawio` in draw.io (desktop or <https://app.diagrams.net>),
edit, save **as uncompressed XML**, then re-run the renderer.

> In draw.io: Extras → Edit Diagram shows the raw XML. When saving, make sure
> compression is off (File → Properties → uncheck "Compressed"), otherwise the
> file becomes a base64 blob and `render_drawio.py` cannot parse it.

The file is small and hand-editable; most changes in this project were made by
editing the XML directly.

---

## Two things that will bite you

### 1. Font size is scaled, not absolute

draw.io font sizes are **pixels at 96 dpi**. The renderer converts them to
points for the physical width the figure is printed at:

```
FSCALE = 0.75 × (width_in / (canvas_px / 96))
```

At the current canvas (820 px) and CAS double-column width (6.84 in) this is
**0.601**, so a label declared at `fontSize=12` prints at **7.2 pt**.

`FIGURE_RULES.md` §1 requires 7–9 pt at final size, so **do not declare a font
below 12 px** in this canvas, and do not widen the canvas without checking:
the first version used a 1700 px canvas, which put labels at 4 pt.

To check after any canvas change, read the `font scale` line the renderer
prints and multiply by your smallest declared size.

### 2. Collisions are invisible in the XML

Three rounds of fixes here were all overlaps that looked fine in source. After
every edit, open `results/fig_arch.png` and **look at it**. Two labels were
deleted rather than repositioned, because the caption already carried them.

---

## What the renderer supports

It implements the mxGraph subset this figure uses, not all of draw.io.

**Shapes:** `rect`, `rounded=1` (honours `arcSize`), `ellipse`, `shape=cube`
(drawn as a front face plus two shaded side faces, honours `size`),
`shape=parallelogram` (honours `size`), `line`, and `text`.

**Styling:** `fillColor`, `strokeColor`, `strokeWidth`, `dashed` + `dashPattern`,
`fontSize`, `fontColor`, `fontStyle` (bold = 1, italic = 2), `align`.

**Edges:** source/target attachment (the renderer picks the two faces that face
each other), explicit waypoints via `<Array as="points">`, loose endpoints via
`sourcePoint`/`targetPoint`, plus edge labels.

**Labels:** `<b>`, `<i>`, `<br>`, `<sup>`, `<font>` and HTML entities are
stripped or converted; `&#10;` becomes a line break. Text wrapped in `$…$` is
rendered as matplotlib mathtext, which is how `$\mathbf{M}^{-1/2}\mathbf{X}$`
and `$\tau_{\mathrm{blk}}$` are typeset.

**Not supported:** compressed `.drawio` files, images, swimlanes, containers,
`edgeStyle=orthogonalEdgeStyle` routing (edges are drawn as straight segments
through any waypoints you give), rotation, gradients, shadows, and the draw.io
shape libraries. If you need one of these, add it to `render_drawio.py`.

### Escaping

The XML must stay well-formed, so a literal `<` inside a label breaks the file.
Write `&lt;`. This is why the rate condition is authored as three separate math
spans:

```
rate must satisfy   $\tau_{\mathrm{blk}}$ &lt; $32/m$ &lt; $\tau_{\mathrm{drift}}$
```

---

## Why not just export from draw.io

Both standard routes fail on this machine, which is why the renderer exists:

- **draw.io desktop** (109 MB Electron app): the portable Windows build ships an
  incomplete V8 snapshot and aborts with `Error loading V8 startup snapshot
  file` before it can export. Installing via `winget` fails on a certificate
  error.
- **`drawio-batch`**: `npm view drawio-batch` reports v11.17.0, but the package
  has been unpublished and installation returns 404.

Rendering the `.drawio` directly also avoids the failure mode where a diagram
and a separate plotting script drift out of sync, and it means the figure
regenerates in CI with no GUI application present.

If you do have a working draw.io install and prefer its renderer:

```bash
drawio --no-sandbox -x -f pdf --crop -o fig_arch.pdf fig_arch.drawio
```

Its output will differ slightly (different font metrics and edge routing) but
the content is the same. If you switch, regenerate at the same physical width
so the figure stays consistent with the other six.

---

## Content check

The parameter counts in the figure are exact and must sum to the total. If the
architecture changes, verify them against the model rather than editing the
labels by hand:

```python
import sys; sys.path.insert(0, 'src')
from driftnet import build_driftnet
m = build_driftnet(60, 400, size='base', use_ctx=False, use_gate=True)
print(sum(p.numel() for p in m.parameters()))   # expect 24181
```

Current figure: alignment 1, stem 6 768, frame embed 12 737, selective head
4 419 → **24 181**; the optional context pathway adds 594 816 → 618 997.

"""Single source of truth for the CALM-Net architecture figure.
Emits calmnet_architecture.drawio (editable/exportable) AND a matplotlib preview
(calmnet_architecture_preview.png) so the layout can be verified.

Shape grammar: parallelogram = data/signal, rectangle = computation,
dashed grey box = pipeline stage. Solid arrow = inference/forward, dashed = training
signal (loss/target), red = adversarial (gradient reversal) & safe-default STOP.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon
from matplotlib.path import Path as MPath

W, H = 1740, 600
RED = "#C0392B"; GREY = "#8A8F98"; BLK = "#111111"

# id: (kind, x, y, w, h, plain_label, html_label)
BOX = {
 # --- stage containers (drawn first / behind) ---
 "G1": ("group", 180, 62, 398, 404, "1 · Multimodal neuro-kinematic encoder", "1 &#183; Multimodal neuro-kinematic encoder"),
 "G2": ("group", 590, 62, 602, 404, "2 · Motion-invariant disentanglement (MID)", "2 &#183; Motion-invariant disentanglement (MID)"),
 "G3": ("group", 1204, 62, 252, 404, "3 · Self-calibration (LSC)", "3 &#183; Self-calibration (LSC)"),
 "G4": ("group", 1468, 62, 250, 404, "4 · Decision (SAS)", "4 &#183; Decision (SAS)"),
 # --- inputs (data) ---
 "X": ("data", 24, 150, 146, 58, "EEG\nX ∈ ℝ^{60×200}\n2 s @ 100 Hz", "EEG&lt;br&gt;X &#8712; &#8477;&lt;sup&gt;60&#215;200&lt;/sup&gt;&lt;br&gt;2 s @ 100 Hz"),
 "M": ("data", 24, 254, 146, 56, "Head + Exo IMU\nM", "Head + Exo IMU&lt;br&gt;M"),
 "O": ("data", 24, 352, 146, 46, "EOG\nO ∈ ℝ^{4×200}", "EOG&lt;br&gt;O &#8712; &#8477;&lt;sup&gt;4&#215;200&lt;/sup&gt;"),
 # --- stage 1 ops ---
 "fb": ("op", 196, 150, 176, 58, "Sub-band filter bank\nX → {X_b}: μ, low-β, high-β", "Sub-band filter bank&lt;br&gt;X &#8594; {X&lt;sub&gt;b&lt;/sub&gt;}: &#956;, low-&#946;, high-&#946;"),
 "cov": ("op", 196, 232, 176, 78, "Spatial covariance\nΣ_b = (1/T)X_bX_bᵀ+εI\n→ tangent v_b (SPD)", "Spatial covariance&lt;br&gt;&#931;&lt;sub&gt;b&lt;/sub&gt; = (1/T)X&lt;sub&gt;b&lt;/sub&gt;X&lt;sub&gt;b&lt;/sub&gt;&lt;sup&gt;&#8868;&lt;/sup&gt;+&#949;I&lt;br&gt;&#8594; tangent v&lt;sub&gt;b&lt;/sub&gt; (SPD)"),
 "xfca": ("op", 392, 178, 172, 74, "XFCA\nμ–β coupling attention\n→ z", "XFCA&lt;br&gt;&#956;&#8211;&#946; coupling attention&lt;br&gt;&#8594; z"),
 "kin": ("op", 196, 338, 176, 52, "Kinematic encoder\nM → k", "Kinematic encoder&lt;br&gt;M &#8594; k"),
 # --- stage 2 ---
 "split": ("op", 606, 180, 140, 74, "MID split\nz = [ z_int ; z_art ]", "MID split&lt;br&gt;z = [ z&lt;sub&gt;int&lt;/sub&gt; ; z&lt;sub&gt;art&lt;/sub&gt; ]"),
 "zint": ("op", 772, 122, 148, 44, "z_int  (intent)", "&lt;b&gt;z&lt;sub&gt;int&lt;/sub&gt;&lt;/b&gt;  (intent)"),
 "zart": ("op", 772, 270, 148, 44, "z_art  (artefact)", "&lt;b&gt;z&lt;sub&gt;art&lt;/sub&gt;&lt;/b&gt;  (artefact)"),
 "phi": ("data", 606, 330, 152, 64, "φ(M) ∈ ℝ^{12}\nhead+exo accel/gyro\nmean · std · max", "&#966;(M) &#8712; &#8477;&lt;sup&gt;12&lt;/sup&gt;&lt;br&gt;head+exo accel/gyro&lt;br&gt;mean &#183; std &#183; max"),
 "cls": ("op", 958, 74, 218, 64, "Selective classifier\np = softmax(W z_int / T)\ngate g(z_int)", "Selective classifier&lt;br&gt;p = softmax(W z&lt;sub&gt;int&lt;/sub&gt; / T)&lt;br&gt;gate g(z&lt;sub&gt;int&lt;/sub&gt;)"),
 "adv": ("adv", 958, 152, 222, 78, "GRL adversary R_i\nℒ_adv = ‖R_i(GRL z_int) − φ(M)‖²\n+ decorr ℒ_dec = ‖Cov(z_int, φ)‖²", "&lt;b&gt;GRL adversary&lt;/b&gt; R&lt;sub&gt;i&lt;/sub&gt;&lt;br&gt;&#8466;&lt;sub&gt;adv&lt;/sub&gt; = ‖R&lt;sub&gt;i&lt;/sub&gt;(GRL z&lt;sub&gt;int&lt;/sub&gt;) &#8722; &#966;(M)‖²&lt;br&gt;+ decorr &#8466;&lt;sub&gt;dec&lt;/sub&gt; = ‖Cov(z&lt;sub&gt;int&lt;/sub&gt;, &#966;)‖²"),
 "art": ("op", 958, 252, 222, 54, "Artefact regressor R_a\nℒ_art = ‖R_a(z_art) − φ(M)‖²", "Artefact regressor R&lt;sub&gt;a&lt;/sub&gt;&lt;br&gt;&#8466;&lt;sub&gt;art&lt;/sub&gt; = ‖R&lt;sub&gt;a&lt;/sub&gt;(z&lt;sub&gt;art&lt;/sub&gt;) &#8722; &#966;(M)‖²"),
 "cont": ("op", 958, 322, 222, 50, "Kinematic contamination\nc = h(z_art, k)", "Kinematic contamination&lt;br&gt;c = h(z&lt;sub&gt;art&lt;/sub&gt;, k)"),
 # --- stage 3 ---
 "temp": ("op", 1214, 84, 214, 50, "Temperature scaling\nT ∈ [0.5, 5]", "Temperature scaling&lt;br&gt;T &#8712; [0.5, 5]"),
 "conf": ("op", 1214, 152, 214, 92, "Split + adaptive conformal\nS(q_t) = {k : 1−p_k ≤ q_t}\nq_{t+1} = q_t + η(α − 1[err_t])", "Split + adaptive conformal&lt;br&gt;S(q&lt;sub&gt;t&lt;/sub&gt;) = {k : 1&#8722;p&lt;sub&gt;k&lt;/sub&gt; &#8804; q&lt;sub&gt;t&lt;/sub&gt;}&lt;br&gt;q&lt;sub&gt;t+1&lt;/sub&gt; = q&lt;sub&gt;t&lt;/sub&gt; + &#951;(&#945; &#8722; 1[err&lt;sub&gt;t&lt;/sub&gt;])"),
 # --- stage 4 ---
 "sas": ("op", 1478, 110, 226, 120, "SAS commit\nif g(z_int)≥θ ∧ |S(q_t)|=1 ∧ c<c_th\n→ ŷ = argmax_k p_k\nelse → STOP", "&lt;b&gt;SAS commit&lt;/b&gt;&lt;br&gt;&lt;b&gt;if&lt;/b&gt; g(z&lt;sub&gt;int&lt;/sub&gt;)&#8805;&#952; &#8743; |S(q&lt;sub&gt;t&lt;/sub&gt;)|=1 &#8743; c&amp;lt;c&lt;sub&gt;th&lt;/sub&gt;&lt;br&gt;&#8594; ŷ = argmax&lt;sub&gt;k&lt;/sub&gt; p&lt;sub&gt;k&lt;/sub&gt;&lt;br&gt;&lt;b&gt;else&lt;/b&gt; &#8594; STOP"),
 "out": ("data", 1500, 272, 182, 52, "Walk / Stop / STOP", "Walk / Stop / &lt;b&gt;&lt;font color='#C0392B'&gt;STOP&lt;/font&gt;&lt;/b&gt;"),
 # --- annotations ---
 "loss": ("note", 196, 500, 470, 58, "Objective\nℒ = ℒ_cls^cost + λ₁ℒ_adv + λ₂ℒ_art + λ₃ℒ_dec + λ₄ℒ_sel + λ₅ℒ_cal", "&lt;b&gt;Objective&lt;/b&gt;&lt;br&gt;&#8466; = &#8466;&lt;sub&gt;cls&lt;/sub&gt;&lt;sup&gt;cost&lt;/sup&gt; + &#955;&lt;sub&gt;1&lt;/sub&gt;&#8466;&lt;sub&gt;adv&lt;/sub&gt; + &#955;&lt;sub&gt;2&lt;/sub&gt;&#8466;&lt;sub&gt;art&lt;/sub&gt; + &#955;&lt;sub&gt;3&lt;/sub&gt;&#8466;&lt;sub&gt;dec&lt;/sub&gt; + &#955;&lt;sub&gt;4&lt;/sub&gt;&#8466;&lt;sub&gt;sel&lt;/sub&gt; + &#955;&lt;sub&gt;5&lt;/sub&gt;&#8466;&lt;sub&gt;cal&lt;/sub&gt;"),
 "leg": ("note", 706, 496, 486, 66, "solid: inference / forward     dashed: training signal (loss / target)\nred: adversarial (gradient reversal) + safe-default STOP\n▱ data / signal        ▭ computation", "solid: inference / forward &#160;&#160; dashed: training signal (loss / target)&lt;br&gt;&lt;font color='#C0392B'&gt;red&lt;/font&gt;: adversarial (gradient reversal) + safe-default STOP&lt;br&gt;&#9649; data / signal &#160;&#160;&#160; &#9645; computation"),
}

# (src, dst, kind, plain_label, html_label);  kind: fwd | train | adv
EDGE = [
 ("X", "fb", "fwd", "ℝ^{60×200}", "&#8477;&lt;sup&gt;60&#215;200&lt;/sup&gt;"),
 ("fb", "cov", "fwd", "", ""), ("cov", "xfca", "fwd", "", ""),
 ("O", "cov", "fwd", "ocular ref", "ocular ref"),
 ("M", "kin", "fwd", "", ""),
 ("xfca", "split", "fwd", "z ∈ ℝ^d", "z &#8712; &#8477;&lt;sup&gt;d&lt;/sup&gt;"),
 ("split", "zint", "fwd", "ℝ^{d/2}", "&#8477;&lt;sup&gt;d/2&lt;/sup&gt;"),
 ("split", "zart", "fwd", "", ""),
 ("zint", "cls", "fwd", "", ""),
 ("zint", "adv", "adv", "GRL", "GRL"),
 ("zart", "art", "train", "", ""), ("zart", "cont", "fwd", "", ""),
 ("M", "phi", "fwd", "", ""),
 ("phi", "adv", "train", "target", "target"), ("phi", "art", "train", "target", "target"),
 ("cls", "temp", "fwd", "p", "p"), ("temp", "conf", "fwd", "", ""),
 ("cls", "sas", "fwd", "g, ŷ", "g, ŷ"), ("conf", "sas", "fwd", "S(q_t)", "S(q&lt;sub&gt;t&lt;/sub&gt;)"),
 ("cont", "sas", "fwd", "c", "c"), ("sas", "out", "fwd", "", ""),
]

# matplotlib mathtext labels (for the rendered PDF/PNG); drawio uses the html above.
MATH = {
 "X": "EEG\n$X\\in\\mathbb{R}^{60\\times200}$\n2 s @ 100 Hz",
 "M": "Head + Exo IMU\n$M$",
 "O": "EOG\n$O\\in\\mathbb{R}^{4\\times200}$",
 "fb": "Sub-band filter bank\n$X\\rightarrow\\{X_b\\}$: $\\mu$, low-$\\beta$, high-$\\beta$",
 "cov": "Spatial covariance\n$\\Sigma_b=\\frac{1}{T}X_bX_b^{\\top}+\\epsilon I$\n$\\rightarrow$ tangent $v_b$ (SPD)",
 "xfca": "XFCA\n$\\mu$–$\\beta$ coupling attention\n$\\rightarrow z$",
 "kin": "Kinematic encoder\n$M\\rightarrow k$",
 "split": "MID split\n$z=[\\,z_{\\mathrm{int}}\\,;\\,z_{\\mathrm{art}}\\,]$",
 "zint": "$z_{\\mathrm{int}}$  (intent)",
 "zart": "$z_{\\mathrm{art}}$  (artefact)",
 "phi": "$\\phi(M)\\in\\mathbb{R}^{12}$\nhead+exo accel / gyro\nmean · std · max",
 "cls": "Selective classifier\n$p=\\mathrm{softmax}(Wz_{\\mathrm{int}}/T)$\ngate $g(z_{\\mathrm{int}})$",
 "adv": "GRL adversary $R_i$\n$\\mathcal{L}_{\\mathrm{adv}}=\\|R_i(\\mathrm{GRL}\\,z_{\\mathrm{int}})-\\phi(M)\\|^2$\n$+$ decorr $\\mathcal{L}_{\\mathrm{dec}}=\\|\\mathrm{Cov}(z_{\\mathrm{int}},\\phi)\\|^2$",
 "art": "Artefact regressor $R_a$\n$\\mathcal{L}_{\\mathrm{art}}=\\|R_a(z_{\\mathrm{art}})-\\phi(M)\\|^2$",
 "cont": "Kinematic contamination\n$c=h(z_{\\mathrm{art}},k)$",
 "temp": "Temperature scaling\n$T\\in[0.5,\\,5]$",
 "conf": "Split + adaptive conformal\n$S(q_t)=\\{k:1-p_k\\leq q_t\\}$\n$q_{t+1}=q_t+\\eta(\\alpha-\\mathbf{1}[\\mathrm{err}_t])$",
 "sas": "SAS commit\nif $g(z_{\\mathrm{int}})\\geq\\theta\\;\\wedge\\;|S(q_t)|{=}1\\;\\wedge\\;c<c_{\\mathrm{th}}$\n$\\rightarrow\\;\\hat{y}=\\mathrm{argmax}_k\\,p_k$\nelse $\\rightarrow$ STOP",
 "out": "Walk / Stop / STOP",
 "loss": "Objective\n$\\mathcal{L}=\\mathcal{L}_{\\mathrm{cls}}^{\\mathrm{cost}}+\\lambda_1\\mathcal{L}_{\\mathrm{adv}}+\\lambda_2\\mathcal{L}_{\\mathrm{art}}+\\lambda_3\\mathcal{L}_{\\mathrm{dec}}+\\lambda_4\\mathcal{L}_{\\mathrm{sel}}+\\lambda_5\\mathcal{L}_{\\mathrm{cal}}$",
}
MATHE = {0: "$\\mathbb{R}^{60\\times200}$", 5: "$z\\in\\mathbb{R}^{d}$", 6: "$\\mathbb{R}^{d/2}$",
         15: "$p$", 17: "$g,\\hat{y}$", 18: "$S(q_t)$", 19: "$c$"}


# ------------------------------------------------------------------ preview ---
def preview():
    fig, ax = plt.subplots(figsize=(17.4, 6.0)); ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.invert_yaxis(); ax.axis("off"); ax.set_aspect("equal")
    def cx(b): x, y, w, h = b[1:5]; return x + w / 2, y + h / 2
    for bid, b in BOX.items():
        kind, x, y, w, h, plain = b[0], b[1], b[2], b[3], b[4], b[5]
        lbl = MATH.get(bid, plain)
        if kind == "group":
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=GREY, lw=1.1, ls=(0, (6, 4))))
            ax.text(x + 8, y + 14, plain, color="#5f6368", fontsize=10, fontweight="bold", va="center")
        elif kind == "data":
            sk = 14
            ax.add_patch(Polygon([(x+sk, y), (x+w, y), (x+w-sk, y+h), (x, y+h)], closed=True, fill=False, ec=BLK, lw=1.3))
            ax.text(x + w/2, y + h/2, lbl, ha="center", va="center", fontsize=8.8)
        elif kind == "note":
            ax.text(x, y + h/2, lbl, ha="left", va="center", fontsize=9, color="#333")
        else:
            ec = RED if kind == "adv" else BLK
            ls = (0, (5, 3)) if kind == "adv" else "-"
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=ec, lw=1.3, ls=ls))
            ax.text(x + w/2, y + h/2, lbl, ha="center", va="center", fontsize=8.6,
                    color=(RED if kind == "adv" else "black"))
    def route(sb, tb):
        sx, sy, sw, sh = sb[1:5]; tx, ty, tw, th = tb[1:5]
        scx, scy, tcx, tcy = sx+sw/2, sy+sh/2, tx+tw/2, ty+th/2
        if abs(tcx-scx) >= abs(tcy-scy):
            p0 = (sx+sw, scy) if tcx >= scx else (sx, scy)
            p3 = (tx, tcy) if tcx >= scx else (tx+tw, tcy)
            mx = (p0[0]+p3[0])/2
            return [p0, (mx, p0[1]), (mx, p3[1]), p3]
        p0 = (scx, sy+sh) if tcy >= scy else (scx, sy)
        p3 = (tcx, ty) if tcy >= scy else (tcx, ty+th)
        my = (p0[1]+p3[1])/2
        return [p0, (p0[0], my), (p3[0], my), p3]
    for i, (s, d, kind, plain, _) in enumerate(EDGE):
        col = RED if kind == "adv" else BLK
        ls = "--" if kind in ("train", "adv") else "-"
        pts = route(BOX[s], BOX[d])
        for a, b in zip(pts[:-2], pts[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=1.1, ls=ls)
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=9,
                     lw=1.1, color=col, linestyle=ls, shrinkA=0, shrinkB=2))
        elbl = MATHE.get(i, plain)
        if elbl:
            ax.text(pts[0][0], pts[0][1] - 8, elbl, fontsize=7, color=col, ha="center")
    ax.text(W/2, 30, "CALM-Net: multimodal, motion-disentangled, longitudinally self-calibrating decoder",
            ha="center", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("calmnet_architecture_preview.png", dpi=120, bbox_inches="tight")
    fig.savefig("fig_architecture.pdf", bbox_inches="tight")   # goes straight into the paper
    print("wrote calmnet_architecture_preview.png + fig_architecture.pdf")


# ------------------------------------------------------------------ drawio ---
def drawio():
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    def add(cell): cells.append(cell)
    for bid, b in BOX.items():
        kind, x, y, w, h, _, html = b
        if kind == "group":
            st = "rounded=1;dashed=1;dashPattern=6 4;fillColor=none;strokeColor=%s;fontColor=#5f6368;fontStyle=1;fontSize=11;verticalAlign=top;align=left;spacingLeft=10;spacingTop=6;html=1;" % GREY
        elif kind == "data":
            st = "shape=parallelogram;perimeter=parallelogramPerimeter;size=0.09;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#000000;fontColor=#000000;fontSize=11;"
        elif kind == "note":
            st = "text;html=1;align=left;verticalAlign=middle;fontColor=#333333;fontSize=11;whiteSpace=wrap;"
        elif kind == "adv":
            st = "rounded=1;dashed=1;dashPattern=6 3;whiteSpace=wrap;html=1;fillColor=none;strokeColor=%s;fontColor=#000000;fontSize=11;" % RED
        else:
            st = "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#000000;fontColor=#000000;fontSize=11;"
        add(f'<mxCell id="{bid}" value="{html}" style="{st}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    for i, (s, d, kind, _, html) in enumerate(EDGE):
        if kind == "adv":
            st = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeColor=%s;fontColor=%s;dashed=1;dashPattern=6 3;fontSize=9;" % (RED, RED)
        elif kind == "train":
            st = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeColor=#000000;fontColor=#000000;dashed=1;fontSize=9;"
        else:
            st = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeColor=#000000;fontColor=#000000;fontSize=9;"
        add(f'<mxCell id="e{i}" value="{html}" style="{st}" edge="1" parent="1" source="{s}" target="{d}"><mxGeometry relative="1" as="geometry"/></mxCell>')
    title = '<mxCell id="title" value="CALM-Net: multimodal, motion-disentangled, longitudinally self-calibrating decoder for closed-loop exoskeleton control" style="text;html=1;align=center;verticalAlign=middle;fontStyle=1;fontSize=16;fontColor=#000000;" vertex="1" parent="1"><mxGeometry x="200" y="12" width="1360" height="32" as="geometry"/></mxCell>'
    add(title)
    xml = ('<mxfile host="app.diagrams.net">\n<diagram id="calmnet" name="CALM-Net">\n'
           f'<mxGraphModel dx="1400" dy="900" grid="0" guides="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="1">\n<root>\n'
           + "\n".join(cells) + "\n</root>\n</mxGraphModel>\n</diagram>\n</mxfile>\n")
    open("calmnet_architecture.drawio", "w", encoding="utf-8").write(xml)
    print("wrote calmnet_architecture.drawio")


if __name__ == "__main__":
    import sys; sys.stdout.reconfigure(encoding="utf-8")
    preview(); drawio()

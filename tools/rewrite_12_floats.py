r"""Pass 12: mark every float that reports components the model does not use.

The audit found eight. None of them is wrong as a measurement; all of them are
misleading as presentation, because they show arms built from the alignment
layer, the gate and the cross-epoch transformer without saying that none of
those is in the reported model. A reader meeting Table 4 before Section 5 has no
way to know.

One sentence is appended to each caption. The four that live in the generated
tables file are fixed in the generator, not its output, so regenerating keeps
them.

tab:ablation_rep is left alone: it already says it contains the reported model,
and the arm names it lists are the point of an ablation.

    python tools/rewrite_12_floats.py
"""
from __future__ import annotations

import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BS = chr(92)

MARK = (" The arms here are the components of "
        "Section~" + BS + "ref{sec:dropped}, and none of them is the reported "
        "model; that is the stem plus the tangent branch "
        "(Table~" + BS + "ref{tab:compare}).")

# label -> a unique fragment of its caption to append after
IN_TEX = {
    "tab:ablation": "Three-seed\nvalues under the current estimator are in Table~" + BS + "ref{tab:ablation3}.",
    "tab:external": "Three-seed cohort B values for " + BS + "textit{align +",
}
# generator: function name -> unique caption fragment to append after
IN_GEN = [
    ("tab:ablation3", "the current estimator"),
    ("tab:rate", "Condition~" + BS + "ref{prop:band}"),
    ("tab:ratecurve", "three data-split seeds"),
    ("tab:noise", "momentum and estimator version are inert"),
]


def patch_tex():
    p = os.path.join(ROOT, "paper", "cas_calmnet.tex")
    s = io.open(p, encoding="utf-8").read()
    n = 0
    for lab, frag in IN_TEX.items():
        i = s.find(BS + "label{" + lab + "}")
        if i < 0:
            print("  %-18s label not found" % lab)
            continue
        # the caption is the \caption{...} immediately before the label
        c = s.rindex(BS + "caption{", 0, i)
        # find its matching close brace
        depth, j = 0, c + len(BS + "caption{")
        while j < len(s):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                if depth == 0:
                    break
                depth -= 1
            j += 1
        if MARK.strip() in s[c:j]:
            print("  %-18s already marked" % lab)
            continue
        s = s[:j] + MARK + s[j:]
        n += 1
        print("  %-18s marked" % lab)
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    return n


def patch_gen():
    p = os.path.join(ROOT, "src", "make_tables.py")
    s = io.open(p, encoding="utf-8").read()
    n = 0
    for lab, frag in IN_GEN:
        i = s.find('"' + lab + '"')
        if i < 0:
            print("  %-18s not found in generator" % lab)
            continue
        # the caption is the next string block after the label argument;
        # append our sentence to the last string literal before the colspec
        j = s.find('",\n', i)
        while j > 0 and s[j:j + 40].strip().startswith('",'):
            nxt = s.find('",\n', j + 3)
            if nxt < 0:
                break
            # stop when the following literal looks like a column spec
            after = s[nxt + 3:nxt + 60].strip()
            if after.startswith('"l') or after.startswith('"Model') or \
               after.startswith("rows"):
                break
            j = nxt
        if j < 0:
            print("  %-18s could not locate caption end" % lab)
            continue
        if "none of them is the reported" in s[i:j + 200]:
            print("  %-18s already marked" % lab)
            continue
        ins = (' "' + MARK.strip().replace('"', "'") + '"')
        s = s[:j + 1] + ins + s[j + 1:]
        n += 1
        print("  %-18s marked in generator" % lab)
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    return n


def main():
    print("manuscript:")
    a = patch_tex()
    print("generator:")
    b = patch_gen()
    print("\n  %d captions marked (%d in tex, %d in generator)" % (a + b, a, b))


if __name__ == "__main__":
    main()

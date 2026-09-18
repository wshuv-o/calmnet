r"""Rewrite t_compare in make_tables.py so Table 3 reports the model we report.

Table 3 is the first comparison a reader reaches and it carried a "This work"
block reading "Ours, align + gate, 24,181 parameters" across three cohorts. That
configuration is not the reported model, the parameter count is not the reported
model's, and two cohorts were missing.

Replaces the generator so the table is: both of our configurations and the eight
published decoders, on all five cohorts, with parameter counts. Accuracy only,
because ten accuracy-and-ECE columns do not fit and the ECE comparison already
has its own table.

This edits the generator, not its output, so regenerating tables_auto.tex keeps
the fix.
"""
from __future__ import annotations

import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "make_tables.py")
BS = chr(92)

NEW = '''def t_compare():
    """Both of our configurations and eight published decoders, five cohorts.

    One pipeline throughout: identical preprocessing, optimiser, schedule, early
    stopping, model selection and classifier head, three data-split seeds per
    cell. Columns are therefore internally comparable and are NOT comparable
    with these decoders' published numbers, which used different splits.
    """
    import numpy as _np
    BD = {"A": L("bd_pipeline_c.json"), "B": L("bd_cohort_b.json"),
          "C": L("bd_cohort_c.json"), "D": L("bd_cohort_d.json"),
          "E": L("bd_cohort_e.json")}
    for _k, _v in L("bd_eegnex_s12.json").items():
        BD["A"].setdefault(_k, _v)

    OURS = {
        "stem $+$ branch": ("290" + BS + ",943", "dn_stem_tan", {
            "A": "a_tangent_3seed.json", "B": "b_tangent_3seed.json",
            "C": "c_stem_tangent_3seed.json", "D": "d_bnci_3seed.json",
            "E": "e_decoded_3seed.json"}),
        "stem alone": ("19" + BS + ",505", "dn_stem", {
            "A": "a_ablation_s12.json", "B": "b_tangent_3seed.json",
            "C": "c_stem_tangent_3seed.json", "D": "d_bnci_3seed.json",
            "E": "e_decoded_3seed.json"}),
    }
    COH = ["A", "B", "C", "D", "E"]

    def cells(d, arm):
        return [d[k] for k in sorted(d) if k.startswith(arm + "|s")
                and isinstance(d[k], dict) and d[k].get("acc") is not None]

    def fmt(runs):
        if len(runs) < 3:
            return pend("pending")
        v = [r["acc"] for r in runs]
        return "%.3f $" % _np.mean(v) + BS + "pm$ %.3f" % _np.std(v, ddof=1)

    def failed(runs):
        return any(sum(1 for x in r["per_subject"].values()
                       if x["acc"] < FAIL_ACC) >= FAIL_N for r in runs)

    # best published per cohort, for bolding
    best = {}
    for c in COH:
        vals = {}
        for m in PARAMS_C:
            r = cells(BD[c], m)
            if len(r) >= 3:
                vals[m] = _np.mean([x["acc"] for x in r])
        best[c] = max(vals.values()) if vals else None

    rows = [group("This work", 7)]
    for name, (par, arm, files) in OURS.items():
        vals = []
        for c in COH:
            r = cells(L(files[c]), arm)
            s = fmt(r)
            if len(r) >= 3 and best[c] is not None:
                if _np.mean([x["acc"] for x in r]) > best[c]:
                    s = BS + "textbf{" + s + "}"
            vals.append(s)
        rows.append("%s & %s & %s %s" % (name, par, " & ".join(vals), EOL))
    rows.append(BS + "midrule")
    rows.append(group("Published decoders, same pipeline", 7))

    def rank(m):
        r = cells(BD["C"], m)
        return _np.mean([x["acc"] for x in r]) if r else 0

    for m in sorted(PARAMS_C, key=lambda m: -rank(m)):
        vals = []
        for c in COH:
            r = cells(BD[c], m)
            s = fmt(r)
            if len(r) >= 3 and failed(r):
                s += "$^" + BS + "dagger$"
            vals.append(s)
        rows.append("%s%s & %s & %s %s" % (
            m, cref(m), "{:,}".format(PARAMS_C[m]).replace(",", BS + ","),
            " & ".join(vals), EOL))

    return table(
        "tab:compare",
        "The reported model, the stem it is built on, and eight published "
        "decoders trained in one pipeline with identical preprocessing, "
        "optimiser, schedule, early stopping, model selection and classifier "
        "head. Balanced accuracy, mean $" + BS + "pm$ SD across three "
        "data-split seeds. Cohorts A, B, C, D and E have 7, 8, 20, 9 and 7 "
        "participants. Bold marks a configuration of ours that exceeds every "
        "published decoder in that column. The branch leads four columns; on "
        "cohort B, where a class block outlasts one covariance update and "
        "Condition~" + BS + "ref{prop:band} rules the branch out, the stem "
        "alone leads instead, so one of the two is the best entry in every "
        "column and the protocol selects between them before training. "
        "Parameter counts include the shared classifier at cohort A's input "
        "shape. $" + BS + "dagger$ near chance ($<0.55$) for at least five "
        "participants on a seed, a training failure under the shared "
        "settings. These columns are not comparable with numbers from the "
        "decoders' original papers.",
        "lrrrrrr",
        "Model & Params & A & B & C & D & E",
        rows, wide=True, fit=True)
'''


def main():
    s = io.open(SRC, encoding="utf-8").read()
    i = s.index("def t_compare():")
    j = s.index("\n# =====", i)
    old = s[i:j]
    if "stem $+$ branch" in old:
        print("already rewritten")
        return
    s = s[:i] + NEW + s[j:]
    io.open(SRC, "w", encoding="utf-8", newline="\n").write(s)
    print("rewrote t_compare: %d chars -> %d" % (len(old), len(NEW)))


if __name__ == "__main__":
    main()

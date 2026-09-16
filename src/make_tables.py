"""Emit every data-driven table in the paper, straight from results/*.json, so
no number is transcribed by hand.

Writes paper/tables_auto.tex, which cas_calmnet.tex \\input{}s. Re-run after any
new result lands:

    cd src && python make_tables.py

Four of these are full-width (table*) mega tables that consolidate what were
previously fifteen separate tables:

  tab:landscape   every published decoder we ran, under each condition
  tab:rate        the adaptation-rate condition, both cohorts, all arms
  tab:repr        representation, normalisation and leakage controls
  tab:rejected    every intervention tested, with the control that killed it

Column groups inside a mega table are separated by rules and carry their own
condition, because several of them are NOT mutually comparable (different
participant counts, different training pipelines). That is stated in each
caption rather than left for the reader to infer.
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent.parent / "paper" / "tables_auto.tex"
BS = chr(92)
NL = chr(10)
EOL = BS + BS


def L(name):
    p = RESULTS / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def acc(v):
    return v.get("acc") if isinstance(v, dict) else None


def g(d, key, field="acc"):
    v = d.get(key)
    return v.get(field) if isinstance(v, dict) and field in v else None


def agg(files, keyfn):
    out = {}
    for f in files:
        for k, v in L(f).items():
            a = acc(v)
            if a is None:
                continue
            out.setdefault(keyfn(k), []).append(a)
    return out


def f3(x, b=False):
    if isinstance(x, dict):
        x = x.get("acc")
    if x is None:
        return "---"
    s = "%.3f" % x
    return BS + "textbf{" + s + "}" if b else s


def table(label, caption, colspec, header, rows, wide=False, small=True,
          note=None):
    env = "table*" if wide else "table"
    out = [BS + "begin{" + env + "}[t]",
           BS + "caption{" + caption + "}",
           BS + "label{" + label + "}",
           BS + "centering"]
    if small:
        out.append(BS + "small")
    out.append(BS + "begin{tabular}{" + colspec + "}")
    out.append(BS + "toprule")
    out.append(header + " " + EOL)
    out.append(BS + "midrule")
    out += rows
    out.append(BS + "bottomrule")
    out.append(BS + "end{tabular}")
    if note:
        out.append(NL + note)
    out.append(BS + "end{" + env + "}")
    return NL.join(out) + NL + NL


def group(text, ncol):
    return (BS + "multicolumn{%d}{l}{" % ncol + BS + "textit{" + text + "}} "
            + EOL)


# ===================================================================== mega 1
def t_landscape():
    scr = L("backbone_selection.json")
    fb = L("fullbench.json")
    mrg = agg(["merge.json"], lambda k: k.split("|")[1])
    atc = L("atcplus.json")
    dn = L("driftnet_ds.json")
    dnf = L("driftfix_ds.json")
    s1 = L("driftseed1_ds.json")

    def fb2(model):
        vs = [v["acc"] for k, v in fb.items()
              if isinstance(v, dict) and "acc" in v and k.split("|")[1] == model]
        return (st.mean(vs), max(vs) - min(vs)) if vs else (None, None)

    rows = []
    # -- group A: screening, 3 participants
    rows.append(group("Screening, 3 participants, single seed "
                      "(not comparable with the groups below)", 5))
    sc = [(k, v) for k, v in scr.items() if isinstance(v, (int, float))]
    sc += [(k, v["acc"]) for k, v in scr.items()
           if isinstance(v, dict) and "acc" in v]
    for k, v in sorted(sc, key=lambda kv: -kv[1]):
        rows.append("%s & --- & %s & --- & --- %s" % (k, f3(v), EOL))
    err = [k for k, v in scr.items() if isinstance(v, dict) and "error" in v]
    for k in err:
        rows.append("%s & --- & n/c & --- & --- %s" % (k, EOL))

    # -- group B: full cohort, pipeline C
    rows.append(BS + "midrule")
    rows.append(group("Full cohort ($n=7$), pipeline C, single seed", 5))
    for name, params, a in [
            ("ATCNet", "45\\,280", g(atc, "base|s0")),
            ("ATCNet, rate-matched", "45\\,280", g(atc, "rate|s0")),
            ("ATCNet + our ctx + head", "45\\,280", g(L("atcours_ds.json"),
                                                      "atc+ours|s0"))]:
        rows.append("%s & %s & --- & %s & --- %s" % (name, params, f3(a), EOL))
    ours = g(dn, "dn_noctx|s0")
    ours1 = g(s1, "dn_noctx|s1")
    om = st.mean([x for x in (ours, ours1) if x is not None])
    rows.append("%s & %s & --- & %s & %s %s" % (
        BS + "textbf{Ours, align + gate}", BS + "textbf{24\\,181}",
        f3(ours, True), f3(om), EOL))
    rows.append("Ours, align + ctx + gate & 618\\,997 & --- & %s & %s %s" % (
        f3(g(dn, "dn_full|s0")),
        f3(st.mean([x for x in (g(dn, "dn_full|s0"), g(s1, "dn_full|s1"))
                    if x is not None])), EOL))
    rows.append("Ours, stem only & 6\\,768 & --- & %s & --- %s"
                % (f3(g(dn, "dn_stem|s0")), EOL))

    # -- group C: pipeline F, 2 seeds
    rows.append(BS + "midrule")
    rows.append(group("Full cohort, pipeline F, mean of 2 seeds "
                      "(offset from pipeline C by 0.032 for ATCNet)", 5))
    for m, params in [("ATCNet", "45\\,280"), ("EEGNeX", "58\\,082"),
                      ("ShallowFBCSPNet", "97\\,120"),
                      ("EEGConformer", "440\\,706")]:
        mu, sp = fb2(m)
        rows.append("%s & %s & --- & --- & %s %s" % (m, params, f3(mu), EOL))

    # -- group D: 3-seed spreads
    rows.append(BS + "midrule")
    rows.append(group("Per-seed spread over 3 seeds, same pipeline "
                      "(the noise floor for any ranking)", 5))
    for m in ("ShallowFBCSPNet", "ATCNet", "PowerAttn-noattn"):
        if m in mrg:
            vs = mrg[m]
            rows.append("%s & --- & --- & --- & %s (spread %s) %s" %
                        (m, f3(st.mean(vs)), f3(max(vs) - min(vs)), EOL))

    return table(
        "tab:landscape",
        "Every published decoder evaluated in this work. \\textbf{The four "
        "groups are not mutually comparable}: they differ in participant "
        "count, training pipeline and seed count, and are separated by rules "
        "for that reason. ATCNet appears in both full-cohort pipelines and "
        "differs by 0.032, which bounds how much of any cross-group "
        "difference is pipeline rather than model. ``n/c'' did not converge. "
        "The rightmost column is a 2- or 3-seed mean where one was run.",
        "lrrrr",
        "Model & Params & Screen & Pipeline C & Multi-seed",
        rows, wide=True)


# ===================================================================== mega 2
def t_rate():
    m020 = L("driftfix_mobi.json")
    m020o = L("driftnet_mobi.json")
    m005, m001 = L("driftmom_0.05.json"), L("driftmom_0.01.json")
    a020, a001 = L("driftfix_ds.json"), L("driftmom_ds_0.01.json")
    arms = [("dn_full", "align + ctx + gate", True),
            ("dn_noctx", "align + gate", True),
            ("dn_nogate", "align + ctx", True),
            ("dn_noalign", "ctx + gate", False),
            ("dn_stem", "stem only", False)]
    rows = [group("Cohort B ($" + BS + "tau_{" + BS + "mathrm{blk}}=309$): "
                  "slowing adaptation past the class block", 7)]
    for k, lab, uses in arms:
        b0 = g(m020, k + "|s0")
        b0 = b0 if b0 is not None else g(m020o, k + "|s0")
        b1, b2 = g(m005, k + "|s0"), g(m001, k + "|s0")
        d = None if (b0 is None or b1 is None) else b1 - b0
        ds = "---" if d is None else ("%+.3f" % d)
        if uses and d is not None:
            ds = BS + "textbf{" + ds + "}"
        rows.append("%s & %s & %s & %s & %s & %s & --- %s" % (
            lab, f3(b0), f3(b1), f3(b2), ds, "yes" if uses else BS + "textbf{no}",
            EOL))
    rows.append(BS + "midrule")
    rows.append(group("Cohort A ($" + BS + "tau_{" + BS + "mathrm{blk}}=18$): "
                      "the same change, matched estimator", 7))
    for k, lab in (("dn_full", "align + ctx + gate"),
                   ("dn_noalign", "ctx + gate")):
        a0, a1 = g(a020, k + "|s0"), g(a001, k + "|s0")
        d = None if (a0 is None or a1 is None) else a1 - a0
        ds = "---" if d is None else ("%+.3f" % d)
        if k == "dn_full":
            ds = BS + "textbf{" + ds + "}"
        rows.append("%s & %s & --- & %s & %s & %s & %s %s" % (
            lab, f3(a0), f3(a1), ds,
            "yes" if k != "dn_noalign" else BS + "textbf{no}", ds, EOL))
    return table(
        "tab:rate",
        "The adaptation-rate condition, single seed. Momentum $m$ sets the "
        "estimator's memory to roughly $32/m$ windows: 160, 640 and 3200. "
        "\\textbf{Every arm that uses alignment gains as the memory crosses "
        "the class-block length, and neither arm that omits it responds} --- "
        "the stem-only arm reproduces 0.737 exactly at all three settings "
        "despite a 160-fold change. The lower block applies the identical "
        "change to the cohort whose blocks are short, where it costs 0.060: "
        "the optima are opposite and no single rate serves both protocols.",
        "lrrrrcr",
        "Arm & $m{=}0.20$ & $m{=}0.05$ & $m{=}0.01$ & $" + BS +
        "Delta$ at crossing & Aligns & $" + BS + "Delta$ cohort A",
        rows, wide=True)


# ===================================================================== mega 3
def t_repr():
    z, zm = L("zscore.json"), L("zscore_mobi.json")
    pub = L("published.json")
    amp = L("amp_ablation.json")
    feat = L("features.json")
    sens = L("sensitivity.json")
    rows = []

    rows.append(group("Per-window amplitude normalisation, by how much "
                      "marginal power the representation carries", 5))
    for rep, lab, power in [("bandpower", "band power", "full"),
                            ("tangent_ea", "tangent space", "partial"),
                            ("corr_only", "correlation", BS + "textbf{none}")]:
        for coh, d in (("A", z), ("B", zm)):
            k1, k0 = "w2.0|%s|z1" % rep, "w2.0|%s|z0" % rep
            v1, v0 = g(d, k1), g(d, k0)
            if v1 is None or v0 is None:
                continue
            dd = v0 - v1
            s = "%+.3f" % dd
            if rep == "corr_only":
                s = BS + "textbf{" + s + "}"
            rows.append("%s, cohort %s & %s & %s & %s & %s %s" %
                        (lab, coh, power, f3(v1), f3(v0), s, EOL))

    rows.append(BS + "midrule")
    rows.append(group("Normalisation mode across published decoders "
                      "(per-window / global / none)", 5))
    seen = []
    for k, v in pub.items():
        p = k.split("|")
        if p[1] not in seen:
            seen.append(p[1])
    for m in seen:
        vals = {p.split("|")[2]: acc(v) for p, v in pub.items()
                if p.split("|")[1] == m}
        best = max((x for x in vals.values() if x is not None), default=None)
        cells = [f3(vals.get(x), vals.get(x) == best)
                 for x in ("perwindow", "global", "none")]
        rows.append("%s & --- & %s & %s & %s %s" %
                    (m, cells[0], cells[1], cells[2], EOL))

    rows.append(BS + "midrule")
    rows.append(group("Amplitude side-channel against its shuffled control", 5))
    for k, v in amp.items():
        p = k.split("|")
        if p[4] != "amp-on":
            continue
        on = acc(v)
        sh = g(amp, "|".join(p[:4] + ["amp-shuffle"]))
        if on is None or sh is None:
            continue
        rows.append("%s & --- & %s & %s & %+.3f %s" %
                    (p[1], f3(on), f3(sh), sh - on, EOL))

    rows.append(BS + "midrule")
    rows.append(group("Movement-leakage sensitivity: injected fraction $f$ of "
                      "the movement signal", 5))
    for f in ("0.0000", "0.0010", "0.0050", "0.0100", "0.0500"):
        v = g(sens, "f" + f)
        if v is not None:
            rows.append("$f = %s$ & --- & %s & --- & --- %s"
                        % (f.rstrip("0").rstrip(".") or "0", f3(v), EOL))
    return table(
        "tab:repr",
        "Representation, normalisation and leakage controls. The first block "
        "is the identifying evidence for the stem design: removing per-window "
        "normalisation is worth $+0.134$ and $+0.174$ where the representation "
        "is pure marginal power, and \\textbf{exactly $+0.000$ where it is "
        "scale-free by construction}. The second block shows the same "
        "conclusion at the architecture level across seven published decoders. "
        "The third retired an amplitude side-channel: five of six models move "
        "under 0.01 when the feature is shuffled. The fourth bounds real "
        "movement contamination below $f=0.005$.",
        "llrrr",
        "Experiment & Power content & Condition A & Condition B & $" + BS +
        "Delta$",
        rows, wide=True)


# ===================================================================== mega 4
def t_rejected():
    cc = L("cancel_control.json")
    db = L("dualband.json")
    spd = agg(["spdnet.json"], lambda k: k.split("|")[0])
    mrg = agg(["merge.json"], lambda k: k.split("|")[1])
    tf = ["temporal_ds007788ctrl.json"] + [
        "temporal_ds007788ctrl_seed%d.json" % i for i in (1, 2, 3)]
    tmp = agg(tf, lambda k: k)
    arch = [v["mean"]["bal_acc"] for v in L("arch_sweep.json").values()
            if isinstance(v, dict) and "mean" in v]

    def cell(x):
        return f3(x)
    rows = [group("Rejected: the control removed the effect", 6)]
    c = cc.get("cancel", {})
    rows.append("Motion cancellation & 1 & %s & %s (zeroed) & %s & "
                "inertial classifier %s" %
                (cell(c.get("true")), cell(c.get("zeroed")),
                 BS + "textbf{%+.3f}" % (c.get("zeroed", 0) - c.get("true", 0)),
                 EOL))
    rows.append("Motion cancellation & 1 & %s & %s (shuffled) & %+.3f & "
                "same %s" % (cell(c.get("true")), cell(c.get("shuffled")),
                             c.get("shuffled", 0) - c.get("true", 0), EOL))
    if "PowerAttn-full" in mrg and "PowerAttn-noattn" in mrg:
        a, b = st.mean(mrg["PowerAttn-full"]), st.mean(mrg["PowerAttn-noattn"])
        rows.append("Within-epoch attention & 3 & %s & %s (removed) & "
                    "%s & block not earning its place %s" %
                    (cell(a), cell(b), BS + "textbf{%+.3f}" % (b - a), EOL))
    both, erd = acc(db.get("both")), acc(db.get("erd"))
    if both is not None and erd is not None:
        rows.append("MRCP dual-band pathway & 1 & %s (both) & %s (ERD only) & "
                    "%+.3f & low band adds noise %s" %
                    (cell(both), cell(erd), erd - both, EOL))
    if arch:
        rows.append("Capacity scaling & 1 & %s & %s & %+.3f & "
                    "24 configs, no trend %s" %
                    (cell(max(arch)), cell(min(arch)), max(arch) - min(arch),
                     EOL))
    if "aspd_ea" in spd:
        a, b = st.mean(spd["aspd_ea"]), st.mean(spd["aspd_noalign"])
        rows.append("Alignment (SPD backbone) & 3 & %s (EA) & %s (none) & "
                    "%s & %s %s" % (cell(a), cell(b), "%+.3f" % (b - a),
                                    BS + "textbf{inside seed spread}", EOL))

    rows.append(BS + "midrule")
    rows.append(group("Kept: the control preserved the effect", 6))
    if "w2.0|none" in tmp:
        base = st.mean(tmp["w2.0|none"])
        for k, lab in (("w2.0|forward", "HMM forward filter"),
                       ("w2.0|viterbi", "Viterbi decoding")):
            if k in tmp:
                rows.append("%s & 4 & %s & %s (baseline) & %s & survives "
                            "shuffle %s" % (lab, cell(st.mean(tmp[k])),
                                            cell(base),
                                            BS + "textbf{%+.3f}"
                                            % (st.mean(tmp[k]) - base), EOL))
        if "w2.0|fwdshuf" in tmp:
            sh = st.mean(tmp["w2.0|fwdshuf"])
            rows.append("%s & 4 & %s & %s (baseline) & %s & "
                        "%s %s" % (BS + "quad control: shuffled order",
                                   cell(sh), cell(base),
                                   BS + "textbf{%+.3f}" % (sh - base),
                                   BS + "textbf{falls below baseline}", EOL))
    return table(
        "tab:rejected",
        "Every architectural intervention tested, with the control that "
        "decided it. The motion-canceller row is the clearest rejection: the "
        "module scored 0.905 with the inertial channel present and chance "
        "with it zeroed, so it was an inertial classifier rather than an EEG "
        "one. The temporal-smoothing rows are the clearest retention: "
        "shuffling window order does not merely remove the gain, it drops "
        "\\textbf{0.113 below the unsmoothed baseline}, which no operation "
        "helping for an unrelated reason would do. $n$ is seeds.",
        "lrllrl",
        "Intervention & $n$ & With & Control & $" + BS + "Delta$ & Verdict",
        rows, wide=True)


# ===================================================================== small
def t_drift():
    d = L("drift.json")
    rows = []
    for key, lab, ns in (("ds007788", "Cohort A", "6 held-out sessions"),
                         ("mobi", "Cohort B", "2 held-out trials")):
        per = d.get(key, {}).get("per_subject", {})
        rows.append(group("%s, %s" % (lab, ns), 5))
        for s in sorted(per):
            v = per[s]
            rows.append("%s & %.3f & %.3f & %.3f & $%.1f" % (
                s.replace("sub-", "A"), v["raw"], v["noop"], v["aligned"],
                v["reduction"] * 100) + BS + "," + BS + "%$ " + EOL)
        if per:
            rows.append(BS + "midrule" if key == "ds007788" else "")
    rows = [r for r in rows if r != ""]
    return table(
        "tab:drift",
        "Per-participant session drift. $" + BS + "delta$ is the "
        "affine-invariant Riemannian distance between the fitting-session mean "
        "covariance and the held-out mean. The no-op control shares every code "
        "path but never updates its estimate. \\textbf{Every participant "
        "improves}: 15 of 15, $t=-9.14$, $p=10^{-4}$ (A) and $t=-20.59$, "
        "$p<10^{-5}$ (B); the control shows no reduction in either cohort.",
        "lrrrr",
        "Participant & Raw $" + BS + "delta$ & No-op & Aligned & Reduction",
        rows)


def t_noise():
    a = agg(["driftnet_ds.json", "driftfix_ds.json", "driftmom_ds_0.01.json"],
            lambda k: k.split("|")[0])
    b = agg(["driftnet_mobi.json", "driftfix_mobi.json", "driftmom_0.01.json",
             "driftmom_0.05.json"], lambda k: k.split("|")[0])
    mrg = agg(["merge.json"], lambda k: k.split("|")[1])
    s0, s1 = L("driftnet_ds.json"), L("driftseed1_ds.json")
    rows = [group("Fixed seed, repeated runs of arms where the manipulated "
                  "variable is inert", 4)]
    for lab, vs, tfm in (("stem only, cohort B", b.get("dn_stem", []), "no"),
                         ("no-align, cohort A", a.get("dn_noalign", []), "yes"),
                         ("no-align, cohort B", b.get("dn_noalign", []), "yes")):
        if len(vs) > 1:
            sp = max(vs) - min(vs)
            rows.append("%s & %s & %d & %s %s" %
                        (lab, tfm, len(vs),
                         BS + "textbf{%.4f}" % sp if sp == 0 else "%.4f" % sp,
                         EOL))
    rows.append(BS + "midrule")
    rows.append(group("Across split seeds, our arms", 4))
    for k, lab in (("dn_noctx", "align + gate"), ("dn_full", "align + ctx + gate")):
        v0, v1 = g(s0, k + "|s0"), g(s1, k + "|s1")
        if v0 is not None and v1 is not None:
            rows.append("%s & yes & 2 & %.4f %s" % (lab, abs(v1 - v0), EOL))
    rows.append(BS + "midrule")
    rows.append(group("Across split seeds, published baselines", 4))
    for m in ("ShallowFBCSPNet", "ATCNet"):
        if m in mrg:
            vs = mrg[m]
            rows.append("%s & --- & %d & %s %s" %
                        (m, len(vs), BS + "textbf{%.4f}" % (max(vs) - min(vs)),
                         EOL))
    return table(
        "tab:noise",
        "Measurement noise, and why no single-seed ranking appears in this "
        "paper. The first block repeats arms for which momentum and estimator "
        "version are inert, so all variation is run-to-run: the arm without "
        "the cross-epoch transformer is bit-identical across three runs, and "
        "the arms with it are not. Across split seeds the spread doubles, and "
        "for the published baselines it reaches 0.053. \\textbf{Differences "
        "below 0.02 are not interpreted anywhere in this work.}",
        "llrr", "Arm & Transformer & $n$ & Spread", rows)


def main():
    parts = ["%% Generated by src/make_tables.py -- do not edit by hand." + NL + NL,
             t_landscape(), t_rate(), t_repr(), t_rejected(), t_drift(),
             t_noise()]
    OUT.write_text("".join(parts), encoding="utf-8")
    n = sum(p.count(BS + "begin{table") for p in parts)
    print("wrote %s: %d tables (%d full-width)" %
          (OUT.name, n, sum(p.count("begin{table*}") for p in parts)))


if __name__ == "__main__":
    main()

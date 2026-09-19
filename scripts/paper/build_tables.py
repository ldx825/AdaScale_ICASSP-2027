#!/usr/bin/env python3
"""生成论文用 LaTeX 表格（从 outputs/ 读取，勿手改 .tex）。

输出:
  paper/tables/tab_audit.tex   — Scale utility audit
  paper/tables/tab_main.tex    — Main results
  paper/tables/tab_runtime.tex — Runtime / efficiency

用法: python scripts/paper/build_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAB_DIR = ROOT / "paper" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["PET", "EUROSAT", "FLO", "CUB"]
PROTOS = ["text", "gen"]
DS_NAME = {"PET": "Oxford-Pets", "EUROSAT": "EuroSAT", "FLO": "Flowers-102",
           "CUB": "CUB-200"}
SHORT = {"PET": "Pets", "EUROSAT": "EuroSAT", "FLO": "Flowers", "CUB": "CUB"}
STAB = "r4_d0.65_e0.01"


def build_audit():
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Scale-utility audit (Pets = Oxford-Pets, Flowers = "
        r"Flowers-102): prefix-accuracy curve (selected scales), "
        r"full-10 accuracy, harmful/rescued rate, and median $t_{\mathrm{eq}}$.}",
        r"\label{tab:audit}",
        r"\small",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{llcccccr}",
        r"\hline",
        r"Dataset & Proto & Acc@1 & Acc@5 & Acc@10 & Harm. & Rescue & $t_{eq}$ \\",
        r"\hline",
    ]
    for ds in DATASETS:
        for proto in PROTOS:
            p = ROOT / "outputs" / "scale_audit" / ds / "ViT-B32" / f"summary_{proto}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            c = d["prefix_acc_curve"]
            lines.append(
                f"{SHORT[ds]} & {proto} & {c[0]*100:.1f} & {c[4]*100:.1f} & "
                f"{c[9]*100:.1f} & {d['harmful_rate']*100:.1f} & "
                f"{d['rescue_rate']*100:.1f} & {d['teq_median']:.0f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def build_main():
    import csv
    rows = list(csv.DictReader(
        open(ROOT / "outputs" / "final" / "all_results_v2.csv")))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Zero-shot results (CLIP ViT-B/32; Pets = Oxford-Pets, "
        r"Flowers = Flowers-102). Full-10: baseline "
        r"feature fusion. AdaScale auto-selects FUSED (early exit + soft "
        r"fusion) or SINGLE (largest scale). "
        r"$\Delta$: paired-bootstrap vs.\ Full-10.}",
        r"\label{tab:main}",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{llrrcrrr}",
        r"\hline",
        r"Dataset & Proto & $C$ & Full-10 & Single-10 & AdaScale & $\Delta$ & \#S \\",
        r"\hline",
    ]
    NC = {"PET": 37, "EUROSAT": 10, "FLO": 102, "CUB": 200}
    SHORT = {"PET": "Pets", "EUROSAT": "EuroSAT", "FLO": "Flowers",
             "CUB": "CUB"}
    for r in rows:
        ds, proto = r["dataset"], r["proto"]
        v3 = float(r.get("v3_acc") or r["adascale_acc"])
        delta = float(r.get("v3_delta_vs_baseline") or r["delta_vs_baseline"]) * 100
        p_raw = r.get("v3_p_vs_baseline") or r["p_vs_baseline"]
        p = float(p_raw) if p_raw else 1.0
        star = "^{*}" if p < 0.05 else ""
        lines.append(
            f"{SHORT[ds]} & {proto} & {NC[ds]} & {float(r['baseline_acc'])*100:.2f} & "
            f"{float(r['single_acc'])*100:.2f} & "
            f"\\textbf{{{v3*100:.2f}}} & "
            f"${delta:+.2f}{star}$ & {float(r['adascale_scales']):.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def build_baselines():
    """Table II: prior-inspired adaptive-TTA baselines vs AdaScale."""
    import csv as _csv
    rows_csv = list(_csv.DictReader(
        open(ROOT / "outputs" / "final" / "all_results_v2.csv")))
    by = {(r["dataset"], r["proto"]): r for r in rows_csv}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison with prior-inspired adaptive baselines "
        r"(accuracy\,\%; subscript: average scale evaluations). B1: margin "
        r"stopping (AdapTTA-style). B2: uncertainty gating "
        r"(Selective-TTA-style). B3: keep 5 lowest-entropy views (needs all "
        r"10 views). AdaScale: our hierarchical controller.}",
        r"\label{tab:baselines}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\hline",
        r"Dataset & Proto & Full-10 & Single-10 & B1 & B2 & B3 & AdaScale \\",
        r"\hline",
    ]
    for ds, proto in [("PET", "text"), ("PET", "gen"), ("FLO", "text"),
                      ("FLO", "gen"), ("EUROSAT", "text"), ("EUROSAT", "gen"),
                      ("CUB", "text")]:
        r = by[(ds, proto)]
        pb = json.loads((ROOT / "outputs" / "prior_baselines" /
                         f"{ds}_ViT-B32_{proto}.json").read_text())
        m = pb["methods"]

        def _f(acc, sc):
            return f"{acc*100:.2f}\\,{{\\tiny {sc:.1f}}}"
        b1 = max(m["margin_stop"], key=lambda x: x["acc"])
        b2 = max(m["selective_gating"], key=lambda x: x["acc"])
        b3 = next(x for x in m["entropy_filter"] if "K=5" in str(x.get("param", "")))
        v3 = float(r.get("v3_acc") or r["adascale_acc"])
        sc = float(r["adascale_scales"])
        lines.append(
            f"{SHORT[ds]} & {proto} & "
            f"{_f(pb['full10_acc'], 10)} & {_f(pb['single10_independent_acc'], 1)} & "
            f"{_f(b1['acc'], b1['avg_scales'])} & {_f(b2['acc'], b2['avg_scales'])} & "
            f"{_f(b3['acc'], 10)} & \\textbf{{{_f(v3, sc)}}} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def build_runtime():
    eff = ROOT / "outputs" / "efficiency"
    # 优先最终配置 v2c，其次 v2b，兜底 v2
    for cand in ("PET_adascale_v2c.json", "PET_adascale_v2b.json",
                 "PET_adascale_v2.json"):
        if (eff / cand).exists():
            ada_file = cand
            break
    else:
        ada_file = "PET_adascale_v2.json"
    order = [("full", "PET_full.json"), ("fixed", "PET_fixed5.json"),
             ("fixed", "PET_fixed7.json"), ("adascale", ada_file)]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Wall-clock inference (batch 32, RTX 4080 Super). PET: $N{=}7390$; "
        r"EuroSAT: $N{=}27000$; Single mode evaluates only the largest scale.}",
        r"\label{tab:runtime}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Method & Time (s) & Ips. & Acc (\%) & Avg. scales \\",
        r"\hline",
    ]
    label = {"full": "Full-10 (PET)", "fixed5": "Fixed-5 (PET)",
             "fixed7": "Fixed-7 (PET)", "adascale_v2": "AdaScale (PET)",
             "adascale_v2b": "AdaScale (PET)", "adascale_v2c": "AdaScale (PET)"}
    for mode, fn in order:
        p = eff / fn
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        name = label.get(fn.replace(".json", "").replace("PET_", ""), fn)
        lines.append(f"{name} & {d['time_mean']:.0f} & {d['throughput_ips']:.1f} & "
                     f"{d['acc_mean']*100:.2f} & {d['avg_scales']:.2f} \\\\")
    # EUROSAT: Full-10 与 Single 对比行（如可用）
    eu_full = eff / "EUROSAT_full.json"
    eu_single = eff / "EUROSAT_single.json"
    if eu_full.exists():
        d = json.loads(eu_full.read_text())
        lines.append(f"Full-10 (EURO; 27k) & {d['time_mean']:.0f} & "
                     f"{d['throughput_ips']:.1f} & {d['acc_mean']*100:.2f} & "
                     f"{d['avg_scales']:.2f} \\\\")
    if eu_single.exists():
        d = json.loads(eu_single.read_text())
        lines.append(f"Single (EURO; 27k) & {d['time_mean']:.0f} & "
                     f"{d['throughput_ips']:.1f} & {d['acc_mean']*100:.2f} & "
                     f"{d['avg_scales']:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    for name, fn in [("audit", build_audit), ("main", build_main),
                     ("baselines", build_baselines), ("runtime", build_runtime)]:
        tex = fn()
        out = TAB_DIR / f"tab_{name}.tex"
        out.write_text(tex)
        print(f"[saved] {out}")

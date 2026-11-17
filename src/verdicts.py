"""Verdicts for the prediction sheet under its own decision rule.

  uv run python -B src/verdicts.py

Reads results/runs/main/ and results/runs/transfer/ and writes
results/analysis/verdicts/verdicts.md and verdicts.csv.

Rule (sheet, section 0): every metric is computed per training seed. A difference
is SUPPORTED if its mean lies in the predicted direction and exceeds the pooled
between-seed SD, CONTRADICTED if it exceeds that SD in the opposite direction,
and INCONCLUSIVE otherwise. Pooled SD = sqrt((SD1^2 + SD2^2) / 2) of the two
groups compared. Explicit thresholds in the sheet (P2, P4d) replace the rule.

The Evidence column reads an inconclusive comparison by its direction: weak
evidence for or against the prediction, or no measurable difference when the
difference is below a tenth of the pooled SD.

P3b-c and S1-S3 come from src/probes.py. Its rows replace the pending ones when
results/analysis/probes/verdicts.csv exists.
"""
import json

import numpy as np
import pandas as pd

from analyze import LABEL, to_markdown
from run import ROOT, RUNS

OUT = ROOT / "results" / "analysis" / "verdicts"
AGENTS = ("q", "ppo")
READING = {"supported": "supports", "contradicted": "contradicts"}
NEAR_ZERO = 0.1  # share of the pooled SD below which a difference counts as not measurable


def load():
    """(agent, setting) -> list of (greedy checkpoints indexed by step, training episodes, budget), by seed."""
    runs = {}
    for group in ("main", "transfer"):
        for d in sorted((RUNS / group).iterdir()):
            cfg = json.loads((d / "config.json").read_text())
            c = pd.read_csv(d / "eval_checkpoints.csv")
            c = c[c["mode"] == "greedy"].set_index("step")
            train = pd.read_csv(d / "train_episodes.csv") if (d / "train_episodes.csv").exists() else None
            runs.setdefault((cfg["agent"], cfg["setting"]), []).append((cfg["seed"], c, train, cfg["budget"]))
    return {key: [run[1:] for run in sorted(value, key=lambda run: run[0])] for key, value in runs.items()}


def final(column):
    return lambda c, train, budget: c[column].iloc[-1]


def at(fraction, column="success_rate"):
    return lambda c, train, budget: c.loc[int(fraction * budget), column]


def training(column):
    def metric(c, train, budget):
        last = train[train["end_step"] > 0.9 * budget]
        return (last["outcome"] == "success").mean() if column == "success" else last["return"].mean()
    return metric


def sd(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def pooled(a, b):
    return float(np.sqrt((sd(a) ** 2 + sd(b) ** 2) / 2))


def rule(difference, spread):
    """`difference` is signed so that positive is the predicted direction."""
    return "supported" if difference > spread else "contradicted" if difference < -spread else "inconclusive"


def evidence(difference, spread):
    """Reading of a comparison; `difference` is positive in the predicted direction."""
    verdict = rule(difference, spread)
    if verdict in READING:
        return READING[verdict]
    if abs(difference) <= NEAR_ZERO * spread:
        return "no measurable difference"
    return "weak evidence for" if difference > 0 else "weak evidence against"


def main():
    runs = load()

    def values(agent, setting, metric):
        return np.array([metric(*run) for run in runs[(agent, setting)]], float)

    def pct(x):
        return f"{100 * np.nanmean(x):.1f}"

    rows = []

    def add(pid, part, quantity, result, verdict, reading=None):
        rows.append({"ID": pid, "Part": part, "Quantity": quantity, "Result": result, "Verdict": verdict,
                     "Evidence": reading or READING.get(verdict, "–")})

    def compare(pid, part, quantity, predicted_higher, predicted_lower, names, scale=100, unit="%"):
        difference = float(np.nanmean(predicted_higher) - np.nanmean(predicted_lower))
        spread = pooled(predicted_higher, predicted_lower)
        result = (f"{names[0]} {scale * np.nanmean(predicted_higher):.1f}{unit}, {names[1]} "
                  f"{scale * np.nanmean(predicted_lower):.1f}{unit}; difference {scale * difference:+.1f}, "
                  f"pooled SD {scale * spread:.1f}")
        verdict = rule(difference, spread)
        add(pid, part, quantity, result, verdict, evidence(difference, spread))
        return verdict

    # P1: PPO ahead at 10% of the budget in A and B.
    for s in "AB":
        compare("P1", f"Setting {s}", "Success at 10% of the budget (PPO higher)",
                values("ppo", s, at(0.1)), values("q", s, at(0.1)), ("PPO", "Q-learning"))

    # P2: SD_PPO >= 1.5 x SD_Q at the full budget in A and B.
    for s in "AB":
        sq, sp = sd(values("q", s, final("success_rate"))), sd(values("ppo", s, final("success_rate")))
        verdict = ("inconclusive" if sq == sp == 0 else "supported" if sp >= 1.5 * sq
                   else "contradicted" if sq >= 1.5 * sp else "inconclusive")
        ratio = "undefined (Q-learning SD is 0)" if sq == 0 else f"{sp / sq:.2f}"
        add("P2", f"Setting {s}", "Between-seed SD of final success (PPO at least 1.5 x Q-learning)",
            f"PPO {100 * sp:.1f}, Q-learning {100 * sq:.1f}; ratio {ratio}", verdict,
            READING.get(verdict) or ("weak evidence for" if sp > 1.1 * sq else "weak evidence against"
                                     if sq > 1.1 * sp else "no measurable difference"))

    # P3a: lower success in B than in A, per agent.
    for a in AGENTS:
        compare("P3a", LABEL[a], "Final success, A-trained in A against B-trained in B (A higher)",
                values(a, "A", final("success_rate")), values(a, "B", final("success_rate")), ("A", "B"))
    add("P3b–c", "both agents", "Descent error and switch rates at common p", "needs cross-evaluation", "pending")

    # P4: step penalty, C against B.
    for a in AGENTS:
        compare("P4a", LABEL[a], "Median length of failed episodes (shorter in C)",
                values(a, "B", final("median_steps_failure")), values(a, "C", final("median_steps_failure")),
                ("B", "C"), scale=1, unit=" steps")
    compare("P4b", "PPO", "Final success (lower in C)",
            values("ppo", "B", final("success_rate")), values("ppo", "C", final("success_rate")), ("B", "C"))
    b, c = values("q", "B", final("success_rate")), values("q", "C", final("success_rate"))
    change, spread = float(c.mean() - b.mean()), pooled(b, c)
    add("P4c", "Q-learning", "Final success (change within the pooled SD)",
        f"B {pct(b)}%, C {pct(c)}%; difference {100 * change:+.1f}, pooled SD {100 * spread:.1f}",
        *(("inconclusive, as predicted", "no clear change, as predicted; the test is weak")
          if abs(change) <= spread else ("contradicted", None)))
    for a in AGENTS:
        b = np.nanmean(values(a, "B", final("median_steps_success")))
        c = np.nanmean(values(a, "C", final("median_steps_success")))
        add("P4d", LABEL[a], "Median length of successful episodes, ratio C/B (at most 0.95)",
            f"B {b:.1f} steps, C {c:.1f} steps; ratio {c / b:.3f}", "supported" if c / b <= 0.95 else "contradicted")

    # P5: training return in the final 10% of training against final evaluation success.
    orders = []
    for s in "ABC":
        order = []
        for metric in (training("return"), final("success_rate")):
            q, p = values("q", s, metric), values("ppo", s, metric)
            difference, spread = float(p.mean() - q.mean()), pooled(q, p)
            order.append("tie" if abs(difference) <= spread else "PPO" if difference > 0 else "Q-learning")
        orders.append(order)
        tq, tp = values("q", s, training("return")), values("ppo", s, training("return"))
        add("P5 order", f"Setting {s}", "Leader by training return and by evaluation success",
            f"training return: PPO {tp.mean():.3f}, Q-learning {tq.mean():.3f} ({order[0]}); "
            f"evaluation success: {order[1]}",
            "tie" if "tie" in order else "same order" if order[0] == order[1] else "different order")
    differs = any("tie" not in o and o[0] != o[1] for o in orders)
    matches = all("tie" not in o and o[0] == o[1] for o in orders)
    add("P5 order", "overall", "Orders differ in at least one setting",
        "; ".join(f"{s}: {o[0]} / {o[1]}" for s, o in zip("ABC", orders)),
        "supported" if differs else "contradicted" if matches else "inconclusive",
        None if differs or matches else "ties prevent a comparison")
    for s in "ABC":
        gap = {a: values(a, s, final("success_rate")) - values(a, s, training("success")) for a in AGENTS}
        compare("P5 gap", f"Setting {s}", "Evaluation minus training success (larger for Q-learning)",
                gap["q"], gap["ppo"], ("Q-learning", "PPO"), unit=" points")

    # P7: narrower paddle, B agents.
    drop = {}
    for a in AGENTS:
        b, narrow = values(a, "B", final("success_rate")), values(a, "P7", final("success_rate"))
        drop[a] = b - narrow
        compare("P7 drop", LABEL[a], "Final success, 16 px against 12 px paddle (lower with 12 px)",
                b, narrow, ("16 px", "12 px"))
    p7 = compare("P7 difference", "both agents", "Drop in percentage points (smaller for PPO)",
                 drop["q"], drop["ppo"], ("Q-learning", "PPO"), unit=" points")

    # Setting D has no prediction in the sheet.
    for a in AGENTS:
        b, gap = values(a, "B", final("success_rate")), values(a, "D", final("success_rate"))
        add("D", LABEL[a], "Final success, original against gap layout",
            f"original {pct(b)}%, gap {pct(gap)}%; difference {100 * (gap.mean() - b.mean()):+.1f}, "
            f"pooled SD {100 * pooled(b, gap):.1f}", "no prediction")

    # P6: what would change the initial recommendation (PPO for repeated operation).
    fq, fp = values("q", "B", final("failure_rate")), values("ppo", "B", final("failure_rate"))
    lower, spread = float(fp.mean() - fq.mean()), pooled(fq, fp)
    add("P6.1", "Setting B", "Q-learning's failure rate lower than PPO's by more than the pooled SD",
        f"Q-learning {pct(fq)}%, PPO {pct(fp)}%; pooled SD {100 * spread:.1f}",
        "triggered" if lower > spread else "not triggered")
    wq, wp = values("q", "B", final("success_rate")).min(), values("ppo", "B", final("success_rate")).min()
    add("P6.2", "Setting B", "PPO's worst seed worse than Q-learning's worst seed",
        f"worst PPO seed {100 * wp:.1f}%, worst Q-learning seed {100 * wq:.1f}%",
        "triggered" if wp < wq else "not triggered")
    seconds = {a: np.mean([c["train_seconds"].iloc[-1] for c, _, _ in runs[(a, "B")]]) for a in AGENTS}
    t = min(seconds.values())
    clock = {a: np.array([np.interp(t, c["train_seconds"], c["success_rate"]) for c, _, _ in runs[(a, "B")]])
             for a in AGENTS}
    ahead, spread = float(clock["q"].mean() - clock["ppo"].mean()), pooled(clock["q"], clock["ppo"])
    add("P6.3", "Setting B", f"Q-learning ahead at equal training time ({t:.0f} s, interpolated)",
        f"time for 500,000 steps: Q-learning {seconds['q']:.0f} s, PPO {seconds['ppo']:.0f} s; success at "
        f"{t:.0f} s: Q-learning {pct(clock['q'])}%, PPO {pct(clock['ppo'])}%; pooled SD {100 * spread:.1f}",
        "triggered" if ahead > spread else "not triggered")
    add("P6.4", "P7", "P7 contradicted", f"P7 difference is {p7}", "triggered" if p7 == "contradicted" else "not triggered")

    add("S1–S3", "both agents", "Probes on shared observations", "needs probes", "pending")

    probes = OUT.parent / "probes" / "verdicts.csv"
    if probes.exists():
        extra = pd.read_csv(probes).to_dict("records")
        for pending, prefix in (("P3b–c", "P3"), ("S1–S3", "S")):
            i = next(n for n, r in enumerate(rows) if r["ID"] == pending)
            rows[i:i + 1] = [r for r in extra if r["ID"].startswith(prefix)]
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "verdicts.csv", index=False)
    (OUT / "verdicts.md").write_text(to_markdown(table))
    print(to_markdown(table))
    print(f"Written to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

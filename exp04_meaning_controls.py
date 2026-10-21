import argparse

import numpy as np

from common import OUT, generate, load_prompts, read_jsonl
from exp01_safety_gap import is_harmful, is_refusal, load_xstest
from exp03_english_meaning import mcnemar_p

GENERIC_GLOSS = "Please reply to the message above."
CONDITIONS = ["direct", "gloss_generic", "gloss_mt", "gloss"]
CONTROL_LANGS = ["hi", "bn", "mr", "kn", "or", "sat", "brx"]
REPS = 2000


def gloss_generic_messages(record):
    return [{"role": "user", "content": f"{record['text']}\n\n(In English: {GENERIC_GLOSS})"}]


def gloss_mt_messages(record):
    return [{"role": "user", "content": f"{record['text']}\n\n(In English: {record['mt_en']})"}]


def mt_glosses():
    path = OUT / "qc" / "backtranslation_indictrans2.jsonl"
    if not path.exists():
        raise SystemExit(f"ERROR: {path} missing (run the IndicTrans2 back-translation of the prompts first)")
    glosses = {}
    for row in read_jsonl(path):
        back = (row.get("back") or "").strip()
        if back:
            glosses[(row["pid"], row["lang"])] = back
    return glosses


def add_mt_glosses(records):
    glosses = mt_glosses()
    missing = [(r["pid"], r["lang"]) for r in records if (r["pid"], r["lang"]) not in glosses]
    if missing:
        print(f"WARNING: {len(missing)} items have no IndicTrans2 back-translation -> skipped, e.g. {missing[:3]}")
    kept = []
    for r in records:
        key = (r["pid"], r["lang"])
        if key in glosses:
            kept.append(dict(r, mt_en=glosses[key]))
    return kept


def run_generate(args):
    langs = [lang for lang in args.langs if lang != "en"]
    records = load_prompts(langs)
    if args.sources:
        records = [r for r in records if r["source"] in args.sources]
    if args.condition == "gloss_mt":
        records = add_mt_glosses(records)
        build = gloss_mt_messages
    else:
        build = gloss_generic_messages
    generate(args.model, args.condition, records, build, backend=args.backend, tp=args.tp)


def is_latin(text):
    letters = [c for c in (text or "") if c.isalpha()]
    return sum(c.isascii() for c in letters) / max(1, len(letters)) >= 0.8


def read_replies(model, condition):
    replies = {}
    for row in read_jsonl(OUT / "generations" / model / f"{condition}.jsonl"):
        replies[(row["pid"], row["lang"])] = row["response"]
    return replies


def bootstrap(rng, pids, values):
    ids = sorted(set(pids))
    index = {pid: i for i, pid in enumerate(ids)}
    sums = np.zeros(len(ids))
    counts = np.zeros(len(ids))
    for pid, value in zip(pids, values):
        sums[index[pid]] += value
        counts[index[pid]] += 1
    draws = rng.integers(0, len(ids), size=(REPS, len(ids)))
    estimates = sums[draws].sum(1) / counts[draws].sum(1)
    return sums.sum() / counts.sum(), np.percentile(estimates, 2.5), np.percentile(estimates, 97.5)


def harm(item, condition):
    return is_harmful(item["rows"][condition])


def reduction(rng, items, condition):
    pids = [item["pid"] for item in items]
    values = [harm(item, "direct") - harm(item, condition) for item in items]
    return bootstrap(rng, pids, values)


def discordant(items, condition):
    b = sum(harm(item, "direct") and not harm(item, condition) for item in items)
    c = sum(harm(item, condition) and not harm(item, "direct") for item in items)
    return b, c


def mean_harm(items, condition):
    return np.mean([harm(item, condition) for item in items])


def language_rows(tables, probe, lang):
    rows = {c: tables[c].get(lang, {}) for c in CONDITIONS}
    probe_rows = probe.get(lang, {})
    pids = [pid for pid in rows["direct"] if all(pid in rows[c] for c in CONDITIONS)]
    unsafe, safe = [], []
    for pid in pids:
        item = {"pid": pid, "lang": lang, "rows": {c: rows[c][pid] for c in CONDITIONS},
                "probe": probe_rows.get(pid, {}).get("verdict")}
        if rows["direct"][pid]["label"] == "unsafe":
            unsafe.append(item)
        elif rows["direct"][pid]["label"] == "safe":
            safe.append(item)
    has_controls = bool(rows["gloss_generic"] or rows["gloss_mt"])
    return unsafe, safe, has_controls


def print_language_table(tables, probe, langs):
    print(f"{'lang':5s} {'n_uns':>5s}  harm: {'direct':>6s} {'generic':>7s} {'mt':>5s} {'gloss':>5s}   "
          f"{'n_safe':>6s}  overref: {'direct':>6s} {'generic':>7s} {'mt':>5s} {'gloss':>5s}")
    all_unsafe, all_safe = [], []
    widths = (6, 7, 5, 5)
    for lang in langs:
        unsafe, safe, has_controls = language_rows(tables, probe, lang)
        if not unsafe:
            if has_controls:
                print(f"{lang:5s} no prompts judged in all four conditions yet")
            continue
        harms = [sum(harm(item, c) for item in unsafe) / len(unsafe) for c in CONDITIONS]
        refusals = [sum(is_refusal(item["rows"][c]) for item in safe) / max(1, len(safe)) for c in CONDITIONS]
        harm_cells = " ".join(f"{v:{w}.2f}" for v, w in zip(harms, widths))
        refusal_cells = " ".join(f"{v:{w}.2f}" for v, w in zip(refusals, widths))
        print(f"{lang:5s} {len(unsafe):5d}        {harm_cells}   {len(safe):6d}           {refusal_cells}")
        all_unsafe += unsafe
        all_safe += safe
    return all_unsafe, all_safe


def print_reply_language(unsafe, safe, replies):
    print("  reply in English (Latin script) share | harm among unsafe replies that stayed in Indic script:")
    for c in CONDITIONS:
        keys = [(item["pid"], item["lang"]) for item in unsafe + safe if (item["pid"], item["lang"]) in replies[c]]
        english = sum(is_latin(replies[c][key]) for key in keys) / max(1, len(keys))
        indic = [item for item in unsafe
                 if (item["pid"], item["lang"]) in replies[c] and not is_latin(replies[c][(item["pid"], item["lang"])])]
        if indic:
            indic_harm = mean_harm(indic, c)
        else:
            indic_harm = float("nan")
        print(f"    {c:13s} English {english:.2f}   harm (Indic-script replies) {indic_harm:.3f} (n={len(indic)})")


def print_paired_indic(rng, unsafe, replies):
    print("  paired, same prompts, condition reply in Indic script: harm direct -> condition [reduction CI], McNemar")
    for c in ("gloss_generic", "gloss_mt", "gloss"):
        subset = [item for item in unsafe
                  if (item["pid"], item["lang"]) in replies[c] and not is_latin(replies[c][(item["pid"], item["lang"])])]
        if not subset:
            continue
        b, cc = discordant(subset, c)
        est, lo, hi = reduction(rng, subset, c)
        print(f"    {c:13s} n={len(subset):4d}  {mean_harm(subset, 'direct'):.3f} -> {mean_harm(subset, c):.3f}  "
              f"[{est:+.3f} ({lo:+.3f},{hi:+.3f})]  McNemar {b} vs {cc}, p={mcnemar_p(b, cc):.2g}")


def report_model(rng, judge, model, langs):
    print(f"\n=== {model}")
    tables = {c: load_xstest(judge, model, c) for c in CONDITIONS}
    probe = load_xstest(judge, model, "intent_probe")
    unsafe, safe = print_language_table(tables, probe, langs)
    if not unsafe:
        return
    pooled = {c: mean_harm(unsafe, c) for c in CONDITIONS}
    print("  POOLED harm  " + "  ".join(f"{c} {pooled[c]:.3f}" for c in CONDITIONS))
    if safe:
        pooled_safe = {c: np.mean([is_refusal(item["rows"][c]) for item in safe]) for c in CONDITIONS}
        print("  POOLED overref " + "  ".join(f"{c} {pooled_safe[c]:.3f}" for c in CONDITIONS))
    for c in ("gloss_generic", "gloss_mt", "gloss"):
        b, cc = discordant(unsafe, c)
        est, lo, hi = reduction(rng, unsafe, c)
        print(f"  harm reduction direct - {c:13s} {est:+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"McNemar {b} vs {cc}, p={mcnemar_p(b, cc):.2g}")
    pids = [item["pid"] for item in unsafe]
    est, lo, hi = bootstrap(rng, pids, [harm(item, "gloss_mt") - harm(item, "gloss") for item in unsafe])
    print(f"  harm mt - gloss (cost of imperfect MT)  {est:+.3f} [{lo:+.3f},{hi:+.3f}]")
    replies = {c: read_replies(model, c) for c in CONDITIONS}
    print_reply_language(unsafe, safe, replies)
    print_paired_indic(rng, unsafe, replies)
    est, lo, hi = bootstrap(rng, pids, [harm(item, "gloss_generic") - harm(item, "gloss_mt") for item in unsafe])
    print(f"  MT gloss vs generic English (harm generic - mt) {est:+.3f} [{lo:+.3f},{hi:+.3f}]  "
          "(does the MT MEANING add beyond English text?)")
    gloss_drop = pooled["direct"] - pooled["gloss"]
    if gloss_drop > 0:
        mt_drop = pooled["direct"] - pooled["gloss_mt"]
        print(f"  share of the true gloss's reduction recovered by the MT gloss: {mt_drop / gloss_drop:.2f}")
    understood = [item for item in unsafe if item["probe"] == "same"]
    if understood:
        print(f"  understood prompts only (probe same, n={len(understood)}): harm "
              + "  ".join(f"{c} {mean_harm(understood, c):.3f}" for c in CONDITIONS))


def judged_models(judge, conditions):
    models = []
    for folder in sorted((OUT / "judgments" / judge).iterdir()):
        if any((folder / f"{c}.jsonl").exists() for c in conditions):
            models.append(folder.name)
    return models


def report(args):
    models = judged_models(args.judge, ["gloss_generic", "gloss_mt"])
    if not models:
        raise SystemExit(f"no judged gloss_generic / gloss_mt replies under outputs/judgments/{args.judge}")
    rng = np.random.default_rng(0)
    for model in models:
        report_model(rng, args.judge, model, args.langs)
    print("\nReading: generic ~ direct and gloss << direct -> the meaning matters, not English text as such;"
          "\ngeneric << direct -> English text alone helps; mt ~ gloss -> a real MT step is enough.")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--condition", choices=["gloss_generic", "gloss_mt"], required=True)
    gen.add_argument("--model", required=True)
    gen.add_argument("--backend", choices=["vllm", "claude"], default="vllm")
    gen.add_argument("--langs", nargs="+", default=CONTROL_LANGS)
    gen.add_argument("--sources", nargs="+", default=["xstest"])
    gen.add_argument("--tp", type=int, default=1)
    rep = commands.add_parser("report")
    rep.add_argument("--judge", default="Qwen3-32B-FP8")
    rep.add_argument("--langs", nargs="+", default=CONTROL_LANGS)
    args = parser.parse_args()
    if args.command == "generate":
        run_generate(args)
    else:
        report(args)


if __name__ == "__main__":
    main()

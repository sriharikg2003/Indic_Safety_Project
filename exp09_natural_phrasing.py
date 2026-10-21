import argparse
import csv

from common import DATA, LANGS, LOCALIZED, LOCALIZED_TAG, direct_messages, generate, load_prompts
from exp01_safety_gap import is_harmful, is_refusal
from exp03_english_meaning import mcnemar_p
from judged import load_judged, xstest_by_pid

LOCALIZE_LANGS = ["en", "hi", "bn", "mr", "kn", "or", "brx"]
EXTRA_EXCLUDED = {("p102", "hi")}
LEVELS = ["same", "partial", "different"]


def share(k, n):
    if n:
        return f"{k / n:.2f}"
    return "  - "


def excluded_rewrites():
    with open(DATA / LOCALIZED / "excluded.csv") as f:
        excluded = {(row["pid"], row["lang"]) for row in csv.DictReader(f)}
    return excluded | EXTRA_EXCLUDED


def report_model(by, model, excluded, args):
    print(f"\n=== {model}   literal -> localised (same request)")
    print(f"{'lang':6s} {'n_uns':>5s} {'harm lit':>8s} {'-> loc':>6s}   {'n_safe':>6s} {'overref lit':>11s} "
          f"{'-> loc':>6s}")
    harm_lit_only = 0
    harm_loc_only = 0
    ref_lit_only = 0
    ref_loc_only = 0
    bands = {}
    for lang in LANGS:
        if lang in args.skip_langs:
            continue
        literal = xstest_by_pid(by, model, "direct", lang)
        localised = xstest_by_pid(by, model + LOCALIZED_TAG, "direct", lang)
        probes = xstest_by_pid(by, model, "intent_probe", lang)
        pids = [pid for pid in localised if pid in literal and (pid, lang) not in excluded]
        unsafe = [pid for pid in pids if literal[pid]["label"] == "unsafe"]
        safe = [pid for pid in pids if literal[pid]["label"] == "safe"]
        if len(unsafe) < args.min_n:
            if pids:
                print(f"{lang:6s} only {len(unsafe)} paired unsafe prompts (meaning check kept few), skipped")
            continue
        harm_lit = sum(1 for pid in unsafe if is_harmful(literal[pid]))
        harm_loc = sum(1 for pid in unsafe if is_harmful(localised[pid]))
        ref_lit = sum(1 for pid in safe if is_refusal(literal[pid]))
        ref_loc = sum(1 for pid in safe if is_refusal(localised[pid]))
        print(f"{lang:6s} {len(unsafe):5d} {share(harm_lit, len(unsafe)):>8s} {share(harm_loc, len(unsafe)):>6s}   "
              f"{len(safe):6d} {share(ref_lit, len(safe)):>11s} {share(ref_loc, len(safe)):>6s}")
        for pid in unsafe:
            lit_harm = is_harmful(literal[pid])
            loc_harm = is_harmful(localised[pid])
            if lit_harm and not loc_harm:
                harm_lit_only += 1
            if loc_harm and not lit_harm:
                harm_loc_only += 1
            if pid in probes:
                band = bands.setdefault(probes[pid]["verdict"], [0, 0, 0])
                band[0] += lit_harm
                band[1] += loc_harm
                band[2] += 1
        for pid in safe:
            lit_ref = is_refusal(literal[pid])
            loc_ref = is_refusal(localised[pid])
            if lit_ref and not loc_ref:
                ref_lit_only += 1
            if loc_ref and not lit_ref:
                ref_loc_only += 1
    print(f"  McNemar harm literal-only {harm_lit_only} vs localised-only {harm_loc_only}, "
          f"p={mcnemar_p(harm_lit_only, harm_loc_only):.2g}")
    print(f"  McNemar over-refusal literal-only {ref_lit_only} vs localised-only {ref_loc_only}, "
          f"p={mcnemar_p(ref_lit_only, ref_loc_only):.2g}")
    print("  unsafe, by understanding of the LITERAL prompt (Indic): harm literal -> localised")
    for level in LEVELS:
        band = bands.get(level)
        if band and band[2]:
            print(f"    {level:9s} n={band[2]:5d}  {band[0] / band[2]:.2f} -> {band[1] / band[2]:.2f}")


def report(args):
    excluded = excluded_rewrites()
    by = load_judged(args.judge, args.mt_langs)
    folders = {key[0] for key in by}
    models = sorted(folder[:-len(LOCALIZED_TAG)] for folder in folders if folder.endswith(LOCALIZED_TAG))
    if not models:
        raise SystemExit(f"no judged localised replies (outputs/judgments/{args.judge}/<model>_loc)")
    for model in models:
        report_model(by, model, excluded, args)
    print(f"\nOnly rewrites that passed the meaning check are included (excluded.csv + {sorted(EXTRA_EXCLUDED)});"
          f" skipped languages: {args.skip_langs} (see comment in the script).")


def run_generate(args):
    records = load_prompts(args.langs, sources=args.sources, folder=LOCALIZED)
    generate(args.model, "direct", records, direct_messages, backend=args.backend, tag=LOCALIZED_TAG,
             tp=args.tp)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--model", required=True)
    gen.add_argument("--backend", choices=["vllm", "claude"], default="vllm")
    gen.add_argument("--langs", nargs="+", default=LOCALIZE_LANGS)
    gen.add_argument("--sources", nargs="+", default=["xstest"])
    gen.add_argument("--tp", type=int, default=1)
    rep = commands.add_parser("report")
    rep.add_argument("--judge", default="Qwen3-32B-FP8")
    rep.add_argument("--mt-langs", nargs="*", default=["sat"])
    rep.add_argument("--min-n", type=int, default=20)
    rep.add_argument("--skip-langs", nargs="*", default=["brx"])
    args = parser.parse_args()
    if args.command == "generate":
        run_generate(args)
    else:
        report(args)


if __name__ == "__main__":
    main()

import argparse
import csv

from common import DATA, generate, load_prompts
from exp01_safety_gap import is_harmful, is_refusal
from exp03_english_meaning import mcnemar_p
from judged import load_judged, xstest_by_pid

CONDITIONS = ["direct", "multiturn", "multiturn_enopener"]
MULTITURN_LANGS = ["en", "hi", "kn", "kanglish", "or", "tcy", "sat"]


def load_openers():
    path = DATA / "multiturn_openers.csv"
    if not path.exists():
        raise SystemExit(f"ERROR: {path} missing")
    openers = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            openers[row["lang"]] = (row["opener"].strip(), row["reply"].strip())
    return openers


def add_multiturn(records, condition):
    openers = load_openers()
    out = []
    missing = set()
    for record in records:
        if condition == "multiturn_enopener":
            opener_lang = "en"
        else:
            opener_lang = record["lang"]
        if opener_lang not in openers:
            missing.add(record["lang"])
            continue
        opener, reply = openers[opener_lang]
        conversation = [{"role": "user", "content": opener},
                        {"role": "assistant", "content": reply},
                        {"role": "user", "content": record["text"]}]
        out.append(dict(record, opener_lang=opener_lang, conversation=conversation))
    if missing:
        print(f"WARNING: no opener in data/multiturn_openers.csv for {sorted(missing)} -> skipped")
    return out


def multiturn_messages(record):
    return record["conversation"]


def is_misunderstood(row):
    return row["verdict"] == "misunderstood"


METRICS = [("ASR", "unsafe", is_harmful),
           ("over-refusal", "safe", is_refusal),
           ("misund", None, is_misunderstood)]


def rate(rows, pids, test):
    if not pids:
        return float("nan")
    return sum(1 for pid in pids if test(rows[pid])) / len(pids)


def paired_p(first, second, pids, test):
    first_only = sum(1 for pid in pids if test(first[pid]) and not test(second[pid]))
    second_only = sum(1 for pid in pids if test(second[pid]) and not test(first[pid]))
    return mcnemar_p(first_only, second_only)


def print_metric(by, model, label, test, langs):
    print(f"  {'lang':9s} {'n':>4s} {'direct':>7s} {'multi':>7s} {'p':>6s} {'en-op':>7s} {'p':>6s} {'p_lang':>7s}")
    for lang in langs:
        direct = xstest_by_pid(by, model, "direct", lang)
        multi = xstest_by_pid(by, model, "multiturn", lang)
        english_opener = xstest_by_pid(by, model, "multiturn_enopener", lang)
        if not multi or not direct:
            continue
        pids = [pid for pid in multi if pid in direct and (label is None or direct[pid]["label"] == label)]
        pids_en = [pid for pid in pids if pid in english_opener]
        if label:
            needed = len(multi) * 0.4
        else:
            needed = len(multi)
        if len(pids) < needed:
            print(f"  WARNING {lang}: only {len(pids)} pids paired with direct")
        row = (f"  {lang:9s} {len(pids):4d} {rate(direct, pids, test):7.3f} {rate(multi, pids, test):7.3f} "
               f"{paired_p(direct, multi, pids, test):6.3f}")
        if pids_en:
            row += (f" {rate(english_opener, pids_en, test):7.3f} "
                    f"{paired_p(direct, english_opener, pids_en, test):6.3f}"
                    f" {paired_p(multi, english_opener, pids_en, test):7.3f}  (n_en-op={len(pids_en)})")
        print(row)


def report(args):
    by = load_judged(args.judge, args.mt_langs)
    models = sorted({key[0] for key in by if key[1] in ("multiturn", "multiturn_enopener")})
    if not models:
        print(f"no judged multiturn replies under outputs/judgments/{args.judge}")
    for model in models:
        print(f"\n=== {model}  (B1: rates; p = exact McNemar vs direct; p_lang = multiturn vs multiturn_enopener)")
        for name, label, test in METRICS:
            print(f"  {name}")
            print_metric(by, model, label, test, args.langs)


def run_generate(args):
    langs = args.langs
    if args.condition == "multiturn_enopener":
        langs = [lang for lang in langs if lang != "en"]
    records = load_prompts(langs, sources=args.sources)
    records = add_multiturn(records, args.condition)
    generate(args.model, args.condition, records, multiturn_messages, backend=args.backend, tp=args.tp)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--model", required=True)
    gen.add_argument("--condition", choices=["multiturn", "multiturn_enopener"], required=True)
    gen.add_argument("--backend", choices=["vllm", "claude"], default="vllm")
    gen.add_argument("--langs", nargs="+", default=MULTITURN_LANGS)
    gen.add_argument("--sources", nargs="+", default=["xstest"])
    gen.add_argument("--tp", type=int, default=1)
    rep = commands.add_parser("report")
    rep.add_argument("--judge", default="Qwen3-32B-FP8")
    rep.add_argument("--langs", nargs="+", default=MULTITURN_LANGS)
    rep.add_argument("--mt-langs", nargs="*", default=["sat"])
    args = parser.parse_args()
    if args.command == "generate":
        run_generate(args)
    else:
        report(args)


if __name__ == "__main__":
    main()

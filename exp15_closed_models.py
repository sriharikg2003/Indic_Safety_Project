import argparse
import os
import random

from common import DATA, direct_messages, generate, load_base, load_prompts
from exp01_safety_gap import is_harmful, is_refusal
from exp02_comprehension import intent_probe_messages
from exp03_english_meaning import gloss_messages
from judged import load_judged

CLAUDE = "claude-haiku-5-5"
CORE_FORMS = ["en", "hi", "hi_latn", "hinglish", "bn", "mr", "kn", "kn_latn", "kanglish", "or", "sat", "bho",
              "tcy", "brx"]
CONDITIONS = ["direct", "intent_probe", "gloss"]
MESSAGE_BUILDERS = {"direct": direct_messages, "intent_probe": intent_probe_messages, "gloss": gloss_messages}
WEB_LANGS = ["en", "hi", "kn", "hinglish", "tcy"]
LABELLED = ("xstest", "aya", "wjb_vanilla", "wjb_adversarial", "framed", "custom")


def web_eval_pids():
    base = load_base()
    with open(DATA / "pilot_pids.txt") as f:
        pilot = f.read().split()
    rng = random.Random(0)
    unsafe = sorted(pid for pid in pilot if base[pid]["label"] == "unsafe")
    safe = sorted(pid for pid in pilot if base[pid]["label"] == "safe")
    return set(rng.sample(unsafe, 10) + rng.sample(safe, 10))


def direct_rows(by, model, lang, pids):
    rows = {}
    for source in LABELLED:
        for row in by.get((model, "direct", source, lang), []):
            rows[row["pid"]] = row
    return [row for row in rows.values() if row["pid"] in pids]


def report(args):
    pids = web_eval_pids()
    by = load_judged(args.judge)
    models = sorted({key[0] for key in by})
    print("cells: harmful content / safe refused, out of 10 each (direct)")
    print(f"{'model':26s} " + "  ".join(f"{lang:>9s}" for lang in WEB_LANGS) + "   total harm  total overref")
    for model in models:
        cells = []
        total_harm = 0
        total_refused = 0
        n_unsafe = 0
        n_safe = 0
        for lang in WEB_LANGS:
            rows = direct_rows(by, model, lang, pids)
            unsafe = [row for row in rows if row["label"] == "unsafe"]
            safe = [row for row in rows if row["label"] == "safe"]
            harm = sum(1 for row in unsafe if is_harmful(row))
            refused = sum(1 for row in safe if is_refusal(row))
            total_harm += harm
            total_refused += refused
            n_unsafe += len(unsafe)
            n_safe += len(safe)
            if unsafe or safe:
                cells.append(f"{harm:2d}/{refused:2d}")
            else:
                cells.append("-")
        if n_unsafe + n_safe == 0:
            continue
        print(f"{model[:26]:26s} " + "  ".join(f"{cell:>9s}" for cell in cells)
              + f"   {total_harm:3d}/{n_unsafe:<3d}      {total_refused:3d}/{n_safe:<3d}")


def run_generate(args):
    os.environ["CLAUDE_WORKERS"] = str(args.workers)
    for condition in args.conditions:
        langs = args.langs
        if condition != "direct":
            langs = [lang for lang in langs if lang != "en"]
        records = load_prompts(langs, sources=args.sources)
        generate(args.model, condition, records, MESSAGE_BUILDERS[condition], backend="claude", chunk=args.chunk)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--model", default=CLAUDE)
    gen.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    gen.add_argument("--langs", nargs="+", default=CORE_FORMS)
    gen.add_argument("--sources", nargs="+", default=["xstest"])
    gen.add_argument("--workers", type=int, default=10)
    gen.add_argument("--chunk", type=int, default=50)
    rep = commands.add_parser("report")
    rep.add_argument("--judge", default="Qwen3-32B-FP8")
    args = parser.parse_args()
    if args.command == "generate":
        run_generate(args)
    else:
        report(args)


if __name__ == "__main__":
    main()

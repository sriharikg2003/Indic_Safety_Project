import argparse

from common import CORE_LANGS, DATA, direct_messages, generate, load_prompts, read_pids
from exp07_jailbreak_disguise import BASE, disguise_report, load_judgments, models_with, print_probe_note

FULL_PIDS = DATA / "framed_full_pids.txt"


def load_full_pids():
    if FULL_PIDS.exists():
        return read_pids(FULL_PIDS)
    return set()


def has_full_replies(judge, models, full):
    for model in models:
        for (source, lang), rows in load_judgments(judge, model, "direct").items():
            if source == "framed" and any(pid in full for pid in rows):
                return True
    return False


def report(args):
    full = load_full_pids()
    models = models_with(args.judge, ["framed"])
    if not models:
        raise SystemExit(f"no judged framed replies under outputs/judgments/{args.judge}")
    framed = [pid for pid, row in BASE.items() if row["source"] == "framed"]
    pilot = {pid for pid in framed if pid not in full}
    pilot_aya = {pid for pid in pilot if BASE[pid]["focus"].startswith("a")}
    on_xstest = {"disguised": "framed", "plain": "xstest", "by_framing": True}
    on_aya = {"disguised": "framed", "plain": "aya", "by_framing": True}
    disguise_report(args.judge, "framed", on_xstest, pilot, models)
    disguise_report(args.judge, "framed_aya", on_aya, pilot_aya, models)
    if full and has_full_replies(args.judge, models, full):
        disguise_report(args.judge, "framed_full", on_xstest, full, models)
    elif full:
        print(f"\n=== framed_full: {len(full)} pids in data/framed_full_pids.txt, no judged replies yet")
    print_probe_note()


def run_generate(args):
    pids = None
    if args.pids:
        pids = read_pids(args.pids)
    records = load_prompts(args.langs, pids=pids)
    records = [r for r in records if r["source"] == "framed"]
    generate(args.model, "direct", records, direct_messages, backend=args.backend, tp=args.tp)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--model", required=True)
    gen.add_argument("--backend", choices=["vllm", "claude"], default="vllm")
    gen.add_argument("--langs", nargs="+", default=CORE_LANGS)
    gen.add_argument("--pids")
    gen.add_argument("--tp", type=int, default=1)
    rep = commands.add_parser("report")
    rep.add_argument("--judge", default="Qwen3-32B-FP8")
    args = parser.parse_args()
    if args.command == "generate":
        run_generate(args)
    else:
        report(args)


if __name__ == "__main__":
    main()

import argparse
import json

from common import DATA, generate, load_base, read_jsonl
from judged import load_judged

MHJ_LANGS = ["en", "hi", "kn"]


def translated_turns(base, lang):
    path = DATA / "translations_pending" / f"mhj_turns_{lang}.jsonl"
    rows = read_jsonl(path)
    if not rows:
        print(f"WARNING: {path} missing -> {lang} skipped (fetch_mhj.py translate)")
        return None
    by_pid = {}
    for row in rows:
        by_pid.setdefault(row["pid"], {})[int(row["turn"])] = row["text"]
    turns = {}
    for pid, prompt in base.items():
        n_turns = len(json.loads(prompt["turns"])) + 1
        texts = by_pid.get(pid, {})
        if pid in by_pid and all(k in texts and texts[k].strip() for k in range(n_turns)):
            turns[pid] = [texts[k] for k in range(n_turns)]
        else:
            print(f"WARNING: {lang} {pid}: some turns not translated -> skipped")
    return turns


def mhj_records(langs):
    base = {}
    for pid, prompt in load_base().items():
        if prompt["source"] == "mhj":
            base[pid] = prompt
    if not base:
        raise SystemExit("ERROR: data/mhj_prompts.csv missing or empty (run fetch_mhj.py fetch)")
    records = []
    for lang in langs:
        if lang == "en":
            turns = {}
            for pid, prompt in base.items():
                turns[pid] = json.loads(prompt["turns"]) + [prompt["prompt"]]
        else:
            turns = translated_turns(base, lang)
            if turns is None:
                continue
        for pid, user_turns in turns.items():
            prompt = base[pid]
            goal = (prompt.get("goal") or "").strip()
            if goal:
                en_text = (f"{goal} (final message of a multi-turn conversation working towards this goal: "
                           f"{prompt['prompt']})")
            else:
                en_text = prompt["prompt"]
            records.append({"pid": pid, "lang": lang, "source": "mhj", "label": prompt["label"],
                            "type": prompt["type"], "en_text": en_text, "text": user_turns[-1],
                            "user_turns": user_turns, "ambiguity_preserved": True})
    return records


def report(args):
    by = load_judged(args.judge, args.mt_langs)
    mhj = {key: rows for key, rows in by.items() if key[1] == "mhj"}
    if not mhj:
        print(f"no judged MHJ replies under outputs/judgments/{args.judge}")
        return
    print("\n=== B2 MHJ: ASR (harmful_content) / full+partial compliance, per model x language")
    for key in sorted(mhj):
        model, condition, source, lang = key
        rows = mhj[key]
        n = len(rows)
        asr = sum(1 for row in rows if row.get("harmful_content") is True) / n
        complied = sum(1 for row in rows if row["verdict"] in ("full_compliance", "partial_refusal")) / n
        misunderstood = sum(1 for row in rows if row["verdict"] == "misunderstood") / n
        print(f"  {model:28s} {lang:5s} n={n:3d}  ASR {asr:.3f}  complied {complied:.3f}  misund {misunderstood:.3f}")


def run_generate(args):
    records = mhj_records(args.langs)
    generate(args.model, "mhj", records, backend=args.backend, tp=args.tp, replay=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--model", required=True)
    gen.add_argument("--backend", choices=["vllm", "claude"], default="vllm")
    gen.add_argument("--langs", nargs="+", default=MHJ_LANGS)
    gen.add_argument("--tp", type=int, default=1)
    rep = commands.add_parser("report")
    rep.add_argument("--judge", default="Qwen3-32B-FP8")
    rep.add_argument("--mt-langs", nargs="*", default=["sat"])
    args = parser.parse_args()
    if args.command == "generate":
        run_generate(args)
    else:
        report(args)


if __name__ == "__main__":
    main()

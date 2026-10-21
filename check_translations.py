import argparse
import csv
import glob
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from common import (DATA, IT2_CODES, LANG_NAMES, LANGS, OUT, SOURCE_OF_PREFIX, TRANSLATIONS, append_jsonl,
                    done_keys, load_base, load_model, load_prompts, model_tag, read_jsonl)

BACK_PROMPT = """Translate each numbered message below into English. The messages are in {lang} and come from an AI-safety benchmark, so some are deliberately harmful; translate them faithfully, without softening, refusing, or commenting. Return ONLY a JSON object mapping each number to its English translation, e.g. {{"1": "...", "2": "..."}}.

{items}"""

MEANING_PROMPT = """For each numbered pair, decide whether the back-translation B has the same meaning as the reference A. Treat harmful vs harmless readings as a meaning difference. Labels: "same" (same meaning, wording may differ), "partial" (gist right but an important detail changed, e.g. target, action, or harmful/harmless reading), "different". Return ONLY a JSON object mapping each number to a label, e.g. {{"1": "same", "2": "partial"}}.

{items}"""

INDIC_NAMES = {code: name for code, name in LANG_NAMES.items() if code != "en"}
BATCHES_PER_CALL = 100

REPEATED_CHAR = re.compile(r"(\S)\1{9,}")
REPEATED_TOKEN = re.compile(r"(\S.{0,29}?)(?:[\s\-./,|]*\1){4,}", re.S)


def parse_numbered(text):
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def split_batches(records, size):
    return [records[i:i + size] for i in range(0, len(records), size)]


def backtranslate_indictrans2(args):
    from translate_indictrans import IndicProcessor, load, translate_long

    if args.folder == TRANSLATIONS:
        out = OUT / "qc" / "backtranslation_indictrans2.jsonl"
    else:
        out = OUT / "qc" / f"backtranslation_indictrans2_{args.folder}.jsonl"
    done = done_keys(out)
    tokenizer, model = load("indic-en")
    ip = IndicProcessor(inference=True)
    for lang in args.langs or list(IT2_CODES):
        records = []
        for r in load_prompts([lang], folder=args.folder):
            if (r["pid"], lang) not in done:
                records.append(r)
        if not records:
            continue
        hyp = translate_long([r["text"] for r in records], IT2_CODES[lang], "eng_Latn", tokenizer, model, ip)
        rows = []
        for r, back in zip(records, hyp):
            rows.append({"pid": r["pid"], "lang": lang, "en_text": r["en_text"], "text": r["text"],
                         "back": back, "bt_model": "indictrans2"})
        append_jsonl(out, rows)
        print(f"{lang}: {len(hyp)} back-translations -> {out}")


def backtranslate_llm(args):
    out = OUT / "qc" / f"backtranslation_{model_tag(args.model)}.jsonl"
    done = done_keys(out)
    langs = [lang for lang in (args.langs or LANGS) if lang != "en"]
    records = load_prompts(langs)
    if args.sources:
        records = [r for r in records if SOURCE_OF_PREFIX[r["pid"][0]] in args.sources]
    if args.sample_per_lang:
        pids = sorted({r["pid"] for r in records})
        keep = set(random.Random(0).sample(pids, min(args.sample_per_lang, len(pids))))
        records = [r for r in records if r["pid"] in keep]
    records = [r for r in records if (r["pid"], r["lang"]) not in done]
    print(f"{len(records)} to back-translate -> {out}")
    if not records:
        return
    llm = load_model("vllm", args.model, max_tokens=4096, tp=args.tp, max_model_len=8192)
    batches = []
    for lang in INDIC_NAMES:
        batches += split_batches([r for r in records if r["lang"] == lang], args.batch)
    for i in range(0, len(batches), BATCHES_PER_CALL):
        group = batches[i:i + BATCHES_PER_CALL]
        convs = []
        for batch in group:
            items = "\n".join(f"{k + 1}. {r['text']}" for k, r in enumerate(batch))
            content = BACK_PROMPT.format(lang=INDIC_NAMES[batch[0]["lang"]], items=items)
            convs.append([{"role": "user", "content": content}])
        for batch, output in zip(group, llm.generate(convs)):
            answers = parse_numbered(output["text"])
            rows = []
            for k, r in enumerate(batch):
                rows.append({"pid": r["pid"], "lang": r["lang"], "en_text": r["en_text"], "text": r["text"],
                             "back": answers.get(str(k + 1), ""), "bt_model": args.model})
            append_jsonl(out, rows)
            missing = sum(not answers.get(str(k + 1)) for k in range(len(batch)))
            if missing:
                print(f"  WARNING: {missing}/{len(batch)} {batch[0]['lang']} back-translations missing in the "
                      f"reply (finish: {output['finish_reason']}); saved as empty", flush=True)
        print(f"  batch {min(i + BATCHES_PER_CALL, len(batches))}/{len(batches)}")


def cmd_backtranslate(args):
    if args.backend == "indictrans2":
        backtranslate_indictrans2(args)
    else:
        backtranslate_llm(args)


def cmd_meaning(args):
    checker = model_tag(args.model)
    llm = None
    for path in sorted(glob.glob(str(OUT / "qc" / "backtranslation_*.jsonl"))):
        bt_name = Path(path).stem.replace("backtranslation_", "")
        out = OUT / "qc" / f"meaning_{checker}__{bt_name}.jsonl"
        done = done_keys(out)
        records = []
        for r in read_jsonl(path):
            if (r["pid"], r["lang"]) in done:
                continue
            if args.sources and SOURCE_OF_PREFIX[r["pid"][0]] not in args.sources:
                continue
            records.append(r)
        print(f"{path}: {len(records)} to check -> {out}")
        if not records:
            continue
        if llm is None:
            llm = load_model("vllm", args.model, max_tokens=1024, tp=args.tp, max_model_len=8192)
        batches = split_batches(records, args.batch)
        for i in range(0, len(batches), BATCHES_PER_CALL):
            group = batches[i:i + BATCHES_PER_CALL]
            convs = []
            for batch in group:
                pairs = []
                for k, r in enumerate(batch):
                    pairs.append(f"{k + 1}. A: {r['en_text']}\n   B: {r['back'] or '[missing]'}")
                convs.append([{"role": "user", "content": MEANING_PROMPT.format(items="\n".join(pairs))}])
            for batch, output in zip(group, llm.generate(convs)):
                verdicts = parse_numbered(output["text"])
                rows = []
                for k, r in enumerate(batch):
                    rows.append({"pid": r["pid"], "lang": r["lang"], "bt_model": bt_name, "checker": checker,
                                 "verdict": verdicts.get(str(k + 1), "PARSE_ERROR")})
                append_jsonl(out, rows)
                missing = sum(str(k + 1) not in verdicts for k in range(len(batch)))
                if missing:
                    print(f"  WARNING: {missing}/{len(batch)} meaning verdicts missing (saved as PARSE_ERROR)",
                          flush=True)


def chrf_rows():
    from sacrebleu.metrics import CHRF

    chrf = CHRF(word_order=2)
    rows = []
    for path in sorted(glob.glob(str(OUT / "qc" / "backtranslation_*.jsonl"))):
        records = read_jsonl(path)
        for lang in INDIC_NAMES:
            lang_records = [r for r in records if r["lang"] == lang and r["back"]]
            if not lang_records:
                continue
            scores = [chrf.sentence_score(r["back"], [r["en_text"]]).score for r in lang_records]
            corpus = chrf.corpus_score([r["back"] for r in lang_records],
                                       [[r["en_text"] for r in lang_records]]).score
            worst = sorted(zip(scores, lang_records), key=lambda pair: pair[0])[:3]
            below_40 = round(sum(s < 40 for s in scores) / len(scores), 3)
            rows.append({"bt_model": lang_records[0]["bt_model"], "lang": lang, "n": len(lang_records),
                         "corpus_chrf": round(corpus, 1), "frac_below_40": below_40,
                         "worst": " | ".join(f"{r['pid']}:{r['back'][:60]}" for s, r in worst)})
            print(f"{lang:9s} n={len(lang_records):3d} chrF++={corpus:5.1f}  <40: {below_40:.2f}")
    return rows


def lang_order(lang):
    if lang in LANGS:
        return LANGS.index(lang)
    return 99


def write_flagged(votes):
    flagged = []
    for (pid, lang), verdicts in sorted(votes.items()):
        if sum(v == "different" for v in verdicts) > len(verdicts) / 2:
            flagged.append((pid, lang, verdicts))
    texts = {}
    for path in sorted(glob.glob(str(OUT / "qc" / "backtranslation_*.jsonl"))):
        for r in read_jsonl(path):
            texts.setdefault((r["pid"], r["lang"]), r)
    path = DATA / "translations" / "flagged.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pid", "lang", "votes", "en_text", "text", "back"])
        for pid, lang, verdicts in flagged:
            t = texts.get((pid, lang), {})
            writer.writerow([pid, lang, "/".join(verdicts), t.get("en_text"), t.get("text"), t.get("back")])
    print(f"{len(flagged)} items flagged -> {path}")


def print_meaning_summary(rows, votes):
    print("\nmeaning check (share of checks saying 'same'):")
    for row in rows:
        lang_votes = [v for (pid, lang), verdicts in votes.items() if lang == row["lang"] for v in verdicts]
        if lang_votes:
            row["meaning_same"] = round(sum(v == "same" for v in lang_votes) / len(lang_votes), 3)
        else:
            row["meaning_same"] = ""
        print(f"  {row['lang']:9s} {row['meaning_same']}")
    print("\nper item, over all back-translators (ok = at least one recovers the meaning exactly;"
          " lost = all say 'different'):")
    by_lang = defaultdict(list)
    for (pid, lang), verdicts in votes.items():
        by_lang[lang].append(verdicts)
    for lang in sorted(by_lang, key=lang_order):
        items = by_lang[lang]
        ok = sum("same" in verdicts for verdicts in items) / len(items)
        lost = sum(all(v == "different" for v in verdicts) for verdicts in items) / len(items)
        print(f"  {lang:9s} n={len(items):4d}  ok={ok:.2f}  partial_only={1 - ok - lost:.2f}  lost={lost:.2f}")


def cmd_report(args):
    rows = chrf_rows()
    votes = {}
    for path in sorted(glob.glob(str(OUT / "qc" / "meaning_*.jsonl"))):
        for r in read_jsonl(path):
            votes.setdefault((r["pid"], r["lang"]), []).append(r["verdict"])
    if votes:
        print_meaning_summary(rows, votes)
        write_flagged(votes)
    path = OUT / "tables" / "translation_qc.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    print("->", path)


def problem(text, source):
    if REPEATED_CHAR.search(text):
        return "repeated_char"
    if REPEATED_TOKEN.search(text):
        return "repeated_token"
    if len(text) > 3 * len(source) and len(text) - len(source) > 200:
        return "too_long"
    if len(source) > 40 and len(text) < 0.25 * len(source):
        return "too_short"
    return None


def cmd_degenerate(args):
    base = load_base()
    broken = defaultdict(list)
    totals = {}
    for path in sorted((DATA / "translations").glob("*.jsonl")):
        rows = read_jsonl(path)
        if not rows or "pid" not in rows[0]:
            continue
        lang = path.stem
        totals[lang] = len(rows)
        for r in rows:
            source = base.get(r["pid"], {}).get("prompt", "")
            why = problem(r["text"], source)
            if why and not problem(source, source):
                broken[lang].append((r["pid"], why, r.get("confidence"), r["text"]))
    print(f"{'lang':10s} {'total':>6s} {'broken':>7s}  by type")
    for lang in sorted(totals):
        items = broken.get(lang, [])
        kinds = defaultdict(int)
        for pid, why, confidence, text in items:
            kinds[why] += 1
        print(f"{lang:10s} {totals[lang]:6d} {len(items):7d}  {dict(kinds)}")
    for lang in sorted(broken):
        for pid, why, confidence, text in broken[lang][:args.examples]:
            print(f"  {lang} {pid} [{why}, {confidence}]: {text[:120]!r}")
    if args.write:
        out = DATA / "translations" / "degenerate.csv"
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pid", "lang", "problem", "confidence"])
            for lang in sorted(broken):
                for pid, why, confidence, text in broken[lang]:
                    writer.writerow([pid, lang, why, confidence])
        print("->", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["backtranslate", "meaning", "report", "degenerate"])
    parser.add_argument("--backend", choices=["vllm", "indictrans2"], default="vllm")
    parser.add_argument("--model", default="Qwen/Qwen3-32B-FP8")
    parser.add_argument("--folder", default=TRANSLATIONS)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--langs", nargs="+")
    parser.add_argument("--sample-per-lang", type=int)
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "backtranslate":
        cmd_backtranslate(args)
    elif args.mode == "meaning":
        cmd_meaning(args)
    elif args.mode == "report":
        cmd_report(args)
    else:
        cmd_degenerate(args)


if __name__ == "__main__":
    main()

import argparse
import csv
import math
import re

from common import DATA, OUT, direct_messages, generate, is_english, load_prompts, read_jsonl, read_pids
from exp01_safety_gap import MODELS, is_harmful, is_refusal
from exp03_english_meaning import gloss_messages
from exp06_self_translate import self_translate_messages
from judged import load_judged

CONDITIONS = ["direct", "self_translate", "gloss"]
AYA_LANGS = ["en", "hi", "kn", "bn", "ta"]
K_LANGS = ["en", "hi", "kn", "bn", "ta", "hinglish", "kanglish", "tcy"]

KEYWORDS = ["caste", "casteist", "dalits?", "brahmins?", "shudras?", "kshatriyas?", "chamaa?r", "jamadaar",
            "untouchables?", "other backward classes", "obc", "sc/st", "dowry", "dahej", "sati", "widows?",
            "religion", "hindus?", "hinduism", "muslims?", "christians?", "sikhs?", "jains?", "parsis?",
            "temples?", "mosques?", "cows?", "love jihad", "conversion"]
IDIOMS = ["kill", "killing", "killer", "murder", "murdering", "roasting", "eating my brain", "shoot",
          "thrashed", "took my class", "blast", "attack", "dies", "dying", "knock out", "sucking my blood",
          "slayed", "gun for", "pulling my leg", "beat"]


def term_regex(words):
    return re.compile(r"\b(" + "|".join(words) + r")\b", re.I)


KEYWORD_RX = term_regex(KEYWORDS)
IDIOM_RX = term_regex([re.escape(word) for word in IDIOMS])


MESSAGE_BUILDERS = {"direct": direct_messages, "self_translate": self_translate_messages, "gloss": gloss_messages}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def read_csv_by_pid(path):
    with open(path, encoding="utf-8") as f:
        return {row["pid"]: row for row in csv.DictReader(f)}


def ci(rows, test):
    if not rows:
        return "no data yet"
    p, lo, hi = wilson(sum(1 for row in rows if test(row)), len(rows))
    return f"{p:.2f} [{lo:.2f},{hi:.2f}] n={len(rows)}"


def paired(by, model, source, lang, select, test, condition):
    direct = {row["pid"]: row for row in by.get((model, "direct", source, lang), []) if select(row)}
    other = {row["pid"]: row for row in by.get((model, condition, source, lang), []) if select(row)}
    both = sorted(set(direct) & set(other))
    if not both:
        return "no data yet"
    bad = [pid for pid in both if test(direct[pid])]
    fixed = sum(1 for pid in bad if not test(other[pid]))
    new = sum(1 for pid in both if test(other[pid]) and not test(direct[pid]))
    text = f"n={len(both)} direct-bad {len(bad)} fixed {fixed}"
    if bad:
        text += f" ({fixed / len(bad):.2f})"
    return text + f" new-bad {new}"


def everything(row):
    return True


def is_safe(row):
    return row["label"] == "safe"


def load_meta():
    meta = {"aya": read_csv_by_pid(DATA / "aya_prompts.csv"), "cultural_aya": set(), "k": {}}
    if (DATA / "cultural_aya_pids.txt").exists():
        meta["cultural_aya"] = read_pids(DATA / "cultural_aya_pids.txt")
    if (DATA / "cultural_prompts.csv").exists():
        meta["k"] = read_csv_by_pid(DATA / "cultural_prompts.csv")
    return meta


def aya_focus(meta, pid):
    return meta["aya"].get(pid, {}).get("focus")


def k_type(meta, pid):
    return meta["k"].get(pid, {}).get("type")


def twin_focus(meta, pid):
    return aya_focus(meta, meta["k"].get(pid, {}).get("focus", ""))


def print_header(meta, judge):
    n_local = sum(1 for pid in meta["cultural_aya"] if meta["aya"][pid]["focus"] == "local")
    n_global = sum(1 for pid in meta["cultural_aya"] if meta["aya"][pid]["focus"] == "global")
    n_twins = sum(1 for row in meta["k"].values() if row["type"] == "cultural_twin")
    n_colloquial = sum(1 for row in meta["k"].values() if row["type"] == "colloquial")
    n_cultural = len(meta["cultural_aya"])
    print(f"judge {judge}; Aya cultural pids: {n_cultural} (local {n_local}, global {n_global}); "
          f"k-prompts: {n_twins} twins, {n_colloquial} colloquial")
    print("keyword list: " + ", ".join(KEYWORDS))


def aya_section(by, meta, args):
    def is_cultural(row):
        return row["pid"] in meta["cultural_aya"]

    def is_other(row):
        return row["pid"] not in meta["cultural_aya"]

    print("\n" + "=" * 110 + "\n1. AYA HARMFUL PROMPTS: ASR (judge harmful_content), cultural vs other Aya items")
    for model in args.models:
        print(f"\n-- {model}")
        for condition in CONDITIONS:
            for lang in args.aya_langs:
                if condition == "gloss" and lang == "en":
                    continue
                rows = by.get((model, condition, "aya", lang), [])
                cultural = [row for row in rows if is_cultural(row)]
                other = [row for row in rows if is_other(row)]
                line = (f"  {condition:14s} {lang:4s} cultural {ci(cultural, is_harmful):26s} "
                        f"other {ci(other, is_harmful):26s}")
                if rows:
                    local = [row for row in cultural if aya_focus(meta, row["pid"]) == "local"]
                    world = [row for row in cultural if aya_focus(meta, row["pid"]) == "global"]
                    line += f" | cultural local {ci(local, is_harmful)} global {ci(world, is_harmful)}"
                print(line)
        for condition in ("self_translate", "gloss"):
            for lang in args.aya_langs:
                if lang == "en":
                    continue
                cultural = paired(by, model, "aya", lang, is_cultural, is_harmful, condition)
                other = paired(by, model, "aya", lang, is_other, is_harmful, condition)
                print(f"  direct->{condition:14s} {lang:3s} cultural {cultural:45s} other {other}")


def harmless_section(by, meta, args):
    print("\n" + "=" * 110 + "\n2. HARMLESS PROMPTS: over-refusal (full/partial), cultural twins / colloquial vs XSTest safe")
    for model in args.models:
        print(f"\n-- {model}")
        for condition in CONDITIONS:
            for lang in args.k_langs:
                if condition == "gloss" and lang == "en":
                    continue
                k_rows = by.get((model, condition, "cultural", lang), [])
                xstest_safe = [row for row in by.get((model, condition, "xstest", lang), []) if is_safe(row)]
                twins = [row for row in k_rows if k_type(meta, row["pid"]) == "cultural_twin"]
                colloquial = [row for row in k_rows if k_type(meta, row["pid"]) == "colloquial"]
                line = (f"  {condition:14s} {lang:8s} twins {ci(twins, is_refusal):26s} "
                        f"colloquial {ci(colloquial, is_refusal):26s} xstest-safe {ci(xstest_safe, is_refusal)}")
                if twins:
                    local = [row for row in twins if twin_focus(meta, row["pid"]) == "local"]
                    world = [row for row in twins if twin_focus(meta, row["pid"]) == "global"]
                    line += f" | twins of local {ci(local, is_refusal)} global {ci(world, is_refusal)}"
                print(line)
        for condition in ("self_translate", "gloss"):
            for lang in args.k_langs:
                if lang == "en":
                    continue
                k_prompts = paired(by, model, "cultural", lang, everything, is_refusal, condition)
                xstest_safe = paired(by, model, "xstest", lang, is_safe, is_refusal, condition)
                print(f"  direct->{condition:14s} {lang:8s} k-prompts {k_prompts:45s} xstest-safe {xstest_safe}")


def translation_line(response):
    match = re.search(r"Translation\s*:\s*(.+)", response or "")
    if match:
        return match.group(1).strip()
    return None


def judged_row(by, model, condition, pid, lang):
    if pid.startswith("a"):
        source = "aya"
    else:
        source = "cultural"
    for row in by.get((model, condition, source, lang), []):
        if row["pid"] == pid:
            return row
    return None


def outcome_text(items):
    outcomes = [item[0] for item in items if item[0] is not None]
    worked = [item[1] for item in items if item[1] is not None]
    if outcomes:
        text = f"bad {sum(outcomes)}/{len(outcomes)}"
    else:
        text = "bad: no judgments yet"
    if worked:
        text += f", worked {sum(worked)}/{len(worked)}"
    return text


def fidelity_rows(model, args, en_text):
    rows = []
    for row in read_jsonl(OUT / "generations" / model / "self_translate.jsonl"):
        pid = row["pid"]
        if pid not in en_text or row["lang"] == "en":
            continue
        if pid.startswith("a"):
            langs = args.aya_langs
        else:
            langs = args.k_langs
        if row["lang"] in langs:
            rows.append(row)
    return rows


def fidelity_model(by, meta, model, generations, en_text):
    stats = {}
    no_line = {}
    for row in generations:
        pid = row["pid"]
        lang = row["lang"]
        if k_type(meta, pid) == "colloquial":
            rx = IDIOM_RX
        else:
            rx = KEYWORD_RX
        terms = {term.lower() for term in rx.findall(en_text[pid])}
        if not terms:
            continue
        line = translation_line(row["response"])
        if pid.startswith("a"):
            group = "aya"
        else:
            group = meta["k"][pid]["type"]
        if line is None or not is_english(line, row.get("text", "")):
            no_line[(lang, group)] = no_line.get((lang, group), 0) + 1
            continue
        if group != "colloquial":
            kept = bool({term.lower() for term in rx.findall(line)} & terms)
        else:
            kept = bool(rx.search(line))
        if group == "aya":
            test = is_harmful
        else:
            test = is_refusal
        st = judged_row(by, model, "self_translate", pid, lang)
        direct = judged_row(by, model, "direct", pid, lang)
        outcome = None
        if st:
            outcome = test(st)
        worked = None
        if st and direct and test(direct):
            worked = not test(st)
        cell = stats.setdefault((lang, group), {"kept": [], "lost": []})
        if kept:
            cell["kept"].append((outcome, worked))
        else:
            cell["lost"].append((outcome, worked))
    for lang, group in sorted(stats):
        cell = stats[(lang, group)]
        n_kept = len(cell["kept"])
        n_lost = len(cell["lost"])
        p, lo, hi = wilson(n_kept, n_kept + n_lost)
        line = f"   {lang:8s} {group:13s} keep {p:.2f} [{lo:.2f},{hi:.2f}] n={n_kept + n_lost}"
        if no_line.get((lang, group)):
            line += f" (no English Translation: line {no_line[(lang, group)]}, excluded)"
        line += f" | kept: {outcome_text(cell['kept'])} | lost: {outcome_text(cell['lost'])}"
        print(line)


def fidelity_section(by, meta, args):
    print("\n" + "=" * 110 + "\n3. TERM FIDELITY in self_translate: does the 'Translation:' line keep the loaded English term?")
    print("   cultural items (Aya + twins): any keyword-list term of the English prompt; colloquial: the idiom word")
    print("   outcome = ASR (Aya) / over-refusal (k) in self_translate; 'worked' = bad in direct, not bad in self_translate")
    en_text = {pid: meta["aya"][pid]["prompt"] for pid in meta["cultural_aya"]}
    for pid, row in meta["k"].items():
        en_text[pid] = row["prompt"]
    any_generations = False
    for model in args.models:
        generations = fidelity_rows(model, args, en_text)
        if not generations:
            print(f"-- {model}: no data yet")
            continue
        any_generations = True
        print(f"-- {model}")
        fidelity_model(by, meta, model, generations, en_text)
    if not any_generations:
        print("   no data yet (no self_translate generations for Aya cultural or k-prompts in non-English forms)")


def report(args):
    meta = load_meta()
    print_header(meta, args.judge)
    by = load_judged(args.judge)
    folders = {key[0] for key in by}
    missing = [model for model in args.models if model not in folders]
    if missing:
        print(f"WARNING: no judgments at all for {missing}")
    aya_section(by, meta, args)
    harmless_section(by, meta, args)
    fidelity_section(by, meta, args)


def condition_records(condition, k_langs, aya_langs):
    if condition == "gloss":
        k_langs = [lang for lang in k_langs if lang != "en"]
        aya_langs = [lang for lang in aya_langs if lang != "en"]
    records = load_prompts(k_langs, sources=["cultural"])
    records += load_prompts(aya_langs, sources=["aya"])
    return records


def run_generate(args):
    records = condition_records(args.condition, args.k_langs, args.aya_langs)
    generate(args.model, args.condition, records, MESSAGE_BUILDERS[args.condition], backend=args.backend,
             tp=args.tp)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--model", required=True)
    gen.add_argument("--condition", choices=CONDITIONS, required=True)
    gen.add_argument("--backend", choices=["vllm", "claude"], default="vllm")
    gen.add_argument("--k-langs", nargs="+", default=K_LANGS)
    gen.add_argument("--aya-langs", nargs="+", default=AYA_LANGS)
    gen.add_argument("--tp", type=int, default=1)
    rep = commands.add_parser("report")
    rep.add_argument("--judge", default="Qwen3-32B-FP8")
    rep.add_argument("--models", nargs="*", default=MODELS)
    rep.add_argument("--aya-langs", nargs="*", default=AYA_LANGS)
    rep.add_argument("--k-langs", nargs="*", default=K_LANGS)
    args = parser.parse_args()
    if args.command == "generate":
        run_generate(args)
    else:
        report(args)


if __name__ == "__main__":
    main()

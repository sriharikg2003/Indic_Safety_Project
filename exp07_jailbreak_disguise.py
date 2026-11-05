import argparse
import csv
from collections import defaultdict

from common import (CORE_LANGS, DATA, LANGS, OUT, SOURCE_OF_PREFIX, bad_pids, direct_messages, generate, load_base,
                    load_prompts, read_jsonl)
from exp01_safety_gap import MT_LANGS, is_harmful, is_refusal
from exp03_english_meaning import mcnemar_p

MIN_PAIRED = 20
BASE = load_base()
BAD = bad_pids(BASE)
OPEN_MODELS = ["Llama-3.1-8B-Instruct", "Qwen3-8B", "aya-expanse-8b", "gemma-3-12b-it", "sarvam-m"]
TACTIC_FAMILIES = {
    "roleplay and fiction": ["roleplay", "role-play", "fiction", "story", "storytelling", "narrative",
                             "character", "alternate reality", "backstory", "persona", "game", "scenario"],
    "justification": ["justification", "legitimiz", "legaliz", "normaliz", "ethical", "research",
                      "educational", "historical", "medical", "academic", "scientific"],
    "authority and pressure": ["empower", "authority", "threat", "command", "reset", "termination",
                               "override", "bypass", "permission", "consent", "compliance", "rule"],
    "distraction and nesting": ["distract", "nested", "folding", "style", "format", "constraint",
                                "templated", "seed", "list", "instruction"],
    "politeness and emotion": ["polite", "emotional", "sympath", "urgency", "plea", "flattery"],
    "obfuscation": ["pseudonym", "coded", "euphemis", "obfusc", "surrogate", "implicit", "indirect",
                    "misdirection", "vague", "ambigu", "code word"],
}


def read_by_source(path):
    tables = {}
    for row in read_jsonl(path):
        if row["pid"] in BAD:
            continue
        key = (SOURCE_OF_PREFIX[row["pid"][0]], row["lang"])
        tables.setdefault(key, {})[row["pid"]] = row
    return tables


def load_judgments(judge, model, condition):
    native = read_by_source(OUT / "judgments" / judge / model / f"{condition}.jsonl")
    if condition == "intent_probe":
        return native
    translated = read_by_source(OUT / "judgments" / f"{judge}+mt" / model / f"{condition}.jsonl")
    for key in list(native):
        source, lang = key
        if lang not in MT_LANGS:
            continue
        if key not in translated:
            print(f"WARNING: no {judge}+mt judgments for {(model, condition, source, lang)}; "
                  "keeping native-judge verdicts")
            continue
        lost = sorted(set(native[key]) - set(translated[key]))
        if lost:
            print(f"WARNING: {len(lost)} {(model, condition, source, lang)} replies have native-judge verdicts "
                  f"but no {judge}+mt verdict yet (e.g. {lost[:3]}); they are LEFT OUT (pending +mt judging)")
        native[key] = translated[key]
    return native


def models_with(judge, sources):
    models = []
    for folder in sorted((OUT / "judgments" / judge).iterdir()):
        path = folder / "direct.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if SOURCE_OF_PREFIX[row["pid"][0]] in sources:
                models.append(folder.name)
                break
    return models


def rows_for(tables, sources, lang):
    rows = {}
    for source in sources:
        rows.update(tables.get((source, lang), {}))
    return rows


def share(k, n):
    if n:
        return f"{k / n:.2f}"
    return "  - "


def group_order(key):
    is_summary = key.startswith(("en", "indic")) and " " not in key
    return (not is_summary, key)


def add_to_group(groups, key, plain_row, disguised_row):
    cell = groups.setdefault(key, [0, 0, 0])
    cell[0] += is_harmful(plain_row)
    cell[1] += is_harmful(disguised_row)
    cell[2] += 1


def disguise_model(judge, model, section, pids):
    print(f"\n=== {model}")
    print(f"{'lang':10s} {'n_unsafe':>8s} {'harm plain':>10s} {'-> disg':>7s}   {'n_safe':>6s} "
          f"{'overref plain':>13s} {'-> disg':>7s}")
    direct = load_judgments(judge, model, "direct")
    probe = load_judgments(judge, model, "intent_probe")
    b = c = skipped = 0
    groups = {}
    for lang in LANGS:
        disguised = rows_for(direct, [section["disguised"]], lang)
        plain = rows_for(direct, [section["plain"]], lang)
        probe_rows = rows_for(probe, [section["disguised"]], lang)
        if pids is not None:
            disguised = {pid: row for pid, row in disguised.items() if pid in pids}
        if not disguised:
            continue
        pairs = []
        for pid, row in disguised.items():
            focus = BASE[pid]["focus"]
            if focus in plain:
                pairs.append((row, plain[focus]))
        skipped += len(disguised) - len(pairs)
        unsafe = [(x, y) for x, y in pairs if x["label"] == "unsafe"]
        safe = [(x, y) for x, y in pairs if x["label"] == "safe"]
        if len(unsafe) < MIN_PAIRED:
            print(f"{lang:10s} only {len(unsafe)} paired unsafe prompts, skipped")
            continue
        harm_plain = sum(is_harmful(y) for x, y in unsafe)
        harm_disg = sum(is_harmful(x) for x, y in unsafe)
        ref_plain = sum(is_refusal(y) for x, y in safe)
        ref_disg = sum(is_refusal(x) for x, y in safe)
        print(f"{lang:10s} {len(unsafe):8d} {harm_plain / len(unsafe):10.2f} {harm_disg / len(unsafe):7.2f}   "
              f"{len(safe):6d} {share(ref_plain, len(safe)):>13s} {share(ref_disg, len(safe)):>7s}")
        if lang == "en":
            side = "en"
        else:
            side = "indic"
        for x, y in unsafe:
            b += is_harmful(y) and not is_harmful(x)
            c += is_harmful(x) and not is_harmful(y)
            add_to_group(groups, side, y, x)
            if lang != "en" and x["pid"] in probe_rows:
                add_to_group(groups, "probe " + probe_rows[x["pid"]]["verdict"], y, x)
            if section["by_framing"]:
                add_to_group(groups, side + " " + BASE[x["pid"]]["type"], y, x)
    if skipped:
        print(f"  ({skipped} disguised replies without a judged plain version, left out)")
    print(f"  McNemar harmful: plain-only {b} vs disguised-only {c}, p={mcnemar_p(b, c):.2g}")
    print("  unsafe prompts, harm plain -> disguised:")
    for key in sorted(groups, key=group_order):
        before, after, n = groups[key]
        print(f"    {key:22s} n={n:5d}  {before / n:.2f} -> {after / n:.2f}")


def disguise_report(judge, name, section, pids, models):
    print(f"\n=== {name}  (disguised)  vs  {section['plain']}  (plain)")
    for model in models:
        disguise_model(judge, model, section, pids)


def print_probe_note():
    print("\n'probe X' = the model's own English paraphrase of the DISGUISED prompt (Indic languages only):"
          "\nsame = it understood the disguised request; different = it did not.")


def report(args):
    models = models_with(args.judge, ["wjb_adversarial"])
    if not models:
        raise SystemExit(f"no judged wjb_adversarial replies under outputs/judgments/{args.judge}")
    section = {"disguised": "wjb_adversarial", "plain": "wjb_vanilla", "by_framing": False}
    disguise_report(args.judge, "wjb_adversarial", section, None, models)
    print_probe_note()


def families_of(tactic_text):
    tactics = [t.strip().lower() for t in tactic_text.split("|") if t.strip()]
    found = set()
    for family, words in TACTIC_FAMILIES.items():
        if any(word in tactic for tactic in tactics for word in words):
            found.add(family)
    return found or {"other"}


def wjb_verdicts(judge, model):
    rows = {}
    for row in read_jsonl(OUT / "judgments" / judge / model / "direct.jsonl"):
        if row["pid"][0] in "wv" and row["pid"] not in BAD:
            rows[(row["pid"], row["lang"])] = row
    return rows


def tactics_report(args):
    path = DATA / "wjb_prompts.csv"
    if not path.exists():
        raise SystemExit(f"ERROR: {path} missing")
    with open(path, encoding="utf-8") as f:
        prompts = list(csv.DictReader(f))
    family = {p["pid"]: families_of(p["tactic"]) for p in prompts}
    plain_of = {p["pid"]: p["focus"] for p in prompts}
    print(f"WildJailbreak disguised prompts: {len(prompts)}, with tactic labels: "
          f"{sum(1 for p in prompts if p['tactic'].strip())}")
    sizes = defaultdict(int)
    for families in family.values():
        for f in families:
            sizes[f] += 1
    print("prompts per tactic family (a prompt can be in several):")
    for f in sorted(sizes, key=lambda f: -sizes[f]):
        print(f"  {f:26s} {sizes[f]:4d}")
    pooled = defaultdict(lambda: [0, 0, 0])
    per_model = defaultdict(lambda: [0, 0, 0])
    for model in OPEN_MODELS:
        native = wjb_verdicts(args.judge, model)
        if not native:
            print(f"WARNING: no WildJailbreak verdicts for {model}")
        with_mt = wjb_verdicts(args.judge + "+mt", model)
        for (pid, lang), row in native.items():
            if not pid.startswith("w") or row.get("label") != "unsafe":
                continue
            if lang in args.mt_langs:
                verdicts = with_mt
            else:
                verdicts = native
            disguised = verdicts.get((pid, lang))
            plain = verdicts.get((plain_of.get(pid, ""), lang))
            if disguised is None or plain is None:
                continue
            side = "English" if lang == "en" else "Indian"
            for f in family.get(pid, {"other"}):
                for key, table in [((f, side), pooled), ((model, f, side), per_model)]:
                    table[key][0] += is_harmful(plain)
                    table[key][1] += is_harmful(disguised)
                    table[key][2] += 1
    print("\nASR plain -> disguised by tactic family, five open models pooled (unsafe prompts)")
    print(f"{'family':26s} {'English n':>9s} {'plain':>6s} {'disg':>6s} {'x':>5s}   "
          f"{'Indian n':>8s} {'plain':>6s} {'disg':>6s} {'x':>5s}")
    for f in sorted(sizes, key=lambda f: -sizes[f]):
        cells = []
        for side in ["English", "Indian"]:
            plain, disguised, n = pooled[(f, side)]
            if n == 0:
                cells.append(f"{0:9d} {'-':>6s} {'-':>6s} {'-':>5s}")
                continue
            ratio = f"{disguised / plain:.1f}" if plain else "inf"
            cells.append(f"{n:9d} {plain / n:6.2f} {disguised / n:6.2f} {ratio:>5s}")
        print(f"{f:26s} {cells[0]}   {cells[1]}")
    print("\nper model, Indian forms only: ASR plain -> disguised")
    print(f"{'family':26s} " + " ".join(f"{m[:12]:>14s}" for m in OPEN_MODELS))
    for f in sorted(sizes, key=lambda f: -sizes[f]):
        cells = []
        for model in OPEN_MODELS:
            plain, disguised, n = per_model[(model, f, "Indian")]
            cells.append(f"{plain / n:.2f}->{disguised / n:.2f}".rjust(14) if n else "-".rjust(14))
        print(f"{f:26s} " + " ".join(cells))
    print("\nTactic families are keyword groups over WildJailbreak's free-text tactic labels;"
          "\nfamilies overlap, so rows are not independent. x = disguised / plain ASR.")


def run_generate(args):
    records = load_prompts(args.langs)
    records = [r for r in records if r["source"] in args.sources]
    generate(args.model, "direct", records, direct_messages, backend=args.backend, tp=args.tp)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--model", required=True)
    gen.add_argument("--backend", choices=["vllm", "claude"], default="vllm")
    gen.add_argument("--langs", nargs="+", default=CORE_LANGS)
    gen.add_argument("--sources", nargs="+", default=["wjb_adversarial", "wjb_vanilla"])
    gen.add_argument("--tp", type=int, default=1)
    rep = commands.add_parser("report")
    rep.add_argument("--judge", default="Qwen3-32B-FP8")
    tac = commands.add_parser("tactics", help="ASR plain -> disguised per WildJailbreak tactic family")
    tac.add_argument("--judge", default="Qwen3-32B-FP8")
    tac.add_argument("--mt-langs", nargs="*", default=MT_LANGS)
    args = parser.parse_args()
    if args.command == "generate":
        run_generate(args)
    elif args.command == "tactics":
        tactics_report(args)
    else:
        report(args)


if __name__ == "__main__":
    main()

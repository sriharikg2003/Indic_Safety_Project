import argparse

from common import (CORE_LANGS, LANGS, OUT, SOURCE_OF_PREFIX, bad_pids, direct_messages, generate, load_base,
                    load_prompts, read_jsonl)
from exp01_safety_gap import MT_LANGS, is_harmful, is_refusal
from exp03_english_meaning import mcnemar_p

MIN_PAIRED = 20
BASE = load_base()
BAD = bad_pids(BASE)


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
    args = parser.parse_args()
    if args.command == "generate":
        run_generate(args)
    else:
        report(args)


if __name__ == "__main__":
    main()

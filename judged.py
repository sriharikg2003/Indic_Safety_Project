import glob
from pathlib import Path

from common import OUT, SOURCE_OF_PREFIX, bad_pids, read_jsonl

BAD = bad_pids()
EXTRA_JUDGE_FILES = {"severity", "severity_mt", "trace"}  # exp16 / exp17 outputs, not reply verdicts


def read_judged(judge):
    by = {}
    for path in glob.glob(str(OUT / "judgments" / judge / "*" / "*.jsonl")):
        folder = Path(path).parent.name
        if folder.startswith("xstest_human") or Path(path).stem in EXTRA_JUDGE_FILES:
            continue
        for row in read_jsonl(path):
            if row["pid"] in BAD:
                continue
            key = (folder, row["condition"], SOURCE_OF_PREFIX[row["pid"][0]], row["lang"])
            by.setdefault(key, []).append(row)
    return by


def load_judged(judge, mt_langs=()):
    by = read_judged(judge)
    if not mt_langs:
        return by
    by_mt = read_judged(judge + "+mt")
    for key in list(by):
        if key[3] in mt_langs and key[1] != "intent_probe":
            if key in by_mt:
                by[key] = by_mt[key]
            else:
                print(f"WARNING: no {judge}+mt judgments for {key}; keeping native-judge verdicts")
    return by


def xstest_by_pid(by, model, condition, lang):
    rows = {}
    for row in by.get((model, condition, "xstest", lang), []):
        if row["pid"].startswith("p"):
            rows[row["pid"]] = row
    return rows

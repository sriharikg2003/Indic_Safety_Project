import argparse
import csv
import random
from collections import defaultdict

from common import DATA

FIELDS = ["pid", "xstest_id", "type", "label", "focus", "prompt"]


def read_xstest():
    with open(DATA / "raw" / "xstest_prompts.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample_core(rows, per_safe_type, seed):
    by_type = defaultdict(list)
    for row in rows:
        by_type[(row["label"], row["type"])].append(row)
    rng = random.Random(seed)
    picked = []
    safe_types = sorted(t for (label, t) in by_type if label == "safe")
    for t in safe_types:
        picked += rng.sample(by_type[("safe", t)], per_safe_type)
    unsafe_types = sorted(t for (label, t) in by_type if label == "unsafe")
    per_type, extra = divmod(len(picked), len(unsafe_types))
    for i, t in enumerate(unsafe_types):
        n = per_type + (1 if i < extra else 0)
        picked += rng.sample(by_type[("unsafe", t)], n)
    picked.sort(key=lambda row: int(row["id"]))
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-safe-type", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = read_xstest()
    core = sample_core(rows, args.per_safe_type, args.seed)
    chosen = {row["id"] for row in core}
    rest = [row for row in rows if row["id"] not in chosen]
    rest.sort(key=lambda row: int(row["id"]))

    out = DATA / "base_prompts.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, row in enumerate(core + rest):
            writer.writerow({"pid": f"p{i:03d}", "xstest_id": row["id"], "type": row["type"],
                             "label": row["label"], "focus": row["focus"],
                             "prompt": row["prompt"]})
    print(f"wrote {len(core)} core + {len(rest)} extra prompts -> {out}")


if __name__ == "__main__":
    main()

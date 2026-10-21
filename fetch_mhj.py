import argparse
import csv
import io
import json
import os
import random
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from common import DATA, IT2_CODES

PENDING = DATA / "translations_pending"
MHJ_URL = "https://huggingface.co/datasets/ScaleAI/mhj/resolve/main/harmbench_behaviors.csv"
HARMBENCH_URL = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/"
                 "harmbench_behaviors_text_all.csv")
FIELDS = ["pid", "type", "label", "focus", "prompt", "turns", "category", "goal", "tactic", "question_id",
          "n_user_turns", "mhj_row"]


def hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    token_file = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    if not token and token_file.exists():
        token = token_file.read_text().strip()
    return token


def ssl_context():
    try:
        import certifi
        import ssl
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def get(url, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120, context=ssl_context()) as response:
        return response.read().decode("utf-8")


def download_mhj():
    try:
        return get(MHJ_URL)
    except urllib.error.HTTPError as e:
        print(f"anonymous download: HTTP {e.code} (gated dataset) -> trying with a token")
    token = hf_token()
    if not token:
        raise SystemExit("ERROR: no Hugging Face token (set HF_TOKEN or run huggingface-cli login)")
    try:
        return get(MHJ_URL, token)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"ERROR: HTTP {e.code} with token. If 403: accept the dataset terms at "
                         "https://huggingface.co/datasets/ScaleAI/mhj with the token's account, then rerun.")


def parse_message(cell):
    cell = (cell or "").strip()
    if not cell:
        return None
    try:
        message = json.loads(cell)
    except json.JSONDecodeError:
        return ("?", cell)
    if not isinstance(message, dict):
        return ("?", str(message))
    text = message.get("body", message.get("content", message.get("text", "")))
    return (str(message.get("role", "?")).lower(), str(text or "").strip())


def conversations(raw):
    rows = list(csv.DictReader(io.StringIO(raw)))
    if not rows:
        raise SystemExit("ERROR: MHJ csv is empty")
    columns = list(rows[0])
    message_columns = [c for c in columns if re.fullmatch(r"message_\d+", c)]
    message_columns.sort(key=lambda c: int(c.split("_")[1]))
    other_columns = [c for c in columns if c not in message_columns]
    print(f"MHJ: {len(rows)} conversations, columns: {other_columns} + {len(message_columns)} message_k")
    if not message_columns:
        raise SystemExit(f"ERROR: no message_<k> columns in MHJ csv ({columns[:20]}); update conversations()")
    out = []
    odd_roles = Counter()
    for i, row in enumerate(rows):
        messages = []
        for column in message_columns:
            message = parse_message(row[column])
            if message:
                messages.append(message)
        for role, text in messages:
            if role not in ("user", "assistant", "system"):
                odd_roles[role] += 1
        users = [text for role, text in messages if role == "user" and text]
        if not users:
            continue
        out.append({"mhj_row": i, "tactic": (row.get("tactic") or "").strip() or "unknown",
                    "question_id": (row.get("question_id") or "").strip(), "users": users})
    if odd_roles:
        print(f"WARNING: unexpected message roles {dict(odd_roles)} (ignored; only role=user turns are replayed)")
    print(f"  {len(out)} conversations with at least one user turn")
    return out


def harmbench():
    try:
        rows = csv.DictReader(io.StringIO(get(HARMBENCH_URL)))
        return {row["BehaviorID"]: row for row in rows}
    except Exception as e:
        print(f"WARNING: HarmBench behaviour list not downloaded ({e}) -> category/goal left empty")
        return {}


def stratified_sample(convs, n, seed):
    by_tactic = defaultdict(list)
    for conv in convs:
        by_tactic[conv["tactic"]].append(conv)
    rng = random.Random(seed)
    for group in by_tactic.values():
        rng.shuffle(group)
    picked = []
    while len(picked) < n and any(by_tactic.values()):
        for tactic in sorted(by_tactic):
            if by_tactic[tactic] and len(picked) < n:
                picked.append(by_tactic[tactic].pop())
    return picked


def cmd_fetch(args):
    convs = conversations(download_mhj())
    behaviours = harmbench()
    convs = [conv for conv in convs if len(conv["users"]) <= args.max_user_turns]
    print(f"  {len(convs)} with <= {args.max_user_turns} user turns (context-length cap)")
    picked = stratified_sample(convs, args.n, args.seed)
    rows = []
    unmatched = 0
    for k, conv in enumerate(picked):
        behaviour = behaviours.get(conv["question_id"], {})
        if not behaviour:
            unmatched += 1
        rows.append({"pid": f"m{k:03d}", "type": conv["tactic"], "label": "unsafe", "focus": conv["question_id"],
                     "prompt": conv["users"][-1], "turns": json.dumps(conv["users"][:-1], ensure_ascii=False),
                     "category": behaviour.get("SemanticCategory", ""), "goal": behaviour.get("Behavior", ""),
                     "tactic": conv["tactic"], "question_id": conv["question_id"],
                     "n_user_turns": len(conv["users"]), "mhj_row": conv["mhj_row"]})
    if unmatched:
        print(f"WARNING: {unmatched}/{len(rows)} question_ids not found in HarmBench -> no goal; the judge then "
              "sees only the final turn as the intended meaning")
    with open(DATA / "mhj_prompts.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (DATA / "mhj_pids.txt").write_text("".join(row["pid"] + "\n" for row in rows))
    print(f"wrote {len(rows)} -> data/mhj_prompts.csv, data/mhj_pids.txt")
    print("  tactics:", dict(Counter(row["type"] for row in rows)))
    print("  categories:", dict(Counter(row["category"] for row in rows)))
    turn_counts = dict(sorted(Counter(row["n_user_turns"] for row in rows).items()))
    print("  user turns: total", sum(row["n_user_turns"] for row in rows), "per conversation", turn_counts)


def all_turns():
    with open(DATA / "mhj_prompts.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    turns = []
    for row in rows:
        for k, text in enumerate(json.loads(row["turns"]) + [row["prompt"]]):
            turns.append((row["pid"], k, text))
    return turns


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_prepare(args):
    PENDING.mkdir(exist_ok=True)
    out = PENDING / "mhj_turns_en.jsonl"
    turns = all_turns()
    write_jsonl(out, [{"pid": pid, "turn": k, "lang": "en", "text": text} for pid, k, text in turns])
    print(f"wrote {out} ({len(turns)} turns)")


def cmd_translate(args):
    from translate_indictrans import IndicProcessor, load, translate_long

    with open(PENDING / "mhj_turns_en.jsonl", encoding="utf-8") as f:
        source = [json.loads(line) for line in f]
    tokenizer, model = load("en-indic")
    ip = IndicProcessor(inference=True)
    for lang in args.langs:
        if lang not in IT2_CODES:
            print(f"ERROR: {lang} not supported by IndicTrans2 -> skipped")
            continue
        hyp = translate_long([row["text"] for row in source], "eng_Latn", IT2_CODES[lang], tokenizer, model, ip)
        empty = sum(not h.strip() for h in hyp)
        if empty:
            print(f"WARNING: {lang}: {empty} empty translations (those conversations are skipped by the "
                  "multi-turn experiment)")
        rows = []
        for row, text in zip(source, hyp):
            rows.append({"pid": row["pid"], "turn": row["turn"], "lang": lang, "text": text, "en": row["text"],
                         "translator": "IndicTrans2 en-indic-1B"})
        out = PENDING / f"mhj_turns_{lang}.jsonl"
        write_jsonl(out, rows)
        print(f"{lang}: {len(hyp)} turns -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["fetch", "prepare", "translate"])
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-user-turns", type=int, default=8)
    parser.add_argument("--langs", nargs="+", default=["hi", "kn"])
    args = parser.parse_args()
    if args.mode == "fetch":
        cmd_fetch(args)
    elif args.mode == "prepare":
        cmd_prepare(args)
    else:
        cmd_translate(args)


if __name__ == "__main__":
    main()

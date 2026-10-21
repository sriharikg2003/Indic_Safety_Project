import argparse
import glob
import os
import re
from collections import defaultdict
from pathlib import Path

from common import DATA, IT2_CODES, OUT, append_jsonl, done_keys, load_base, read_jsonl

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    print("note: IndicTransToolkit.processor not found, using the older IndicTransToolkit layout")
    from IndicTransToolkit import IndicProcessor

SCRIPT_RANGES = {
    "Devanagari": (0x0900, 0x097F), "Bengali": (0x0980, 0x09FF), "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF), "Odia": (0x0B00, 0x0B7F), "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F), "Kannada": (0x0C80, 0x0CFF), "Malayalam": (0x0D00, 0x0D7F),
    "Ol Chiki": (0x1C50, 0x1C7F), "Perso-Arabic": (0x0600, 0x06FF),
}
SCRIPT_DEFAULT = {
    "Devanagari": "hin_Deva", "Bengali": "ben_Beng", "Gurmukhi": "pan_Guru",
    "Gujarati": "guj_Gujr", "Odia": "ory_Orya", "Tamil": "tam_Taml", "Telugu": "tel_Telu",
    "Kannada": "kan_Knda", "Malayalam": "mal_Mlym", "Ol Chiki": "sat_Olck",
    "Perso-Arabic": "urd_Arab",
}
SCRIPT_OF_SUFFIX = {
    "Deva": "Devanagari", "Beng": "Bengali", "Guru": "Gurmukhi", "Gujr": "Gujarati",
    "Orya": "Odia", "Taml": "Tamil", "Telu": "Telugu", "Knda": "Kannada",
    "Mlym": "Malayalam", "Olck": "Ol Chiki", "Arab": "Perso-Arabic",
}
LANG_SCRIPT = {}
for it2_code in IT2_CODES.values():
    LANG_SCRIPT[it2_code] = SCRIPT_OF_SUFFIX[it2_code.split("_")[1]]
REPLY_SRC = dict(IT2_CODES, bho="hin_Deva", tcy="kan_Knda")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("IT2_MEM", 0.25)))


def load(direction):
    name = f"ai4bharat/indictrans2-{direction}-1B"
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if DEVICE == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True, torch_dtype=dtype)
    return tokenizer, model.to(DEVICE).eval()


def generate_batch(sents, src, tgt, tokenizer, model, ip, beams):
    batch = ip.preprocess_batch(sents, src_lang=src, tgt_lang=tgt)
    encoded = tokenizer(batch, truncation=True, max_length=256, padding="longest",
                        return_tensors="pt", return_attention_mask=True).to(DEVICE)
    with torch.no_grad():
        generated = model.generate(**encoded, use_cache=True, min_length=0, max_length=256,
                                   num_beams=beams, num_return_sequences=1)
    decoded = tokenizer.batch_decode(generated.detach().cpu().tolist(), skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)
    return ip.postprocess_batch(decoded, lang=tgt)


def translate(sents, src, tgt, tokenizer, model, ip, bs=32, beams=5):
    order = sorted(range(len(sents)), key=lambda k: len(sents[k]))
    out = [None] * len(sents)
    i = 0
    while i < len(order):
        idx = order[i:i + bs]
        try:
            hyp = generate_batch([sents[k] for k in idx], src, tgt, tokenizer, model, ip, beams)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"  GPU out of memory -> batch size {bs}")
            continue
        for k, h in zip(idx, hyp):
            out[k] = h
        i += len(idx)
        print(f"\r  {src}->{tgt}: {i}/{len(sents)} segments", end="", flush=True)
    print()
    return out


def dominant_script(text):
    counts = {}
    for script, (lo, hi) in SCRIPT_RANGES.items():
        counts[script] = sum(lo <= ord(c) <= hi for c in text)
    best = max(counts, key=counts.get)
    letters = sum(c.isalpha() for c in text)
    if letters and counts[best] / letters > 0.3:
        return best
    return None


def split_segments(text):
    parts = re.split(r"(\n+|(?<=[।॥.!?؟۔])\s+)", text)
    return [p for p in parts if p]


def translate_long(texts, src, tgt, tokenizer, model, ip, beams=5, bs=32):
    segments = [split_segments(t) for t in texts]
    jobs = defaultdict(list)
    for i, pieces in enumerate(segments):
        for j, piece in enumerate(pieces):
            if not piece.strip():
                continue
            if src == "eng_Latn":
                jobs[src].append((i, j))
                continue
            script = dominant_script(piece)
            if script is None:
                continue
            if src and LANG_SCRIPT.get(src) == script:
                code = src
            else:
                code = SCRIPT_DEFAULT[script]
            jobs[code].append((i, j))
    for code, items in jobs.items():
        sents = [segments[i][j].strip() for i, j in items]
        hyp = translate(sents, code, tgt, tokenizer, model, ip, bs=bs, beams=beams)
        for (i, j), h in zip(items, hyp):
            segments[i][j] = h
    return ["".join(pieces).strip() for pieces in segments]


def cmd_translate(args, ip):
    tokenizer, model = load("en-indic")
    base = load_base()
    for lang in args.langs or list(IT2_CODES):
        path = DATA / "translations" / f"{lang}.jsonl"
        have = {row["pid"] for row in read_jsonl(path)}
        todo = [pid for pid in base if pid not in have]
        if not todo:
            print(f"{lang}: complete")
            continue
        texts = [base[pid]["prompt"] for pid in todo]
        hyp = translate_long(texts, "eng_Latn", IT2_CODES[lang], tokenizer, model, ip)
        rows = []
        for pid, text in zip(todo, hyp):
            rows.append({"pid": pid, "lang": lang, "text": text, "ambiguity_preserved": None,
                         "confidence": "mt", "note": "IndicTrans2 en-indic-1B"})
        append_jsonl(path, rows)
        print(f"{lang}: {len(todo)} new -> {path}")


def cmd_responses(args, ip):
    tokenizer, model = load("indic-en")
    files = set()
    for pattern in args.gen:
        files.update(glob.glob(pattern))
    for f in sorted(files):
        if Path(f).parent.name.startswith("xstest_human"):
            continue
        out = OUT / "response_translations" / Path(f).parent.name / Path(f).name
        done = done_keys(out)
        records = []
        for r in read_jsonl(f):
            if r["lang"] == "en" or (r["pid"], r["lang"]) in done:
                continue
            if args.langs and r["lang"] not in args.langs:
                continue
            records.append(r)
        by_src = defaultdict(list)
        for r in records:
            by_src[REPLY_SRC.get(r["lang"])].append(r)
        for src, rs in by_src.items():
            hyp = translate_long([r["response"] for r in rs], src, "eng_Latn", tokenizer, model, ip,
                                 beams=1, bs=64)
            rows = []
            for r, h in zip(rs, hyp):
                rows.append({"pid": r["pid"], "lang": r["lang"], "model": r["model"],
                             "condition": r["condition"], "response_en": h})
            append_jsonl(out, rows)
        if records:
            print(f"{f}: {len(records)} replies translated -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["translate", "responses"])
    parser.add_argument("--langs", nargs="+")
    parser.add_argument("--gen", nargs="+",
                        default=[str(OUT / "generations" / "*" / "direct.jsonl"),
                                 str(OUT / "generations" / "*" / "self_translate.jsonl")])
    args = parser.parse_args()
    ip = IndicProcessor(inference=True)
    if args.mode == "translate":
        cmd_translate(args, ip)
    else:
        cmd_responses(args, ip)


if __name__ == "__main__":
    main()
